from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.centralized_attention_mtsp import CentralizedAttentionMTSP
from training.imitation_train import cpu_state_dict, set_seed
from training.mixed_imitation_train import (
    evaluate_decoded_mixed,
    find_dataset_files,
    make_loaders,
    teacher_forced_loss,
)


def closed_route_makespan(
    depots: Tensor,
    tasks: Tensor,
    actions: Tensor,
) -> Tensor:
    """
    Generic closed-route MinMax cost.

    Works for variable:
        batch size
        number of agents
        number of tasks
    """
    batch_size, num_agents, _ = depots.shape
    device = depots.device

    current_positions = depots.clone()

    route_lengths = torch.zeros(
        batch_size,
        num_agents,
        device=device,
    )

    batch_indices = torch.arange(
        batch_size,
        device=device,
    )

    for step in range(actions.shape[1]):
        agent_ids = actions[:, step, 0].long()
        task_ids = actions[:, step, 1].long()

        task_positions = tasks[
            batch_indices,
            task_ids,
        ]

        previous_positions = current_positions[
            batch_indices,
            agent_ids,
        ]

        travel_distance = torch.linalg.vector_norm(
            task_positions - previous_positions,
            dim=-1,
        )

        agent_one_hot = torch.nn.functional.one_hot(
            agent_ids,
            num_classes=num_agents,
        ).float()

        route_lengths = (
            route_lengths
            + agent_one_hot * travel_distance.unsqueeze(1)
        )

        current_positions = (
            current_positions
            * (1.0 - agent_one_hot.unsqueeze(-1))
            + task_positions.unsqueeze(1)
            * agent_one_hot.unsqueeze(-1)
        )

    return_distance = torch.linalg.vector_norm(
        current_positions - depots,
        dim=-1,
    )

    closed_lengths = route_lengths + return_distance

    return closed_lengths.max(dim=1).values


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CentralizedAttentionMTSP, dict[str, int | float]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_config = checkpoint["model_config"]

    model = CentralizedAttentionMTSP(
        **model_config,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])

    return model, model_config


def save_checkpoint(
    path: Path,
    model: CentralizedAttentionMTSP,
    model_config: dict[str, int | float],
    epoch: int,
    metrics: dict[str, float],
    init_checkpoint: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_config": model_config,
            "state_dict": cpu_state_dict(model),
            "metrics": metrics,
            "init_checkpoint": str(init_checkpoint),
        },
        path,
    )


def imitation_weight(
    epoch: int,
    initial_weight: float,
    decay_epochs: int,
) -> float:
    if initial_weight <= 0.0 or decay_epochs <= 0:
        return 0.0

    if epoch >= decay_epochs:
        return 0.0

    return initial_weight * (
        1.0 - epoch / float(decay_epochs)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed-shape self-critical RL for variable-agent mTSP."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/variable_mtsp"),
    )

    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/mixed_imitation_variable_mtsp/"
            "mixed_imitation_best_decode.pt"
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="mixed_rl_variable_mtsp",
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--imitation-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--imitation-decay-epochs",
        type=int,
        default=15,
    )

    parser.add_argument("--decode-every", type=int, default=5)

    parser.add_argument(
        "--decode-instances-per-shape",
        type=int,
        default=50,
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    if not args.init_checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.init_checkpoint}"
        )

    set_seed(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pin_memory = device.type == "cuda"

    train_specs = find_dataset_files(args.data_dir, "train")
    val_specs = find_dataset_files(args.data_dir, "val")

    train_loaders = make_loaders(
        specs=train_specs,
        batch_size=args.batch_size,
        augment_labels=True,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loaders = make_loaders(
        specs=val_specs,
        batch_size=args.batch_size,
        augment_labels=False,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model, model_config = load_model(
        checkpoint_path=args.init_checkpoint,
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path("checkpoints") / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = checkpoint_dir / "mixed_rl_metrics.csv"

    shape_keys = sorted(train_loaders.keys())

    best_overall_gap = float("inf")

    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Initial checkpoint: {args.init_checkpoint}")
    print(f"Run name: {args.run_name}")
    print("Training shapes:")
    for key in shape_keys:
        print(
            f"  {key[0]} agents, {key[1]} tasks: "
            f"{len(train_loaders[key].dataset)} train instances"
        )

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "epoch",
            "learning_rate",
            "imitation_weight",
            "sampled_makespan_m",
            "greedy_baseline_makespan_m",
            "sample_beats_baseline_fraction",
            "policy_loss",
            "imitation_loss",
            "entropy",
            "total_loss",
            "overall_gap_pct",
            "overall_decoded_makespan_m",
            "overall_expert_makespan_m",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            # Use eval mode to disable dropout during sampled-vs-greedy comparison.
            # Gradients still work.
            model.eval()

            current_imitation_weight = imitation_weight(
                epoch=epoch - 1,
                initial_weight=args.imitation_weight,
                decay_epochs=args.imitation_decay_epochs,
            )

            sampled_cost_sum = 0.0
            baseline_cost_sum = 0.0
            win_count = 0
            instance_count = 0

            policy_loss_sum = 0.0
            imitation_loss_sum = 0.0
            entropy_sum = 0.0
            total_loss_sum = 0.0

            epoch_shape_keys = shape_keys.copy()
            random.shuffle(epoch_shape_keys)

            for shape_key in epoch_shape_keys:
                loader = train_loaders[shape_key]

                for batch in loader:
                    depots = batch["depots"].to(
                        device,
                        non_blocking=True,
                    )

                    tasks = batch["tasks"].to(
                        device,
                        non_blocking=True,
                    )

                    with torch.no_grad():
                        greedy_actions = model.decode(
                            depots=depots,
                            tasks=tasks,
                        )

                        greedy_costs = closed_route_makespan(
                            depots=depots,
                            tasks=tasks,
                            actions=greedy_actions,
                        )

                    sampled_actions, log_probs, entropies = model.sample(
                        depots=depots,
                        tasks=tasks,
                    )

                    with torch.no_grad():
                        sampled_costs = closed_route_makespan(
                            depots=depots,
                            tasks=tasks,
                            actions=sampled_actions,
                        )

                        raw_advantages = sampled_costs - greedy_costs

                        advantage_scale = raw_advantages.std(
                            unbiased=False
                        ).clamp_min(1.0)

                        advantages = raw_advantages / advantage_scale

                    sequence_log_probs = log_probs.sum(dim=1)

                    policy_loss = (
                        advantages.detach() * sequence_log_probs
                    ).mean()

                    entropy = entropies.mean()

                    bc_loss, _ = teacher_forced_loss(
                        model=model,
                        batch=batch,
                        device=device,
                    )

                    total_loss = (
                        policy_loss
                        - args.entropy_coef * entropy
                        + current_imitation_weight * bc_loss
                    )

                    optimizer.zero_grad(set_to_none=True)

                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    optimizer.step()

                    batch_size = tasks.shape[0]

                    sampled_cost_sum += (
                        sampled_costs.mean().item() * batch_size
                    )

                    baseline_cost_sum += (
                        greedy_costs.mean().item() * batch_size
                    )

                    win_count += int(
                        (sampled_costs < greedy_costs).sum().item()
                    )

                    instance_count += batch_size

                    policy_loss_sum += policy_loss.item() * batch_size
                    imitation_loss_sum += bc_loss.item() * batch_size
                    entropy_sum += entropy.item() * batch_size
                    total_loss_sum += total_loss.item() * batch_size

            row: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "imitation_weight": current_imitation_weight,
                "sampled_makespan_m": (
                    sampled_cost_sum / instance_count
                ),
                "greedy_baseline_makespan_m": (
                    baseline_cost_sum / instance_count
                ),
                "sample_beats_baseline_fraction": (
                    win_count / instance_count
                ),
                "policy_loss": policy_loss_sum / instance_count,
                "imitation_loss": imitation_loss_sum / instance_count,
                "entropy": entropy_sum / instance_count,
                "total_loss": total_loss_sum / instance_count,
                "overall_gap_pct": float("nan"),
                "overall_decoded_makespan_m": float("nan"),
                "overall_expert_makespan_m": float("nan"),
            }

            should_decode = (
                epoch % args.decode_every == 0
                or epoch == args.epochs
            )

            if should_decode:
                decode_metrics = evaluate_decoded_mixed(
                    model=model,
                    loaders=val_loaders,
                    device=device,
                    max_instances_per_shape=(
                        args.decode_instances_per_shape
                    ),
                )

                row["overall_gap_pct"] = decode_metrics[
                    "overall_gap_pct"
                ]

                row["overall_decoded_makespan_m"] = decode_metrics[
                    "overall_decoded_makespan_m"
                ]

                row["overall_expert_makespan_m"] = decode_metrics[
                    "overall_expert_makespan_m"
                ]

                if decode_metrics["overall_gap_pct"] < best_overall_gap:
                    best_overall_gap = decode_metrics[
                        "overall_gap_pct"
                    ]

                    save_checkpoint(
                        checkpoint_dir / "mixed_rl_best_decode.pt",
                        model=model,
                        model_config=model_config,
                        epoch=epoch,
                        metrics=decode_metrics,
                        init_checkpoint=args.init_checkpoint,
                    )

            writer.writerow(row)
            file.flush()

            message = (
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"sample={row['sampled_makespan_m']:.2f} m | "
                f"greedy baseline="
                f"{row['greedy_baseline_makespan_m']:.2f} m | "
                f"wins="
                f"{100.0 * row['sample_beats_baseline_fraction']:.1f}%"
            )

            if should_decode:
                message += (
                    f" | decoded gap="
                    f"{row['overall_gap_pct']:+.2f}%"
                )

            print(message)

    save_checkpoint(
        checkpoint_dir / "mixed_rl_last.pt",
        model=model,
        model_config=model_config,
        epoch=args.epochs,
        metrics={
            "best_overall_gap_pct": best_overall_gap,
        },
        init_checkpoint=args.init_checkpoint,
    )

    print()
    print("Mixed RL fine-tuning complete.")
    print(
        "Best checkpoint: "
        f"{checkpoint_dir / 'mixed_rl_best_decode.pt'}"
    )
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
