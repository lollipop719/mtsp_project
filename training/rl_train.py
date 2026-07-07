from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.centralized_attention_mtsp import (
    CentralizedAttentionMTSP,
)
from training.imitation_train import (
    MTSPDataset,
    evaluate_greedy_decode,
    set_seed,
    teacher_forced_loss,
)


def closed_route_makespan(
    depots: Tensor,
    tasks: Tensor,
    actions: Tensor,
) -> Tensor:
    """
    Return one closed-route MinMax cost per batch instance.

    Every drone starts at its own depot, visits its assigned tasks,
    and returns to that same depot.
    """
    batch_size, num_drones, _ = depots.shape
    device = depots.device

    current_positions = depots.clone()

    route_lengths = torch.zeros(
        batch_size,
        num_drones,
        device=device,
    )

    batch_indices = torch.arange(
        batch_size,
        device=device,
    )

    for step in range(actions.shape[1]):
        drone_ids = actions[:, step, 0].long()
        task_ids = actions[:, step, 1].long()

        task_positions = tasks[
            batch_indices,
            task_ids,
        ]

        previous_positions = current_positions[
            batch_indices,
            drone_ids,
        ]

        travel_distance = torch.linalg.vector_norm(
            task_positions - previous_positions,
            dim=-1,
        )

        drone_one_hot = torch.nn.functional.one_hot(
            drone_ids,
            num_classes=num_drones,
        ).float()

        route_lengths = (
            route_lengths
            + drone_one_hot * travel_distance.unsqueeze(1)
        )

        current_positions = (
            current_positions
            * (1.0 - drone_one_hot.unsqueeze(-1))
            + task_positions.unsqueeze(1)
            * drone_one_hot.unsqueeze(-1)
        )

    return_distance = torch.linalg.vector_norm(
        current_positions - depots,
        dim=-1,
    )

    closed_route_lengths = route_lengths + return_distance

    return closed_route_lengths.max(dim=1).values


def cpu_state_dict(
    model: torch.nn.Module,
) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


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
        1.0 - (epoch / float(decay_epochs))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-critical RL fine-tuning for centralized mTSP."
    )

    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/imitation_15task_3drone/"
            "centralized_attention_best_decode.pt"
        ),
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path(
            "data/mtsp_3drones_15tasks_train.npz"
        ),
    )

    parser.add_argument(
        "--val",
        type=Path,
        default=Path(
            "data/mtsp_3drones_15tasks_val.npz"
        ),
    )

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--imitation-weight",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--imitation-decay-epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--decode-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--decode-instances",
        type=int,
        default=250,
    )


    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument(
        "--run-name",
        type=str,
        default="rl_15task_3drone",
    )

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

    train_dataset = MTSPDataset(
        args.train,
        augment_labels=True,
    )

    val_dataset = MTSPDataset(
        args.val,
        augment_labels=False,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model, model_config = load_model(
        args.init_checkpoint,
        device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path("checkpoints") / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = checkpoint_dir / "rl_training_metrics.csv"

    best_decoded_makespan = float("inf")

    print(f"Device: {device}")
    print(f"Initial checkpoint: {args.init_checkpoint}")
    print(f"Training instances: {len(train_dataset)}")
    print(f"Validation instances: {len(val_dataset)}")

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "epoch",
            "learning_rate",
            "imitation_weight",
            "mean_sampled_makespan_m",
            "mean_greedy_baseline_makespan_m",
            "sample_beats_baseline_fraction",
            "policy_loss",
            "imitation_loss",
            "entropy",
            "total_loss",
            "decoded_makespan_m",
            "decoded_total_distance_m",
            "expert_makespan_m",
            "expert_total_distance_m",
            "makespan_gap_pct",
            "total_distance_gap_pct",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            # Eval mode disables dropout but gradients still work.
            # It makes sampled-policy and greedy-baseline comparison stable.
            model.eval()

            sampled_cost_sum = 0.0
            baseline_cost_sum = 0.0
            win_count = 0
            instance_count = 0

            policy_loss_sum = 0.0
            imitation_loss_sum = 0.0
            entropy_sum = 0.0
            total_loss_sum = 0.0

            current_imitation_weight = imitation_weight(
                epoch=epoch - 1,
                initial_weight=args.imitation_weight,
                decay_epochs=args.imitation_decay_epochs,
            )

            for batch in train_loader:
                depots = batch["depots"].to(
                    device,
                    non_blocking=True,
                )

                tasks = batch["tasks"].to(
                    device,
                    non_blocking=True,
                )

                # Greedy route is the self-critical baseline.
                with torch.no_grad():
                    greedy_actions = model.decode(
                        depots=depots,
                        tasks=tasks,
                    )

                    greedy_costs = closed_route_makespan(
                        depots,
                        tasks,
                        greedy_actions,
                    )

                sampled_actions, log_probs, entropies = model.sample(
                    depots=depots,
                    tasks=tasks,
                )

                with torch.no_grad():
                    sampled_costs = closed_route_makespan(
                        depots,
                        tasks,
                        sampled_actions,
                    )

                    raw_advantages = (
                        sampled_costs - greedy_costs
                    )

                    # Keep sign relative to baseline, but stabilize scale.
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

                policy_loss_sum += (
                    policy_loss.item() * batch_size
                )

                imitation_loss_sum += (
                    bc_loss.item() * batch_size
                )

                entropy_sum += entropy.item() * batch_size
                total_loss_sum += total_loss.item() * batch_size

            row: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "imitation_weight": current_imitation_weight,
                "mean_sampled_makespan_m": (
                    sampled_cost_sum / instance_count
                ),
                "mean_greedy_baseline_makespan_m": (
                    baseline_cost_sum / instance_count
                ),
                "sample_beats_baseline_fraction": (
                    win_count / instance_count
                ),
                "policy_loss": policy_loss_sum / instance_count,
                "imitation_loss": imitation_loss_sum / instance_count,
                "entropy": entropy_sum / instance_count,
                "total_loss": total_loss_sum / instance_count,
                "decoded_makespan_m": float("nan"),
                "decoded_total_distance_m": float("nan"),
                "expert_makespan_m": float("nan"),
                "expert_total_distance_m": float("nan"),
                "makespan_gap_pct": float("nan"),
                "total_distance_gap_pct": float("nan"),
            }

            should_decode = (
                epoch % args.decode_every == 0
                or epoch == args.epochs
            )

            if should_decode:
                decoded_metrics = evaluate_greedy_decode(
                    model=model,
                    loader=val_loader,
                    device=device,
                    max_instances=args.decode_instances,
                )

                row.update(decoded_metrics)

                if (
                    decoded_metrics["decoded_makespan_m"]
                    < best_decoded_makespan
                ):
                    best_decoded_makespan = (
                        decoded_metrics["decoded_makespan_m"]
                    )

                    save_checkpoint(
                        checkpoint_dir
                        / "centralized_attention_rl_best_decode.pt",
                        model=model,
                        model_config=model_config,
                        epoch=epoch,
                        metrics=decoded_metrics,
                        init_checkpoint=args.init_checkpoint,
                    )

            writer.writerow(row)
            file.flush()

            message = (
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"sample={row['mean_sampled_makespan_m']:.2f} m | "
                f"greedy baseline="
                f"{row['mean_greedy_baseline_makespan_m']:.2f} m | "
                f"wins={100.0 * row['sample_beats_baseline_fraction']:.1f}%"
            )

            if should_decode:
                message += (
                    f" | decoded="
                    f"{row['decoded_makespan_m']:.2f} m "
                    f"({row['makespan_gap_pct']:+.2f}% vs OR-Tools)"
                )

            print(message)

    save_checkpoint(
        checkpoint_dir / "centralized_attention_rl_last.pt",
        model=model,
        model_config=model_config,
        epoch=args.epochs,
        metrics={
            "best_decoded_makespan_m": best_decoded_makespan,
        },
        init_checkpoint=args.init_checkpoint,
    )

    print("\nRL fine-tuning complete.")
    print(
        "Best checkpoint: "
        "checkpoints/rl_15task_3drone/"
        "centralized_attention_rl_best_decode.pt"
    )


if __name__ == "__main__":
    main()