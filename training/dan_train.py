from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from env.dan_async_mtsp_env import (
    DANAsyncMTSPEnv,
    DANMTSPInstance,
    DANPolicyOutput,
    nearest_task_policy,
)
from models.dan_mtsp import (
    DANMTSPPolicy,
    DANPolicyRunner,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def parse_int_range(text: str) -> tuple[int, int]:
    if ":" in text:
        start_text, end_text = text.split(":", maxsplit=1)
        return int(start_text), int(end_text)

    value = int(text)
    return value, value


def sample_problem_size(
    *,
    agent_range: tuple[int, int],
    task_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[int, int]:
    min_agents, max_agents = agent_range
    min_tasks, max_tasks = task_range

    num_agents = int(
        rng.integers(min_agents, max_agents + 1)
    )

    num_tasks = int(
        rng.integers(min_tasks, max_tasks + 1)
    )

    return num_agents, num_tasks


def rollout_sampled(
    *,
    model: DANMTSPPolicy,
    device: torch.device,
    instance: DANMTSPInstance,
    num_agents: int,
    num_tasks: int,
    delta_g: float,
) -> tuple[float, Tensor, Tensor, int]:
    env = DANAsyncMTSPEnv(
        num_agents=num_agents,
        num_tasks=num_tasks,
        delta_g=delta_g,
    )

    policy = DANPolicyRunner(
        model=model,
        device=device,
        mode="sample",
    )

    result = env.rollout(
        policy,
        instance=instance,
    )

    if len(result.log_probs) == 0:
        raise RuntimeError("Sampled rollout produced no log_probs.")

    log_prob_sum = torch.stack(result.log_probs).sum()

    if len(result.entropies) > 0:
        entropy_sum = torch.stack(result.entropies).sum()
    else:
        entropy_sum = torch.zeros(
            (),
            device=device,
        )

    return (
        float(result.makespan),
        log_prob_sum,
        entropy_sum,
        int(len(result.decisions)),
    )


@torch.no_grad()
def rollout_greedy(
    *,
    model: DANMTSPPolicy,
    device: torch.device,
    instance: DANMTSPInstance,
    num_agents: int,
    num_tasks: int,
    delta_g: float,
) -> float:
    env = DANAsyncMTSPEnv(
        num_agents=num_agents,
        num_tasks=num_tasks,
        delta_g=delta_g,
    )

    policy = DANPolicyRunner(
        model=model,
        device=device,
        mode="greedy",
    )

    result = env.rollout(
        policy,
        instance=instance,
    )

    return float(result.makespan)


def rollout_nearest(
    *,
    instance: DANMTSPInstance,
    num_agents: int,
    num_tasks: int,
    delta_g: float,
) -> float:
    env = DANAsyncMTSPEnv(
        num_agents=num_agents,
        num_tasks=num_tasks,
        delta_g=delta_g,
    )

    result = env.rollout(
        nearest_task_policy,
        instance=instance,
    )

    return float(result.makespan)


@torch.no_grad()
def evaluate_dan_greedy(
    *,
    model: DANMTSPPolicy,
    device: torch.device,
    settings: list[tuple[int, int]],
    instances_per_setting: int,
    delta_g: float,
    seed: int,
) -> dict[str, float]:
    model.eval()

    rng = np.random.default_rng(seed)

    metrics: dict[str, float] = {}

    all_dan_costs: list[float] = []
    all_nearest_costs: list[float] = []
    all_gaps_to_nearest: list[float] = []

    for num_agents, num_tasks in settings:
        dan_costs: list[float] = []
        nearest_costs: list[float] = []
        gaps: list[float] = []

        for _ in range(instances_per_setting):
            env = DANAsyncMTSPEnv(
                num_agents=num_agents,
                num_tasks=num_tasks,
                delta_g=delta_g,
            )

            instance = env.generate_instance(
                seed=int(rng.integers(0, 2**31 - 1))
            )

            dan_cost = rollout_greedy(
                model=model,
                device=device,
                instance=instance,
                num_agents=num_agents,
                num_tasks=num_tasks,
                delta_g=delta_g,
            )

            nearest_cost = rollout_nearest(
                instance=instance,
                num_agents=num_agents,
                num_tasks=num_tasks,
                delta_g=delta_g,
            )

            gap_to_nearest = (
                100.0 * (dan_cost - nearest_cost) / nearest_cost
                if nearest_cost > 0.0
                else float("nan")
            )

            dan_costs.append(dan_cost)
            nearest_costs.append(nearest_cost)
            gaps.append(gap_to_nearest)

            all_dan_costs.append(dan_cost)
            all_nearest_costs.append(nearest_cost)
            all_gaps_to_nearest.append(gap_to_nearest)

        key = f"{num_agents}a_{num_tasks}t"

        metrics[f"{key}_dan_greedy_makespan"] = float(
            np.mean(dan_costs)
        )

        metrics[f"{key}_nearest_makespan"] = float(
            np.mean(nearest_costs)
        )

        metrics[f"{key}_gap_to_nearest_pct"] = float(
            np.mean(gaps)
        )

    metrics["overall_dan_greedy_makespan"] = float(
        np.mean(all_dan_costs)
    )

    metrics["overall_nearest_makespan"] = float(
        np.mean(all_nearest_costs)
    )

    metrics["overall_gap_to_nearest_pct"] = float(
        np.mean(all_gaps_to_nearest)
    )

    return metrics


def parse_eval_settings(text: str) -> list[tuple[int, int]]:
    settings: list[tuple[int, int]] = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        if "x" not in item:
            raise ValueError(
                "Eval settings must look like '2x10,3x15'."
            )

        agents_text, tasks_text = item.split("x", maxsplit=1)
        settings.append(
            (int(agents_text), int(tasks_text))
        )

    return settings


def save_checkpoint(
    path: Path,
    *,
    model: DANMTSPPolicy,
    optimizer: Adam,
    step: int,
    config: dict,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "model_config": {
                "embedding_dim": model.embedding_dim,
                "num_heads": model.num_heads,
                "num_encoder_layers": model.num_encoder_layers,
                "dropout": model.dropout,
                "tanh_clipping": model.tanh_clipping,
            },
            "state_dict": cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_config": config,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DAN with REINFORCE and greedy rollout baseline."
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="dan_small",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to load model weights from before training.",
    )

    parser.add_argument(
        "--agent-range",
        type=str,
        default="2:5",
        help="Inclusive range, e.g. 2:5 or fixed value like 3.",
    )

    parser.add_argument(
        "--task-range",
        type=str,
        default="10:25",
        help="Inclusive range, e.g. 10:25 or fixed value like 20.",
    )

    parser.add_argument(
        "--delta-g",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.96,
    )

    parser.add_argument(
        "--lr-decay-every",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--advantage-normalization",
        action="store_true",
        help="Normalize advantages inside each update batch.",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tanh-clipping",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--eval-settings",
        type=str,
        default="2x10,3x15,5x20",
    )

    parser.add_argument(
        "--eval-instances",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    agent_range = parse_int_range(args.agent_range)
    task_range = parse_int_range(args.task_range)
    eval_settings = parse_eval_settings(args.eval_settings)

    rng = np.random.default_rng(args.seed)

    model = DANMTSPPolicy(
        embedding_dim=args.embedding_dim,
        num_heads=args.heads,
        num_encoder_layers=args.layers,
        dropout=args.dropout,
        tanh_clipping=args.tanh_clipping,
    ).to(device)

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        saved_model_config = checkpoint.get("model_config", {})
        current_model_config = {
            "embedding_dim": args.embedding_dim,
            "num_heads": args.heads,
            "num_encoder_layers": args.layers,
            "dropout": args.dropout,
            "tanh_clipping": args.tanh_clipping,
        }

        if saved_model_config != current_model_config:
            print("WARNING: checkpoint model_config differs from current args.")
            print(f"  checkpoint: {saved_model_config}")
            print(f"  current:    {current_model_config}")

        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded model weights from: {args.resume}")

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    checkpoint_dir = Path("checkpoints") / args.run_name
    output_dir = Path("outputs") / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "dan_train_metrics.csv"
    config_path = output_dir / "dan_train_config.json"

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["device"] = str(device)

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    best_eval_gap = float("inf")

    print(f"Device: {device}")
    print(f"Run name: {args.run_name}")
    print(f"Agent range: {agent_range}")
    print(f"Task range: {task_range}")
    print(f"Episodes/update: {args.episodes_per_update}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Metrics CSV: {metrics_csv}")
    print()

    with metrics_csv.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "step",
            "learning_rate",
            "train_sampled_makespan",
            "train_greedy_baseline_makespan",
            "train_advantage",
            "train_sample_beats_baseline_fraction",
            "train_policy_loss",
            "train_entropy",
            "train_total_loss",
            "overall_dan_greedy_makespan",
            "overall_nearest_makespan",
            "overall_gap_to_nearest_pct",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        started_at = time.perf_counter()

        for step in range(1, args.steps + 1):
            model.train()

            episode_losses: list[Tensor] = []
            episode_entropies: list[Tensor] = []

            sampled_costs: list[float] = []
            baseline_costs: list[float] = []
            advantages: list[float] = []
            log_prob_sums: list[Tensor] = []
            decision_counts: list[int] = []

            for _ in range(args.episodes_per_update):
                num_agents, num_tasks = sample_problem_size(
                    agent_range=agent_range,
                    task_range=task_range,
                    rng=rng,
                )

                env = DANAsyncMTSPEnv(
                    num_agents=num_agents,
                    num_tasks=num_tasks,
                    delta_g=args.delta_g,
                )

                instance = env.generate_instance(
                    seed=int(rng.integers(0, 2**31 - 1))
                )

                # Greedy rollout baseline on the exact same instance.
                model.eval()

                baseline_cost = rollout_greedy(
                    model=model,
                    device=device,
                    instance=instance,
                    num_agents=num_agents,
                    num_tasks=num_tasks,
                    delta_g=args.delta_g,
                )

                # Sampled rollout with gradients.
                model.train()

                sampled_cost, log_prob_sum, entropy_sum, decision_count = (
                    rollout_sampled(
                        model=model,
                        device=device,
                        instance=instance,
                        num_agents=num_agents,
                        num_tasks=num_tasks,
                        delta_g=args.delta_g,
                    )
                )

                advantage = sampled_cost - baseline_cost

                sampled_costs.append(sampled_cost)
                baseline_costs.append(baseline_cost)
                advantages.append(advantage)
                log_prob_sums.append(log_prob_sum)
                episode_entropies.append(entropy_sum)
                decision_counts.append(decision_count)

            advantage_tensor = torch.tensor(
                advantages,
                dtype=torch.float32,
                device=device,
            )

            if args.advantage_normalization and len(advantages) > 1:
                advantage_tensor = (
                    advantage_tensor - advantage_tensor.mean()
                ) / advantage_tensor.std(unbiased=False).clamp_min(1e-6)

            for index, log_prob_sum in enumerate(log_prob_sums):
                policy_loss = advantage_tensor[index].detach() * log_prob_sum
                entropy_loss = -args.entropy_coef * episode_entropies[index]
                episode_losses.append(policy_loss + entropy_loss)

            total_loss = torch.stack(episode_losses).mean()

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            if step % args.lr_decay_every == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= args.lr_decay

            sampled_mean = float(np.mean(sampled_costs))
            baseline_mean = float(np.mean(baseline_costs))
            advantage_mean = float(np.mean(advantages))
            beat_fraction = float(
                np.mean(
                    [
                        sampled < baseline
                        for sampled, baseline in zip(
                            sampled_costs,
                            baseline_costs,
                        )
                    ]
                )
            )

            entropy_mean = float(
                torch.stack(episode_entropies).mean().detach().cpu()
            )

            row = {
                "step": step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_sampled_makespan": sampled_mean,
                "train_greedy_baseline_makespan": baseline_mean,
                "train_advantage": advantage_mean,
                "train_sample_beats_baseline_fraction": beat_fraction,
                "train_policy_loss": float(
                    (
                        total_loss
                        + args.entropy_coef
                        * torch.stack(episode_entropies).mean()
                    ).detach().cpu()
                ),
                "train_entropy": entropy_mean,
                "train_total_loss": float(total_loss.detach().cpu()),
                "overall_dan_greedy_makespan": float("nan"),
                "overall_nearest_makespan": float("nan"),
                "overall_gap_to_nearest_pct": float("nan"),
            }

            should_eval = (
                step % args.eval_every == 0
                or step == 1
                or step == args.steps
            )

            if should_eval:
                eval_metrics = evaluate_dan_greedy(
                    model=model,
                    device=device,
                    settings=eval_settings,
                    instances_per_setting=args.eval_instances,
                    delta_g=args.delta_g,
                    seed=args.seed + 1_000_000 + step,
                )

                row["overall_dan_greedy_makespan"] = eval_metrics[
                    "overall_dan_greedy_makespan"
                ]

                row["overall_nearest_makespan"] = eval_metrics[
                    "overall_nearest_makespan"
                ]

                row["overall_gap_to_nearest_pct"] = eval_metrics[
                    "overall_gap_to_nearest_pct"
                ]

                if (
                    eval_metrics["overall_gap_to_nearest_pct"]
                    < best_eval_gap
                ):
                    best_eval_gap = eval_metrics[
                        "overall_gap_to_nearest_pct"
                    ]

                    save_checkpoint(
                        checkpoint_dir / "dan_best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        config=config,
                        metrics=eval_metrics,
                    )

            if (
                step % args.checkpoint_every == 0
                or step == args.steps
            ):
                save_checkpoint(
                    checkpoint_dir / f"dan_step_{step}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    metrics=row,
                )

            writer.writerow(row)
            file.flush()

            if step % 25 == 0 or should_eval:
                elapsed = time.perf_counter() - started_at

                message = (
                    f"Step {step:05d}/{args.steps} | "
                    f"sample={sampled_mean:.4f} | "
                    f"greedy_base={baseline_mean:.4f} | "
                    f"adv={advantage_mean:+.4f} | "
                    f"wins={100.0 * beat_fraction:.1f}% | "
                    f"loss={row['train_total_loss']:+.4f} | "
                    f"elapsed={elapsed / 60.0:.1f} min"
                )

                if should_eval:
                    message += (
                        f" | eval DAN={row['overall_dan_greedy_makespan']:.4f}, "
                        f"nearest={row['overall_nearest_makespan']:.4f}, "
                        f"gap={row['overall_gap_to_nearest_pct']:+.2f}%"
                    )

                print(message)

    final_metrics = {
        "best_eval_gap_to_nearest_pct": best_eval_gap,
    }

    save_checkpoint(
        checkpoint_dir / "dan_last.pt",
        model=model,
        optimizer=optimizer,
        step=args.steps,
        config=config,
        metrics=final_metrics,
    )

    print()
    print("DAN training complete.")
    print(f"Best checkpoint: {checkpoint_dir / 'dan_best.pt'}")
    print(f"Last checkpoint: {checkpoint_dir / 'dan_last.pt'}")
    print(f"Metrics CSV: {metrics_csv}")


if __name__ == "__main__":
    main()
