from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from baselines.greedy import greedy_minmax_insertion
from env.mtsp_2d_env import MTSPInstance, evaluate_routes
from evaluation.evaluate import plot_routes
from models.centralized_attention_mtsp import CentralizedAttentionMTSP


def actions_to_routes(
    actions: np.ndarray,
    num_drones: int,
) -> list[list[int]]:
    """Convert [[drone_id, task_id], ...] into one ordered route per drone."""
    routes: list[list[int]] = [[] for _ in range(num_drones)]

    for drone_id, task_id in actions:
        routes[int(drone_id)].append(int(task_id))

    return routes


def percent_gap(value: float, reference: float) -> float:
    if reference <= 0.0:
        return 0.0

    return 100.0 * (value - reference) / reference


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> CentralizedAttentionMTSP:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = CentralizedAttentionMTSP(
        **checkpoint["model_config"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a learned centralized mTSP policy."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/centralized_attention_best_decode.pt"
        ),
    )

    parser.add_argument(
        "--test",
        type=Path,
        default=Path(
            "data/mtsp_3drones_15tasks_test.npz"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="0 means evaluate the entire test set.",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Which test case to plot.",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not args.test.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {args.test}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with np.load(args.test, allow_pickle=False) as data:
        seeds = data["seeds"]
        depots = data["depots"].astype(np.float32)
        tasks = data["tasks"].astype(np.float32)
        teacher_actions = data["teacher_actions"].astype(np.int64)
        expert_makespans = data["makespans"].astype(np.float32)
        expert_totals = data["total_distances"].astype(np.float32)

    total_available = len(depots)

    if args.max_instances > 0:
        num_instances = min(args.max_instances, total_available)
    else:
        num_instances = total_available

    if not 0 <= args.sample_index < num_instances:
        raise ValueError(
            f"--sample-index must be between 0 and {num_instances - 1}."
        )

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    output_dir = (
        Path("outputs")
        / f"learned_eval_{args.checkpoint.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test instances: {num_instances}")

    start_time = time.perf_counter()

    for batch_start in range(0, num_instances, args.batch_size):
        batch_end = min(
            batch_start + args.batch_size,
            num_instances,
        )

        depots_batch = torch.from_numpy(
            depots[batch_start:batch_end]
        ).to(device)

        tasks_batch = torch.from_numpy(
            tasks[batch_start:batch_end]
        ).to(device)

        with torch.no_grad():
            decoded_actions = model.decode(
                depots=depots_batch,
                tasks=tasks_batch,
            ).cpu().numpy()

        for local_index, actions in enumerate(decoded_actions):
            index = batch_start + local_index

            instance = MTSPInstance(
                depots=depots[index].astype(np.float64),
                tasks=tasks[index].astype(np.float64),
            )

            learned_routes = actions_to_routes(
                actions,
                num_drones=instance.num_drones,
            )

            greedy_routes = greedy_minmax_insertion(instance)

            learned_metrics = evaluate_routes(
                instance,
                learned_routes,
            )

            greedy_metrics = evaluate_routes(
                instance,
                greedy_routes,
            )

            expert_makespan = float(expert_makespans[index])
            expert_total = float(expert_totals[index])

            rows.append(
                {
                    "seed": int(seeds[index]),
                    "learned_makespan_m": learned_metrics.makespan,
                    "learned_total_distance_m": (
                        learned_metrics.total_distance
                    ),
                    "greedy_makespan_m": greedy_metrics.makespan,
                    "greedy_total_distance_m": (
                        greedy_metrics.total_distance
                    ),
                    "ortools_makespan_m": expert_makespan,
                    "ortools_total_distance_m": expert_total,
                    "learned_makespan_gap_to_ortools_pct": percent_gap(
                        learned_metrics.makespan,
                        expert_makespan,
                    ),
                    "greedy_makespan_gap_to_ortools_pct": percent_gap(
                        greedy_metrics.makespan,
                        expert_makespan,
                    ),
                    "learned_total_gap_to_ortools_pct": percent_gap(
                        learned_metrics.total_distance,
                        expert_total,
                    ),
                    "greedy_total_gap_to_ortools_pct": percent_gap(
                        greedy_metrics.total_distance,
                        expert_total,
                    ),
                }
            )

        print(
            f"\rEvaluated {batch_end}/{num_instances}",
            end="",
            flush=True,
        )

    print()

    elapsed_s = time.perf_counter() - start_time

    def average(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    summary = {
        "num_instances": num_instances,
        "runtime_s": elapsed_s,
        "learned_mean_makespan_m": average("learned_makespan_m"),
        "greedy_mean_makespan_m": average("greedy_makespan_m"),
        "ortools_mean_makespan_m": average("ortools_makespan_m"),
        "learned_mean_total_distance_m": average(
            "learned_total_distance_m"
        ),
        "greedy_mean_total_distance_m": average(
            "greedy_total_distance_m"
        ),
        "ortools_mean_total_distance_m": average(
            "ortools_total_distance_m"
        ),
        "learned_mean_makespan_gap_to_ortools_pct": average(
            "learned_makespan_gap_to_ortools_pct"
        ),
        "greedy_mean_makespan_gap_to_ortools_pct": average(
            "greedy_makespan_gap_to_ortools_pct"
        ),
        "learned_mean_total_gap_to_ortools_pct": average(
            "learned_total_gap_to_ortools_pct"
        ),
        "greedy_mean_total_gap_to_ortools_pct": average(
            "greedy_total_gap_to_ortools_pct"
        ),
    }

    csv_path = output_dir / "per_instance.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    sample_index = args.sample_index

    sample_instance = MTSPInstance(
        depots=depots[sample_index].astype(np.float64),
        tasks=tasks[sample_index].astype(np.float64),
    )

    sample_depots = torch.from_numpy(
        depots[sample_index:sample_index + 1]
    ).to(device)

    sample_tasks = torch.from_numpy(
        tasks[sample_index:sample_index + 1]
    ).to(device)

    with torch.no_grad():
        sample_learned_actions = model.decode(
            depots=sample_depots,
            tasks=sample_tasks,
        ).cpu().numpy()[0]

    sample_learned_routes = actions_to_routes(
        sample_learned_actions,
        num_drones=sample_instance.num_drones,
    )

    sample_greedy_routes = greedy_minmax_insertion(
        sample_instance
    )

    sample_ortools_routes = actions_to_routes(
        teacher_actions[sample_index],
        num_drones=sample_instance.num_drones,
    )

    plot_routes(
        sample_instance,
        sample_learned_routes,
        title="Learned Centralized Policy",
        output_path=output_dir / "sample_learned.png",
    )

    plot_routes(
        sample_instance,
        sample_greedy_routes,
        title="Greedy MinMax Insertion",
        output_path=output_dir / "sample_greedy.png",
    )

    plot_routes(
        sample_instance,
        sample_ortools_routes,
        title="OR-Tools Expert",
        output_path=output_dir / "sample_ortools.png",
    )

    print("\nTEST SUMMARY")
    print("-" * 68)
    print(
        f"Learned makespan: "
        f"{summary['learned_mean_makespan_m']:.2f} m "
        f"({summary['learned_mean_makespan_gap_to_ortools_pct']:+.2f}% "
        f"vs OR-Tools)"
    )
    print(
        f"Greedy makespan:  "
        f"{summary['greedy_mean_makespan_m']:.2f} m "
        f"({summary['greedy_mean_makespan_gap_to_ortools_pct']:+.2f}% "
        f"vs OR-Tools)"
    )
    print(
        f"OR-Tools makespan: "
        f"{summary['ortools_mean_makespan_m']:.2f} m"
    )
    print(
        f"\nSaved results in: {output_dir}"
    )
    print(f"  {csv_path.name}")
    print(f"  {summary_path.name}")
    print("  sample_learned.png")
    print("  sample_greedy.png")
    print("  sample_ortools.png")


if __name__ == "__main__":
    main()