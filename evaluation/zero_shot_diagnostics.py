from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from env.mtsp_2d_env import MTSPInstance
from evaluation.zero_shot_evaluate import (
    actions_to_routes,
    generate_zero_shot_instance,
    load_model,
    parse_int_list,
    parse_seeds,
    percent_gap,
    solve_ortools_minmax_generic,
)


def route_lengths(
    instance: MTSPInstance,
    routes: list[list[int]],
) -> np.ndarray:
    depots = np.asarray(instance.depots, dtype=np.float64)
    tasks = np.asarray(instance.tasks, dtype=np.float64)

    lengths = np.zeros(len(routes), dtype=np.float64)

    for agent_id, route in enumerate(routes):
        depot = depots[agent_id]
        current = depot.copy()

        for task_id in route:
            task = tasks[task_id]
            lengths[agent_id] += np.linalg.norm(task - current)
            current = task

        lengths[agent_id] += np.linalg.norm(depot - current)

    return lengths


def route_task_counts(
    routes: list[list[int]],
) -> np.ndarray:
    return np.asarray(
        [len(route) for route in routes],
        dtype=np.float64,
    )


def summarize_distribution(
    values: np.ndarray,
) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def compute_route_stats(
    instance: MTSPInstance,
    routes: list[list[int]],
) -> dict[str, float]:
    counts = route_task_counts(routes)
    lengths = route_lengths(instance, routes)

    total_length = float(np.sum(lengths))
    max_length = float(np.max(lengths))
    min_length = float(np.min(lengths))

    used_agents = int(np.sum(counts > 0))
    empty_agents = int(np.sum(counts == 0))

    longest_agent = int(np.argmax(lengths))
    busiest_agent = int(np.argmax(counts))

    return {
        "used_agents": used_agents,
        "empty_agents": empty_agents,
        "task_count_min": float(np.min(counts)),
        "task_count_max": float(np.max(counts)),
        "task_count_range": float(np.max(counts) - np.min(counts)),
        "task_count_std": float(np.std(counts)),
        "route_length_min_m": min_length,
        "route_length_max_m": max_length,
        "route_length_range_m": max_length - min_length,
        "route_length_std_m": float(np.std(lengths)),
        "max_route_share_of_total": (
            max_length / total_length if total_length > 0.0 else 0.0
        ),
        "longest_agent": longest_agent,
        "busiest_agent": busiest_agent,
    }


def format_counts(
    routes: list[list[int]],
) -> str:
    return "[" + ", ".join(str(len(route)) for route in routes) + "]"


def format_lengths(
    lengths: np.ndarray,
) -> str:
    return "[" + ", ".join(f"{value:.1f}" for value in lengths) + "]"


def write_csv(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose learned mTSP zero-shot route usage."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/final_3drone_15task/"
            "centralized_attention_mtsp_15task.pt"
        ),
    )

    parser.add_argument(
        "--agents",
        type=str,
        default="2,3,4,5",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default="15",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0:30",
    )

    parser.add_argument(
        "--workspace-size",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--ortools-time-limit",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/zero_shot_diagnostics"),
    )

    args = parser.parse_args()

    agent_counts = parse_int_list(args.agents)
    task_counts = parse_int_list(args.tasks)
    seeds = parse_seeds(args.seeds)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    rows: list[dict[str, float | int | str]] = []

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    for num_agents in agent_counts:
        for num_tasks in task_counts:
            print(
                f"Diagnosing: {num_agents} agents, {num_tasks} tasks"
            )

            for seed in seeds:
                instance = generate_zero_shot_instance(
                    num_agents=num_agents,
                    num_tasks=num_tasks,
                    seed=seed,
                    workspace_size=args.workspace_size,
                )

                depots_tensor = torch.from_numpy(
                    instance.depots.astype(np.float32)[None, ...]
                ).to(device)

                tasks_tensor = torch.from_numpy(
                    instance.tasks.astype(np.float32)[None, ...]
                ).to(device)

                with torch.inference_mode():
                    learned_actions = model.decode(
                        depots=depots_tensor,
                        tasks=tasks_tensor,
                    )[0].cpu().numpy()

                learned_routes = actions_to_routes(
                    learned_actions,
                    num_agents=num_agents,
                    num_tasks=num_tasks,
                )

                learned_lengths = route_lengths(
                    instance,
                    learned_routes,
                )

                learned_makespan = float(np.max(learned_lengths))

                learned_stats = compute_route_stats(
                    instance,
                    learned_routes,
                )

                ortools_routes = solve_ortools_minmax_generic(
                    instance,
                    time_limit_s=args.ortools_time_limit,
                )

                if ortools_routes is None:
                    ortools_makespan = float("nan")
                    ortools_counts = ""
                    ortools_lengths_text = ""
                else:
                    ortools_lengths = route_lengths(
                        instance,
                        ortools_routes,
                    )
                    ortools_makespan = float(np.max(ortools_lengths))
                    ortools_counts = format_counts(ortools_routes)
                    ortools_lengths_text = format_lengths(ortools_lengths)

                row: dict[str, float | int | str] = {
                    "num_agents": num_agents,
                    "num_tasks": num_tasks,
                    "seed": seed,
                    "learned_makespan_m": learned_makespan,
                    "ortools_makespan_m": ortools_makespan,
                    "learned_gap_to_ortools_pct": percent_gap(
                        learned_makespan,
                        ortools_makespan,
                    ),
                    "learned_task_counts": format_counts(
                        learned_routes
                    ),
                    "learned_route_lengths_m": format_lengths(
                        learned_lengths
                    ),
                    "ortools_task_counts": ortools_counts,
                    "ortools_route_lengths_m": ortools_lengths_text,
                }

                row.update(learned_stats)
                rows.append(row)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_instance_path = output_dir / "per_instance_diagnostics.csv"
    summary_path = output_dir / "summary_diagnostics.csv"
    summary_json_path = output_dir / "summary_diagnostics.json"

    write_csv(per_instance_path, rows)

    grouped: dict[tuple[int, int], list[dict[str, float | int | str]]] = defaultdict(list)

    for row in rows:
        key = (
            int(row["num_agents"]),
            int(row["num_tasks"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, float | int]] = []

    for (num_agents, num_tasks), group in sorted(grouped.items()):
        def mean(key: str) -> float:
            return float(
                np.nanmean(
                    np.asarray(
                        [float(row[key]) for row in group],
                        dtype=np.float64,
                    )
                )
            )

        summary_rows.append(
            {
                "num_agents": num_agents,
                "num_tasks": num_tasks,
                "num_instances": len(group),
                "learned_makespan_m": mean("learned_makespan_m"),
                "ortools_makespan_m": mean("ortools_makespan_m"),
                "learned_gap_to_ortools_pct": mean(
                    "learned_gap_to_ortools_pct"
                ),
                "used_agents": mean("used_agents"),
                "empty_agents": mean("empty_agents"),
                "task_count_range": mean("task_count_range"),
                "task_count_std": mean("task_count_std"),
                "route_length_range_m": mean("route_length_range_m"),
                "route_length_std_m": mean("route_length_std_m"),
                "max_route_share_of_total": mean(
                    "max_route_share_of_total"
                ),
            }
        )

    write_csv(summary_path, summary_rows)

    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary_rows, file, indent=2)

    print()
    print("DIAGNOSTIC SUMMARY")
    print("-" * 105)

    for row in summary_rows:
        print(
            f"{int(row['num_agents'])} agents, "
            f"{int(row['num_tasks'])} tasks | "
            f"gap={row['learned_gap_to_ortools_pct']:+.2f}% | "
            f"used_agents={row['used_agents']:.2f} | "
            f"empty_agents={row['empty_agents']:.2f} | "
            f"task_count_range={row['task_count_range']:.2f} | "
            f"route_length_range={row['route_length_range_m']:.2f} m | "
            f"max_route_share={100.0 * row['max_route_share_of_total']:.1f}%"
        )

    print()
    print(f"Saved: {per_instance_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_json_path}")


if __name__ == "__main__":
    main()
