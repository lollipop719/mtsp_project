from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from baselines.greedy import greedy_minmax_insertion, round_robin_routes
from baselines.ortools_solver import solve_ortools_minmax
from env.mtsp_2d_env import evaluate_routes, generate_random_instance


def parse_task_counts(spec: str) -> list[int]:
    values = [int(value.strip()) for value in spec.split(",") if value.strip()]

    if not values or any(value < 1 for value in values):
        raise ValueError("Task counts must be positive integers.")

    return values


def parse_seeds(spec: str) -> list[int]:
    """
    Accept either:
      0:10      -> [0, 1, ..., 9]
      1,4,7     -> [1, 4, 7]
    """
    if ":" in spec:
        start_text, stop_text = spec.split(":", maxsplit=1)
        start = int(start_text)
        stop = int(stop_text)

        if stop <= start:
            raise ValueError("For A:B seed syntax, B must be greater than A.")

        return list(range(start, stop))

    values = [int(value.strip()) for value in spec.split(",") if value.strip()]

    if not values:
        raise ValueError("At least one seed is required.")

    return values


def percent_gap(value: float, reference: float) -> float:
    """Positive means worse than the reference solution."""
    if reference <= 0.0:
        return 0.0

    return 100.0 * (value - reference) / reference


def write_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    case_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)

    for row in case_rows:
        key = (int(row["num_tasks"]), str(row["solver"]))
        grouped[key].append(row)

    summary_rows: list[dict[str, object]] = []

    for (num_tasks, solver), rows in sorted(grouped.items()):
        def values(column: str) -> list[float]:
            return [float(row[column]) for row in rows]

        makespans = values("makespan_m")
        totals = values("total_distance_m")
        runtimes = values("runtime_s")
        makespan_gaps = values("makespan_gap_to_ortools_pct")
        total_gaps = values("total_gap_to_ortools_pct")

        summary_rows.append(
            {
                "num_tasks": num_tasks,
                "solver": solver,
                "num_instances": len(rows),
                "mean_makespan_m": mean(makespans),
                "std_makespan_m": pstdev(makespans) if len(makespans) > 1 else 0.0,
                "mean_total_distance_m": mean(totals),
                "std_total_distance_m": pstdev(totals) if len(totals) > 1 else 0.0,
                "mean_runtime_s": mean(runtimes),
                "std_runtime_s": pstdev(runtimes) if len(runtimes) > 1 else 0.0,
                "mean_makespan_gap_to_ortools_pct": mean(makespan_gaps),
                "mean_total_gap_to_ortools_pct": mean(total_gaps),
            }
        )

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark three-drone centralized mTSP baselines."
    )
    parser.add_argument(
        "--task-counts",
        type=str,
        default="10,15,20",
        help='Comma-separated list, for example "10,15,20".',
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0:10",
        help='Seed range "0:10" or list "0,1,2".',
    )
    parser.add_argument(
        "--ortools-time-limit",
        type=int,
        default=2,
        help="OR-Tools search time per instance, in seconds.",
    )
    args = parser.parse_args()

    task_counts = parse_task_counts(args.task_counts)
    seeds = parse_seeds(args.seeds)

    if args.ortools_time_limit < 1:
        raise ValueError("--ortools-time-limit must be at least 1.")

    task_tag = "-".join(str(value) for value in task_counts)
    seed_tag = f"{min(seeds)}-{max(seeds)}"

    output_dir = Path("outputs") / (
        f"benchmark_tasks_{task_tag}_seeds_{seed_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, object]] = []

    total_cases = len(task_counts) * len(seeds)
    case_number = 0

    for num_tasks in task_counts:
        for seed in seeds:
            case_number += 1
            print(
                f"[{case_number}/{total_cases}] "
                f"tasks={num_tasks}, seed={seed}"
            )

            instance = generate_random_instance(
                num_tasks=num_tasks,
                seed=seed,
            )

            solver_outputs: dict[str, tuple[float, float, float]] = {}

            solver_start = time.perf_counter()
            round_robin_metrics = evaluate_routes(
                instance,
                round_robin_routes(instance),
            )
            solver_outputs["round_robin"] = (
                round_robin_metrics.makespan,
                round_robin_metrics.total_distance,
                time.perf_counter() - solver_start,
            )

            solver_start = time.perf_counter()
            greedy_metrics = evaluate_routes(
                instance,
                greedy_minmax_insertion(instance),
            )
            solver_outputs["greedy"] = (
                greedy_metrics.makespan,
                greedy_metrics.total_distance,
                time.perf_counter() - solver_start,
            )

            solver_start = time.perf_counter()
            ortools_metrics = evaluate_routes(
                instance,
                solve_ortools_minmax(
                    instance,
                    time_limit_s=args.ortools_time_limit,
                ),
            )
            solver_outputs["ortools"] = (
                ortools_metrics.makespan,
                ortools_metrics.total_distance,
                time.perf_counter() - solver_start,
            )

            ortools_makespan = ortools_metrics.makespan
            ortools_total = ortools_metrics.total_distance

            for solver_name, (makespan, total_distance, runtime_s) in (
                solver_outputs.items()
            ):
                case_rows.append(
                    {
                        "num_tasks": num_tasks,
                        "seed": seed,
                        "solver": solver_name,
                        "makespan_m": makespan,
                        "total_distance_m": total_distance,
                        "runtime_s": runtime_s,
                        "makespan_gap_to_ortools_pct": percent_gap(
                            makespan,
                            ortools_makespan,
                        ),
                        "total_gap_to_ortools_pct": percent_gap(
                            total_distance,
                            ortools_total,
                        ),
                    }
                )

    summary_rows = summarize(case_rows)

    write_csv(case_rows, output_dir / "per_instance_results.csv")
    write_csv(summary_rows, output_dir / "summary.csv")

    metadata = {
        "task_counts": task_counts,
        "seeds": seeds,
        "ortools_time_limit_s": args.ortools_time_limit,
        "summary": summary_rows,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nSUMMARY")
    print("-" * 88)

    for row in summary_rows:
        print(
            f"tasks={row['num_tasks']:>2} | "
            f"{row['solver']:<11} | "
            f"mean makespan={row['mean_makespan_m']:.2f} m | "
            f"gap={row['mean_makespan_gap_to_ortools_pct']:.2f}% | "
            f"runtime={row['mean_runtime_s']:.3f} s"
        )

    print(f"\nSaved benchmark files in: {output_dir}")
    print("  per_instance_results.csv")
    print("  summary.csv")
    print("  summary.json")


if __name__ == "__main__":
    main()