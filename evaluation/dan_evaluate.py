from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from env.dan_async_mtsp_env import (
    DANAsyncMTSPEnv,
    DANMTSPInstance,
    compute_route_lengths,
    nearest_task_policy,
)
from env.mtsp_2d_env import MTSPInstance, evaluate_routes
from evaluation.zero_shot_evaluate import (
    parse_int_list,
    parse_seeds,
    solve_ortools_minmax_generic,
)
from models.dan_mtsp import DANMTSPPolicy, DANPolicyRunner


def parse_settings(text: str) -> list[tuple[int, int]]:
    settings: list[tuple[int, int]] = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        if "x" not in item:
            raise ValueError(
                "Settings must look like '2x10,3x15,5x20'."
            )

        agents_text, tasks_text = item.split("x", maxsplit=1)

        settings.append(
            (int(agents_text), int(tasks_text))
        )

    return settings


def load_dan_model(
    checkpoint_path: Path,
    device: torch.device,
) -> DANMTSPPolicy:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = DANMTSPPolicy(
        **checkpoint["model_config"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def dan_rollout(
    *,
    model: DANMTSPPolicy,
    device: torch.device,
    instance: DANMTSPInstance,
    num_agents: int,
    num_tasks: int,
    delta_g: float,
    mode: str,
) -> tuple[float, list[list[int]], float]:
    env = DANAsyncMTSPEnv(
        num_agents=num_agents,
        num_tasks=num_tasks,
        delta_g=delta_g,
    )

    policy = DANPolicyRunner(
        model=model,
        device=device,
        mode=mode,
    )

    started_at = time.perf_counter()

    with torch.no_grad():
        result = env.rollout(
            policy,
            instance=instance,
        )

    elapsed_ms = 1000.0 * (time.perf_counter() - started_at)

    return result.makespan, result.routes, elapsed_ms


def dan_sampling_best(
    *,
    model: DANMTSPPolicy,
    device: torch.device,
    instance: DANMTSPInstance,
    num_agents: int,
    num_tasks: int,
    delta_g: float,
    num_samples: int,
) -> tuple[float, list[list[int]], float]:
    best_makespan = float("inf")
    best_routes: list[list[int]] | None = None

    started_at = time.perf_counter()

    for _ in range(num_samples):
        makespan, routes, _ = dan_rollout(
            model=model,
            device=device,
            instance=instance,
            num_agents=num_agents,
            num_tasks=num_tasks,
            delta_g=delta_g,
            mode="sample",
        )

        if makespan < best_makespan:
            best_makespan = makespan
            best_routes = routes

    elapsed_ms = 1000.0 * (time.perf_counter() - started_at)

    if best_routes is None:
        raise RuntimeError("DAN sampling did not produce any routes.")

    return best_makespan, best_routes, elapsed_ms


def nearest_rollout(
    *,
    instance: DANMTSPInstance,
    num_agents: int,
    num_tasks: int,
    delta_g: float,
) -> tuple[float, list[list[int]], float]:
    env = DANAsyncMTSPEnv(
        num_agents=num_agents,
        num_tasks=num_tasks,
        delta_g=delta_g,
    )

    started_at = time.perf_counter()

    result = env.rollout(
        nearest_task_policy,
        instance=instance,
    )

    elapsed_ms = 1000.0 * (time.perf_counter() - started_at)

    return result.makespan, result.routes, elapsed_ms


def ortools_solution(
    *,
    instance: DANMTSPInstance,
    num_agents: int,
    time_limit_s: float,
) -> tuple[float, list[list[int]] | None, float]:
    # OR-Tools expects one start/end depot per vehicle.
    # We create separate depot nodes at the same common depot coordinate.
    depots = np.repeat(
        instance.depot_xy[None, :],
        repeats=num_agents,
        axis=0,
    )

    mtsp_instance = MTSPInstance(
        depots=depots,
        tasks=instance.tasks_xy,
    )

    started_at = time.perf_counter()

    routes = solve_ortools_minmax_generic(
        mtsp_instance,
        time_limit_s=time_limit_s,
    )

    elapsed_ms = 1000.0 * (time.perf_counter() - started_at)

    if routes is None:
        return float("nan"), None, elapsed_ms

    metrics = evaluate_routes(
        mtsp_instance,
        routes,
    )

    return float(metrics.makespan), routes, elapsed_ms


def percent_gap(
    value: float,
    reference: float,
) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0.0:
        return float("nan")

    return 100.0 * (value - reference) / reference


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


def summarize_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int]]:
    grouped: dict[tuple[int, int], list[dict[str, float | int | str]]] = defaultdict(list)

    for row in rows:
        key = (
            int(row["num_agents"]),
            int(row["num_tasks"]),
        )
        grouped[key].append(row)

    summaries: list[dict[str, float | int]] = []

    for (num_agents, num_tasks), group in sorted(grouped.items()):
        def mean(key: str) -> float:
            values = np.asarray(
                [float(row[key]) for row in group],
                dtype=np.float64,
            )
            return float(np.nanmean(values))

        summaries.append(
            {
                "num_agents": num_agents,
                "num_tasks": num_tasks,
                "num_instances": len(group),
                "dan_greedy_makespan": mean("dan_greedy_makespan"),
                "dan_sampling_makespan": mean("dan_sampling_makespan"),
                "nearest_makespan": mean("nearest_makespan"),
                "ortools_makespan": mean("ortools_makespan"),
                "dan_greedy_gap_to_ortools_pct": mean(
                    "dan_greedy_gap_to_ortools_pct"
                ),
                "dan_sampling_gap_to_ortools_pct": mean(
                    "dan_sampling_gap_to_ortools_pct"
                ),
                "nearest_gap_to_ortools_pct": mean(
                    "nearest_gap_to_ortools_pct"
                ),
                "dan_greedy_runtime_ms": mean("dan_greedy_runtime_ms"),
                "dan_sampling_runtime_ms": mean("dan_sampling_runtime_ms"),
                "nearest_runtime_ms": mean("nearest_runtime_ms"),
                "ortools_runtime_ms": mean("ortools_runtime_ms"),
            }
        )

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained DAN on asynchronous common-depot mTSP."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/dan_2to5_agents_10to25_tasks/dan_best.pt"
        ),
    )

    parser.add_argument(
        "--settings",
        type=str,
        default="2x10,2x15,2x20,3x10,3x15,3x20,4x10,4x15,4x20,5x10,5x15,5x20",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0:50",
    )

    parser.add_argument(
        "--delta-g",
        type=float,
        default=0.01,
        help="DAN paper uses finer delta_g during testing.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Number of sampled DAN rollouts per instance.",
    )

    parser.add_argument(
        "--ortools-time-limit",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dan_eval"),
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    settings = parse_settings(args.settings)
    seeds = parse_seeds(args.seeds)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_dan_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    rows: list[dict[str, float | int | str]] = []

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Settings: {settings}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} instances)")
    print(f"DAN samples: {args.samples}")
    print(f"delta_g: {args.delta_g}")
    print()

    for num_agents, num_tasks in settings:
        print(f"Evaluating {num_agents} agents, {num_tasks} tasks")

        for seed in seeds:
            env = DANAsyncMTSPEnv(
                num_agents=num_agents,
                num_tasks=num_tasks,
                delta_g=args.delta_g,
            )

            instance = env.generate_instance(seed=seed)

            dan_greedy_makespan, dan_greedy_routes, dan_greedy_ms = dan_rollout(
                model=model,
                device=device,
                instance=instance,
                num_agents=num_agents,
                num_tasks=num_tasks,
                delta_g=args.delta_g,
                mode="greedy",
            )

            (
                dan_sampling_makespan,
                dan_sampling_routes,
                dan_sampling_ms,
            ) = dan_sampling_best(
                model=model,
                device=device,
                instance=instance,
                num_agents=num_agents,
                num_tasks=num_tasks,
                delta_g=args.delta_g,
                num_samples=args.samples,
            )

            nearest_makespan, nearest_routes, nearest_ms = nearest_rollout(
                instance=instance,
                num_agents=num_agents,
                num_tasks=num_tasks,
                delta_g=args.delta_g,
            )

            ortools_makespan, ortools_routes, ortools_ms = ortools_solution(
                instance=instance,
                num_agents=num_agents,
                time_limit_s=args.ortools_time_limit,
            )

            row = {
                "num_agents": num_agents,
                "num_tasks": num_tasks,
                "seed": seed,
                "dan_greedy_makespan": dan_greedy_makespan,
                "dan_sampling_makespan": dan_sampling_makespan,
                "nearest_makespan": nearest_makespan,
                "ortools_makespan": ortools_makespan,
                "dan_greedy_gap_to_ortools_pct": percent_gap(
                    dan_greedy_makespan,
                    ortools_makespan,
                ),
                "dan_sampling_gap_to_ortools_pct": percent_gap(
                    dan_sampling_makespan,
                    ortools_makespan,
                ),
                "nearest_gap_to_ortools_pct": percent_gap(
                    nearest_makespan,
                    ortools_makespan,
                ),
                "dan_greedy_runtime_ms": dan_greedy_ms,
                "dan_sampling_runtime_ms": dan_sampling_ms,
                "nearest_runtime_ms": nearest_ms,
                "ortools_runtime_ms": ortools_ms,
            }

            rows.append(row)

    summaries = summarize_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_instance_path = args.output_dir / "per_instance.csv"
    summary_path = args.output_dir / "summary.csv"
    summary_json_path = args.output_dir / "summary.json"

    write_csv(per_instance_path, rows)
    write_csv(summary_path, summaries)

    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print()
    print("DAN EVALUATION SUMMARY")
    print("-" * 110)

    for summary in summaries:
        print(
            f"{int(summary['num_agents'])} agents, "
            f"{int(summary['num_tasks'])} tasks | "
            f"DAN-g={summary['dan_greedy_makespan']:.4f} "
            f"({summary['dan_greedy_gap_to_ortools_pct']:+.2f}%) | "
            f"DAN-s{args.samples}={summary['dan_sampling_makespan']:.4f} "
            f"({summary['dan_sampling_gap_to_ortools_pct']:+.2f}%) | "
            f"nearest={summary['nearest_makespan']:.4f} "
            f"({summary['nearest_gap_to_ortools_pct']:+.2f}%) | "
            f"OR={summary['ortools_makespan']:.4f}"
        )

    print()
    print(f"Saved: {per_instance_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_json_path}")


if __name__ == "__main__":
    main()
