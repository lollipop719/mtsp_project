from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from baselines.ortools_solver import solve_ortools_minmax
from env.mtsp_2d_env import evaluate_routes, generate_random_instance


def routes_to_teacher_actions(
    routes: list[list[int]],
    num_tasks: int,
) -> np.ndarray:
    """
    Convert three ordered routes into one deterministic action sequence.

    Example:
        Drone 1: [4, 7]
        Drone 2: [1]
        Drone 3: [3, 8]

    becomes:
        [(0, 4), (1, 1), (2, 3), (0, 7), (2, 8)]

    Each row is [drone_id, task_id], using zero-based indices.
    """
    actions: list[tuple[int, int]] = []
    max_route_length = max(len(route) for route in routes)

    for route_position in range(max_route_length):
        for drone_id, route in enumerate(routes):
            if route_position < len(route):
                actions.append((drone_id, route[route_position]))

    action_array = np.asarray(actions, dtype=np.int64)

    if action_array.shape != (num_tasks, 2):
        raise RuntimeError(
            "Teacher action sequence has an unexpected shape: "
            f"{action_array.shape}."
        )

    assigned_tasks = sorted(action_array[:, 1].tolist())

    if assigned_tasks != list(range(num_tasks)):
        raise RuntimeError(
            "Teacher actions must contain every task exactly once."
        )

    return action_array


def solve_one_example(
    job: tuple[int, int, int, int],
) -> tuple[int, dict[str, np.ndarray]]:
    """
    Worker function.

    job = (dataset_index, seed, num_tasks, ortools_time_limit_s)
    """
    dataset_index, seed, num_tasks, time_limit_s = job

    instance = generate_random_instance(
        num_tasks=num_tasks,
        seed=seed,
    )

    routes = solve_ortools_minmax(
        instance,
        time_limit_s=time_limit_s,
    )

    metrics = evaluate_routes(instance, routes)
    teacher_actions = routes_to_teacher_actions(
        routes,
        num_tasks=num_tasks,
    )

    example = {
        "seed": np.asarray(seed, dtype=np.int64),
        "depots": instance.depots.astype(np.float32),
        "tasks": instance.tasks.astype(np.float32),
        "teacher_actions": teacher_actions,
        "route_lengths": metrics.per_drone_distance.astype(np.float32),
        "makespan": np.asarray(metrics.makespan, dtype=np.float32),
        "total_distance": np.asarray(
            metrics.total_distance,
            dtype=np.float32,
        ),
    }

    return dataset_index, example


def generate_split(
    split_name: str,
    num_examples: int,
    start_seed: int,
    num_tasks: int,
    time_limit_s: int,
    workers: int,
) -> list[dict[str, np.ndarray]]:
    if num_examples < 1:
        return []

    jobs = [
        (
            dataset_index,
            start_seed + dataset_index,
            num_tasks,
            time_limit_s,
        )
        for dataset_index in range(num_examples)
    ]

    results: list[dict[str, np.ndarray] | None] = [None] * num_examples

    print(
        f"\nGenerating {split_name}: "
        f"{num_examples} examples, {workers} worker(s)"
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(solve_one_example, job): job[0]
            for job in jobs
        }

        completed = 0

        for future in as_completed(future_to_index):
            dataset_index, example = future.result()

            results[dataset_index] = example
            completed += 1

            print(
                f"\r{split_name}: {completed}/{num_examples}",
                end="",
                flush=True,
            )

    print()

    return [
        example
        for example in results
        if example is not None
    ]


def save_split(
    split_name: str,
    examples: list[dict[str, np.ndarray]],
    output_dir: Path,
    num_tasks: int,
) -> Path:
    if not examples:
        raise ValueError(f"No examples generated for split '{split_name}'.")

    output_path = output_dir / (
        f"mtsp_3drones_{num_tasks}tasks_{split_name}.npz"
    )

    np.savez_compressed(
        output_path,
        seeds=np.asarray(
            [example["seed"] for example in examples],
            dtype=np.int64,
        ),
        depots=np.stack(
            [example["depots"] for example in examples]
        ),
        tasks=np.stack(
            [example["tasks"] for example in examples]
        ),
        teacher_actions=np.stack(
            [example["teacher_actions"] for example in examples]
        ),
        route_lengths=np.stack(
            [example["route_lengths"] for example in examples]
        ),
        makespans=np.asarray(
            [example["makespan"] for example in examples],
            dtype=np.float32,
        ),
        total_distances=np.asarray(
            [example["total_distance"] for example in examples],
            dtype=np.float32,
        ),
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OR-Tools labels for 3-drone mTSP training."
    )

    parser.add_argument("--tasks", type=int, default=15)
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ortools-time-limit", type=int, default=1)

    default_workers = max(
        1,
        min(8, os.cpu_count() or 1),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
    )

    args = parser.parse_args()

    if args.tasks < 3:
        raise ValueError("Use at least 3 tasks.")

    if args.ortools_time_limit < 1:
        raise ValueError("--ortools-time-limit must be at least 1.")

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = [
        ("train", args.train, args.seed),
        ("val", args.val, args.seed + 1_000_000),
        ("test", args.test, args.seed + 2_000_000),
    ]

    saved_paths: dict[str, str] = {}

    for split_name, num_examples, split_seed in split_specs:
        if num_examples == 0:
            continue

        examples = generate_split(
            split_name=split_name,
            num_examples=num_examples,
            start_seed=split_seed,
            num_tasks=args.tasks,
            time_limit_s=args.ortools_time_limit,
            workers=args.workers,
        )

        saved_path = save_split(
            split_name=split_name,
            examples=examples,
            output_dir=output_dir,
            num_tasks=args.tasks,
        )

        saved_paths[split_name] = str(saved_path)

    metadata: dict[str, Any] = {
        "problem": "Centralized 3-drone multi-depot MinMax mTSP",
        "num_drones": 3,
        "num_tasks": args.tasks,
        "solver": "OR-Tools",
        "ortools_time_limit_s": args.ortools_time_limit,
        "action_encoding": (
            "Each row is [drone_id, task_id]. "
            "Routes are interleaved by route position, "
            "then drone index."
        ),
        "splits": saved_paths,
    }

    metadata_path = output_dir / (
        f"mtsp_3drones_{args.tasks}tasks_metadata.json"
    )

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nSaved:")
    for split_name, saved_path in saved_paths.items():
        print(f"  {split_name}: {saved_path}")

    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()