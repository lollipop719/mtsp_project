from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from env.mtsp_2d_env import MTSPInstance, evaluate_routes
from evaluation.zero_shot_evaluate import (
    generate_zero_shot_instance,
    parse_int_list,
    solve_ortools_minmax_generic,
)


def canonical_interleaved_actions(
    routes: list[list[int]],
) -> np.ndarray:
    """
    Convert per-agent routes into one teacher action sequence.

    Example:
        D1 route: [3, 5]
        D2 route: [1]
        D3 route: [0, 2]

    Sequence:
        (D1,3), (D2,1), (D3,0), (D1,5), (D3,2)

    This preserves each agent's internal route order while giving the
    policy one joint action per decoding step.
    """
    num_agents = len(routes)
    positions = [0] * num_agents
    actions: list[tuple[int, int]] = []

    total_tasks = sum(len(route) for route in routes)

    while len(actions) < total_tasks:
        for agent_id in range(num_agents):
            if positions[agent_id] >= len(routes[agent_id]):
                continue

            task_id = routes[agent_id][positions[agent_id]]
            actions.append((agent_id, task_id))
            positions[agent_id] += 1

    return np.asarray(actions, dtype=np.int64)


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

    return lengths.astype(np.float32)


def solve_one_instance(
    *,
    num_agents: int,
    num_tasks: int,
    seed: int,
    workspace_size: float,
    ortools_time_limit: float,
) -> dict[str, np.ndarray | int | float]:
    instance = generate_zero_shot_instance(
        num_agents=num_agents,
        num_tasks=num_tasks,
        seed=seed,
        workspace_size=workspace_size,
    )

    routes = solve_ortools_minmax_generic(
        instance,
        time_limit_s=ortools_time_limit,
    )

    if routes is None:
        raise RuntimeError(
            f"OR-Tools failed for agents={num_agents}, "
            f"tasks={num_tasks}, seed={seed}."
        )

    actions = canonical_interleaved_actions(routes)

    assigned = sorted(int(action[1]) for action in actions)

    if assigned != list(range(num_tasks)):
        raise RuntimeError(
            f"Invalid teacher actions for seed={seed}: {assigned}"
        )

    metrics = evaluate_routes(
        instance,
        routes,
    )

    lengths = route_lengths(
        instance,
        routes,
    )

    return {
        "seed": seed,
        "depots": instance.depots.astype(np.float32),
        "tasks": instance.tasks.astype(np.float32),
        "teacher_actions": actions.astype(np.int64),
        "route_lengths": lengths,
        "makespan": float(metrics.makespan),
        "total_distance": float(metrics.total_distance),
    }


def generate_split(
    *,
    num_agents: int,
    num_tasks: int,
    split_name: str,
    num_instances: int,
    base_seed: int,
    workspace_size: float,
    ortools_time_limit: float,
    workers: int,
    output_dir: Path,
) -> Path:
    output_path = (
        output_dir
        / f"mtsp_{num_agents}agents_{num_tasks}tasks_{split_name}.npz"
    )

    seeds = [
        base_seed + index
        for index in range(num_instances)
    ]

    print(
        f"Generating {output_path.name}: "
        f"{num_instances} instances..."
    )

    started_at = time.perf_counter()

    results: list[dict[str, np.ndarray | int | float] | None] = [
        None
    ] * num_instances

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                solve_one_instance,
                num_agents=num_agents,
                num_tasks=num_tasks,
                seed=seed,
                workspace_size=workspace_size,
                ortools_time_limit=ortools_time_limit,
            ): index
            for index, seed in enumerate(seeds)
        }

        completed = 0

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

            completed += 1

            if completed % 25 == 0 or completed == num_instances:
                print(
                    f"  {completed}/{num_instances}",
                    flush=True,
                )

    depots = np.stack(
        [result["depots"] for result in results if result is not None],
        axis=0,
    )

    tasks = np.stack(
        [result["tasks"] for result in results if result is not None],
        axis=0,
    )

    teacher_actions = np.stack(
        [
            result["teacher_actions"]
            for result in results
            if result is not None
        ],
        axis=0,
    )

    route_lengths_array = np.stack(
        [
            result["route_lengths"]
            for result in results
            if result is not None
        ],
        axis=0,
    )

    makespans = np.asarray(
        [
            result["makespan"]
            for result in results
            if result is not None
        ],
        dtype=np.float32,
    )

    total_distances = np.asarray(
        [
            result["total_distance"]
            for result in results
            if result is not None
        ],
        dtype=np.float32,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        seeds=np.asarray(seeds, dtype=np.int64),
        depots=depots,
        tasks=tasks,
        teacher_actions=teacher_actions,
        route_lengths=route_lengths_array,
        makespans=makespans,
        total_distances=total_distances,
        num_agents=np.asarray(num_agents, dtype=np.int64),
        num_tasks=np.asarray(num_tasks, dtype=np.int64),
        workspace_size=np.asarray(workspace_size, dtype=np.float32),
        ortools_time_limit=np.asarray(
            ortools_time_limit,
            dtype=np.float32,
        ),
    )

    elapsed = time.perf_counter() - started_at

    print(
        f"Saved {output_path} "
        f"in {elapsed:.1f} s"
    )

    return output_path


def inspect_npz(path: Path) -> dict[str, list[int]]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "depots": list(data["depots"].shape),
            "tasks": list(data["tasks"].shape),
            "teacher_actions": list(data["teacher_actions"].shape),
            "route_lengths": list(data["route_lengths"].shape),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate variable-agent variable-task OR-Tools datasets."
    )

    parser.add_argument(
        "--agents",
        type=str,
        default="2,3,4,5",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default="10,15,20,25",
    )

    parser.add_argument(
        "--train",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--val",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--test",
        type=int,
        default=100,
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
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/variable_mtsp"),
    )

    args = parser.parse_args()

    agent_counts = parse_int_list(args.agents)
    task_counts = parse_int_list(args.tasks)

    all_paths: list[Path] = []

    config = {
        "agents": agent_counts,
        "tasks": task_counts,
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "workspace_size": args.workspace_size,
        "ortools_time_limit": args.ortools_time_limit,
        "base_seed": args.base_seed,
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (args.output_dir / "dataset_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    for num_agents in agent_counts:
        for num_tasks in task_counts:
            setting_offset = (
                num_agents * 1_000_000
                + num_tasks * 10_000
            )

            split_specs = [
                (
                    "train",
                    args.train,
                    args.base_seed + setting_offset,
                ),
                (
                    "val",
                    args.val,
                    args.base_seed + setting_offset + 3_000_000,
                ),
                (
                    "test",
                    args.test,
                    args.base_seed + setting_offset + 6_000_000,
                ),
            ]

            for split_name, count, split_seed in split_specs:
                if count <= 0:
                    continue

                path = generate_split(
                    num_agents=num_agents,
                    num_tasks=num_tasks,
                    split_name=split_name,
                    num_instances=count,
                    base_seed=split_seed,
                    workspace_size=args.workspace_size,
                    ortools_time_limit=args.ortools_time_limit,
                    workers=args.workers,
                    output_dir=args.output_dir,
                )

                all_paths.append(path)

    print()
    print("Generated datasets:")
    print("-" * 80)

    for path in all_paths:
        shapes = inspect_npz(path)
        print(f"{path}")
        print(f"  {shapes}")

    print()
    print(f"Config: {args.output_dir / 'dataset_config.json'}")


if __name__ == "__main__":
    main()
