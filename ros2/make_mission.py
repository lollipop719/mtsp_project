from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baselines.greedy import greedy_minmax_insertion
from env.mtsp_2d_env import MTSPInstance, evaluate_routes
from evaluation.zero_shot_diagnostics import route_lengths
from evaluation.zero_shot_evaluate import (
    actions_to_routes,
    load_model,
    make_depots,
    solve_ortools_minmax_generic,
)
from ros2.mission_config import (
    GAZEBO_WORLD_OFFSET_ENU_M,
    PLANNER_TO_GAZEBO_SCALE,
    PROJECT_ROOT,
    WORKSPACE_SIZE_M,
    planner_to_gazebo_enu,
)


def make_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def generate_tasks(
    *,
    seed: int,
    num_tasks: int,
    workspace_size: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(
        low=1.0,
        high=workspace_size - 1.0,
        size=(num_tasks, 2),
    ).astype(np.float64)


def learned_attention_routes(
    *,
    instance: MTSPInstance,
    checkpoint_path: Path,
) -> list[list[int]]:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    depots_tensor = torch.from_numpy(
        instance.depots.astype(np.float32)[None, ...]
    ).to(device)

    tasks_tensor = torch.from_numpy(
        instance.tasks.astype(np.float32)[None, ...]
    ).to(device)

    with torch.inference_mode():
        actions = model.decode(
            depots=depots_tensor,
            tasks=tasks_tensor,
        )[0].cpu().numpy()

    return actions_to_routes(
        actions=actions,
        num_agents=instance.depots.shape[0],
        num_tasks=instance.tasks.shape[0],
    )


def make_routes(
    *,
    planner: str,
    instance: MTSPInstance,
    checkpoint_path: Path | None,
    ortools_time_limit_s: float,
) -> list[list[int]]:
    if planner == "learned_attention":
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required for learned_attention.")
        return learned_attention_routes(
            instance=instance,
            checkpoint_path=checkpoint_path,
        )

    if planner == "greedy":
        return greedy_minmax_insertion(instance)

    if planner == "ortools":
        routes = solve_ortools_minmax_generic(
            instance,
            time_limit_s=ortools_time_limit_s,
        )
        if routes is None:
            raise RuntimeError("OR-Tools failed to find a solution.")
        return routes

    if planner == "dan":
        raise NotImplementedError(
            "DAN planner is not implemented yet. Later it should return "
            "the same routes_task_ids format."
        )

    raise ValueError(f"Unknown planner: {planner}")


def validate_routes(
    *,
    routes: list[list[int]],
    num_agents: int,
    num_tasks: int,
) -> None:
    if len(routes) != num_agents:
        raise ValueError(
            f"Expected {num_agents} routes, got {len(routes)}."
        )

    assigned = sorted(
        int(task_id)
        for route in routes
        for task_id in route
    )

    expected = list(range(num_tasks))

    if assigned != expected:
        raise ValueError(
            f"Invalid routes. Assigned tasks={assigned}, expected={expected}"
        )


def make_mission_dict(
    *,
    planner: str,
    seed: int,
    instance: MTSPInstance,
    routes: list[list[int]],
    checkpoint_path: Path | None,
    ortools_time_limit_s: float,
) -> dict:
    metrics = evaluate_routes(instance, routes)
    lengths = route_lengths(instance, routes)

    task_to_agent: dict[int, int] = {}

    for agent_id, route in enumerate(routes):
        for task_id in route:
            task_to_agent[int(task_id)] = int(agent_id)

    depot_positions_enu = [
        planner_to_gazebo_enu(depot_xy).tolist()
        for depot_xy in instance.depots
    ]

    task_positions_enu = [
        planner_to_gazebo_enu(task_xy).tolist()
        for task_xy in instance.tasks
    ]

    num_agents = int(instance.depots.shape[0])
    num_tasks = int(instance.tasks.shape[0])

    return {
        "planner_name": planner,
        "seed": seed,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "ortools_time_limit_s": ortools_time_limit_s,
        "planner": {
            "seed": seed,
            "planner_name": planner,
            "num_agents": num_agents,
            "num_drones": num_agents,
            "num_tasks": num_tasks,
            "workspace_size_m": float(WORKSPACE_SIZE_M),
            "depots_xy": instance.depots.tolist(),
            "tasks_xy": instance.tasks.tolist(),
            "routes_task_ids": routes,
            "task_to_agent": task_to_agent,
            "route_lengths_m": lengths.tolist(),
            "makespan_m": float(metrics.makespan),
            "total_distance_m": float(metrics.total_distance),

            # Backward-compatible names for older scripts.
            "learned_makespan_m": float(metrics.makespan),
            "total_route_distance_m": float(metrics.total_distance),
        },
        "gazebo": {
            "planner_to_gazebo_scale": float(PLANNER_TO_GAZEBO_SCALE),
            "world_offset_enu_m": GAZEBO_WORLD_OFFSET_ENU_M.tolist(),
            "depot_positions_enu_m": depot_positions_enu,
            "task_positions_enu_m": task_positions_enu,
        },
    }


def print_mission_summary(mission: dict) -> None:
    planner_data = mission["planner"]

    print()
    print("=" * 80)
    print("MISSION GENERATED")
    print("=" * 80)
    print(f"Planner: {mission['planner_name']}")
    print(f"Seed: {mission['seed']}")
    print(f"Agents: {planner_data['num_agents']}")
    print(f"Tasks: {planner_data['num_tasks']}")
    print(f"Makespan: {planner_data['makespan_m']:.2f} m")
    print(f"Total distance: {planner_data['total_distance_m']:.2f} m")
    print()

    for agent_id, route in enumerate(planner_data["routes_task_ids"]):
        route_text = [int(task_id) + 1 for task_id in route]
        length = planner_data["route_lengths_m"][agent_id]

        print(
            f"Drone {agent_id + 1}: "
            f"tasks={route_text}, "
            f"route_length={length:.2f} m"
        )

    print("=" * 80)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Gazebo mission JSON from a selected planner."
    )

    parser.add_argument(
        "--planner",
        type=str,
        required=True,
        choices=["learned_attention", "greedy", "ortools", "dan"],
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/mixed_rl_variable_mtsp/"
            "mixed_rl_best_decode.pt"
        ),
    )

    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-tasks", type=int, default=15)
    parser.add_argument("--ortools-time-limit-s", type=float, default=10.0)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gazebo_missions/mission_generated.json"),
    )

    args = parser.parse_args()

    if args.num_agents < 1:
        raise ValueError("--num-agents must be at least 1.")

    if args.num_tasks < 1:
        raise ValueError("--num-tasks must be at least 1.")

    checkpoint_path = None

    if args.planner == "learned_attention":
        checkpoint_path = make_project_path(args.checkpoint)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

    output_path = make_project_path(args.output)

    depots_xy = make_depots(
        num_agents=args.num_agents,
        workspace_size=float(WORKSPACE_SIZE_M),
    )

    tasks_xy = generate_tasks(
        seed=args.seed,
        num_tasks=args.num_tasks,
        workspace_size=float(WORKSPACE_SIZE_M),
    )

    instance = MTSPInstance(
        depots=depots_xy,
        tasks=tasks_xy,
    )

    routes = make_routes(
        planner=args.planner,
        instance=instance,
        checkpoint_path=checkpoint_path,
        ortools_time_limit_s=args.ortools_time_limit_s,
    )

    validate_routes(
        routes=routes,
        num_agents=args.num_agents,
        num_tasks=args.num_tasks,
    )

    mission = make_mission_dict(
        planner=args.planner,
        seed=args.seed,
        instance=instance,
        routes=routes,
        checkpoint_path=checkpoint_path,
        ortools_time_limit_s=args.ortools_time_limit_s,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(mission, file, indent=2)

    print_mission_summary(mission)
    print(f"Saved mission: {output_path}")


if __name__ == "__main__":
    main()
