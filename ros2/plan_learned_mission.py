from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env.mtsp_2d_env import (
    evaluate_routes,
    generate_random_instance,
)
from models.centralized_attention_mtsp import (
    CentralizedAttentionMTSP,
)
from ros2.mission_config import (
    DEFAULT_MISSION_SEED,
    DRONES,
    NUM_TASKS,
    PROJECT_ROOT,
    WORKSPACE_SIZE_M,
    PLANNER_TO_GAZEBO_SCALE,
    GAZEBO_WORLD_OFFSET_ENU_M,
    planner_to_gazebo_enu,
    resolve_checkpoint_path,
)


def actions_to_routes(
    actions: np.ndarray,
    num_drones: int,
) -> list[list[int]]:
    routes: list[list[int]] = [[] for _ in range(num_drones)]

    for drone_id, task_id in actions:
        routes[int(drone_id)].append(int(task_id))

    return routes


def validate_routes(
    routes: list[list[int]],
    num_tasks: int,
) -> None:
    assigned_tasks = [
        task_id
        for route in routes
        for task_id in route
    ]

    if sorted(assigned_tasks) != list(range(num_tasks)):
        raise ValueError(
            "Routes must assign every task exactly once."
        )


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


def make_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one learned 3-drone Gazebo mission."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MISSION_SEED,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    if args.output is None:
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "gazebo_missions"
            / f"mission_seed_{args.seed}.json"
        )
    else:
        output_path = make_project_path(args.output)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    instance = generate_random_instance(
        num_tasks=NUM_TASKS,
        seed=args.seed,
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
        decoded_actions = model.decode(
            depots=depots_tensor,
            tasks=tasks_tensor,
        )[0].cpu().numpy()

    routes = actions_to_routes(
        decoded_actions,
        num_drones=len(DRONES),
    )

    validate_routes(
        routes=routes,
        num_tasks=instance.num_tasks,
    )

    metrics = evaluate_routes(
        instance=instance,
        routes=routes,
    )

    gazebo_depots = planner_to_gazebo_enu(instance.depots)
    gazebo_tasks = planner_to_gazebo_enu(instance.tasks)

    mission = {
        "format_version": 1,
        "task_indexing": "zero_based",
        "planner": {
            "seed": args.seed,
            "num_tasks": instance.num_tasks,
            "workspace_size_m": WORKSPACE_SIZE_M,
            "depots_xy": instance.depots.tolist(),
            "tasks_xy": instance.tasks.tolist(),
            "decoded_actions": decoded_actions.tolist(),
            "routes_task_ids": routes,
            "metrics": {
                "makespan_m": metrics.makespan,
                "total_distance_m": metrics.total_distance,
            },
        },
        "gazebo": {
            "planner_to_gazebo_scale": PLANNER_TO_GAZEBO_SCALE,
            "world_offset_enu_m": GAZEBO_WORLD_OFFSET_ENU_M.tolist(),
            "depot_positions_enu_m": gazebo_depots.tolist(),
            "task_positions_enu_m": gazebo_tasks.tolist(),
        },
        "drones": [
            {
                "name": spec.name,
                "namespace": spec.namespace,
                "target_system": spec.target_system,
                "depot_index": spec.depot_index,
                "cruise_altitude_m": spec.cruise_altitude_m,
            }
            for spec in DRONES
        ],
        "checkpoint": str(checkpoint_path),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(mission, file, indent=2)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mission JSON: {output_path}")
    print()

    for drone_index, route in enumerate(routes, start=1):
        human_task_ids = [task_id + 1 for task_id in route]

        print(
            f"Drone {drone_index}: "
            f"tasks={human_task_ids}"
        )

    print()
    print(
        f"Learned makespan: "
        f"{metrics.makespan:.2f} m"
    )
    print(
        f"Total route distance: "
        f"{metrics.total_distance:.2f} m"
    )
    print()
    print("Gazebo depot positions [east, north]:")
    for index, position in enumerate(gazebo_depots, start=1):
        print(
            f"Drone {index}: "
            f"[{position[0]:.2f}, {position[1]:.2f}]"
        )


if __name__ == "__main__":
    main()