from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a DAN task-world JSON for live asynchronous planning."
    )

    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--depot-x", type=float, default=0.5)
    parser.add_argument("--depot-y", type=float, default=0.5)
    parser.add_argument(
        "--planner-size",
        type=float,
        default=20.0,
        help="Unit-square coordinates are multiplied by this for planner/Gazebo frame later.",
    )
    parser.add_argument(
        "--spawn-radius",
        type=float,
        default=0.8,
        help="Small planner-frame radius around the common depot for PX4 spawn pads.",
    )
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.num_agents < 1:
        raise ValueError("--num-agents must be at least 1.")

    if args.num_tasks < 1:
        raise ValueError("--num-tasks must be at least 1.")

    rng = np.random.default_rng(args.seed)

    depot_unit = np.asarray(
        [args.depot_x, args.depot_y],
        dtype=np.float64,
    )

    tasks_unit = rng.uniform(
        low=0.0,
        high=1.0,
        size=(args.num_tasks, 2),
    ).astype(np.float64)

    depot_planner = depot_unit * args.planner_size
    tasks_planner = tasks_unit * args.planner_size

    # PX4/Gazebo cannot safely spawn all drones at exactly the same XY.
    # These are small physical spawn pads around the common DAN depot.
    # DAN inference still uses depot_xy_unit as the common depot.
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        num=args.num_agents,
        endpoint=False,
    )
    spawn_offsets = np.stack(
        [
            args.spawn_radius * np.cos(angles),
            args.spawn_radius * np.sin(angles),
        ],
        axis=1,
    )
    depots_planner = depot_planner[None, :] + spawn_offsets

    world = {
        "schema": "dan_task_world_v1",
        "planner_name": "dan_live_async",
        "seed": args.seed,
        "num_agents": args.num_agents,
        "num_tasks": args.num_tasks,
        "coordinate_frame": {
            "dan_model_frame": "unit_square_[0,1]^2",
            "planner_frame": f"[0,{args.planner_size}]^2",
            "planner_size": args.planner_size,
            "note": (
                "DAN inference should use unit coordinates. "
                "Planner coordinates are included for later Gazebo/PX4 conversion."
            ),
        },
        "communication": {
            "mode": "global",
            "note": (
                "First DAN reproduction uses global task mask and global agent states, "
                "matching the paper's fully observable assumption."
            ),
        },
        "depot_xy_unit": depot_unit.tolist(),
        "tasks_xy_unit": tasks_unit.tolist(),
        "depot_xy_planner": depot_planner.tolist(),
        "tasks_xy_planner": tasks_planner.tolist(),

        # Compatibility aliases for the existing simulation pipeline.
        # depots_xy are physical spawn pads, not DAN's mathematical common depot.
        "depots_xy": depots_planner.tolist(),
        "tasks_xy": tasks_planner.tolist(),
        "routes_task_ids": [[] for _ in range(args.num_agents)],

        # Old centralized simulation pipeline expects mission["planner"].
        # For DAN, this is only used to spawn PX4 vehicles and visuals.
        "planner": {
            "planner_name": "dan_live_async",
            "num_agents": args.num_agents,
            "num_drones": args.num_agents,
            "num_tasks": args.num_tasks,
            "depots_xy": depots_planner.tolist(),
            "tasks_xy": tasks_planner.tolist(),
            "depot_xy": depot_planner.tolist(),
            "common_depot_xy": depot_planner.tolist(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(world, file, indent=2)

    print(f"Saved DAN world: {args.output}")
    print(f"Agents: {args.num_agents}")
    print(f"Tasks:  {args.num_tasks}")
    print(f"Depot unit:    {depot_unit.tolist()}")
    print(f"Depot planner: {depot_planner.tolist()}")


if __name__ == "__main__":
    main()
