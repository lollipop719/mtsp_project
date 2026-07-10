from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from env.dan_async_mtsp_env import (
    DANAsyncMTSPEnv,
    DANMTSPInstance,
)
from models.dan_mtsp import (
    DANMTSPPolicy,
    DANPolicyRunner,
)


def load_model(
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


def load_world(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        world = json.load(file)

    if world.get("schema") != "dan_task_world_v1":
        raise ValueError(
            f"Expected schema dan_task_world_v1, got {world.get('schema')}"
        )

    return world


def make_instance_from_world(world: dict) -> DANMTSPInstance:
    return DANMTSPInstance(
        depot_xy=np.asarray(world["depot_xy_unit"], dtype=np.float64),
        tasks_xy=np.asarray(world["tasks_xy_unit"], dtype=np.float64),
    )


def run_live_dan_rollout(
    *,
    model: DANMTSPPolicy,
    device: torch.device,
    instance: DANMTSPInstance,
    num_agents: int,
    delta_g: float,
    mode: str,
) -> dict:
    env = DANAsyncMTSPEnv(
        num_agents=num_agents,
        num_tasks=instance.num_tasks,
        delta_g=delta_g,
    )

    policy = DANPolicyRunner(
        model=model,
        device=device,
        mode=mode,
    )

    env.reset(instance=instance)

    decision_events: list[dict] = []

    started_at = time.perf_counter()

    with torch.no_grad():
        while not env.all_tasks_visited():
            for agent_id in range(num_agents):
                if env.all_tasks_visited():
                    break

                assert env.remaining_travel_times is not None

                if env.remaining_travel_times[agent_id] > 1e-12:
                    continue

                observation = env.get_observation(agent_id)
                output = policy(observation)

                task_id = int(output.action)

                if not observation.action_mask[task_id]:
                    valid = np.flatnonzero(observation.action_mask).tolist()
                    raise RuntimeError(
                        f"Invalid DAN action: task {task_id}, valid={valid}"
                    )

                from_xy = observation.current_position.copy()
                to_xy = instance.tasks_xy[task_id].copy()
                decision_time = float(env.elapsed_time)

                distance = env.step_decision(
                    agent_id=agent_id,
                    task_id=task_id,
                )

                decision_events.append(
                    {
                        "decision_index": len(decision_events),
                        "time": decision_time,
                        "agent_id": agent_id,
                        "agent_label": f"drone_{agent_id + 1}",
                        "task_id": task_id,
                        "task_label": f"task_{task_id + 1}",
                        "from_xy_unit": from_xy.tolist(),
                        "to_xy_unit": to_xy.tolist(),
                        "travel_distance_unit": float(distance),
                    }
                )

            if not env.all_tasks_visited():
                env.advance_time()

    result = env.finish_rollout()
    runtime_ms = 1000.0 * (time.perf_counter() - started_at)

    return {
        "makespan": float(result.makespan),
        "total_distance": float(result.total_distance),
        "route_lengths": result.route_lengths.tolist(),
        "outbound_lengths": result.outbound_lengths.tolist(),
        "return_lengths": result.return_lengths.tolist(),
        "routes_task_ids": result.routes,
        "decision_events": decision_events,
        "elapsed_async_time": float(result.elapsed_time),
        "num_ticks": int(result.num_ticks),
        "runtime_ms": float(runtime_ms),
    }


def route_xy(
    *,
    depot_xy: np.ndarray,
    tasks_xy: np.ndarray,
    route: list[int],
) -> list[list[float]]:
    points = [depot_xy.tolist()]

    for task_id in route:
        points.append(tasks_xy[task_id].tolist())

    points.append(depot_xy.tolist())
    return points


def add_route_coordinates(
    *,
    rollout: dict,
    world: dict,
) -> None:
    depot_unit = np.asarray(world["depot_xy_unit"], dtype=np.float64)
    tasks_unit = np.asarray(world["tasks_xy_unit"], dtype=np.float64)

    depot_planner = np.asarray(world["depot_xy_planner"], dtype=np.float64)
    tasks_planner = np.asarray(world["tasks_xy_planner"], dtype=np.float64)

    rollout["routes_xy_unit"] = [
        route_xy(
            depot_xy=depot_unit,
            tasks_xy=tasks_unit,
            route=route,
        )
        for route in rollout["routes_task_ids"]
    ]

    rollout["routes_xy_planner"] = [
        route_xy(
            depot_xy=depot_planner,
            tasks_xy=tasks_planner,
            route=route,
        )
        for route in rollout["routes_task_ids"]
    ]


def plot_rollout(
    *,
    world: dict,
    rollout: dict,
    output_path: Path,
) -> None:
    depot = np.asarray(world["depot_xy_unit"], dtype=np.float64)
    tasks = np.asarray(world["tasks_xy_unit"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        tasks[:, 0],
        tasks[:, 1],
        marker="o",
        label="tasks",
    )

    for task_id, xy in enumerate(tasks):
        ax.text(
            xy[0],
            xy[1],
            str(task_id + 1),
            fontsize=8,
        )

    ax.scatter(
        [depot[0]],
        [depot[1]],
        marker="*",
        s=160,
        label="depot",
    )

    for agent_id, route_points in enumerate(rollout["routes_xy_unit"]):
        pts = np.asarray(route_points, dtype=np.float64)

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            marker=".",
            label=f"drone {agent_id + 1}",
        )

    ax.set_title(
        f"DAN live async rollout | makespan={rollout['makespan']:.4f}"
    )
    ax.set_xlabel("x unit")
    ax.set_ylabel("y unit")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained DAN as a live asynchronous planner offline."
    )

    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)

    parser.add_argument(
        "--mode",
        choices=["greedy", "sample", "best"],
        default="best",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Only used when --mode best.",
    )

    parser.add_argument(
        "--delta-g",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, default=None)

    args = parser.parse_args()

    if not args.world.exists():
        raise FileNotFoundError(args.world)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    world = load_world(args.world)
    instance = make_instance_from_world(world)

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    num_agents = int(world["num_agents"])

    print(f"Device: {device}")
    print(f"World: {args.world}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Agents: {num_agents}")
    print(f"Tasks: {instance.num_tasks}")
    print(f"Mode: {args.mode}")
    print()

    if args.mode in {"greedy", "sample"}:
        rollout = run_live_dan_rollout(
            model=model,
            device=device,
            instance=instance,
            num_agents=num_agents,
            delta_g=args.delta_g,
            mode=args.mode,
        )
        rollout["decode_mode"] = args.mode
        rollout["sample_index"] = 0

    else:
        best_rollout: dict | None = None

        started_at = time.perf_counter()

        for sample_index in range(args.samples):
            rollout_candidate = run_live_dan_rollout(
                model=model,
                device=device,
                instance=instance,
                num_agents=num_agents,
                delta_g=args.delta_g,
                mode="sample",
            )

            rollout_candidate["decode_mode"] = "sample"
            rollout_candidate["sample_index"] = sample_index

            if (
                best_rollout is None
                or rollout_candidate["makespan"] < best_rollout["makespan"]
            ):
                best_rollout = rollout_candidate

        if best_rollout is None:
            raise RuntimeError("No rollout candidates were generated.")

        rollout = best_rollout
        rollout["decode_mode"] = f"best_of_{args.samples}"
        rollout["total_sampling_runtime_ms"] = (
            1000.0 * (time.perf_counter() - started_at)
        )

    add_route_coordinates(
        rollout=rollout,
        world=world,
    )

    output_data = {
        "schema": "dan_offline_rollout_v1",
        "world": world,
        "checkpoint": str(args.checkpoint),
        "delta_g": args.delta_g,
        "seed": args.seed,
        "rollout": rollout,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

    if args.plot is not None:
        plot_rollout(
            world=world,
            rollout=rollout,
            output_path=args.plot,
        )

    print("DAN OFFLINE LIVE-ROLLOUT SUMMARY")
    print("-" * 80)

    for agent_id, route in enumerate(rollout["routes_task_ids"]):
        readable = [task_id + 1 for task_id in route]
        length = rollout["route_lengths"][agent_id]
        print(
            f"Drone {agent_id + 1}: "
            f"tasks={readable}, "
            f"length={length:.4f}"
        )

    print()
    print(f"Makespan:       {rollout['makespan']:.4f}")
    print(f"Total distance: {rollout['total_distance']:.4f}")
    print(f"Decisions:      {len(rollout['decision_events'])}")
    print(f"Decode mode:    {rollout['decode_mode']}")

    if "total_sampling_runtime_ms" in rollout:
        print(f"Runtime:        {rollout['total_sampling_runtime_ms']:.2f} ms")
    else:
        print(f"Runtime:        {rollout['runtime_ms']:.2f} ms")

    print()
    print(f"Saved rollout JSON: {args.output}")

    if args.plot is not None:
        print(f"Saved route plot:   {args.plot}")


if __name__ == "__main__":
    main()
