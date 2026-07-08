from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from baselines.greedy import greedy_minmax_insertion
from env.mtsp_2d_env import MTSPInstance, evaluate_routes
from models.centralized_attention_mtsp import CentralizedAttentionMTSP


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_seeds(text: str) -> list[int]:
    if ":" in text:
        start_text, end_text = text.split(":", maxsplit=1)
        return list(range(int(start_text), int(end_text)))

    return parse_int_list(text)


def make_depots(
    num_agents: int,
    workspace_size: float = 20.0,
) -> np.ndarray:
    """
    Deterministic depot layouts for zero-shot testing.

    For 3 agents, we keep the training depot layout.
    For other counts, we use simple symmetric layouts.
    """
    if num_agents == 1:
        return np.array(
            [[workspace_size / 2.0, workspace_size / 2.0]],
            dtype=np.float64,
        )

    if num_agents == 2:
        return np.array(
            [
                [2.0, 2.0],
                [workspace_size - 2.0, 2.0],
            ],
            dtype=np.float64,
        )

    if num_agents == 3:
        return np.array(
            [
                [2.0, 2.0],
                [workspace_size - 2.0, 2.0],
                [workspace_size / 2.0, workspace_size - 3.0],
            ],
            dtype=np.float64,
        )

    if num_agents == 4:
        return np.array(
            [
                [2.0, 2.0],
                [workspace_size - 2.0, 2.0],
                [workspace_size - 2.0, workspace_size - 2.0],
                [2.0, workspace_size - 2.0],
            ],
            dtype=np.float64,
        )

    if num_agents == 5:
        return np.array(
            [
                [2.0, 2.0],
                [workspace_size - 2.0, 2.0],
                [workspace_size - 2.0, workspace_size - 2.0],
                [2.0, workspace_size - 2.0],
                [workspace_size / 2.0, workspace_size / 2.0],
            ],
            dtype=np.float64,
        )

    center = np.array(
        [workspace_size / 2.0, workspace_size / 2.0],
        dtype=np.float64,
    )

    radius = 0.40 * workspace_size

    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        num_agents,
        endpoint=False,
    )

    depots = np.stack(
        [
            center
            + radius
            * np.array(
                [math.cos(angle), math.sin(angle)],
                dtype=np.float64,
            )
            for angle in angles
        ],
        axis=0,
    )

    return depots.astype(np.float64)


def generate_zero_shot_instance(
    *,
    num_agents: int,
    num_tasks: int,
    seed: int,
    workspace_size: float = 20.0,
) -> MTSPInstance:
    rng = np.random.default_rng(seed)

    depots = make_depots(
        num_agents=num_agents,
        workspace_size=workspace_size,
    )

    tasks = rng.uniform(
        low=1.0,
        high=workspace_size - 1.0,
        size=(num_tasks, 2),
    ).astype(np.float64)

    return MTSPInstance(
        depots=depots,
        tasks=tasks,
    )


def actions_to_routes(
    actions: np.ndarray,
    num_agents: int,
    num_tasks: int,
) -> list[list[int]]:
    routes: list[list[int]] = [[] for _ in range(num_agents)]

    seen: set[int] = set()

    for drone_id, task_id in actions:
        drone = int(drone_id)
        task = int(task_id)

        if not 0 <= drone < num_agents:
            raise ValueError(f"Invalid drone id: {drone}")

        if not 0 <= task < num_tasks:
            raise ValueError(f"Invalid task id: {task}")

        if task in seen:
            raise ValueError(f"Duplicate task assignment: {task}")

        routes[drone].append(task)
        seen.add(task)

    if seen != set(range(num_tasks)):
        missing = sorted(set(range(num_tasks)) - seen)
        raise ValueError(f"Missing tasks: {missing}")

    return routes


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


def solve_ortools_minmax_generic(
    instance: MTSPInstance,
    time_limit_s: float,
) -> list[list[int]] | None:
    """
    Generic multi-depot MinMax mTSP OR-Tools reference.

    Works for variable numbers of agents and tasks.
    """
    depots = np.asarray(instance.depots, dtype=np.float64)
    tasks = np.asarray(instance.tasks, dtype=np.float64)

    num_agents = depots.shape[0]
    num_tasks = tasks.shape[0]

    locations = np.concatenate(
        [depots, tasks],
        axis=0,
    )

    num_locations = locations.shape[0]

    starts = list(range(num_agents))
    ends = list(range(num_agents))

    manager = pywrapcp.RoutingIndexManager(
        num_locations,
        num_agents,
        starts,
        ends,
    )

    routing = pywrapcp.RoutingModel(manager)

    scale = 1000

    distance_matrix = np.zeros(
        (num_locations, num_locations),
        dtype=np.int64,
    )

    for i in range(num_locations):
        for j in range(num_locations):
            distance = np.linalg.norm(
                locations[i] - locations[j]
            )

            distance_matrix[i, j] = int(round(scale * distance))

    def distance_callback(
        from_index: int,
        to_index: int,
    ) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return int(distance_matrix[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    max_distance = int(
        round(
            scale
            * 10.0
            * 20.0
            * max(num_tasks, 1)
        )
    )

    routing.AddDimension(
        transit_callback_index,
        0,
        max_distance,
        True,
        "Distance",
    )

    distance_dimension = routing.GetDimensionOrDie("Distance")

    # Encourage MinMax route balancing.
    distance_dimension.SetGlobalSpanCostCoefficient(
        1_000_000
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = int(time_limit_s)
    search_parameters.time_limit.nanos = int(
        (time_limit_s - int(time_limit_s)) * 1_000_000_000
    )

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return None

    routes: list[list[int]] = []

    for vehicle_id in range(num_agents):
        route: list[int] = []

        index = routing.Start(vehicle_id)

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)

            if node >= num_agents:
                route.append(node - num_agents)

            index = solution.Value(routing.NextVar(index))

        routes.append(route)

    return routes


def percent_gap(
    value: float,
    reference: float,
) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return float("nan")

    return 100.0 * (value - reference) / reference


def summarize_rows(
    rows: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    groups: dict[tuple[int, int], list[dict[str, float | int]]] = defaultdict(list)

    for row in rows:
        key = (
            int(row["num_agents"]),
            int(row["num_tasks"]),
        )
        groups[key].append(row)

    summaries: list[dict[str, float | int]] = []

    for (num_agents, num_tasks), group_rows in sorted(groups.items()):
        def mean(key: str) -> float:
            values = np.array(
                [float(row[key]) for row in group_rows],
                dtype=np.float64,
            )
            return float(np.nanmean(values))

        summaries.append(
            {
                "num_agents": num_agents,
                "num_tasks": num_tasks,
                "num_instances": len(group_rows),
                "learned_makespan_m": mean("learned_makespan_m"),
                "greedy_makespan_m": mean("greedy_makespan_m"),
                "ortools_makespan_m": mean("ortools_makespan_m"),
                "learned_gap_to_ortools_pct": mean(
                    "learned_gap_to_ortools_pct"
                ),
                "greedy_gap_to_ortools_pct": mean(
                    "greedy_gap_to_ortools_pct"
                ),
                "learned_inference_ms": mean("learned_inference_ms"),
                "ortools_runtime_ms": mean("ortools_runtime_ms"),
            }
        )

    return summaries


def write_csv(
    path: Path,
    rows: list[dict[str, float | int]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of learned mTSP model."
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
        default="3",
        help="Comma-separated agent counts, e.g. 2,3,4,5.",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default="10,15,20,25",
        help="Comma-separated task counts.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0:50",
        help="Either comma list like 1,2,3 or range start:end.",
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
        default=Path("outputs/zero_shot"),
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

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

    rows: list[dict[str, float | int]] = []

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Agents: {agent_counts}")
    print(f"Tasks: {task_counts}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} instances per setting)")
    print()

    for num_agents in agent_counts:
        for num_tasks in task_counts:
            print(
                f"Evaluating zero-shot: "
                f"{num_agents} agents, {num_tasks} tasks"
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

                torch.cuda.synchronize() if device.type == "cuda" else None
                learned_start = time.perf_counter()

                with torch.inference_mode():
                    learned_actions = model.decode(
                        depots=depots_tensor,
                        tasks=tasks_tensor,
                    )[0].cpu().numpy()

                torch.cuda.synchronize() if device.type == "cuda" else None
                learned_elapsed_ms = (
                    time.perf_counter() - learned_start
                ) * 1000.0

                learned_routes = actions_to_routes(
                    learned_actions,
                    num_agents=num_agents,
                    num_tasks=num_tasks,
                )

                learned_metrics = evaluate_routes(
                    instance,
                    learned_routes,
                )

                try:
                    greedy_routes = greedy_minmax_insertion(instance)
                    greedy_metrics = evaluate_routes(
                        instance,
                        greedy_routes,
                    )
                    greedy_makespan = greedy_metrics.makespan
                except Exception:
                    greedy_makespan = float("nan")

                ortools_start = time.perf_counter()

                ortools_routes = solve_ortools_minmax_generic(
                    instance,
                    time_limit_s=args.ortools_time_limit,
                )

                ortools_elapsed_ms = (
                    time.perf_counter() - ortools_start
                ) * 1000.0

                if ortools_routes is None:
                    ortools_makespan = float("nan")
                else:
                    ortools_metrics = evaluate_routes(
                        instance,
                        ortools_routes,
                    )
                    ortools_makespan = ortools_metrics.makespan

                row = {
                    "num_agents": num_agents,
                    "num_tasks": num_tasks,
                    "seed": seed,
                    "learned_makespan_m": learned_metrics.makespan,
                    "greedy_makespan_m": greedy_makespan,
                    "ortools_makespan_m": ortools_makespan,
                    "learned_gap_to_ortools_pct": percent_gap(
                        learned_metrics.makespan,
                        ortools_makespan,
                    ),
                    "greedy_gap_to_ortools_pct": percent_gap(
                        greedy_makespan,
                        ortools_makespan,
                    ),
                    "learned_inference_ms": learned_elapsed_ms,
                    "ortools_runtime_ms": ortools_elapsed_ms,
                }

                rows.append(row)

    summaries = summarize_rows(rows)

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_instance_path = output_dir / "per_instance.csv"
    summary_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"

    write_csv(per_instance_path, rows)
    write_csv(summary_path, summaries)

    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print()
    print("ZERO-SHOT SUMMARY")
    print("-" * 86)

    for summary in summaries:
        print(
            f"{int(summary['num_agents'])} agents, "
            f"{int(summary['num_tasks'])} tasks | "
            f"learned={summary['learned_makespan_m']:.2f} m "
            f"({summary['learned_gap_to_ortools_pct']:+.2f}% vs OR-Tools) | "
            f"greedy={summary['greedy_makespan_m']:.2f} m "
            f"({summary['greedy_gap_to_ortools_pct']:+.2f}%) | "
            f"OR-Tools={summary['ortools_makespan_m']:.2f} m"
        )

    print()
    print(f"Saved: {per_instance_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_json_path}")


if __name__ == "__main__":
    main()
