from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from env.mtsp_2d_env import (
    MTSPInstance,
    RouteMetrics,
    evaluate_routes,
    generate_random_instance,
    route_distance,
    validate_routes,
)


Routes = list[list[int]]


def partial_route_lengths(
    instance: MTSPInstance,
    routes: Sequence[Sequence[int]],
) -> np.ndarray:
    """
    Route lengths for a partial assignment.

    Unlike evaluate_routes(), this allows unassigned tasks because greedy
    construction happens one task at a time.
    """
    return np.asarray(
        [
            route_distance(instance, drone_id, route)
            for drone_id, route in enumerate(routes)
        ],
        dtype=np.float64,
    )


def partial_objective(
    instance: MTSPInstance,
    routes: Sequence[Sequence[int]],
) -> tuple[float, float, float]:
    """
    Objective for an incomplete solution.

    Returns:
        current makespan,
        current total distance,
        route-length imbalance.

    The imbalance is only a deterministic tie-breaker after makespan and
    total distance.
    """
    lengths = partial_route_lengths(instance, routes)

    makespan = float(np.max(lengths))
    total_distance = float(np.sum(lengths))
    imbalance = float(np.max(lengths) - np.min(lengths))

    return makespan, total_distance, imbalance


def greedy_minmax_insertion(instance: MTSPInstance) -> Routes:
    """
    Centralized greedy construction heuristic.

    At each step:
      1. Consider every remaining task.
      2. Try inserting it into every possible position of every drone route.
      3. Keep the insertion with the best fleet-level partial objective.

    This is a baseline, not an exact solver.
    """
    routes: Routes = [[] for _ in range(instance.num_drones)]
    unassigned_tasks = set(range(instance.num_tasks))

    while unassigned_tasks:
        best_routes: Routes | None = None
        best_task_id: int | None = None
        best_key: tuple[float, float, float, int, int, int] | None = None

        for task_id in sorted(unassigned_tasks):
            for drone_id in range(instance.num_drones):
                for insert_index in range(len(routes[drone_id]) + 1):
                    candidate_routes = [route.copy() for route in routes]
                    candidate_routes[drone_id].insert(
                        insert_index,
                        task_id,
                    )

                    makespan, total_distance, imbalance = partial_objective(
                        instance,
                        candidate_routes,
                    )

                    # Last three entries make equal-cost choices repeatable.
                    key = (
                        makespan,
                        total_distance,
                        imbalance,
                        task_id,
                        drone_id,
                        insert_index,
                    )

                    if best_key is None or key < best_key:
                        best_key = key
                        best_routes = candidate_routes
                        best_task_id = task_id

        if best_routes is None or best_task_id is None:
            raise RuntimeError("Greedy construction failed unexpectedly.")

        routes = best_routes
        unassigned_tasks.remove(best_task_id)

    validate_routes(instance, routes)
    return routes


def round_robin_routes(instance: MTSPInstance) -> Routes:
    """The simple valid-but-weak assignment used in the environment self-test."""
    return [
        list(range(drone_id, instance.num_tasks, instance.num_drones))
        for drone_id in range(instance.num_drones)
    ]


def print_solution(
    name: str,
    instance: MTSPInstance,
    routes: Routes,
) -> RouteMetrics:
    metrics = evaluate_routes(instance, routes)

    print(f"\n{name}")
    print("-" * 60)

    for drone_id, route in enumerate(routes):
        one_indexed_tasks = [task_id + 1 for task_id in route]

        print(
            f"Drone {drone_id + 1}: "
            f"tasks={one_indexed_tasks} | "
            f"distance={metrics.per_drone_distance[drone_id]:.2f} m"
        )

    print(f"Makespan: {metrics.makespan:.2f} m")
    print(f"Total distance: {metrics.total_distance:.2f} m")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    instance = generate_random_instance(
        num_tasks=args.tasks,
        seed=args.seed,
    )

    round_robin = round_robin_routes(instance)
    greedy = greedy_minmax_insertion(instance)

    round_robin_metrics = print_solution(
        "ROUND-ROBIN BASELINE",
        instance,
        round_robin,
    )

    greedy_metrics = print_solution(
        "CENTRALIZED GREEDY MINMAX INSERTION",
        instance,
        greedy,
    )

    makespan_improvement = (
        round_robin_metrics.makespan - greedy_metrics.makespan
    )

    total_improvement = (
        round_robin_metrics.total_distance - greedy_metrics.total_distance
    )

    print("\nIMPROVEMENT OVER ROUND-ROBIN")
    print("-" * 60)
    print(f"Makespan reduction: {makespan_improvement:.2f} m")
    print(f"Total-distance reduction: {total_improvement:.2f} m")


if __name__ == "__main__":
    main()