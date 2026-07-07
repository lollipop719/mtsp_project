from __future__ import annotations

import argparse

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from env.mtsp_2d_env import (
    MTSPInstance,
    evaluate_routes,
    generate_random_instance,
    validate_routes,
)


Routes = list[list[int]]

# OR-Tools uses integer costs. 1 meter becomes 1000 cost units.
DISTANCE_SCALE = 1000


def build_distance_matrix(instance: MTSPInstance) -> np.ndarray:
    """
    Node layout:
      nodes 0..2            = Drone 1/2/3 depots
      nodes 3..(3+N-1)      = task locations
    """
    locations = np.vstack([instance.depots, instance.tasks])

    distances_m = np.linalg.norm(
        locations[:, None, :] - locations[None, :, :],
        axis=-1,
    )

    return np.rint(distances_m * DISTANCE_SCALE).astype(np.int64)


def solve_ortools_minmax(
    instance: MTSPInstance,
    time_limit_s: int = 5,
) -> Routes:
    """
    Solve a 3-drone multi-depot mTSP.

    Each drone starts and ends at its own depot.
    Every task is visited exactly once.

    Primary objective:
        minimize the longest route

    Secondary objective:
        minimize total travel distance
    """
    if time_limit_s < 1:
        raise ValueError("time_limit_s must be at least 1.")

    num_drones = instance.num_drones
    num_depots = num_drones
    num_nodes = num_depots + instance.num_tasks

    distance_matrix = build_distance_matrix(instance)

    # Vehicle i starts and ends at depot node i.
    starts = list(range(num_drones))
    ends = list(range(num_drones))

    manager = pywrapcp.RoutingIndexManager(
        num_nodes,
        num_drones,
        starts,
        ends,
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return int(distance_matrix[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    # The normal arc cost is total fleet distance.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    max_arc_cost = int(distance_matrix.max())

    # A route can visit every task, then return to its depot.
    max_route_cost = max_arc_cost * (instance.num_tasks + 1)

    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        max_route_cost,
        True,  # route distance starts at zero
        "Distance",
    )

    distance_dimension = routing.GetDimensionOrDie("Distance")

    # Because all routes start at cumulative distance 0, the global span is
    # exactly the longest route distance.
    #
    # Choose a coefficient larger than every possible total-distance
    # difference, making makespan minimization dominate total-distance cost.
    max_total_cost = max_arc_cost * (
        instance.num_tasks + instance.num_drones
    )
    span_coefficient = max_total_cost + 1

    distance_dimension.SetGlobalSpanCostCoefficient(span_coefficient)

    search_parameters = (
        pywrapcp.DefaultRoutingSearchParameters()
    )

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = time_limit_s

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        raise RuntimeError("OR-Tools could not find a solution.")

    routes: Routes = []

    for drone_id in range(num_drones):
        route: list[int] = []
        index = routing.Start(drone_id)

        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))

            if routing.IsEnd(next_index):
                break

            node = manager.IndexToNode(next_index)

            # Depot nodes are 0, 1, 2. Task nodes begin at 3.
            if node >= num_depots:
                route.append(node - num_depots)

            index = next_index

        routes.append(route)

    validate_routes(instance, routes)
    return routes


def print_solution(
    instance: MTSPInstance,
    routes: Routes,
) -> None:
    metrics = evaluate_routes(instance, routes)

    print("\nOR-TOOLS MINMAX SOLUTION")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-limit", type=int, default=5)
    args = parser.parse_args()

    instance = generate_random_instance(
        num_tasks=args.tasks,
        seed=args.seed,
    )

    routes = solve_ortools_minmax(
        instance,
        time_limit_s=args.time_limit,
    )

    print_solution(instance, routes)


if __name__ == "__main__":
    main()