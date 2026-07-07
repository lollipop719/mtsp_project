from __future__ import annotations

import argparse
from pathlib import Path

from baselines.greedy import greedy_minmax_insertion, round_robin_routes
from baselines.ortools_solver import solve_ortools_minmax
from env.mtsp_2d_env import generate_random_instance
from evaluation.evaluate import plot_routes, save_solution_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ortools-time-limit", type=int, default=5)
    args = parser.parse_args()

    instance = generate_random_instance(
        num_tasks=args.tasks,
        seed=args.seed,
    )

    solutions = {
        "round_robin": round_robin_routes(instance),
        "greedy": greedy_minmax_insertion(instance),
        "ortools": solve_ortools_minmax(
            instance,
            time_limit_s=args.ortools_time_limit,
        ),
    }

    output_dir = Path("outputs") / f"tasks_{args.tasks}_seed_{args.seed}"

    print(f"Saving results in: {output_dir}")

    for solver_name, routes in solutions.items():
        metrics = plot_routes(
            instance=instance,
            routes=routes,
            title=solver_name.replace("_", " ").title(),
            output_path=output_dir / f"{solver_name}.png",
        )

        save_solution_json(
            instance=instance,
            routes=routes,
            solver_name=solver_name,
            output_path=output_dir / f"{solver_name}.json",
        )

        print(
            f"{solver_name:>12}: "
            f"makespan={metrics.makespan:.2f} m | "
            f"total={metrics.total_distance:.2f} m"
        )


if __name__ == "__main__":
    main()