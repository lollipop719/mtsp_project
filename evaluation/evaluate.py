from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

# Allows plots to be saved inside Docker without needing a GUI backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from env.mtsp_2d_env import MTSPInstance, RouteMetrics, evaluate_routes


Routes = Sequence[Sequence[int]]


def plot_routes(
    instance: MTSPInstance,
    routes: Routes,
    title: str,
    output_path: str | Path,
) -> RouteMetrics:
    """
    Save a 2D visualization of three drone routes.

    Each route starts and ends at its corresponding depot.
    """
    metrics = evaluate_routes(instance, routes)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 8))

    # Task locations.
    axis.scatter(
        instance.tasks[:, 0],
        instance.tasks[:, 1],
        marker="o",
        s=70,
        label="Tasks",
        zorder=3,
    )

    for task_id, task_position in enumerate(instance.tasks):
        axis.annotate(
            str(task_id + 1),
            xy=(task_position[0], task_position[1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Drone start/end depots.
    axis.scatter(
        instance.depots[:, 0],
        instance.depots[:, 1],
        marker="s",
        s=140,
        label="Depots",
        zorder=4,
    )

    for drone_id, depot_position in enumerate(instance.depots):
        axis.annotate(
            f"D{drone_id + 1}",
            xy=(depot_position[0], depot_position[1]),
            xytext=(6, -14),
            textcoords="offset points",
            fontweight="bold",
        )

    # Matplotlib automatically gives each drone a distinct default line color.
    for drone_id, route in enumerate(routes):
        points = [instance.depots[drone_id]]

        for task_id in route:
            points.append(instance.tasks[task_id])

        points.append(instance.depots[drone_id])
        points_array = np.asarray(points, dtype=np.float64)

        axis.plot(
            points_array[:, 0],
            points_array[:, 1],
            marker="o",
            linewidth=2,
            label=(
                f"Drone {drone_id + 1} "
                f"({metrics.per_drone_distance[drone_id]:.2f} m)"
            ),
        )

    axis.set_title(
        f"{title}\n"
        f"Makespan: {metrics.makespan:.2f} m | "
        f"Total: {metrics.total_distance:.2f} m"
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(0, 20)
    axis.set_ylim(0, 20)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    return metrics


def save_solution_json(
    instance: MTSPInstance,
    routes: Routes,
    solver_name: str,
    output_path: str | Path,
) -> None:
    """Save one route solution and its metrics for later comparisons."""
    metrics = evaluate_routes(instance, routes)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "solver": solver_name,
        "depots": instance.depots.tolist(),
        "tasks": instance.tasks.tolist(),
        "routes_zero_indexed": [list(route) for route in routes],
        "routes_one_indexed": [
            [task_id + 1 for task_id in route]
            for route in routes
        ],
        "per_drone_distance_m": metrics.per_drone_distance.tolist(),
        "makespan_m": metrics.makespan,
        "total_distance_m": metrics.total_distance,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)