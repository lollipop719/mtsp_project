from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ros2.mission_config import planner_to_gazebo_enu


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_task_positions_planner(mission_path: Path) -> np.ndarray:
    with mission_path.open("r", encoding="utf-8") as file:
        mission: dict[str, Any] = json.load(file)

    planner = mission.get("planner", {})

    if "tasks_xy_planner" in mission:
        tasks = mission["tasks_xy_planner"]
    elif "tasks_xy" in mission:
        tasks = mission["tasks_xy"]
    elif "tasks_xy" in planner:
        tasks = planner["tasks_xy"]
    elif "tasks_xy_unit" in mission:
        planner_size = float(
            mission.get("coordinate_frame", {}).get("planner_size", 20.0)
        )
        tasks = np.asarray(
            mission["tasks_xy_unit"],
            dtype=np.float64,
        ) * planner_size
    else:
        raise KeyError(
            "Mission does not contain tasks_xy_planner, tasks_xy, "
            "planner.tasks_xy, or tasks_xy_unit."
        )

    tasks_array = np.asarray(tasks, dtype=np.float64)

    if tasks_array.ndim != 2 or tasks_array.shape[1] != 2:
        raise ValueError(
            f"Task positions must have shape (N, 2), got {tasks_array.shape}"
        )

    return tasks_array


def find_create_service(timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    last_error = ""

    while time.monotonic() < deadline:
        result = subprocess.run(
            ["gz", "service", "-l"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if result.returncode == 0:
            for raw_line in result.stdout.splitlines():
                service = raw_line.strip()

                if service.startswith("/world/") and service.endswith("/create"):
                    print(f"Gazebo create service detected: {service}")
                    return service
        else:
            last_error = result.stderr.strip()

        time.sleep(1.0)

    raise RuntimeError(
        "Timed out waiting for Gazebo create service. "
        f"Last error: {last_error}"
    )


def make_marker_visual(
    task_number: int,
    east: float,
    north: float,
    *,
    radius_m: float,
    height_m: float,
    red: float,
    green: float,
    blue: float,
    alpha: float,
) -> str:
    z = height_m / 2.0

    return f"""
      <visual name="task_marker_{task_number:03d}">
        <pose>{east:.6f} {north:.6f} {z:.6f} 0 0 0</pose>

        <geometry>
          <cylinder>
            <radius>{radius_m:.6f}</radius>
            <length>{height_m:.6f}</length>
          </cylinder>
        </geometry>

        <material>
          <ambient>{red:.4f} {green:.4f} {blue:.4f} {alpha:.4f}</ambient>
          <diffuse>{red:.4f} {green:.4f} {blue:.4f} {alpha:.4f}</diffuse>
          <specular>0.15 0.15 0.15 1.0</specular>
        </material>

        <cast_shadows>false</cast_shadows>
      </visual>
"""


def build_markers_sdf(
    task_positions_enu: np.ndarray,
    *,
    radius_m: float,
    height_m: float,
) -> str:
    visuals: list[str] = []

    # Bright orange-red so they remain visible from the Gazebo camera.
    marker_color = (1.0, 0.15, 0.02, 1.0)

    for task_index, task_enu in enumerate(task_positions_enu, start=1):
        east = float(task_enu[0])
        north = float(task_enu[1])

        visuals.append(
            make_marker_visual(
                task_index,
                east,
                north,
                radius_m=radius_m,
                height_m=height_m,
                red=marker_color[0],
                green=marker_color[1],
                blue=marker_color[2],
                alpha=marker_color[3],
            )
        )

    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="dan_static_task_markers">
    <static>true</static>

    <link name="task_marker_visuals">
{''.join(visuals)}
    </link>
  </model>
</sdf>
"""


def spawn_sdf(
    *,
    create_service: str,
    sdf_path: Path,
) -> None:
    request = (
        f'sdf_filename: "{sdf_path}", '
        'name: "dan_static_task_markers"'
    )

    command = [
        "gz",
        "service",
        "-s",
        create_service,
        "--reqtype",
        "gz.msgs.EntityFactory",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "10000",
        "--req",
        request,
    ]

    print("Spawning one static Gazebo model containing all task markers...")
    print(" ".join(command[:-1]))
    print(f"Request: {request}")

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    output = "\n".join(
        part for part in (result.stdout.strip(), result.stderr.strip()) if part
    )

    if output:
        print(output)

    if result.returncode != 0:
        raise RuntimeError(
            f"Gazebo marker spawn command failed with code {result.returncode}"
        )

    if "data: true" not in output.lower():
        raise RuntimeError(
            "Gazebo did not confirm successful marker creation. "
            "Use --clean-start to ensure an older marker model is not present."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spawn fixed task-location markers into a running Gazebo world."
    )

    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--service-timeout-s", type=float, default=60.0)

    parser.add_argument(
        "--radius-m",
        type=float,
        default=0.24,
        help="Radius of every task marker.",
    )
    parser.add_argument(
        "--height-m",
        type=float,
        default=0.10,
        help="Height of every task marker.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mission_path = resolve_path(args.mission)

    if not mission_path.exists():
        raise FileNotFoundError(f"Mission file not found: {mission_path}")

    task_positions_planner = load_task_positions_planner(mission_path)

    task_positions_enu = np.stack(
        [
            np.asarray(
                planner_to_gazebo_enu(task_xy),
                dtype=np.float64,
            )
            for task_xy in task_positions_planner
        ],
        axis=0,
    )

    output_dir = PROJECT_ROOT / "outputs" / "gazebo_markers"
    output_dir.mkdir(parents=True, exist_ok=True)

    sdf_path = output_dir / f"{mission_path.stem}_static_task_markers.sdf"

    sdf_text = build_markers_sdf(
        task_positions_enu,
        radius_m=args.radius_m,
        height_m=args.height_m,
    )
    sdf_path.write_text(sdf_text, encoding="utf-8")

    print(f"Mission: {mission_path}")
    print(f"Number of task markers: {len(task_positions_enu)}")
    print(f"Generated SDF: {sdf_path}")

    for index, task_enu in enumerate(task_positions_enu, start=1):
        print(
            f"  Task {index:02d}: "
            f"Gazebo ENU=({task_enu[0]:.3f}, {task_enu[1]:.3f})"
        )

    create_service = find_create_service(args.service_timeout_s)

    spawn_sdf(
        create_service=create_service,
        sdf_path=sdf_path,
    )

    print("Static task markers successfully spawned.")
    print("Marker spawner is exiting; no marker node remains running.")


if __name__ == "__main__":
    main()
