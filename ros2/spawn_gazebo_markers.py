from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from ros2.mission_config import PROJECT_ROOT, planner_to_gazebo_enu


def make_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_command(
    command: list[str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if check and result.returncode != 0:
        print("Command failed:")
        print(" ".join(command))
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError("Command failed.")

    return result


def detect_world_create_service() -> str:
    result = run_command(["gz", "service", "-l"])

    candidates = [
        service.strip()
        for service in result.stdout.splitlines()
        if service.strip().startswith("/world/")
        and service.strip().endswith("/create")
    ]

    if not candidates:
        raise RuntimeError(
            "Could not find a Gazebo create service. Is Gazebo running?"
        )

    return candidates[0]


def material_sdf(
    red: float,
    green: float,
    blue: float,
    alpha: float = 1.0,
) -> str:
    return f"""
        <material>
          <ambient>{red} {green} {blue} {alpha}</ambient>
          <diffuse>{red} {green} {blue} {alpha}</diffuse>
          <specular>0.2 0.2 0.2 {alpha}</specular>
        </material>
    """


def sphere_model_sdf(
    name: str,
    radius: float,
    color: tuple[float, float, float],
) -> str:
    red, green, blue = color

    return f"""
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>{radius}</radius>
          </sphere>
        </geometry>
        {material_sdf(red, green, blue)}
      </visual>
    </link>
  </model>
</sdf>
"""


def box_model_sdf(
    name: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> str:
    red, green, blue = color
    sx, sy, sz = size

    return f"""
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
        {material_sdf(red, green, blue)}
      </visual>
    </link>
  </model>
</sdf>
"""


def spawn_sdf_model(
    *,
    create_service: str,
    name: str,
    sdf: str,
    position: tuple[float, float, float],
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    x, y, z = position
    roll, pitch, yaw = rpy

    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    request = (
        f'name: "{name}", '
        f'allow_renaming: true, '
        f'sdf: {json.dumps(sdf)}, '
        f'pose: {{ '
        f'position: {{ x: {x:.6f}, y: {y:.6f}, z: {z:.6f} }}, '
        f'orientation: {{ x: {qx:.8f}, y: {qy:.8f}, '
        f'z: {qz:.8f}, w: {qw:.8f} }} '
        f'}}'
    )

    result = run_command(
        [
            "gz",
            "service",
            "-s",
            create_service,
            "--reqtype",
            "gz.msgs.EntityFactory",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            request,
        ],
        check=False,
    )

    if result.returncode != 0:
        print("Failed to spawn:", name)
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Failed to spawn Gazebo model.")


def agent_color(index: int) -> tuple[float, float, float]:
    colors = [
        (1.0, 0.05, 0.05),
        (0.05, 1.0, 0.05),
        (0.05, 0.25, 1.0),
        (1.0, 0.65, 0.05),
        (0.65, 0.05, 1.0),
        (0.05, 1.0, 1.0),
        (1.0, 0.05, 0.6),
        (0.5, 0.5, 0.5),
    ]
    return colors[index % len(colors)]


def assigned_agent_by_task(routes: list[list[int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}

    for agent_index, route in enumerate(routes):
        for task_id in route:
            mapping[int(task_id)] = agent_index

    return mapping


def spawn_route_segment(
    *,
    create_service: str,
    name: str,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    color: tuple[float, float, float],
) -> None:
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)

    delta = end - start
    length = float(np.linalg.norm(delta))

    if length < 1e-6:
        return

    midpoint = 0.5 * (start + end)
    yaw = math.atan2(float(delta[1]), float(delta[0]))

    sdf = box_model_sdf(
        name=name,
        size=(length, 0.07, 0.035),
        color=color,
    )

    spawn_sdf_model(
        create_service=create_service,
        name=name,
        sdf=sdf,
        position=(
            float(midpoint[0]),
            float(midpoint[1]),
            0.06,
        ),
        rpy=(0.0, 0.0, yaw),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spawn static mTSP visuals into Gazebo."
    )

    parser.add_argument(
        "--mission",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gazebo_missions"
        / "mission_seed_20260707.json",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    mission_path = make_project_path(args.mission)

    if not mission_path.exists():
        raise FileNotFoundError(f"Mission JSON not found: {mission_path}")

    if args.prefix is None:
        args.prefix = f"mtsp_{int(time.time())}"

    with mission_path.open("r", encoding="utf-8") as file:
        mission = json.load(file)

    depots_xy = np.asarray(
        mission["planner"]["depots_xy"],
        dtype=np.float64,
    )

    tasks_xy = np.asarray(
        mission["planner"]["tasks_xy"],
        dtype=np.float64,
    )

    routes = [
        [int(task_id) for task_id in route]
        for route in mission["planner"]["routes_task_ids"]
    ]

    create_service = detect_world_create_service()

    print(f"Using Gazebo create service: {create_service}")
    print(f"Spawn prefix: {args.prefix}")

    task_to_agent = assigned_agent_by_task(routes)

    for agent_index, depot_xy in enumerate(depots_xy):
        depot_enu = planner_to_gazebo_enu(depot_xy)
        color = agent_color(agent_index)
        name = f"{args.prefix}_depot_D{agent_index + 1}"

        spawn_sdf_model(
            create_service=create_service,
            name=name,
            sdf=box_model_sdf(
                name=name,
                size=(0.70, 0.70, 0.40),
                color=color,
            ),
            position=(
                float(depot_enu[0]),
                float(depot_enu[1]),
                0.20,
            ),
        )

    for task_id, task_xy in enumerate(tasks_xy):
        task_enu = planner_to_gazebo_enu(task_xy)
        assigned_agent = task_to_agent[task_id]
        color = agent_color(assigned_agent)
        name = f"{args.prefix}_task_T{task_id + 1}"

        spawn_sdf_model(
            create_service=create_service,
            name=name,
            sdf=sphere_model_sdf(
                name=name,
                radius=0.23,
                color=color,
            ),
            position=(
                float(task_enu[0]),
                float(task_enu[1]),
                0.25,
            ),
        )

    for agent_index, route in enumerate(routes):
        color = agent_color(agent_index)
        depot_enu = planner_to_gazebo_enu(depots_xy[agent_index])

        points = [depot_enu]

        for task_id in route:
            points.append(planner_to_gazebo_enu(tasks_xy[task_id]))

        points.append(depot_enu)

        for segment_index in range(len(points) - 1):
            spawn_route_segment(
                create_service=create_service,
                name=(
                    f"{args.prefix}_route_D{agent_index + 1}_"
                    f"S{segment_index + 1}"
                ),
                start_xy=points[segment_index],
                end_xy=points[segment_index + 1],
                color=color,
            )

    print("Done.")
    print(
        "Gazebo should now show depot cubes, task spheres, and route strips "
        "for all agents."
    )


if __name__ == "__main__":
    main()
