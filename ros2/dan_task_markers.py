from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleLocalPosition


PLANNER_TO_GAZEBO_SCALE = 0.80
GAZEBO_WORLD_OFFSET_ENU_M = np.asarray([-8.0, -8.0], dtype=float)


def planner_to_gazebo_enu(xy: np.ndarray) -> np.ndarray:
    return GAZEBO_WORLD_OFFSET_ENU_M + PLANNER_TO_GAZEBO_SCALE * xy


def rgba(values: tuple[float, float, float, float]) -> str:
    return " ".join(str(v) for v in values)


def cylinder_sdf(
    name: str,
    x: float,
    y: float,
    z: float,
    radius: float,
    length: float,
    color: tuple[float, float, float, float],
) -> str:
    color_txt = rgba(color)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius:.4f}</radius>
            <length>{length:.4f}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{color_txt}</ambient>
          <diffuse>{color_txt}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


class DANTaskMarkers(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("dan_task_markers")
        self.args = args

        world = json.loads(Path(args.mission).read_text())

        tasks_planner = world.get("tasks_xy")
        if tasks_planner is None:
            tasks_planner = world["planner"]["tasks_xy"]

        depots_planner = world.get("depots_xy")
        if depots_planner is None:
            depots_planner = world["planner"]["depots_xy"]

        self.tasks_enu = np.asarray(
            [planner_to_gazebo_enu(np.asarray(xy, dtype=float)) for xy in tasks_planner],
            dtype=float,
        )
        self.depots_enu = np.asarray(
            [planner_to_gazebo_enu(np.asarray(xy, dtype=float)) for xy in depots_planner],
            dtype=float,
        )

        self.num_agents = args.num_agents or len(self.depots_enu)
        self.positions_enu: list[np.ndarray | None] = [None for _ in range(self.num_agents)]
        self.finished = np.zeros(len(self.tasks_enu), dtype=bool)

        self.marker_dir = Path("/tmp/dan_gazebo_markers")
        self.marker_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.num_agents):
            self.create_subscription(
                VehicleLocalPosition,
                f"/px4_{i + 1}/fmu/out/vehicle_local_position_v1",
                self.make_position_callback(i),
                10,
            )

        self.static_markers_spawned = False
        self.marker_queue: list[tuple[str, str]] = []
        self.airborne_since_s: float | None = None
        self.last_marker_spawn_s: float = 0.0
        self.timer = self.create_timer(0.25, self.timer_callback)

        self.get_logger().info(
            f"Marker node ready. Waiting for drones to become airborne before spawning markers. "
            f"Watching {self.num_agents} drones and {len(self.tasks_enu)} tasks."
        )

    def make_position_callback(self, index: int):
        def callback(msg: VehicleLocalPosition) -> None:
            # PX4 local position is NED relative to that vehicle's home:
            # x=north, y=east, z=down.
            home = self.depots_enu[index]
            current_enu = home + np.asarray([msg.y, msg.x], dtype=float)
            altitude = -float(msg.z)
            self.positions_enu[index] = np.asarray(
                [current_enu[0], current_enu[1], altitude],
                dtype=float,
            )

        return callback

    def build_static_marker_queue(self) -> None:
        # Orange task discs.
        for task_id, xy in enumerate(self.tasks_enu, start=1):
            name = f"dan_task_{task_id:03d}"
            sdf = cylinder_sdf(
                name=name,
                x=float(xy[0]),
                y=float(xy[1]),
                z=0.035,
                radius=self.args.task_radius,
                length=0.07,
                color=(1.0, 0.45, 0.05, 1.0),
            )
            self.marker_queue.append((name, sdf))

        # Blue depot discs.
        for drone_id, xy in enumerate(self.depots_enu, start=1):
            name = f"dan_depot_{drone_id:02d}"
            sdf = cylinder_sdf(
                name=name,
                x=float(xy[0]),
                y=float(xy[1]),
                z=0.025,
                radius=self.args.depot_radius,
                length=0.05,
                color=(0.05, 0.25, 1.0, 1.0),
            )
            self.marker_queue.append((name, sdf))

    def timer_callback(self) -> None:
        now = self.get_clock().now().nanoseconds * 1.0e-9

        if not self.static_markers_spawned:
            positions_ready = all(p is not None for p in self.positions_enu)
            all_airborne = (
                positions_ready
                and all(float(p[2]) >= self.args.spawn_min_altitude for p in self.positions_enu if p is not None)
            )

            if all_airborne:
                if self.airborne_since_s is None:
                    self.airborne_since_s = now
                    self.get_logger().info("All drones are above marker spawn altitude. Waiting for stability...")
                elif now - self.airborne_since_s >= self.args.spawn_stable_s:
                    self.build_static_marker_queue()
                    self.static_markers_spawned = True
                    self.get_logger().info(
                        f"Starting slow marker spawn. queued={len(self.marker_queue)}"
                    )
            else:
                self.airborne_since_s = None

            return

        if self.marker_queue:
            if now - self.last_marker_spawn_s >= self.args.spawn_period_s:
                name, sdf = self.marker_queue.pop(0)
                self.spawn_sdf(name, sdf)
                self.last_marker_spawn_s = now
            return

        self.check_finished_tasks()

    def check_finished_tasks(self) -> None:
        available_positions = [p for p in self.positions_enu if p is not None]
        if not available_positions:
            return

        for task_idx, task_xy in enumerate(self.tasks_enu):
            if self.finished[task_idx]:
                continue

            for pos in available_positions:
                if pos[2] < self.args.min_altitude:
                    continue

                dist = float(np.linalg.norm(pos[:2] - task_xy))
                if dist <= self.args.done_radius:
                    self.finished[task_idx] = True
                    self.spawn_done_marker(task_idx, task_xy)
                    self.get_logger().info(
                        f"Task {task_idx + 1} marked finished. dist={dist:.2f} m"
                    )
                    break

    def spawn_done_marker(self, task_idx: int, xy: np.ndarray) -> None:
        # Green larger disc slightly above the orange task marker.
        task_id = task_idx + 1
        name = f"dan_task_done_{task_id:03d}"
        sdf = cylinder_sdf(
            name=name,
            x=float(xy[0]),
            y=float(xy[1]),
            z=0.11,
            radius=self.args.done_radius_visual,
            length=0.08,
            color=(0.0, 1.0, 0.15, 1.0),
        )
        self.spawn_sdf(name, sdf)

    def spawn_sdf(self, name: str, sdf_text: str) -> None:
        sdf_path = self.marker_dir / f"{name}.sdf"
        sdf_path.write_text(sdf_text)

        cmd = [
            "gz",
            "service",
            "-s",
            f"/world/{self.args.world}/create",
            "--reqtype",
            "gz.msgs.EntityFactory",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "1000",
            "--req",
            f'sdf_filename: "{sdf_path}"',
        ]

        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            self.get_logger().warn(
                f"Failed to spawn {name}. stderr={result.stderr.strip()}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--world", default="default")
    parser.add_argument("--done-radius", type=float, default=0.75)
    parser.add_argument("--min-altitude", type=float, default=1.0)
    parser.add_argument(
        "--spawn-min-altitude",
        type=float,
        default=2.7,
        help="Spawn static task/depot markers only after all drones are above this altitude.",
    )
    parser.add_argument(
        "--spawn-stable-s",
        type=float,
        default=5.0,
        help="After all drones pass spawn-min-altitude, wait this many seconds before spawning markers.",
    )
    parser.add_argument(
        "--spawn-period-s",
        type=float,
        default=0.25,
        help="Delay between Gazebo marker spawn requests.",
    )
    parser.add_argument("--task-radius", type=float, default=0.18)
    parser.add_argument("--depot-radius", type=float, default=0.28)
    parser.add_argument("--done-radius-visual", type=float, default=0.28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = DANTaskMarkers(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
