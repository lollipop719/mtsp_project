from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np


PLANNER_TO_GAZEBO_SCALE = 0.80
GAZEBO_WORLD_OFFSET_ENU_M = np.asarray([-8.0, -8.0], dtype=float)


def planner_to_gazebo_enu(xy: np.ndarray) -> np.ndarray:
    return GAZEBO_WORLD_OFFSET_ENU_M + PLANNER_TO_GAZEBO_SCALE * xy


def cylinder_sdf(
    name: str,
    x: float,
    y: float,
    z: float,
    radius: float,
    length: float,
    color: tuple[float, float, float, float],
) -> str:
    color_txt = " ".join(str(v) for v in color)
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


class GazeboMissionMarkers:
    def __init__(self, mission_path: str, world: str = "default") -> None:
        self.mission_path = Path(mission_path)
        self.world = world

        mission = json.loads(self.mission_path.read_text())

        tasks_planner = mission.get("tasks_xy")
        if tasks_planner is None:
            tasks_planner = mission["planner"]["tasks_xy"]

        depots_planner = mission.get("depots_xy")
        if depots_planner is None:
            depots_planner = mission["planner"]["depots_xy"]

        self.tasks_enu = np.asarray(
            [planner_to_gazebo_enu(np.asarray(xy, dtype=float)) for xy in tasks_planner],
            dtype=float,
        )
        self.depots_enu = np.asarray(
            [planner_to_gazebo_enu(np.asarray(xy, dtype=float)) for xy in depots_planner],
            dtype=float,
        )

        self.tmp_dir = Path("/tmp/dan_gazebo_markers")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.static_spawn_requested = False
        self.done_spawned: set[int] = set()

        # Important: all Gazebo service calls run in this background worker.
        # This prevents marker spawning from blocking PX4 setpoint publication.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.static_future: Future | None = None

    def spawn_static_markers(self) -> None:
        if self.static_spawn_requested:
            return

        self.static_spawn_requested = True
        self.static_future = self.executor.submit(self._spawn_static_markers_sync)

    def mark_task_done(self, task_index_zero_based: int) -> None:
        if task_index_zero_based in self.done_spawned:
            return

        if task_index_zero_based < 0 or task_index_zero_based >= len(self.tasks_enu):
            return

        self.done_spawned.add(task_index_zero_based)
        self.executor.submit(self._mark_task_done_sync, task_index_zero_based)

    def _spawn_static_markers_sync(self) -> None:
        print("[DAN markers] Spawning depot/task markers inside Gazebo asynchronously...")

        for depot_id, xy in enumerate(self.depots_enu, start=1):
            name = f"dan_depot_{depot_id:02d}"
            sdf = cylinder_sdf(
                name=name,
                x=float(xy[0]),
                y=float(xy[1]),
                z=0.025,
                radius=0.28,
                length=0.05,
                color=(0.05, 0.25, 1.0, 1.0),
            )
            self.spawn_sdf(name, sdf)
            time.sleep(0.08)

        for task_id, xy in enumerate(self.tasks_enu, start=1):
            name = f"dan_task_{task_id:03d}"
            sdf = cylinder_sdf(
                name=name,
                x=float(xy[0]),
                y=float(xy[1]),
                z=0.035,
                radius=0.18,
                length=0.07,
                color=(1.0, 0.45, 0.05, 1.0),
            )
            self.spawn_sdf(name, sdf)
            time.sleep(0.08)

        print("[DAN markers] Static markers spawned.")

    def _mark_task_done_sync(self, task_index_zero_based: int) -> None:
        xy = self.tasks_enu[task_index_zero_based]
        task_id = task_index_zero_based + 1

        name = f"dan_task_done_{task_id:03d}"
        sdf = cylinder_sdf(
            name=name,
            x=float(xy[0]),
            y=float(xy[1]),
            z=0.12,
            radius=0.30,
            length=0.08,
            color=(0.0, 1.0, 0.15, 1.0),
        )

        self.spawn_sdf(name, sdf)
        print(f"[DAN markers] Task {task_id} marked done.")

    def spawn_sdf(self, name: str, sdf_text: str) -> None:
        sdf_path = self.tmp_dir / f"{name}.sdf"
        sdf_path.write_text(sdf_text)

        cmd = [
            "gz",
            "service",
            "-s",
            f"/world/{self.world}/create",
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
            print(f"[DAN markers] WARNING: failed to spawn {name}")
            if result.stderr.strip():
                print(result.stderr.strip())
