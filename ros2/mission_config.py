from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]

NUM_DRONES: Final = 3
NUM_TASKS: Final = 15
WORKSPACE_SIZE_M: Final = 20.0
DEFAULT_MISSION_SEED: Final = 20260707

# Planner coordinates are 0–20 m. This controls their physical size in Gazebo.
PLANNER_TO_GAZEBO_SCALE: Final = 0.80

# Gazebo world frame is ENU: [east, north].
GAZEBO_WORLD_OFFSET_ENU_M: Final = np.array(
    [-8.0, -8.0],
    dtype=np.float64,
)

CRUISE_ALTITUDES_M: Final = (3.0, 4.0, 5.0)

WAYPOINT_TOLERANCE_M: Final = 0.25
WAYPOINT_DWELL_S: Final = 1.50
OFFBOARD_HEARTBEAT_HZ: Final = 10.0
OFFBOARD_WARMUP_S: Final = 2.0


@dataclass(frozen=True)
class DroneSpec:
    name: str
    namespace: str
    target_system: int
    depot_index: int
    cruise_altitude_m: float


DRONES: Final[tuple[DroneSpec, ...]] = (
    DroneSpec(
        name="drone_1",
        namespace="/px4_1/fmu",
        target_system=2,
        depot_index=0,
        cruise_altitude_m=CRUISE_ALTITUDES_M[0],
    ),
    DroneSpec(
        name="drone_2",
        namespace="/px4_2/fmu",
        target_system=3,
        depot_index=1,
        cruise_altitude_m=CRUISE_ALTITUDES_M[1],
    ),
    DroneSpec(
        name="drone_3",
        namespace="/px4_3/fmu",
        target_system=4,
        depot_index=2,
        cruise_altitude_m=CRUISE_ALTITUDES_M[2],
    ),
)


def resolve_checkpoint_path(
    explicit_path: str | Path | None = None,
) -> Path:
    """Find the final RL checkpoint, unless an explicit path is supplied."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        return path

    candidates = [
        PROJECT_ROOT
        / "checkpoints"
        / "final_3drone_15task"
        / "centralized_attention_mtsp_15task.pt",
        PROJECT_ROOT
        / "checkpoints"
        / "rl_15task_3drone_finetune"
        / "centralized_attention_rl_best_decode.pt",
        PROJECT_ROOT
        / "checkpoints"
        / "rl_15task_3drone"
        / "centralized_attention_rl_best_decode.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find a final learned checkpoint. "
        "Pass --checkpoint explicitly."
    )


def planner_to_gazebo_enu(
    planner_points_xy: np.ndarray,
) -> np.ndarray:
    """
    Convert planner [x, y] coordinates to Gazebo world [east, north].

    Planner x -> Gazebo east
    Planner y -> Gazebo north
    """
    points = np.asarray(planner_points_xy, dtype=np.float64)

    if points.shape[-1] != 2:
        raise ValueError("Expected points with final dimension 2.")

    return (
        GAZEBO_WORLD_OFFSET_ENU_M
        + PLANNER_TO_GAZEBO_SCALE * points
    )


def planner_delta_to_local_ned(
    planner_delta_xy: np.ndarray,
) -> np.ndarray:
    """
    Convert a planner-frame displacement into PX4 local NED displacement.

    Planner/Gazebo:
        x = east
        y = north

    PX4 local:
        x = north
        y = east
        z = down
    """
    delta = np.asarray(planner_delta_xy, dtype=np.float64)

    if delta.shape != (2,):
        raise ValueError("Expected one [x, y] planner displacement.")

    east_delta = PLANNER_TO_GAZEBO_SCALE * delta[0]
    north_delta = PLANNER_TO_GAZEBO_SCALE * delta[1]

    return np.array(
        [north_delta, east_delta, 0.0],
        dtype=np.float64,
    )


def planner_point_to_local_ned(
    planner_point_xy: np.ndarray,
    planner_depot_xy: np.ndarray,
    home_local_ned: np.ndarray,
    cruise_altitude_m: float,
) -> np.ndarray:
    """
    Produce a PX4-local NED waypoint.

    `home_local_ned` is captured from telemetry when the mission begins,
    rather than assuming the initial PX4 local pose is exactly [0, 0, 0].
    """
    point = np.asarray(planner_point_xy, dtype=np.float64)
    depot = np.asarray(planner_depot_xy, dtype=np.float64)
    home = np.asarray(home_local_ned, dtype=np.float64)

    local_target = home + planner_delta_to_local_ned(point - depot)

    # PX4 local z is Down. Going upward means a more negative z.
    local_target[2] = home[2] - cruise_altitude_m

    return local_target


def home_hover_target(
    home_local_ned: np.ndarray,
    cruise_altitude_m: float,
) -> np.ndarray:
    """Waypoint directly above a drone's own depot."""
    home = np.asarray(home_local_ned, dtype=np.float64).copy()
    home[2] -= cruise_altitude_m
    return home


if __name__ == "__main__":
    planner_depots = np.array(
        [
            [2.0, 2.0],
            [18.0, 2.0],
            [10.0, 17.0],
        ],
        dtype=np.float64,
    )

    gazebo_depots = planner_to_gazebo_enu(planner_depots)

    print("Planner depot -> Gazebo ENU")
    for index, point in enumerate(gazebo_depots, start=1):
        print(
            f"Drone {index}: "
            f"east={point[0]:.2f}, north={point[1]:.2f}"
        )