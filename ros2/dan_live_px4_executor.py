from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

from env.dan_async_mtsp_env import DANObservation
from models.dan_mtsp import DANMTSPPolicy, DANPolicyRunner

try:
    # Prefer the project's canonical coordinate conversion.
    from ros2.mission_config import planner_to_gazebo_enu
except ImportError:
    # Fallback keeps this file usable if mission_config is temporarily unavailable.
    PLANNER_TO_GAZEBO_SCALE = 0.80
    GAZEBO_WORLD_OFFSET_ENU_M = np.asarray([-8.0, -8.0], dtype=np.float64)

    def planner_to_gazebo_enu(xy_planner: Any) -> np.ndarray:
        xy = np.asarray(xy_planner, dtype=np.float64)
        return GAZEBO_WORLD_OFFSET_ENU_M + PLANNER_TO_GAZEBO_SCALE * xy


class Phase(str, Enum):
    WAIT_FOR_TELEMETRY = "WAIT_FOR_TELEMETRY"
    WARMUP = "WARMUP"
    TAKEOFF = "TAKEOFF"
    HOVER_READY = "HOVER_READY"
    ROUTE = "ROUTE"
    RETURN = "RETURN"
    LAND = "LAND"
    DONE = "DONE"


@dataclass
class DroneRuntime:
    index: int
    namespace: str
    target_system: int
    home_enu: np.ndarray
    altitude_m: float

    local_position: VehicleLocalPosition | None = None
    vehicle_status: VehicleStatus | None = None

    phase: Phase = Phase.WAIT_FOR_TELEMETRY
    phase_started_s: float = 0.0

    current_task_id: int | None = None
    current_target_enu: np.ndarray | None = None
    route_task_ids: list[int] = field(default_factory=list)

    condition_started_s: float | None = None
    last_arm_offboard_command_s: float = -1.0e9
    last_land_command_s: float = -1.0e9
    land_started_s: float | None = None

    def has_telemetry(self) -> bool:
        # VehicleStatus is useful for diagnostics and landing detection, but it
        # must not prevent startup if that topic arrives later than position.
        return self.local_position is not None


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_world(path: Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        world = json.load(file)

    if world.get("schema") != "dan_task_world_v1":
        raise ValueError(
            f"Expected mission schema 'dan_task_world_v1', "
            f"got {world.get('schema')!r}"
        )

    return world


def load_dan_model(
    checkpoint_path: Path,
    device: torch.device,
) -> DANMTSPPolicy:
    resolved = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(
        resolved,
        map_location=device,
        weights_only=False,
    )

    if "model_config" not in checkpoint or "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint {resolved} must contain 'model_config' and 'state_dict'"
        )

    model = DANMTSPPolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


class DANLivePX4Executor(Node):
    """Non-blocking live DAN executor for multiple PX4 SITL vehicles."""

    CONTROL_PERIOD_S = 0.05  # 20 Hz

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("dan_live_px4_executor")

        self.args = args
        self.world = load_world(args.mission)

        mission_num_agents = int(
            self.world.get(
                "num_agents",
                self.world.get("planner", {}).get("num_agents", 0),
            )
        )

        if args.num_drones is None:
            self.num_agents = mission_num_agents
        else:
            if args.num_drones < 1 or args.num_drones > mission_num_agents:
                raise ValueError(
                    f"--num-drones must be between 1 and {mission_num_agents}"
                )
            self.num_agents = args.num_drones
        self.num_tasks = int(
            self.world.get(
                "num_tasks",
                self.world.get("planner", {}).get("num_tasks", 0),
            )
        )

        if self.num_agents <= 0:
            raise ValueError("Mission contains no agents")
        if self.num_tasks <= 0:
            raise ValueError("Mission contains no tasks")

        self.tasks_unit = self._load_tasks_unit()
        self.tasks_planner = self._load_tasks_planner()
        self.depots_planner = self._load_depots_planner()
        self.initial_agent_positions_unit = self._load_initial_agent_positions_unit()

        if self.tasks_unit.shape != (self.num_tasks, 2):
            raise ValueError(
                f"tasks_xy_unit has shape {self.tasks_unit.shape}, "
                f"expected ({self.num_tasks}, 2)"
            )
        if self.tasks_planner.shape != (self.num_tasks, 2):
            raise ValueError(
                f"Planner task coordinates have shape {self.tasks_planner.shape}, "
                f"expected ({self.num_tasks}, 2)"
            )
        if self.depots_planner.shape != (self.num_agents, 2):
            raise ValueError(
                f"Planner depot coordinates have shape {self.depots_planner.shape}, "
                f"expected ({self.num_agents}, 2)"
            )

        self.tasks_enu = np.stack(
            [
                np.asarray(planner_to_gazebo_enu(xy), dtype=np.float64)
                for xy in self.tasks_planner
            ],
            axis=0,
        )
        self.depots_enu = np.stack(
            [
                np.asarray(planner_to_gazebo_enu(xy), dtype=np.float64)
                for xy in self.depots_planner
            ],
            axis=0,
        )

        self.device = self._make_device(args.device)
        self.model = load_dan_model(args.checkpoint, self.device)
        self.policy = DANPolicyRunner(
            model=self.model,
            device=self.device,
            mode=args.decode_mode,
        )

        # Claimed means unavailable immediately, even before physically visited.
        self.claimed_mask = np.zeros(self.num_tasks, dtype=bool)

        # Match DAN training logic: when an agent selects a task, its model
        # position immediately becomes that selected task.
        self.model_positions_unit = self.initial_agent_positions_unit.copy()

        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.drones: list[DroneRuntime] = []
        self.offboard_publishers: list[Any] = []
        self.setpoint_publishers: list[Any] = []
        self.command_publishers: list[Any] = []
        self._subscriptions: list[Any] = []

        for index in range(self.num_agents):
            namespace = f"/px4_{index + 1}/fmu"
            altitude_m = args.altitude_base + args.altitude_step * index

            drone = DroneRuntime(
                index=index,
                namespace=namespace,
                target_system=index + 2,
                home_enu=self.depots_enu[index].copy(),
                altitude_m=altitude_m,
            )
            self.drones.append(drone)

            self.offboard_publishers.append(
                self.create_publisher(
                    OffboardControlMode,
                    f"{namespace}/in/offboard_control_mode",
                    self.qos,
                )
            )
            self.setpoint_publishers.append(
                self.create_publisher(
                    TrajectorySetpoint,
                    f"{namespace}/in/trajectory_setpoint",
                    self.qos,
                )
            )
            self.command_publishers.append(
                self.create_publisher(
                    VehicleCommand,
                    f"{namespace}/in/vehicle_command",
                    self.qos,
                )
            )

            self._subscriptions.append(
                self.create_subscription(
                    VehicleLocalPosition,
                    f"{namespace}/out/vehicle_local_position_v1",
                    self._make_local_position_callback(index),
                    self.qos,
                )
            )
            self._subscriptions.append(
                self.create_subscription(
                    VehicleStatus,
                    f"{namespace}/out/vehicle_status_v1",
                    self._make_vehicle_status_callback(index),
                    self.qos,
                )
            )

        self.started_s = self.now_s()
        self.last_progress_log_s = self.started_s
        self.routing_started = False
        self.all_landing_started_s: float | None = None
        self.shutdown_requested = False
        self.summary_printed = False

        self._state_handlers: dict[Phase, Callable[[DroneRuntime, float], None]] = {
            Phase.WAIT_FOR_TELEMETRY: self._update_wait_for_telemetry,
            Phase.WARMUP: self._update_warmup,
            Phase.TAKEOFF: self._update_takeoff,
            Phase.HOVER_READY: self._update_hover_ready,
            Phase.ROUTE: self._update_route,
            Phase.RETURN: self._update_return,
            Phase.LAND: self._update_land,
            Phase.DONE: self._update_done,
        }

        self.get_logger().info(
            f"Loaded DAN world: agents={self.num_agents}, tasks={self.num_tasks}"
        )
        self.get_logger().info(
            f"Checkpoint: {resolve_project_path(args.checkpoint)}"
        )
        self.get_logger().info(
            f"Decode mode={args.decode_mode}, device={self.device}, "
            f"execute={args.execute}, hover_only={args.hover_only}"
        )

        for drone in self.drones:
            self.get_logger().info(
                f"Drone {drone.index + 1}: namespace={drone.namespace}, "
                f"sysid={drone.target_system}, "
                f"home_enu={drone.home_enu.tolist()}, "
                f"altitude={drone.altitude_m:.2f} m"
            )

        self.started_s = self.now_s()
        self.last_print_s = self.started_s

        # Sequential takeoff control.
        # Only this drone index is allowed to leave WARMUP and enter TAKEOFF.
        self.next_takeoff_index = 0
        self.next_takeoff_allowed_s = self.started_s
        self.takeoff_interval_s = 5.0

        self.timer = self.create_timer(
            self.CONTROL_PERIOD_S,
            self.control_tick,
        )

    # ------------------------------------------------------------------
    # Mission loading
    # ------------------------------------------------------------------

    def _load_tasks_unit(self) -> np.ndarray:
        if "tasks_xy_unit" not in self.world:
            raise KeyError("Mission is missing tasks_xy_unit")
        return np.asarray(self.world["tasks_xy_unit"], dtype=np.float64)

    def _load_tasks_planner(self) -> np.ndarray:
        if "tasks_xy_planner" in self.world:
            return np.asarray(self.world["tasks_xy_planner"], dtype=np.float64)

        if "tasks_xy" in self.world:
            return np.asarray(self.world["tasks_xy"], dtype=np.float64)

        planner = self.world.get("planner", {})
        if "tasks_xy" in planner:
            return np.asarray(planner["tasks_xy"], dtype=np.float64)

        raise KeyError(
            "Mission must contain tasks_xy_planner, tasks_xy, "
            "or planner.tasks_xy"
        )

    def _load_depots_planner(self) -> np.ndarray:
        planner = self.world.get("planner", {})

        depots = self.world.get("depots_xy")
        if depots is None:
            depots = planner.get("depots_xy")

        if depots is not None:
            array = np.asarray(depots, dtype=np.float64)
            if array.ndim == 1:
                array = np.repeat(array[None, :], self.num_agents, axis=0)
            return array

        common = self.world.get("depot_xy_planner")
        if common is None:
            common = planner.get("common_depot_xy", planner.get("depot_xy"))

        if common is None:
            raise KeyError(
                "Mission must contain depots_xy or a common planner depot"
            )

        common_array = np.asarray(common, dtype=np.float64)
        return np.repeat(common_array[None, :], self.num_agents, axis=0)

    def _load_initial_agent_positions_unit(self) -> np.ndarray:
        # Most DAN worlds use one common unit-square depot.
        common = self.world.get("depot_xy_unit")
        if common is not None:
            common_array = np.asarray(common, dtype=np.float64)
            return np.repeat(common_array[None, :], self.num_agents, axis=0)

        # Fallback: convert each planner depot from [0, planner_size]^2.
        planner_size = float(
            self.world.get("coordinate_frame", {}).get("planner_size", 20.0)
        )
        if planner_size <= 0.0:
            raise ValueError("planner_size must be positive")

        return self.depots_planner / planner_size

    # ------------------------------------------------------------------
    # ROS callbacks and timing
    # ------------------------------------------------------------------

    def _make_local_position_callback(
        self,
        index: int,
    ) -> Callable[[VehicleLocalPosition], None]:
        def callback(msg: VehicleLocalPosition) -> None:
            self.drones[index].local_position = msg

        return callback

    def _make_vehicle_status_callback(
        self,
        index: int,
    ) -> Callable[[VehicleStatus], None]:
        def callback(msg: VehicleStatus) -> None:
            self.drones[index].vehicle_status = msg

        return callback

    def _make_device(self, text: str) -> torch.device:
        if text == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(text)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return device

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds // 1000)

    # ------------------------------------------------------------------
    # PX4 coordinate and telemetry helpers
    # ------------------------------------------------------------------

    def current_enu(self, drone: DroneRuntime) -> np.ndarray:
        if drone.local_position is None:
            return drone.home_enu.copy()

        # PX4 local coordinates are NED:
        # x=north, y=east, z=down.
        north = float(drone.local_position.x)
        east = float(drone.local_position.y)

        return drone.home_enu + np.asarray([east, north], dtype=np.float64)

    def current_altitude(self, drone: DroneRuntime) -> float:
        if drone.local_position is None:
            return 0.0
        return max(0.0, -float(drone.local_position.z))

    def enu_to_local_ned_setpoint(
        self,
        drone: DroneRuntime,
        target_enu: np.ndarray,
        altitude_m: float,
    ) -> list[float]:
        delta_enu = np.asarray(target_enu, dtype=np.float64) - drone.home_enu
        east = float(delta_enu[0])
        north = float(delta_enu[1])
        down = -float(altitude_m)
        return [north, east, down]

    def horizontal_distance(
        self,
        drone: DroneRuntime,
        target_enu: np.ndarray,
    ) -> float:
        return float(np.linalg.norm(self.current_enu(drone) - target_enu))

    def target_reached(
        self,
        drone: DroneRuntime,
        target_enu: np.ndarray,
        altitude_m: float,
        tolerance_m: float,
    ) -> bool:
        horizontal_error = self.horizontal_distance(drone, target_enu)
        vertical_error = abs(self.current_altitude(drone) - altitude_m)
        error_3d = math.hypot(horizontal_error, vertical_error)
        return error_3d <= tolerance_m

    def condition_held(
        self,
        drone: DroneRuntime,
        condition: bool,
        dwell_s: float,
        now: float,
    ) -> bool:
        """Return true only if condition remains continuously true for dwell_s."""
        if not condition:
            drone.condition_started_s = None
            return False

        if drone.condition_started_s is None:
            drone.condition_started_s = now
            return dwell_s <= 0.0

        return (now - drone.condition_started_s) >= dwell_s

    # ------------------------------------------------------------------
    # PX4 publishers and commands
    # ------------------------------------------------------------------

    def publish_offboard_mode(self, drone: DroneRuntime) -> None:
        msg = OffboardControlMode()
        msg.timestamp = self.timestamp_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_publishers[drone.index].publish(msg)

    def publish_setpoint(
        self,
        drone: DroneRuntime,
        target_enu: np.ndarray,
        altitude_m: float,
    ) -> None:
        msg = TrajectorySetpoint()
        msg.timestamp = self.timestamp_us()
        msg.position = self.enu_to_local_ned_setpoint(
            drone,
            target_enu,
            altitude_m,
        )
        msg.yaw = float("nan")
        self.setpoint_publishers[drone.index].publish(msg)

    def publish_vehicle_command(
        self,
        drone: DroneRuntime,
        command: int,
        *,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ) -> None:
        if not self.args.execute:
            return

        msg = VehicleCommand()
        msg.timestamp = self.timestamp_us()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.command = int(command)
        msg.target_system = int(drone.target_system)
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.command_publishers[drone.index].publish(msg)

    def send_arm_and_offboard(self, drone: DroneRuntime, now: float) -> None:
        # Keep the ordering used by the previously working executor.
        self.publish_vehicle_command(
            drone,
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,  # PX4 custom mode: OFFBOARD
        )
        self.publish_vehicle_command(
            drone,
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        drone.last_arm_offboard_command_s = now

    def send_land(self, drone: DroneRuntime, now: float) -> None:
        self.publish_vehicle_command(
            drone,
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
        )
        drone.last_land_command_s = now
        if drone.land_started_s is None:
            drone.land_started_s = now

    # ------------------------------------------------------------------
    # DAN policy
    # ------------------------------------------------------------------

    def remaining_travel_distance_unit(self, drone: DroneRuntime) -> float:
        if drone.current_task_id is None:
            return 0.0

        current = self.current_enu(drone)
        target = self.tasks_enu[drone.current_task_id]

        # ENU task distance is 0.8 * planner distance, and planner coordinates
        # span 20 units. Divide by 16 to express distance in unit-square scale.
        planner_size = float(
            self.world.get("coordinate_frame", {}).get("planner_size", 20.0)
        )
        enu_span_m = 0.80 * planner_size
        if enu_span_m <= 0.0:
            return 0.0

        return float(np.linalg.norm(target - current) / enu_span_m)

    def make_dan_observation(self, agent_id: int) -> DANObservation:
        current_position = self.model_positions_unit[agent_id].copy()

        cities_relative = (
            self.tasks_unit - current_position[None, :]
        ).astype(np.float64)

        remaining_travel_times = np.asarray(
            [
                self.remaining_travel_distance_unit(drone)
                for drone in self.drones
            ],
            dtype=np.float64,
        )

        agent_positions_relative = (
            self.model_positions_unit - current_position[None, :]
        ).astype(np.float64)

        agents_relative = np.concatenate(
            [
                agent_positions_relative,
                remaining_travel_times[:, None],
            ],
            axis=1,
        ).astype(np.float64)

        return DANObservation(
            deciding_agent=agent_id,
            cities_relative=cities_relative,
            agents_relative=agents_relative,
            visited_mask=self.claimed_mask.copy(),
            action_mask=(~self.claimed_mask).copy(),
            current_position=current_position,
            agent_positions=self.model_positions_unit.copy(),
            remaining_travel_times=remaining_travel_times.copy(),
        )

    def _extract_policy_action(self, output: Any) -> int:
        if hasattr(output, "action"):
            return int(output.action)

        if isinstance(output, dict) and "action" in output:
            return int(output["action"])

        if isinstance(output, (int, np.integer)):
            return int(output)

        if torch.is_tensor(output) and output.numel() == 1:
            return int(output.item())

        raise TypeError(
            "DANPolicyRunner output must expose an action field or scalar action; "
            f"got {type(output).__name__}"
        )

    def assign_next_task(self, drone: DroneRuntime) -> bool:
        if bool(np.all(self.claimed_mask)):
            return False

        observation = self.make_dan_observation(drone.index)
        valid_task_ids = np.flatnonzero(observation.action_mask)

        if valid_task_ids.size == 0:
            return False

        with torch.inference_mode():
            output = self.policy(observation)

        task_id = self._extract_policy_action(output)

        if task_id < 0 or task_id >= self.num_tasks:
            self.get_logger().error(
                f"Drone {drone.index + 1}: DAN returned invalid task index "
                f"{task_id}; using first valid task"
            )
            task_id = int(valid_task_ids[0])

        if self.claimed_mask[task_id]:
            # A stale/invalid model result must never create duplicate ownership.
            fallback = int(valid_task_ids[0])
            self.get_logger().warning(
                f"Drone {drone.index + 1}: DAN selected already-claimed "
                f"Task {task_id + 1}; falling back to Task {fallback + 1}"
            )
            task_id = fallback

        self.claimed_mask[task_id] = True
        drone.current_task_id = task_id
        drone.current_target_enu = self.tasks_enu[task_id].copy()
        drone.route_task_ids.append(task_id)
        drone.condition_started_s = None

        # Intentional claimed-state approximation used during live inference.
        self.model_positions_unit[drone.index] = self.tasks_unit[task_id]

        self.get_logger().info(
            f"Drone {drone.index + 1}: assigned Task {task_id + 1} | "
            f"claimed {int(np.count_nonzero(self.claimed_mask))}/{self.num_tasks}"
        )
        return True

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def set_phase(self, drone: DroneRuntime, phase, now: float | None = None) -> None:
        """Set phase, compatible with both Phase enum values and plain strings."""
        if now is None:
            now = self.now_s()

        # If the file defines a Phase enum, try to normalize string phases.
        # If normalization fails, keep the original string.
        try:
            if isinstance(phase, str) and "Phase" in globals():
                try:
                    phase = Phase(phase)
                except Exception:
                    try:
                        phase = Phase[phase]
                    except Exception:
                        pass
        except Exception:
            pass

        if drone.phase == phase:
            return

        drone.phase = phase
        drone.phase_started_s = now

        # Support both old and newer DroneRuntime field names.
        if hasattr(drone, "condition_started_s"):
            drone.condition_started_s = None
        if hasattr(drone, "reached_started_s"):
            drone.reached_started_s = None

        # Newer executor versions may track landing start time.
        try:
            is_land = phase == Phase.LAND
        except Exception:
            is_land = str(phase) == "LAND"

        if is_land and hasattr(drone, "land_started_s"):
            drone.land_started_s = now

        self.get_logger().info(
            f"Drone {drone.index + 1}: phase -> {phase}"
        )


    def _publish_flight_hold(
        self,
        drone: DroneRuntime,
        target_enu: np.ndarray,
        altitude_m: float,
    ) -> None:
        # Called every timer tick in all active flight states.
        self.publish_offboard_mode(drone)
        self.publish_setpoint(drone, target_enu, altitude_m)

    def _update_wait_for_telemetry(
        self,
        drone: DroneRuntime,
        now: float,
    ) -> None:
        if drone.has_telemetry():
            self.set_phase(drone, Phase.WARMUP, now)

    def _update_warmup(self, drone: DroneRuntime, now: float) -> None:
        self._publish_flight_hold(
            drone,
            drone.home_enu,
            drone.altitude_m,
        )

        if (now - drone.phase_started_s) >= self.args.warmup_s:
            self.send_arm_and_offboard(drone, now)
            self.set_phase(drone, Phase.TAKEOFF, now)

    def _update_takeoff(self, drone: DroneRuntime, now: float) -> None:
        self._publish_flight_hold(
            drone,
            drone.home_enu,
            drone.altitude_m,
        )

        if (
            now - drone.last_arm_offboard_command_s
            >= self.args.command_retry_s
        ):
            self.send_arm_and_offboard(drone, now)

        altitude_ready = (
            self.current_altitude(drone)
            >= drone.altitude_m - self.args.takeoff_altitude_tolerance
        )

        if self.condition_held(
            drone,
            altitude_ready,
            self.args.takeoff_dwell_s,
            now,
        ):
            self.set_phase(drone, Phase.HOVER_READY, now)

    def _update_hover_ready(self, drone: DroneRuntime, now: float) -> None:
        self._publish_flight_hold(
            drone,
            drone.home_enu,
            drone.altitude_m,
        )

    def _update_route(self, drone: DroneRuntime, now: float) -> None:
        if drone.current_task_id is None:
            if not self.assign_next_task(drone):
                self.model_positions_unit[drone.index] = (
                    self.initial_agent_positions_unit[drone.index]
                )
                drone.current_target_enu = drone.home_enu.copy()
                self.set_phase(drone, Phase.RETURN, now)
                self._update_return(drone, now)
                return

        assert drone.current_task_id is not None
        assert drone.current_target_enu is not None

        self._publish_flight_hold(
            drone,
            drone.current_target_enu,
            drone.altitude_m,
        )

        reached = self.target_reached(
            drone,
            drone.current_target_enu,
            drone.altitude_m,
            self.args.waypoint_tolerance,
        )

        if self.condition_held(
            drone,
            reached,
            self.args.waypoint_dwell_s,
            now,
        ):
            task_id = drone.current_task_id
            self.get_logger().info(
                f"Drone {drone.index + 1}: reached Task {task_id + 1}"
            )
            drone.current_task_id = None
            drone.current_target_enu = None
            drone.condition_started_s = None

    def _update_return(self, drone: DroneRuntime, now: float) -> None:
        self._publish_flight_hold(
            drone,
            drone.home_enu,
            drone.altitude_m,
        )

        reached_home = self.target_reached(
            drone,
            drone.home_enu,
            drone.altitude_m,
            self.args.waypoint_tolerance,
        )

        if self.condition_held(
            drone,
            reached_home,
            self.args.waypoint_dwell_s,
            now,
        ):
            self.send_land(drone, now)
            self.set_phase(drone, Phase.LAND, now)

    def _is_disarmed(self, drone: DroneRuntime) -> bool:
        status = drone.vehicle_status
        if status is None:
            return False

        disarmed_value = getattr(
            VehicleStatus,
            "ARMING_STATE_DISARMED",
            1,
        )
        return int(status.arming_state) == int(disarmed_value)

    def _update_land(self, drone: DroneRuntime, now: float) -> None:
        # Keep the ROS streams alive. Publishing the offboard messages does not
        # itself switch PX4 back to offboard; no SET_MODE command is sent here.
        self.publish_offboard_mode(drone)
        self.publish_setpoint(
            drone,
            drone.home_enu,
            max(0.0, self.current_altitude(drone)),
        )

        if (now - drone.last_land_command_s) >= self.args.land_retry_s:
            self.send_land(drone, now)

        altitude_landed = (
            self.current_altitude(drone)
            <= self.args.landing_altitude_tolerance
        )
        landed = altitude_landed or self._is_disarmed(drone)

        if self.condition_held(
            drone,
            landed,
            self.args.landing_dwell_s,
            now,
        ):
            self.set_phase(drone, Phase.DONE, now)
            return

        if (
            drone.land_started_s is not None
            and now - drone.land_started_s >= self.args.per_drone_land_timeout_s
        ):
            self.get_logger().warning(
                f"Drone {drone.index + 1}: landing detection timeout; "
                "marking DONE after repeated LAND commands"
            )
            self.set_phase(drone, Phase.DONE, now)

    def _update_done(self, drone: DroneRuntime, now: float) -> None:
        # Intentionally no offboard setpoint after confirmed landing.
        return

    def _start_routing_if_ready(self, now: float) -> None:
        if self.routing_started:
            return

        if not all(
            drone.phase == Phase.HOVER_READY
            for drone in self.drones
        ):
            return

        if self.args.hover_only:
            self.get_logger().info(
                "All drones reached HOVER_READY. Hover-only test is stable."
            )
            self.routing_started = True
            return

        self.get_logger().info(
            "All drones reached HOVER_READY. Starting DAN routing."
        )
        self.routing_started = True

        for drone in self.drones:
            self.set_phase(drone, Phase.ROUTE, now)

    # ------------------------------------------------------------------
    # Mission supervision
    # ------------------------------------------------------------------

    def _handle_timeout(self, now: float) -> None:
        self.get_logger().error(
            "Mission timeout reached. Sending LAND to every drone."
        )

        for drone in self.drones:
            if drone.phase != Phase.DONE:
                self.send_land(drone, now)
                self.set_phase(drone, Phase.LAND, now)

        self._request_shutdown("Mission timeout; landing commands sent.")

    def _update_global_finish(self, now: float) -> None:
        if self.args.hover_only:
            return

        if all(drone.phase == Phase.DONE for drone in self.drones):
            self._request_shutdown("Mission complete. All drones are DONE.")
            return

        all_landing_or_done = all(
            drone.phase in (Phase.LAND, Phase.DONE)
            for drone in self.drones
        )

        if not all_landing_or_done:
            self.all_landing_started_s = None
            return

        if self.all_landing_started_s is None:
            self.all_landing_started_s = now
            self.get_logger().info(
                "All drones are in LAND/DONE. Waiting for landing completion."
            )
            return

        if (
            now - self.all_landing_started_s
            >= self.args.finish_after_return_s
        ):
            self._request_shutdown(
                "Mission complete. All drones returned and entered LAND/DONE."
            )

    def _request_shutdown(self, reason: str) -> None:
        if self.shutdown_requested:
            return

        self.shutdown_requested = True
        self.get_logger().info(reason)
        self.print_summary()

        if rclpy.ok():
            rclpy.shutdown()

    def _log_progress(self, now: float) -> None:
        if now - self.last_progress_log_s < self.args.progress_log_period_s:
            return

        self.last_progress_log_s = now

        phases = ", ".join(
            f"D{drone.index + 1}:{drone.phase.value}"
            for drone in self.drones
        )
        telemetry = ", ".join(
            (
                f"D{drone.index + 1}:"
                f"lp={'Y' if drone.local_position is not None else 'N'}"
                f"/st={'Y' if drone.vehicle_status is not None else 'N'}"
                f"/alt={self.current_altitude(drone):.1f}"
            )
            for drone in self.drones
        )

        self.get_logger().info(
            f"Progress: claimed "
            f"{int(np.count_nonzero(self.claimed_mask))}/{self.num_tasks} | "
            f"{phases} | {telemetry}"
        )

    def arm_and_offboard(self, drone: DroneRuntime) -> None:
        """Compatibility wrapper for older control_tick() code."""
        if hasattr(self, "send_arm_and_offboard"):
            try:
                self.send_arm_and_offboard(drone, self.now_s())
            except TypeError:
                self.send_arm_and_offboard(drone)
            return

        raise AttributeError("No send_arm_and_offboard() method exists.")

    def land(self, drone: DroneRuntime) -> None:
        """Compatibility wrapper for older control_tick() code."""
        now = self.now_s()

        for method_name in (
            "send_land_command",
            "send_land",
            "send_land_command_once",
        ):
            method = getattr(self, method_name, None)
            if method is None:
                continue

            try:
                method(drone, now)
            except TypeError:
                method(drone)
            return

        raise AttributeError("No land/send_land method exists.")

    # ------------------------------------------------------------------
    # Compatibility wrappers
    # ------------------------------------------------------------------

    def drone_current_altitude(self, drone: DroneRuntime) -> float:
        """Compatibility wrapper for older control_tick() code."""
        method = getattr(self, "current_altitude", None)
        if method is not None:
            return float(method(drone))

        if drone.local_position is None:
            return 0.0
        return -float(drone.local_position.z)

    def drone_current_enu(self, drone: DroneRuntime) -> np.ndarray:
        """Compatibility wrapper for older control_tick() code."""
        method = getattr(self, "current_enu", None)
        if method is not None:
            return np.asarray(method(drone), dtype=np.float64)

        if drone.local_position is None:
            return drone.home_enu.copy()

        # PX4 local position is NED: x=north, y=east, z=down.
        east = float(drone.local_position.y)
        north = float(drone.local_position.x)
        return drone.home_enu + np.asarray([east, north], dtype=np.float64)

    def target_reached(
        self,
        drone: DroneRuntime,
        target_enu: np.ndarray,
        altitude_m: float,
        tolerance_m: float,
    ) -> bool:
        """Compatibility target check used by older control_tick() code."""
        if drone.local_position is None:
            return False

        current_enu = self.drone_current_enu(drone)
        current_altitude = self.drone_current_altitude(drone)

        horizontal_error = float(np.linalg.norm(current_enu - target_enu))
        vertical_error = abs(current_altitude - altitude_m)

        return float(np.sqrt(horizontal_error**2 + vertical_error**2)) <= tolerance_m

    def dwell_satisfied(
        self,
        drone: DroneRuntime,
        reached_now: bool,
    ) -> bool:
        now = self.now_s()

        if hasattr(drone, "reached_started_s"):
            attr = "reached_started_s"
        elif hasattr(drone, "condition_started_s"):
            attr = "condition_started_s"
        else:
            setattr(drone, "reached_started_s", None)
            attr = "reached_started_s"

        started = getattr(drone, attr, None)

        if not reached_now:
            setattr(drone, attr, None)
            return False

        if started is None:
            setattr(drone, attr, now)
            return False

        return (now - started) >= self.args.waypoint_dwell_s


    def control_tick(self) -> None:
        now = self.now_s()

        def phase_is(drone: DroneRuntime, name: str) -> bool:
            value = getattr(drone.phase, "value", drone.phase)
            return value == name

        def reset_dwell(drone: DroneRuntime) -> None:
            if hasattr(drone, "reached_started_s"):
                drone.reached_started_s = None
            if hasattr(drone, "condition_started_s"):
                drone.condition_started_s = None

        def hold(drone: DroneRuntime, target_enu: np.ndarray, altitude_m: float) -> None:
            self.publish_offboard_mode(drone)
            self.publish_setpoint(drone, target_enu, altitude_m)

        if now - self.started_s > self.args.mission_timeout_s:
            self.get_logger().error("Mission timeout. Sending land commands.")
            for drone in self.drones:
                self.land(drone)
            rclpy.shutdown()
            return

        routes_started = bool(
            getattr(self, "_routes_started", False)
            or getattr(self, "routing_started", False)
        )

        # Start DAN routing only after all drones completed serial takeoff.
        if (not self.args.hover_only) and (not routes_started):
            if all(phase_is(drone, "HOVER") for drone in self.drones):
                self._routes_started = True
                self.routing_started = True
                self.get_logger().info(
                    "All drones reached staging HOVER. Starting DAN routing."
                )
                for drone in self.drones:
                    self.set_phase(drone, "ROUTE")

        all_done = True

        for drone in self.drones:
            if not drone.has_telemetry():
                all_done = False
                continue

            routes_started = bool(
                getattr(self, "_routes_started", False)
                or getattr(self, "routing_started", False)
            )

            # Serial-control gate:
            # before routing starts, only drones whose turn has arrived may receive
            # offboard/setpoint streams. Future drones stay completely silent.
            active_before_routing = (
                drone.index < self.next_takeoff_index
                or (
                    drone.index == self.next_takeoff_index
                    and now >= self.next_takeoff_allowed_s
                )
            )

            if (
                not routes_started
                and not active_before_routing
                and (
                    phase_is(drone, "WAIT_FOR_TELEMETRY")
                    or phase_is(drone, "WARMUP")
                )
            ):
                all_done = False
                continue

            if phase_is(drone, "WAIT_FOR_TELEMETRY"):
                self.set_phase(drone, "WARMUP")

            if phase_is(drone, "WARMUP"):
                hold(drone, drone.home_enu, drone.altitude_m)

                warmup_done = now - drone.phase_started_s >= self.args.warmup_s
                is_next_drone = drone.index == self.next_takeoff_index

                if warmup_done and is_next_drone:
                    self.get_logger().info(
                        f"Serial control: starting Drone {drone.index + 1}"
                    )
                    self.arm_and_offboard(drone)
                    drone.last_arm_command_s = now
                    self.set_phase(drone, "TAKEOFF")

                all_done = False
                continue

            if phase_is(drone, "TAKEOFF"):
                hold(drone, drone.home_enu, drone.altitude_m)

                if self.args.execute and (now - drone.last_arm_command_s) >= 1.0:
                    self.arm_and_offboard(drone)
                    drone.last_arm_command_s = now

                current_altitude = self.drone_current_altitude(drone)
                reached = current_altitude >= (
                    drone.altitude_m - self.args.takeoff_altitude_tolerance
                )

                if self.dwell_satisfied(drone, reached):
                    self.set_phase(drone, "HOVER")

                    if drone.index == self.next_takeoff_index:
                        self.next_takeoff_index += 1
                        self.next_takeoff_allowed_s = now + self.takeoff_interval_s

                        if self.next_takeoff_index < self.num_agents:
                            self.get_logger().info(
                                f"Serial control: Drone {drone.index + 1} is HOVER. "
                                f"Drone {self.next_takeoff_index + 1} will start in "
                                f"{self.takeoff_interval_s:.1f} s."
                            )
                        else:
                            self.get_logger().info(
                                "Serial control: all drones have reached HOVER."
                            )

                all_done = False
                continue

            if phase_is(drone, "HOVER"):
                hold(drone, drone.home_enu, drone.altitude_m)
                all_done = False
                continue

            if phase_is(drone, "ROUTE"):
                if drone.current_task_id is None:
                    assigned = self.assign_next_task(drone)

                    if not assigned:
                        depot_unit = getattr(self, "depot_unit", None)
                        if depot_unit is None:
                            depot_unit = np.asarray(
                                self.world.get("depot_xy_unit", [0.5, 0.5]),
                                dtype=np.float64,
                            )
                            self.depot_unit = depot_unit

                        self.model_positions_unit[drone.index] = depot_unit
                        drone.current_target_enu = drone.home_enu.copy()
                        self.set_phase(drone, "RETURN")
                    else:
                        assert drone.current_target_enu is not None

                if phase_is(drone, "ROUTE") and drone.current_target_enu is not None:
                    hold(drone, drone.current_target_enu, drone.altitude_m)

                    reached = self.target_reached(
                        drone,
                        drone.current_target_enu,
                        drone.altitude_m,
                        self.args.waypoint_tolerance,
                    )

                    if self.dwell_satisfied(drone, reached):
                        assert drone.current_task_id is not None
                        self.get_logger().info(
                            f"Drone {drone.index + 1}: reached Task {drone.current_task_id + 1}"
                        )
                        drone.current_task_id = None
                        drone.current_target_enu = None
                        reset_dwell(drone)

                all_done = False
                continue

            if phase_is(drone, "RETURN"):
                hold(drone, drone.home_enu, drone.altitude_m)

                reached = self.target_reached(
                    drone,
                    drone.home_enu,
                    drone.altitude_m,
                    self.args.waypoint_tolerance,
                )

                if self.dwell_satisfied(drone, reached):
                    self.land(drone)
                    drone.landed_command_sent = True
                    self.set_phase(drone, "LAND")

                all_done = False
                continue

            if phase_is(drone, "LAND"):
                if not getattr(drone, "landed_command_sent", False):
                    self.land(drone)
                    drone.landed_command_sent = True

                hold(drone, drone.home_enu, drone.altitude_m)
                all_done = False
                continue

            if not phase_is(drone, "DONE"):
                all_done = False

        if now - self.last_print_s >= 5.0:
            self.last_print_s = now
            claimed = int(np.sum(self.claimed_mask))
            phases = ", ".join(
                f"D{drone.index + 1}:{getattr(drone.phase, 'value', drone.phase)}"
                for drone in self.drones
            )

            telemetry = ", ".join(
                (
                    f"D{drone.index + 1}:"
                    f"lp={'Y' if drone.local_position is not None else 'N'}"
                    f"/st={'Y' if drone.vehicle_status is not None else 'N'}"
                    f"/alt={self.drone_current_altitude(drone):.2f}"
                    f"/tgt={drone.altitude_m:.2f}"
                )
                for drone in self.drones
            )

            self.get_logger().info(
                f"Progress: claimed {claimed}/{self.num_tasks} | {phases} | {telemetry}"
            )

            if (not self.args.hover_only) and all(
                phase_is(drone, "LAND") or phase_is(drone, "DONE")
                for drone in self.drones
            ):
                if not hasattr(self, "_all_returned_since_s"):
                    self._all_returned_since_s = now
                    self.get_logger().info(
                        "All drones are in LAND/DONE. Waiting before shutdown..."
                    )
                elif now - self._all_returned_since_s >= self.args.finish_after_return_s:
                    self.get_logger().info("Mission complete. Shutting down executor.")
                    rclpy.shutdown()
                    return
            else:
                if hasattr(self, "_all_returned_since_s"):
                    delattr(self, "_all_returned_since_s")

        if all_done:
            self.print_summary()
            rclpy.shutdown()

    def print_summary(self) -> None:
        if self.summary_printed:
            return

        self.summary_printed = True
        self.get_logger().info("DAN live PX4 mission summary")

        for drone in self.drones:
            readable_tasks = [
                task_id + 1 for task_id in drone.route_task_ids
            ]
            self.get_logger().info(
                f"Drone {drone.index + 1}: "
                f"phase={drone.phase.value}, tasks={readable_tasks}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust live DAN PX4 multi-drone executor."
    )

    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/"
            "dan_refine_2to5_agents_15to25_tasks/"
            "dan_best.pt"
        ),
    )

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hover-only", action="store_true")

    parser.add_argument(
        "--num-drones",
        type=int,
        default=None,
        help="Number of drones to control. Default: use the mission file's num_agents.",
    )

    parser.add_argument(
        "--decode-mode",
        choices=["greedy", "sample"],
        default="greedy",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    parser.add_argument("--mission-timeout-s", type=float, default=600.0)
    parser.add_argument("--finish-after-return-s", type=float, default=10.0)

    parser.add_argument("--waypoint-tolerance", type=float, default=0.6)
    parser.add_argument("--waypoint-dwell-s", type=float, default=1.5)

    parser.add_argument("--warmup-s", type=float, default=4.0)
    parser.add_argument("--command-retry-s", type=float, default=1.0)

    parser.add_argument(
        "--takeoff-altitude-tolerance",
        type=float,
        default=0.7,
    )
    parser.add_argument("--takeoff-dwell-s", type=float, default=1.5)

    parser.add_argument("--altitude-base", type=float, default=3.0)
    parser.add_argument(
        "--altitude-step",
        type=float,
        default=0.8,
        help="Per-drone altitude separation. Keep <= 0.8 for five drones.",
    )

    parser.add_argument("--land-retry-s", type=float, default=1.0)
    parser.add_argument(
        "--landing-altitude-tolerance",
        type=float,
        default=0.25,
    )
    parser.add_argument("--landing-dwell-s", type=float, default=1.0)
    parser.add_argument(
        "--per-drone-land-timeout-s",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--progress-log-period-s",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    if args.altitude_step > 0.8:
        parser.error("--altitude-step should be <= 0.8 for reliable takeoff")
    if args.altitude_base <= 0.0:
        parser.error("--altitude-base must be positive")
    if args.warmup_s < 1.0:
        parser.error("--warmup-s should be at least 1.0 second")
    if args.command_retry_s <= 0.0:
        parser.error("--command-retry-s must be positive")

    return args


def main() -> None:
    args = parse_args()

    rclpy.init()
    node = DANLivePX4Executor(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # rclpy.shutdown() may already have been called inside control_tick()
        # after mission completion. In that case, destroy_node() can raise a
        # harmless cleanup ValueError in ROS 2 Jazzy. Do not make the whole
        # simulation pipeline fail after a successful mission.
        try:
            node.destroy_node()
        except ValueError as exc:
            print(f"Ignoring ROS cleanup ValueError after shutdown: {exc}")

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()