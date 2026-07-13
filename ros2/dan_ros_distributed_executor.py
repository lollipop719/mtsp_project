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
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import String

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
    from ros2.mission_config import planner_to_gazebo_enu
except ImportError:
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
        return self.local_position is not None


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_world(path: Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        world = json.load(file)

    if world.get("schema") != "dan_task_world_v1":
        raise ValueError(
            f"Expected mission schema 'dan_task_world_v1', got {world.get('schema')!r}"
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

    model = DANMTSPPolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def extract_policy_action(output: Any) -> int:
    if hasattr(output, "action"):
        return int(output.action)

    if isinstance(output, dict) and "action" in output:
        return int(output["action"])

    if isinstance(output, (int, np.integer)):
        return int(output)

    if torch.is_tensor(output) and output.numel() == 1:
        return int(output.item())

    raise TypeError(f"Could not extract action from policy output type {type(output)}")


class DroneDecisionNode(Node):
    """
    One ROS decision node per drone.

    Each node owns its own local claimed_mask and local model_positions_unit.
    The coordinator does not broadcast the updated claimed mask anymore.
    Instead, every drone listens to /dan/task_claim and updates local state.
    """

    def __init__(
        self,
        agent_id: int,
        checkpoint: Path,
        device: torch.device,
        decode_mode: str,
    ) -> None:
        super().__init__(f"dan_drone_decision_{agent_id + 1}")

        self.agent_id = agent_id
        self.device = device

        model = load_dan_model(checkpoint, device)
        self.policy = DANPolicyRunner(
            model=model,
            device=device,
            mode=decode_mode,
        )

        self.num_agents: int | None = None
        self.num_tasks: int | None = None
        self.tasks_unit: np.ndarray | None = None
        self.claimed_mask: np.ndarray | None = None
        self.model_positions_unit: np.ndarray | None = None

        self.world_ready = False
        self.latest_token: dict[str, Any] | None = None
        self.published_rounds: set[int] = set()
        self.seen_claim_keys: set[tuple[int, int]] = set()

        self.world_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.dan_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.claim_pub = self.create_publisher(
            String,
            "/dan/task_claim",
            self.dan_qos,
        )

        self.create_subscription(
            String,
            "/dan/world_info",
            self._on_world_info,
            self.world_qos,
        )

        self.create_subscription(
            String,
            "/dan/decision_token",
            self._on_token,
            self.dan_qos,
        )

        self.create_subscription(
            String,
            "/dan/task_claim",
            self._on_claim,
            self.dan_qos,
        )

        self.timer = self.create_timer(0.05, self._try_decide)

        self.get_logger().info(
            f"DroneDecisionNode {agent_id + 1} ready on device={device}"
        )

    def _on_world_info(self, msg: String) -> None:
        data = json.loads(msg.data)

        if data.get("type") != "world_info":
            return

        if self.world_ready:
            return

        self.num_agents = int(data["num_agents"])
        self.num_tasks = int(data["num_tasks"])
        self.tasks_unit = np.asarray(data["tasks_unit"], dtype=np.float64)
        self.claimed_mask = np.zeros(self.num_tasks, dtype=bool)
        self.model_positions_unit = np.asarray(
            data["initial_agent_positions_unit"],
            dtype=np.float64,
        )

        self.world_ready = True

        self.get_logger().info(
            f"Drone {self.agent_id + 1}: received initial world info "
            f"agents={self.num_agents}, tasks={self.num_tasks}"
        )

    def _on_token(self, msg: String) -> None:
        token = json.loads(msg.data)

        if token.get("type") != "decision_token":
            return

        self.latest_token = token
        self._try_decide()

    def _on_claim(self, msg: String) -> None:
        try:
            claim = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if claim.get("type") != "task_claim":
            return

        self._apply_claim(claim)

    def _apply_claim(self, claim: dict[str, Any]) -> None:
        if not self.world_ready:
            return

        assert self.claimed_mask is not None
        assert self.model_positions_unit is not None
        assert self.tasks_unit is not None
        assert self.num_tasks is not None
        assert self.num_agents is not None

        round_id = int(claim.get("round", -1))
        agent_id = int(claim.get("agent_id", -1))
        task_id = int(claim.get("task_id", -1))

        claim_key = (round_id, agent_id)
        if claim_key in self.seen_claim_keys:
            return

        self.seen_claim_keys.add(claim_key)

        if agent_id < 0 or agent_id >= self.num_agents:
            return

        if task_id < 0:
            self.get_logger().info(
                f"Drone {self.agent_id + 1}: heard Drone {agent_id + 1} no-task claim"
            )
            return

        if task_id >= self.num_tasks:
            self.get_logger().warning(
                f"Drone {self.agent_id + 1}: heard invalid claim "
                f"D{agent_id + 1} -> Task {task_id + 1}"
            )
            return

        if self.claimed_mask[task_id]:
            self.get_logger().warning(
                f"Drone {self.agent_id + 1}: heard duplicate claim "
                f"D{agent_id + 1} -> Task {task_id + 1}; already locally claimed"
            )
            return

        self.claimed_mask[task_id] = True
        self.model_positions_unit[agent_id] = self.tasks_unit[task_id]

        self.get_logger().info(
            f"Drone {self.agent_id + 1}: updated local claimed_mask from "
            f"D{agent_id + 1} claim -> Task {task_id + 1}"
        )

    def _try_decide(self) -> None:
        if not self.world_ready or self.latest_token is None:
            return

        assert self.num_agents is not None
        assert self.num_tasks is not None
        assert self.tasks_unit is not None
        assert self.claimed_mask is not None
        assert self.model_positions_unit is not None

        token_agent = int(self.latest_token.get("agent_id", -1))
        round_id = int(self.latest_token.get("round", -1))

        if token_agent != self.agent_id:
            return

        if round_id in self.published_rounds:
            return

        valid_task_ids = np.flatnonzero(~self.claimed_mask)

        if valid_task_ids.size == 0:
            task_id = -1
        else:
            current_position = self.model_positions_unit[self.agent_id].copy()

            cities_relative = (
                self.tasks_unit - current_position[None, :]
            ).astype(np.float64)

            agent_positions_relative = (
                self.model_positions_unit - current_position[None, :]
            ).astype(np.float64)

            # In this local-claim version, the decision nodes no longer receive
            # continuous global travel-time updates. This intentionally uses the
            # claimed-state approximation: once an agent claims a task, every node
            # updates that agent's model position immediately.
            remaining_travel_times = np.zeros(self.num_agents, dtype=np.float64)

            agents_relative = np.concatenate(
                [
                    agent_positions_relative,
                    remaining_travel_times[:, None],
                ],
                axis=1,
            ).astype(np.float64)

            observation = DANObservation(
                deciding_agent=self.agent_id,
                cities_relative=cities_relative,
                agents_relative=agents_relative,
                visited_mask=self.claimed_mask.copy(),
                action_mask=(~self.claimed_mask).copy(),
                current_position=current_position,
                agent_positions=self.model_positions_unit.copy(),
                remaining_travel_times=remaining_travel_times.copy(),
            )

            with torch.inference_mode():
                output = self.policy(observation)

            task_id = extract_policy_action(output)

            if (
                task_id < 0
                or task_id >= self.num_tasks
                or self.claimed_mask[task_id]
            ):
                fallback = int(valid_task_ids[0])
                self.get_logger().warning(
                    f"Drone {self.agent_id + 1}: DAN returned invalid/stale "
                    f"Task {task_id + 1 if task_id >= 0 else task_id}; "
                    f"falling back locally to Task {fallback + 1}"
                )
                task_id = fallback

        claim = {
            "type": "task_claim",
            "round": round_id,
            "agent_id": self.agent_id,
            "task_id": int(task_id),
        }

        msg = String()
        msg.data = json.dumps(claim)
        self.claim_pub.publish(msg)

        self.published_rounds.add(round_id)

        # Apply own claim immediately; the echoed topic message will be ignored
        # because seen_claim_keys already includes this round/agent pair.
        self._apply_claim(claim)

        if task_id >= 0:
            self.get_logger().info(
                f"Drone {self.agent_id + 1}: locally selected and broadcast "
                f"Task {task_id + 1}"
            )
        else:
            self.get_logger().info(
                f"Drone {self.agent_id + 1}: broadcast no-task claim"
            )


class DistributedDANPX4Executor(Node):
    """
    PX4 flight coordinator.

    It does not run the DAN policy. It only:
      - publishes initial world/task info
      - sends decision tokens
      - receives task claims
      - commands PX4 setpoints
    """

    CONTROL_PERIOD_S = 0.05

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("dan_distributed_px4_executor")

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

        if self.depots_planner.shape[0] < self.num_agents:
            raise ValueError(
                f"Planner depot coordinates have shape {self.depots_planner.shape}; "
                f"need at least {self.num_agents} depots"
            )

        self.depots_planner = self.depots_planner[: self.num_agents]
        self.initial_agent_positions_unit = (
            self.initial_agent_positions_unit[: self.num_agents]
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

        # Coordinator keeps a canonical physical execution record,
        # but it does not use DAN to choose tasks.
        self.claimed_mask = np.zeros(self.num_tasks, dtype=bool)
        self.model_positions_unit = self.initial_agent_positions_unit.copy()

        self.px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.world_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.dan_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.drones: list[DroneRuntime] = []
        self.offboard_publishers: list[Any] = []
        self.setpoint_publishers: list[Any] = []
        self.command_publishers: list[Any] = []

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
                    self.px4_qos,
                )
            )
            self.setpoint_publishers.append(
                self.create_publisher(
                    TrajectorySetpoint,
                    f"{namespace}/in/trajectory_setpoint",
                    self.px4_qos,
                )
            )
            self.command_publishers.append(
                self.create_publisher(
                    VehicleCommand,
                    f"{namespace}/in/vehicle_command",
                    self.px4_qos,
                )
            )

            self.create_subscription(
                VehicleLocalPosition,
                f"{namespace}/out/vehicle_local_position_v1",
                self._make_local_position_callback(index),
                self.px4_qos,
            )
            self.create_subscription(
                VehicleStatus,
                f"{namespace}/out/vehicle_status_v1",
                self._make_vehicle_status_callback(index),
                self.px4_qos,
            )

        self.world_pub = self.create_publisher(
            String,
            "/dan/world_info",
            self.world_qos,
        )
        self.token_pub = self.create_publisher(
            String,
            "/dan/decision_token",
            self.dan_qos,
        )
        self.create_subscription(
            String,
            "/dan/task_claim",
            self._on_claim,
            self.dan_qos,
        )

        self.started_s = self.now_s()
        self.last_progress_log_s = self.started_s
        self.last_world_info_s = -1.0e9
        self.last_token_publish_s = -1.0e9

        self.routing_started = False
        self.shutdown_requested = False
        self.summary_printed = False
        self.all_landing_started_s: float | None = None

        self.next_takeoff_index = 0
        self.next_takeoff_allowed_s = self.started_s
        self.takeoff_interval_s = args.takeoff_interval_s

        self.pending_decision_agent: int | None = None
        self.decision_round = 0

        self.timer = self.create_timer(
            self.CONTROL_PERIOD_S,
            self.control_tick,
        )

        self.get_logger().info(
            f"Loaded local-claim distributed world: agents={self.num_agents}, tasks={self.num_tasks}"
        )
        self.get_logger().info(
            f"Coordinator does PX4 control + world/token only. DAN runs inside DroneDecisionNodes."
        )
        self.get_logger().info(
            f"execute={args.execute}, hover_only={args.hover_only}"
        )

        for drone in self.drones:
            self.get_logger().info(
                f"Drone {drone.index + 1}: namespace={drone.namespace}, "
                f"sysid={drone.target_system}, home_enu={drone.home_enu.tolist()}, "
                f"altitude={drone.altitude_m:.2f} m"
            )

    # ------------------------------------------------------------------
    # Mission loading
    # ------------------------------------------------------------------

    def _load_tasks_unit(self) -> np.ndarray:
        return np.asarray(self.world["tasks_xy_unit"], dtype=np.float64)

    def _load_tasks_planner(self) -> np.ndarray:
        if "tasks_xy_planner" in self.world:
            return np.asarray(self.world["tasks_xy_planner"], dtype=np.float64)
        if "tasks_xy" in self.world:
            return np.asarray(self.world["tasks_xy"], dtype=np.float64)
        planner = self.world.get("planner", {})
        if "tasks_xy" in planner:
            return np.asarray(planner["tasks_xy"], dtype=np.float64)
        raise KeyError("Mission is missing task planner coordinates")

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
            raise KeyError("Mission must contain depots_xy or common depot")

        common_array = np.asarray(common, dtype=np.float64)
        return np.repeat(common_array[None, :], self.num_agents, axis=0)

    def _load_initial_agent_positions_unit(self) -> np.ndarray:
        common = self.world.get("depot_xy_unit")
        if common is not None:
            common_array = np.asarray(common, dtype=np.float64)
            return np.repeat(common_array[None, :], self.num_agents, axis=0)

        planner_size = float(
            self.world.get("coordinate_frame", {}).get("planner_size", 20.0)
        )
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

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds // 1000)

    # ------------------------------------------------------------------
    # PX4 helpers
    # ------------------------------------------------------------------

    def current_enu(self, drone: DroneRuntime) -> np.ndarray:
        if drone.local_position is None:
            return drone.home_enu.copy()

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

    def target_reached(
        self,
        drone: DroneRuntime,
        target_enu: np.ndarray,
        altitude_m: float,
        tolerance_m: float,
    ) -> bool:
        horizontal_error = float(np.linalg.norm(self.current_enu(drone) - target_enu))
        vertical_error = abs(self.current_altitude(drone) - altitude_m)
        return math.hypot(horizontal_error, vertical_error) <= tolerance_m

    def condition_held(
        self,
        drone: DroneRuntime,
        condition: bool,
        dwell_s: float,
        now: float,
    ) -> bool:
        if not condition:
            drone.condition_started_s = None
            return False

        if drone.condition_started_s is None:
            drone.condition_started_s = now
            return dwell_s <= 0.0

        return (now - drone.condition_started_s) >= dwell_s

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
    ) -> None:
        if not self.args.execute:
            return

        msg = VehicleCommand()
        msg.timestamp = self.timestamp_us()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = int(drone.target_system)
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.command_publishers[drone.index].publish(msg)

    def send_arm_and_offboard(self, drone: DroneRuntime, now: float) -> None:
        self.publish_vehicle_command(
            drone,
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
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

    def set_phase(self, drone: DroneRuntime, phase: Phase, now: float) -> None:
        if drone.phase == phase:
            return

        drone.phase = phase
        drone.phase_started_s = now
        drone.condition_started_s = None

        if phase == Phase.LAND:
            drone.land_started_s = now

        self.get_logger().info(
            f"Drone {drone.index + 1}: phase -> {phase.value}"
        )

    def hold(
        self,
        drone: DroneRuntime,
        target_enu: np.ndarray,
        altitude_m: float,
    ) -> None:
        self.publish_offboard_mode(drone)
        self.publish_setpoint(drone, target_enu, altitude_m)

    # ------------------------------------------------------------------
    # Initial world broadcast and token/claim communication
    # ------------------------------------------------------------------

    def publish_world_info(self) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "type": "world_info",
                "num_agents": self.num_agents,
                "num_tasks": self.num_tasks,
                "tasks_unit": self.tasks_unit.tolist(),
                "initial_agent_positions_unit": self.initial_agent_positions_unit.tolist(),
            }
        )
        self.world_pub.publish(msg)
        self.last_world_info_s = self.now_s()

    def request_claim(self, drone: DroneRuntime, now: float) -> None:
        if self.pending_decision_agent is not None:
            return

        if bool(np.all(self.claimed_mask)):
            return

        self.pending_decision_agent = drone.index
        self.decision_round += 1

        token = {
            "type": "decision_token",
            "round": self.decision_round,
            "agent_id": drone.index,
        }

        msg = String()
        msg.data = json.dumps(token)
        self.token_pub.publish(msg)
        self.last_token_publish_s = now

        self.get_logger().info(
            f"Coordinator token -> Drone {drone.index + 1}, round {self.decision_round}"
        )

    def _republish_pending_token_if_needed(self, now: float) -> None:
        if self.pending_decision_agent is None:
            return

        if now - self.last_token_publish_s < 0.5:
            return

        token = {
            "type": "decision_token",
            "round": self.decision_round,
            "agent_id": self.pending_decision_agent,
        }

        msg = String()
        msg.data = json.dumps(token)
        self.token_pub.publish(msg)
        self.last_token_publish_s = now

    def _on_claim(self, msg: String) -> None:
        try:
            claim = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Coordinator received malformed task_claim JSON")
            return

        if claim.get("type") != "task_claim":
            return

        round_id = int(claim.get("round", -1))
        agent_id = int(claim.get("agent_id", -1))
        task_id = int(claim.get("task_id", -1))

        if round_id != self.decision_round:
            return

        if self.pending_decision_agent != agent_id:
            return

        if agent_id < 0 or agent_id >= self.num_agents:
            return

        drone = self.drones[agent_id]

        if task_id < 0:
            self.pending_decision_agent = None
            self.get_logger().info(
                f"Coordinator received no-task claim from Drone {agent_id + 1}"
            )
            return

        if task_id >= self.num_tasks:
            self.get_logger().error(
                f"Coordinator rejected invalid claim: Drone {agent_id + 1} -> Task {task_id + 1}"
            )
            self.pending_decision_agent = None
            return

        if self.claimed_mask[task_id]:
            self.get_logger().error(
                f"Coordinator rejected duplicate claim: Drone {agent_id + 1} -> Task {task_id + 1}"
            )
            self.pending_decision_agent = None
            return

        self.claimed_mask[task_id] = True
        drone.current_task_id = task_id
        drone.current_target_enu = self.tasks_enu[task_id].copy()
        drone.route_task_ids.append(task_id)
        drone.condition_started_s = None

        self.model_positions_unit[agent_id] = self.tasks_unit[task_id]
        self.pending_decision_agent = None

        self.get_logger().info(
            f"Coordinator accepted claim: Drone {agent_id + 1} -> Task {task_id + 1} | "
            f"claimed {int(np.count_nonzero(self.claimed_mask))}/{self.num_tasks}"
        )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def control_tick(self) -> None:
        now = self.now_s()

        if self.shutdown_requested:
            return

        # Static world info only. Repeated for robustness, but it does not
        # contain the changing claimed_mask.
        if now - self.last_world_info_s >= 1.0:
            self.publish_world_info()

        self._republish_pending_token_if_needed(now)

        if now - self.started_s > self.args.mission_timeout_s:
            self._handle_timeout(now)
            return

        self._start_routing_if_ready(now)

        for drone in self.drones:
            if not drone.has_telemetry():
                continue

            active_before_routing = (
                drone.index < self.next_takeoff_index
                or (
                    drone.index == self.next_takeoff_index
                    and now >= self.next_takeoff_allowed_s
                )
            )

            if (
                not self.routing_started
                and drone.phase in (Phase.WAIT_FOR_TELEMETRY, Phase.WARMUP)
                and not active_before_routing
            ):
                continue

            if drone.phase == Phase.WAIT_FOR_TELEMETRY:
                self._update_wait_for_telemetry(drone, now)
            elif drone.phase == Phase.WARMUP:
                self._update_warmup(drone, now)
            elif drone.phase == Phase.TAKEOFF:
                self._update_takeoff(drone, now)
            elif drone.phase == Phase.HOVER_READY:
                self._update_hover_ready(drone, now)
            elif drone.phase == Phase.ROUTE:
                self._update_route(drone, now)
            elif drone.phase == Phase.RETURN:
                self._update_return(drone, now)
            elif drone.phase == Phase.LAND:
                self._update_land(drone, now)

        self._log_progress(now)
        self._update_global_finish(now)

    def _update_wait_for_telemetry(self, drone: DroneRuntime, now: float) -> None:
        self.set_phase(drone, Phase.WARMUP, now)

    def _update_warmup(self, drone: DroneRuntime, now: float) -> None:
        self.hold(drone, drone.home_enu, drone.altitude_m)

        if now - drone.phase_started_s >= self.args.warmup_s:
            self.get_logger().info(
                f"Serial local-claim control: starting Drone {drone.index + 1}"
            )
            self.send_arm_and_offboard(drone, now)
            self.set_phase(drone, Phase.TAKEOFF, now)

    def _update_takeoff(self, drone: DroneRuntime, now: float) -> None:
        self.hold(drone, drone.home_enu, drone.altitude_m)

        retry_s = 0.25 if self.current_altitude(drone) < 0.5 else self.args.command_retry_s
        if now - drone.last_arm_offboard_command_s >= retry_s:
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

            if drone.index == self.next_takeoff_index:
                self.next_takeoff_index += 1
                self.next_takeoff_allowed_s = now + self.takeoff_interval_s

                if self.next_takeoff_index < self.num_agents:
                    self.get_logger().info(
                        f"Serial local-claim control: Drone {drone.index + 1} is HOVER_READY. "
                        f"Drone {self.next_takeoff_index + 1} starts in "
                        f"{self.takeoff_interval_s:.1f} s."
                    )
                else:
                    self.get_logger().info(
                        "Serial local-claim control: all drones reached HOVER_READY."
                    )

    def _update_hover_ready(self, drone: DroneRuntime, now: float) -> None:
        self.hold(drone, drone.home_enu, drone.altitude_m)

    def _start_routing_if_ready(self, now: float) -> None:
        if self.routing_started:
            return

        if not all(drone.phase == Phase.HOVER_READY for drone in self.drones):
            return

        if self.args.hover_only:
            self.get_logger().info(
                "All drones reached HOVER_READY. Hover-only test is stable."
            )
            self.routing_started = True
            return

        self.routing_started = True
        self.get_logger().info(
            "All drones reached HOVER_READY. Starting local-claim ROS-distributed DAN routing."
        )

        for drone in self.drones:
            self.set_phase(drone, Phase.ROUTE, now)

    def _update_route(self, drone: DroneRuntime, now: float) -> None:
        if drone.current_task_id is None:
            if bool(np.all(self.claimed_mask)):
                self.model_positions_unit[drone.index] = (
                    self.initial_agent_positions_unit[drone.index]
                )
                drone.current_target_enu = drone.home_enu.copy()
                self.set_phase(drone, Phase.RETURN, now)
                self._update_return(drone, now)
                return

            self.request_claim(drone, now)
            self.hold(drone, self.current_enu(drone), drone.altitude_m)
            return

        assert drone.current_target_enu is not None

        self.hold(
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
        self.hold(drone, drone.home_enu, drone.altitude_m)

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

    def _update_land(self, drone: DroneRuntime, now: float) -> None:
        self.publish_offboard_mode(drone)
        self.publish_setpoint(
            drone,
            drone.home_enu,
            max(0.0, self.current_altitude(drone)),
        )

        if now - drone.last_land_command_s >= self.args.land_retry_s:
            self.send_land(drone, now)

        landed = (
            self.current_altitude(drone)
            <= self.args.landing_altitude_tolerance
        )

        if self.condition_held(
            drone,
            landed,
            self.args.landing_dwell_s,
            now,
        ):
            self.set_phase(drone, Phase.DONE, now)

    def _handle_timeout(self, now: float) -> None:
        self.get_logger().error("Mission timeout reached. Sending LAND to all drones.")

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
                "All drones are in LAND/DONE. Waiting before shutdown..."
            )
            return

        if now - self.all_landing_started_s >= self.args.finish_after_return_s:
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
                f"/alt={self.current_altitude(drone):.2f}"
                f"/tgt={drone.altitude_m:.2f}"
            )
            for drone in self.drones
        )

        pending = (
            "none"
            if self.pending_decision_agent is None
            else f"D{self.pending_decision_agent + 1}"
        )

        self.get_logger().info(
            f"Progress: claimed {int(np.count_nonzero(self.claimed_mask))}/{self.num_tasks} | "
            f"pending={pending} | {phases} | {telemetry}"
        )

    def print_summary(self) -> None:
        if self.summary_printed:
            return

        self.summary_printed = True
        self.get_logger().info("Local-claim ROS-distributed DAN PX4 mission summary")

        for drone in self.drones:
            readable_tasks = [task_id + 1 for task_id in drone.route_task_ids]
            self.get_logger().info(
                f"Drone {drone.index + 1}: phase={drone.phase.value}, tasks={readable_tasks}"
            )


def make_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROS-distributed DAN with local per-drone claim state."
    )

    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/dan_refine_2to5_agents_15to25_tasks/dan_best.pt"
        ),
    )

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hover-only", action="store_true")
    parser.add_argument("--num-drones", type=int, default=None)

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

    parser.add_argument("--mission-timeout-s", type=float, default=700.0)
    parser.add_argument("--finish-after-return-s", type=float, default=10.0)

    parser.add_argument("--waypoint-tolerance", type=float, default=0.6)
    parser.add_argument("--waypoint-dwell-s", type=float, default=1.5)

    parser.add_argument("--warmup-s", type=float, default=4.0)
    parser.add_argument("--command-retry-s", type=float, default=1.0)

    parser.add_argument("--takeoff-altitude-tolerance", type=float, default=0.7)
    parser.add_argument("--takeoff-dwell-s", type=float, default=1.5)
    parser.add_argument("--takeoff-interval-s", type=float, default=5.0)

    parser.add_argument("--altitude-base", type=float, default=3.0)
    parser.add_argument("--altitude-step", type=float, default=0.35)

    parser.add_argument("--land-retry-s", type=float, default=1.0)
    parser.add_argument("--landing-altitude-tolerance", type=float, default=0.25)
    parser.add_argument("--landing-dwell-s", type=float, default=1.0)

    parser.add_argument("--progress-log-period-s", type=float, default=5.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = make_device(args.device)

    rclpy.init()

    flight_node = DistributedDANPX4Executor(args)
    decision_nodes = [
        DroneDecisionNode(
            agent_id=index,
            checkpoint=args.checkpoint,
            device=device,
            decode_mode=args.decode_mode,
        )
        for index in range(flight_node.num_agents)
    ]

    executor = MultiThreadedExecutor()
    executor.add_node(flight_node)
    for node in decision_nodes:
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in decision_nodes:
            try:
                executor.remove_node(node)
                node.destroy_node()
            except Exception:
                pass

        try:
            executor.remove_node(flight_node)
            flight_node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
