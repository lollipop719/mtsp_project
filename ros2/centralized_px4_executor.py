from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np
import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from ros2.mission_config import (
    DRONES,
    NUM_DRONES,
    OFFBOARD_HEARTBEAT_HZ,
    OFFBOARD_WARMUP_S,
    PROJECT_ROOT,
    WAYPOINT_TOLERANCE_M,
    DroneSpec,
    home_hover_target,
    planner_delta_to_local_ned,
    planner_point_to_local_ned,
    planner_to_gazebo_enu,
)


class MissionPhase(Enum):
    WAIT_FOR_TELEMETRY = auto()
    WARMUP = auto()
    TAKEOFF = auto()
    EXECUTE = auto()
    LANDING = auto()
    DONE = auto()


@dataclass(frozen=True)
class Mission:
    path: Path
    seed: int
    depots_xy: np.ndarray
    tasks_xy: np.ndarray
    routes: list[list[int]]


@dataclass
class DroneRuntime:
    spec: DroneSpec
    route: list[int]

    position_ned: np.ndarray | None = None
    home_ned: np.ndarray | None = None

    position_valid: bool = False
    arming_state: int | None = None
    nav_state: int | None = None

    route_index: int = 0
    land_command_sent: bool = False


def make_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_mission(path: Path) -> Mission:
    path = make_project_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Mission JSON not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    planner = data["planner"]

    depots_xy = np.asarray(
        planner["depots_xy"],
        dtype=np.float64,
    )

    tasks_xy = np.asarray(
        planner["tasks_xy"],
        dtype=np.float64,
    )

    routes = [
        [int(task_id) for task_id in route]
        for route in planner["routes_task_ids"]
    ]

    if depots_xy.shape != (NUM_DRONES, 2):
        raise ValueError(
            f"Expected depots shape ({NUM_DRONES}, 2), "
            f"got {depots_xy.shape}."
        )

    if tasks_xy.ndim != 2 or tasks_xy.shape[1] != 2:
        raise ValueError("Tasks must have shape [num_tasks, 2].")

    if len(routes) != NUM_DRONES:
        raise ValueError(
            f"Expected {NUM_DRONES} routes, got {len(routes)}."
        )

    assigned = [
        task_id
        for route in routes
        for task_id in route
    ]

    expected = list(range(len(tasks_xy)))

    if sorted(assigned) != expected:
        raise ValueError(
            "Mission routes must assign every task exactly once."
        )

    return Mission(
        path=path,
        seed=int(planner["seed"]),
        depots_xy=depots_xy,
        tasks_xy=tasks_xy,
        routes=routes,
    )


def print_mission_preview(mission: Mission) -> None:
    gazebo_depots = planner_to_gazebo_enu(
        mission.depots_xy
    )

    print(f"Mission: {mission.path}")
    print(f"Seed: {mission.seed}")
    print()

    for spec, route in zip(DRONES, mission.routes):
        depot = mission.depots_xy[spec.depot_index]

        print(
            f"{spec.name} | "
            f"namespace={spec.namespace} | "
            f"PX4 system ID={spec.target_system}"
        )

        print(
            "  Gazebo depot [east, north]: "
            f"[{gazebo_depots[spec.depot_index][0]:.2f}, "
            f"{gazebo_depots[spec.depot_index][1]:.2f}]"
        )

        print(
            f"  Cruise altitude: "
            f"{spec.cruise_altitude_m:.1f} m"
        )

        for task_id in route:
            planner_task = mission.tasks_xy[task_id]
            gazebo_task = planner_to_gazebo_enu(
                planner_task
            )

            local_delta_ned = planner_delta_to_local_ned(
                planner_task - depot
            )

            print(
                f"  Task {task_id + 1:02d}: "
                f"planner=[{planner_task[0]:.2f}, "
                f"{planner_task[1]:.2f}] | "
                f"Gazebo=[{gazebo_task[0]:.2f}, "
                f"{gazebo_task[1]:.2f}] | "
                f"relative PX4 NED="
                f"[{local_delta_ned[0]:.2f}, "
                f"{local_delta_ned[1]:.2f}, "
                f"{local_delta_ned[2]:.2f}]"
            )

        print("  Final step: return to own depot, then land.")
        print()


class CentralizedPX4Executor(Node):
    def __init__(
        self,
        mission: Mission,
        waypoint_tolerance_m: float,
        mission_timeout_s: float,
    ) -> None:
        super().__init__("centralized_px4_executor")

        self.mission = mission
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.mission_timeout_s = mission_timeout_s

        self.phase = MissionPhase.WAIT_FOR_TELEMETRY
        self.phase_started_at = time.monotonic()

        self.warmup_ticks = 0
        self.last_mode_arm_request_at = -float("inf")

        self.runtimes: dict[str, DroneRuntime] = {
            spec.name: DroneRuntime(
                spec=spec,
                route=mission.routes[spec.depot_index],
            )
            for spec in DRONES
        }

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_mode_publishers = {}
        self.trajectory_publishers = {}
        self.command_publishers = {}
        self.subscriptions = []

        for spec in DRONES:
            self.offboard_mode_publishers[spec.name] = (
                self.create_publisher(
                    OffboardControlMode,
                    f"{spec.namespace}/in/offboard_control_mode",
                    qos,
                )
            )

            self.trajectory_publishers[spec.name] = (
                self.create_publisher(
                    TrajectorySetpoint,
                    f"{spec.namespace}/in/trajectory_setpoint",
                    qos,
                )
            )

            self.command_publishers[spec.name] = (
                self.create_publisher(
                    VehicleCommand,
                    f"{spec.namespace}/in/vehicle_command",
                    qos,
                )
            )

            self.subscriptions.append(
                self.create_subscription(
                    VehicleLocalPosition,
                    f"{spec.namespace}/out/"
                    "vehicle_local_position_v1",
                    lambda message, drone_name=spec.name:
                    self.on_local_position(
                        drone_name,
                        message,
                    ),
                    qos,
                )
            )

            self.subscriptions.append(
                self.create_subscription(
                    VehicleStatus,
                    f"{spec.namespace}/out/vehicle_status_v1",
                    lambda message, drone_name=spec.name:
                    self.on_vehicle_status(
                        drone_name,
                        message,
                    ),
                    qos,
                )
            )

        self.timer = self.create_timer(
            1.0 / OFFBOARD_HEARTBEAT_HZ,
            self.tick,
        )

        self.get_logger().info(
            "Executor created. Waiting for local-position "
            "telemetry from all three drones."
        )

    def timestamp_us(self) -> int:
        return int(
            self.get_clock().now().nanoseconds // 1_000
        )

    def set_phase(self, phase: MissionPhase) -> None:
        self.phase = phase
        self.phase_started_at = time.monotonic()

        self.get_logger().info(
            f"Mission phase: {phase.name}"
        )

    def on_local_position(
        self,
        drone_name: str,
        message: VehicleLocalPosition,
    ) -> None:
        runtime = self.runtimes[drone_name]

        position = np.array(
            [message.x, message.y, message.z],
            dtype=np.float64,
        )

        xy_valid = bool(
            getattr(message, "xy_valid", True)
        )

        z_valid = bool(
            getattr(message, "z_valid", True)
        )

        if (
            xy_valid
            and z_valid
            and np.all(np.isfinite(position))
        ):
            runtime.position_ned = position
            runtime.position_valid = True

    def on_vehicle_status(
        self,
        drone_name: str,
        message: VehicleStatus,
    ) -> None:
        runtime = self.runtimes[drone_name]

        runtime.arming_state = int(message.arming_state)
        runtime.nav_state = int(message.nav_state)

    def all_positions_ready(self) -> bool:
        return all(
            runtime.position_valid
            and runtime.position_ned is not None
            for runtime in self.runtimes.values()
        )

    def capture_home_positions(self) -> None:
        for runtime in self.runtimes.values():
            assert runtime.position_ned is not None

            runtime.home_ned = runtime.position_ned.copy()

            self.get_logger().info(
                f"{runtime.spec.name} home NED captured: "
                f"[{runtime.home_ned[0]:.2f}, "
                f"{runtime.home_ned[1]:.2f}, "
                f"{runtime.home_ned[2]:.2f}]"
            )

    def hover_target(
        self,
        runtime: DroneRuntime,
    ) -> np.ndarray:
        if runtime.home_ned is None:
            raise RuntimeError("Home position is unavailable.")

        return home_hover_target(
            runtime.home_ned,
            runtime.spec.cruise_altitude_m,
        )

    def target_for_runtime(
        self,
        runtime: DroneRuntime,
    ) -> np.ndarray:
        if runtime.home_ned is None:
            raise RuntimeError("Home position is unavailable.")

        if self.phase in {
            MissionPhase.WARMUP,
            MissionPhase.TAKEOFF,
        }:
            return self.hover_target(runtime)

        if (
            self.phase == MissionPhase.EXECUTE
            and runtime.route_index < len(runtime.route)
        ):
            task_id = runtime.route[runtime.route_index]

            planner_depot = self.mission.depots_xy[
                runtime.spec.depot_index
            ]

            planner_task = self.mission.tasks_xy[task_id]

            return planner_point_to_local_ned(
                planner_point_xy=planner_task,
                planner_depot_xy=planner_depot,
                home_local_ned=runtime.home_ned,
                cruise_altitude_m=runtime.spec.cruise_altitude_m,
            )

        return self.hover_target(runtime)

    def publish_offboard_target(
        self,
        runtime: DroneRuntime,
        target_ned: np.ndarray,
    ) -> None:
        mode_message = OffboardControlMode()
        mode_message.timestamp = self.timestamp_us()
        mode_message.position = True
        mode_message.velocity = False
        mode_message.acceleration = False
        mode_message.attitude = False
        mode_message.body_rate = False

        self.offboard_mode_publishers[
            runtime.spec.name
        ].publish(mode_message)

        trajectory_message = TrajectorySetpoint()
        trajectory_message.timestamp = self.timestamp_us()
        trajectory_message.position = [
            float(target_ned[0]),
            float(target_ned[1]),
            float(target_ned[2]),
        ]
        trajectory_message.velocity = [
            float("nan"),
            float("nan"),
            float("nan"),
        ]
        trajectory_message.acceleration = [
            float("nan"),
            float("nan"),
            float("nan"),
        ]
        trajectory_message.yaw = float("nan")
        trajectory_message.yawspeed = float("nan")

        self.trajectory_publishers[
            runtime.spec.name
        ].publish(trajectory_message)

    def publish_vehicle_command(
        self,
        runtime: DroneRuntime,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
    ) -> None:
        message = VehicleCommand()
        message.timestamp = self.timestamp_us()

        message.param1 = param1
        message.param2 = param2

        message.command = command
        message.target_system = runtime.spec.target_system
        message.target_component = 1

        message.source_system = 1
        message.source_component = 1
        message.from_external = True

        self.command_publishers[
            runtime.spec.name
        ].publish(message)

    def request_offboard_and_arm(self) -> None:
        now = time.monotonic()

        if now - self.last_mode_arm_request_at < 1.0:
            return

        self.last_mode_arm_request_at = now

        for runtime in self.runtimes.values():
            if runtime.land_command_sent:
                continue

            self.publish_vehicle_command(
                runtime=runtime,
                command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=6.0,
            )

            self.publish_vehicle_command(
                runtime=runtime,
                command=(
                    VehicleCommand
                    .VEHICLE_CMD_COMPONENT_ARM_DISARM
                ),
                param1=1.0,
            )

    def position_reached(
        self,
        runtime: DroneRuntime,
        target_ned: np.ndarray,
    ) -> bool:
        if runtime.position_ned is None:
            return False

        distance = np.linalg.norm(
            runtime.position_ned - target_ned
        )

        return distance <= self.waypoint_tolerance_m

    def all_drones_at_hover_target(self) -> bool:
        return all(
            self.position_reached(
                runtime,
                self.hover_target(runtime),
            )
            for runtime in self.runtimes.values()
        )

    def publish_all_active_targets(self) -> None:
        for runtime in self.runtimes.values():
            if runtime.land_command_sent:
                continue

            target = self.target_for_runtime(runtime)

            self.publish_offboard_target(
                runtime=runtime,
                target_ned=target,
            )

    def advance_routes(self) -> None:
        for runtime in self.runtimes.values():
            if runtime.land_command_sent:
                continue

            target = self.target_for_runtime(runtime)

            if not self.position_reached(
                runtime,
                target,
            ):
                continue

            if runtime.route_index < len(runtime.route):
                task_id = runtime.route[runtime.route_index]

                self.get_logger().info(
                    f"{runtime.spec.name} reached "
                    f"Task {task_id + 1}."
                )

                runtime.route_index += 1

                if runtime.route_index == len(runtime.route):
                    self.get_logger().info(
                        f"{runtime.spec.name} completed its tasks "
                        "and is returning to its depot."
                    )

                continue

            self.send_land_command(runtime)

    def send_land_command(
        self,
        runtime: DroneRuntime,
    ) -> None:
        if runtime.land_command_sent:
            return

        self.get_logger().info(
            f"{runtime.spec.name} is above its depot. "
            "Sending land command."
        )

        self.publish_vehicle_command(
            runtime=runtime,
            command=VehicleCommand.VEHICLE_CMD_NAV_LAND,
        )

        runtime.land_command_sent = True

    def vehicle_is_armed(
        self,
        runtime: DroneRuntime,
    ) -> bool:
        armed_state = getattr(
            VehicleStatus,
            "ARMING_STATE_ARMED",
            2,
        )

        return runtime.arming_state == armed_state

    def all_vehicles_disarmed(self) -> bool:
        return all(
            runtime.arming_state is not None
            and not self.vehicle_is_armed(runtime)
            for runtime in self.runtimes.values()
        )

    def emergency_land_all(self) -> None:
        self.get_logger().error(
            "Mission timeout reached. Landing all drones "
            "at their current locations."
        )

        for runtime in self.runtimes.values():
            self.send_land_command(runtime)

        self.set_phase(MissionPhase.LANDING)

    def tick(self) -> None:
        if self.phase == MissionPhase.DONE:
            return

        if self.phase == MissionPhase.WAIT_FOR_TELEMETRY:
            if self.all_positions_ready():
                self.capture_home_positions()
                self.set_phase(MissionPhase.WARMUP)

            return

        elapsed = time.monotonic() - self.phase_started_at

        if (
            self.phase
            not in {MissionPhase.LANDING, MissionPhase.DONE}
            and elapsed > self.mission_timeout_s
        ):
            self.emergency_land_all()
            return

        if self.phase == MissionPhase.WARMUP:
            self.publish_all_active_targets()

            self.warmup_ticks += 1

            needed_ticks = math.ceil(
                OFFBOARD_WARMUP_S
                * OFFBOARD_HEARTBEAT_HZ
            )

            if self.warmup_ticks >= needed_ticks:
                self.set_phase(MissionPhase.TAKEOFF)
                self.request_offboard_and_arm()

            return

        if self.phase == MissionPhase.TAKEOFF:
            self.publish_all_active_targets()
            self.request_offboard_and_arm()

            if self.all_drones_at_hover_target():
                self.get_logger().info(
                    "All drones reached cruise altitude. "
                    "Starting route execution."
                )

                self.set_phase(MissionPhase.EXECUTE)

            return

        if self.phase == MissionPhase.EXECUTE:
            self.publish_all_active_targets()
            self.advance_routes()

            if all(
                runtime.land_command_sent
                for runtime in self.runtimes.values()
            ):
                self.set_phase(MissionPhase.LANDING)

            return

        if self.phase == MissionPhase.LANDING:
            if self.all_vehicles_disarmed():
                self.get_logger().info(
                    "Mission complete. All drones disarmed."
                )

                self.set_phase(MissionPhase.DONE)
                self.timer.cancel()
                rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute one learned three-drone PX4 mission."
    )

    parser.add_argument(
        "--mission",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "gazebo_missions"
            / "mission_seed_20260707.json"
        ),
    )

    parser.add_argument(
        "--waypoint-tolerance",
        type=float,
        default=WAYPOINT_TOLERANCE_M,
    )

    parser.add_argument(
        "--mission-timeout-s",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the mission and exit without connecting to PX4.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Arm the drones and run the mission.",
    )

    args = parser.parse_args()

    mission = load_mission(args.mission)

    if args.dry_run:
        print_mission_preview(mission)
        return

    if not args.execute:
        parser.error(
            "Choose --dry-run or --execute. "
            "Execution requires explicit confirmation."
        )

    rclpy.init()

    node = CentralizedPX4Executor(
        mission=mission,
        waypoint_tolerance_m=args.waypoint_tolerance,
        mission_timeout_s=args.mission_timeout_s,
    )

    print_mission_preview(mission)

    node.get_logger().info(
        "Mission execution enabled. Do not use QGroundControl "
        "virtual joystick controls while this node is running."
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().warning(
            "Executor interrupted. PX4 will handle Offboard-loss "
            "failsafe according to its configured behavior."
        )

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()