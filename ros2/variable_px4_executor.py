from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
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
from visualization_msgs.msg import Marker, MarkerArray

from ros2.mission_config import PROJECT_ROOT, PLANNER_TO_GAZEBO_SCALE


@dataclass(frozen=True)
class DroneSpec:
    index: int
    name: str
    namespace: str
    target_system: int
    cruise_altitude_m: float
    depot_xy: np.ndarray
    route: list[int]


@dataclass
class DroneRuntime:
    spec: DroneSpec
    home_ned: np.ndarray | None = None
    current_ned: np.ndarray | None = None
    route_index: int = 0
    arrival_started_at: float | None = None
    land_command_sent: bool = False
    landed: bool = False
    status: VehicleStatus | None = None


@dataclass(frozen=True)
class Mission:
    path: Path
    seed: int
    depots_xy: np.ndarray
    tasks_xy: np.ndarray
    routes: list[list[int]]


def make_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_mission(path: Path) -> Mission:
    mission_path = make_project_path(path)

    with mission_path.open("r", encoding="utf-8") as file:
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

    if len(routes) != depots_xy.shape[0]:
        raise ValueError(
            f"Number of routes ({len(routes)}) does not match "
            f"number of depots ({depots_xy.shape[0]})."
        )

    assigned = sorted(task_id for route in routes for task_id in route)
    expected = list(range(tasks_xy.shape[0]))

    if assigned != expected:
        raise ValueError(
            f"Routes do not assign each task exactly once. "
            f"assigned={assigned}, expected={expected}"
        )

    return Mission(
        path=mission_path,
        seed=int(planner.get("seed", data.get("seed", 0))),
        depots_xy=depots_xy,
        tasks_xy=tasks_xy,
        routes=routes,
    )


def make_drone_specs(mission: Mission) -> list[DroneSpec]:
    specs: list[DroneSpec] = []

    for index, route in enumerate(mission.routes):
        specs.append(
            DroneSpec(
                index=index,
                name=f"drone_{index + 1}",
                namespace=f"/px4_{index + 1}/fmu",
                target_system=index + 2,
                # Keep small altitude separation between drones.
                # The old version used 3.0 + index, which made 4/5-drone
                # missions climb too high for visualization.
                cruise_altitude_m=3.0 + 0.4 * float(index),
                depot_xy=mission.depots_xy[index],
                route=route,
            )
        )

    return specs


class VariablePX4Executor(Node):
    def __init__(
        self,
        *,
        mission: Mission,
        waypoint_tolerance_m: float,
        waypoint_dwell_s: float,
        mission_timeout_s: float,
        hover_only: bool,
        execute: bool,
    ) -> None:
        super().__init__("variable_px4_executor")

        self.mission = mission
        self.specs = make_drone_specs(mission)
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.waypoint_dwell_s = waypoint_dwell_s
        self.mission_timeout_s = mission_timeout_s
        self.hover_only = hover_only
        self.execute = execute

        self.runtimes = {
            spec.index: DroneRuntime(spec=spec)
            for spec in self.specs
        }

        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_publishers = {}
        self.setpoint_publishers = {}
        self.command_publishers = {}

        self.subscriptions_by_drone = []

        for spec in self.specs:
            self.offboard_publishers[spec.index] = self.create_publisher(
                OffboardControlMode,
                f"{spec.namespace}/in/offboard_control_mode",
                self.qos,
            )

            self.setpoint_publishers[spec.index] = self.create_publisher(
                TrajectorySetpoint,
                f"{spec.namespace}/in/trajectory_setpoint",
                self.qos,
            )

            self.command_publishers[spec.index] = self.create_publisher(
                VehicleCommand,
                f"{spec.namespace}/in/vehicle_command",
                self.qos,
            )

            self.subscriptions_by_drone.append(
                self.create_subscription(
                    VehicleLocalPosition,
                    f"{spec.namespace}/out/vehicle_local_position_v1",
                    self.make_position_callback(spec.index),
                    self.qos,
                )
            )

            self.subscriptions_by_drone.append(
                self.create_subscription(
                    VehicleStatus,
                    f"{spec.namespace}/out/vehicle_status_v1",
                    self.make_status_callback(spec.index),
                    self.qos,
                )
            )

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            "/mtsp_visualization",
            10,
        )

        self.phase = "WAIT_FOR_TELEMETRY"
        self.phase_started_at = time.monotonic()
        self.mission_started_at: float | None = None
        self.offboard_command_sent = False
        self.arm_command_sent = False

        self.control_timer = self.create_timer(
            0.05,
            self.control_tick,
        )

        self.marker_timer = self.create_timer(
            0.5,
            self.publish_markers,
        )

        self.print_mission_summary()

    def make_position_callback(self, drone_index: int):
        def callback(msg: VehicleLocalPosition) -> None:
            runtime = self.runtimes[drone_index]

            current = np.array(
                [float(msg.x), float(msg.y), float(msg.z)],
                dtype=np.float64,
            )

            runtime.current_ned = current

            if runtime.home_ned is None and np.all(np.isfinite(current)):
                runtime.home_ned = current.copy()
                self.get_logger().info(
                    f"{runtime.spec.name} home NED captured: "
                    f"{runtime.home_ned.tolist()}"
                )

        return callback

    def make_status_callback(self, drone_index: int):
        def callback(msg: VehicleStatus) -> None:
            self.runtimes[drone_index].status = msg

        return callback

    def now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def print_mission_summary(self) -> None:
        print(f"Mission: {self.mission.path}")
        print(f"Seed: {self.mission.seed}")
        print(f"Agents: {len(self.specs)}")
        print(f"Tasks: {self.mission.tasks_xy.shape[0]}")
        print()

        for spec in self.specs:
            print(
                f"{spec.name} | namespace={spec.namespace} | "
                f"PX4 system ID={spec.target_system}"
            )
            print(f"  Planner depot: {spec.depot_xy.tolist()}")
            print(f"  Cruise altitude: {spec.cruise_altitude_m:.1f} m")

            for task_id in spec.route:
                task_xy = self.mission.tasks_xy[task_id]
                relative = self.relative_task_ned(spec, task_id)
                print(
                    f"  Task {task_id + 1:02d}: "
                    f"planner={task_xy.tolist()} | "
                    f"relative PX4 NED="
                    f"[{relative[0]:.2f}, {relative[1]:.2f}, 0.00]"
                )

            print("  Final step: return to own depot, then land.")
            print()

        if self.hover_only:
            self.get_logger().warn(
                "Hover-only mode enabled: drones will take off, hover, then land."
            )

        if self.execute:
            self.get_logger().info(
                "Mission execution enabled. Do not use QGC joystick controls."
            )
        else:
            self.get_logger().warn(
                "Dry-run mode: setpoints will be computed but commands are not sent."
            )

    def relative_task_ned(
        self,
        spec: DroneSpec,
        task_id: int,
    ) -> np.ndarray:
        task_xy = self.mission.tasks_xy[task_id]
        delta_xy = task_xy - spec.depot_xy

        # Planner x = East, planner y = North.
        north = PLANNER_TO_GAZEBO_SCALE * float(delta_xy[1])
        east = PLANNER_TO_GAZEBO_SCALE * float(delta_xy[0])

        return np.array([north, east, 0.0], dtype=np.float64)

    def target_for_runtime(self, runtime: DroneRuntime) -> np.ndarray:
        spec = runtime.spec

        if runtime.home_ned is None:
            return np.zeros(3, dtype=np.float64)

        target = runtime.home_ned.copy()

        if self.hover_only:
            relative = np.array(
                [0.0, 0.0, -spec.cruise_altitude_m],
                dtype=np.float64,
            )
            return target + relative

        if runtime.route_index < len(spec.route):
            task_id = spec.route[runtime.route_index]
            relative = self.relative_task_ned(spec, task_id)
        else:
            relative = np.zeros(3, dtype=np.float64)

        relative[2] = -spec.cruise_altitude_m

        return target + relative

    def depot_hover_target(self, runtime: DroneRuntime) -> np.ndarray:
        if runtime.home_ned is None:
            return np.zeros(3, dtype=np.float64)

        return runtime.home_ned + np.array(
            [0.0, 0.0, -runtime.spec.cruise_altitude_m],
            dtype=np.float64,
        )

    def publish_offboard_mode(self, runtime: DroneRuntime) -> None:
        msg = OffboardControlMode()
        msg.timestamp = self.now_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False

        self.offboard_publishers[runtime.spec.index].publish(msg)

    def publish_setpoint(
        self,
        runtime: DroneRuntime,
        target_ned: np.ndarray,
    ) -> None:
        msg = TrajectorySetpoint()
        msg.timestamp = self.now_us()
        msg.position = [
            float(target_ned[0]),
            float(target_ned[1]),
            float(target_ned[2]),
        ]
        msg.yaw = float("nan")

        self.setpoint_publishers[runtime.spec.index].publish(msg)

    def publish_vehicle_command(
        self,
        runtime: DroneRuntime,
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
        if not self.execute:
            return

        msg = VehicleCommand()
        msg.timestamp = self.now_us()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.command = int(command)
        msg.target_system = int(runtime.spec.target_system)
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.command_publishers[runtime.spec.index].publish(msg)

    def send_offboard_command(self, runtime: DroneRuntime) -> None:
        self.publish_vehicle_command(
            runtime,
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )

    def send_arm_command(self, runtime: DroneRuntime) -> None:
        self.publish_vehicle_command(
            runtime,
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )

    def send_land_command(self, runtime: DroneRuntime) -> None:
        if runtime.land_command_sent:
            return

        self.get_logger().info(
            f"{runtime.spec.name} landing."
        )

        self.publish_vehicle_command(
            runtime,
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
        )

        runtime.land_command_sent = True

    def all_have_home(self) -> bool:
        return all(
            runtime.home_ned is not None
            for runtime in self.runtimes.values()
        )

    def position_reached(
        self,
        runtime: DroneRuntime,
        target_ned: np.ndarray,
    ) -> bool:
        if runtime.current_ned is None:
            return False

        error = runtime.current_ned - target_ned

        horizontal = math.hypot(
            float(error[0]),
            float(error[1]),
        )

        vertical = abs(float(error[2]))

        return (
            horizontal <= self.waypoint_tolerance_m
            and vertical <= 0.35
        )

    def all_at_takeoff_altitude(self) -> bool:
        for runtime in self.runtimes.values():
            target = self.depot_hover_target(runtime)
            if not self.position_reached(runtime, target):
                return False
        return True

    def all_land_commands_sent(self) -> bool:
        return all(
            runtime.land_command_sent
            for runtime in self.runtimes.values()
        )

    def control_tick(self) -> None:
        now = time.monotonic()

        if self.mission_started_at is not None:
            if now - self.mission_started_at > self.mission_timeout_s:
                self.get_logger().error("Mission timeout reached.")
                rclpy.shutdown()
                return

        for runtime in self.runtimes.values():
            if runtime.home_ned is None:
                continue

            self.publish_offboard_mode(runtime)

            if runtime.land_command_sent:
                continue

            # Before ROUTE phase, always hold above the drone's own depot.
            # Otherwise the drone starts flying toward its first task during takeoff.
            if self.phase in {"WAIT_FOR_TELEMETRY", "WARMUP", "TAKEOFF"}:
                target = self.depot_hover_target(runtime)
            else:
                target = self.target_for_runtime(runtime)

            self.publish_setpoint(runtime, target)

        if self.phase == "WAIT_FOR_TELEMETRY":
            if self.all_have_home():
                self.phase = "WARMUP"
                self.phase_started_at = now
                self.mission_started_at = now
                self.get_logger().info("Mission phase: WARMUP")
            return

        if self.phase == "WARMUP":
            if now - self.phase_started_at >= 2.0:
                for runtime in self.runtimes.values():
                    self.send_offboard_command(runtime)
                    self.send_arm_command(runtime)

                self.offboard_command_sent = True
                self.arm_command_sent = True
                self.phase = "TAKEOFF"
                self.phase_started_at = now
                self.get_logger().info("Mission phase: TAKEOFF")
            return

        if self.phase == "TAKEOFF":
            # Keep sending mode/arm briefly because PX4 can miss early commands.
            if now - self.phase_started_at < 5.0:
                for runtime in self.runtimes.values():
                    self.send_offboard_command(runtime)
                    self.send_arm_command(runtime)

            if self.all_at_takeoff_altitude():
                self.phase = "ROUTE"
                self.phase_started_at = now
                self.get_logger().info("Mission phase: ROUTE")

                if self.hover_only:
                    self.get_logger().info(
                        "Hover-only: holding briefly before landing."
                    )
            return

        if self.phase == "ROUTE":
            if self.hover_only:
                if now - self.phase_started_at >= max(3.0, self.waypoint_dwell_s):
                    for runtime in self.runtimes.values():
                        self.send_land_command(runtime)

                    self.phase = "LAND"
                    self.phase_started_at = now
                    self.get_logger().info("Mission phase: LAND")
                return

            self.advance_routes()

            if self.all_land_commands_sent():
                self.phase = "LAND"
                self.phase_started_at = now
                self.get_logger().info("Mission phase: LAND")
            return

        if self.phase == "LAND":
            if now - self.phase_started_at >= 8.0:
                self.get_logger().info("Mission complete.")
                rclpy.shutdown()

    def advance_routes(self) -> None:
        now = time.monotonic()

        for runtime in self.runtimes.values():
            if runtime.land_command_sent:
                continue

            target = self.target_for_runtime(runtime)

            # Important:
            # Once the drone has entered the waypoint radius once, do not reset
            # the dwell timer just because PX4 drifts slightly outside the tight
            # radius. Otherwise the route can get stuck at a task forever.
            if runtime.arrival_started_at is None:
                if not self.position_reached(runtime, target):
                    continue

                runtime.arrival_started_at = now

                if runtime.route_index < len(runtime.spec.route):
                    task_id = runtime.spec.route[runtime.route_index]
                    self.get_logger().info(
                        f"{runtime.spec.name} reached Task {task_id + 1}; "
                        f"holding for {self.waypoint_dwell_s:.1f} s."
                    )
                else:
                    self.get_logger().info(
                        f"{runtime.spec.name} reached depot hover point; "
                        f"holding for {self.waypoint_dwell_s:.1f} s "
                        "before landing."
                    )

                continue

            if now - runtime.arrival_started_at < self.waypoint_dwell_s:
                continue

            runtime.arrival_started_at = None

            if runtime.route_index < len(runtime.spec.route):
                task_id = runtime.spec.route[runtime.route_index]
                runtime.route_index += 1

                self.get_logger().info(
                    f"{runtime.spec.name} accepted Task {task_id + 1} "
                    "and is moving to next target."
                )

                if runtime.route_index < len(runtime.spec.route):
                    next_task_id = runtime.spec.route[runtime.route_index]
                    self.get_logger().info(
                        f"{runtime.spec.name} next target: "
                        f"Task {next_task_id + 1}."
                    )
                else:
                    self.get_logger().info(
                        f"{runtime.spec.name} completed its tasks "
                        "and is returning to depot."
                    )

                continue

            self.send_land_command(runtime)

    def publish_markers(self) -> None:
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        marker_id = 0

        def add_marker(marker: Marker) -> None:
            nonlocal marker_id
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.id = marker_id
            marker_id += 1
            markers.markers.append(marker)

        colors = [
            (1.0, 0.05, 0.05, 1.0),
            (0.05, 1.0, 0.05, 1.0),
            (0.05, 0.25, 1.0, 1.0),
            (1.0, 0.65, 0.05, 1.0),
            (0.65, 0.05, 1.0, 1.0),
            (0.05, 1.0, 1.0, 1.0),
        ]

        # This RViz marker is minimal: current drone positions only.
        for runtime in self.runtimes.values():
            if runtime.current_ned is None or runtime.home_ned is None:
                continue

            relative = runtime.current_ned - runtime.home_ned

            # NED -> ENU relative for visualization.
            east = float(relative[1])
            north = float(relative[0])
            altitude = float(-relative[2])

            color = colors[runtime.spec.index % len(colors)]

            marker = Marker()
            marker.ns = "drone_positions"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = east
            marker.pose.position.y = north
            marker.pose.position.z = altitude
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.45
            marker.scale.y = 0.45
            marker.scale.z = 0.45
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            add_marker(marker)

        if markers.markers:
            self.marker_publisher.publish(markers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Variable-N PX4 offboard executor for mTSP missions."
    )

    parser.add_argument(
        "--mission",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    parser.add_argument(
        "--hover-only",
        action="store_true",
    )

    parser.add_argument(
        "--waypoint-tolerance",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--waypoint-dwell-s",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--mission-timeout-s",
        type=float,
        default=300.0,
    )

    args = parser.parse_args()

    mission = load_mission(args.mission)

    rclpy.init()

    node = VariablePX4Executor(
        mission=mission,
        waypoint_tolerance_m=args.waypoint_tolerance,
        waypoint_dwell_s=args.waypoint_dwell_s,
        mission_timeout_s=args.mission_timeout_s,
        hover_only=args.hover_only,
        execute=args.execute,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn(
            "Executor interrupted. PX4 will handle Offboard-loss failsafe."
        )
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
