from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PX4_ROOT = Path("/workspace/PX4-Autopilot")
PX4_BIN = PX4_ROOT / "build/px4_sitl_default/bin/px4"

sys.path.insert(0, str(PROJECT_ROOT))

from ros2.mission_config import planner_to_gazebo_enu


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def clean_old_processes() -> None:
    print("Cleaning old simulation processes...")

    patterns = [
        "python.*-m ros2.dan_live_executor",
        "python.*-m ros2.dan_live_px4_executor",
        "MicroXRCEAgent",
        "px4_sitl_default/bin/px4",
        "PX4-Autopilot.*bin/px4",
        "gz sim",
        "gz gui",
        "ruby.*gz",
    ]

    for pattern in patterns:
        subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
    time.sleep(2.0)

    for pattern in patterns:
        subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)

    time.sleep(1.0)
    print("Cleanup done.")


def load_world(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_depots_planner(world: dict) -> np.ndarray:
    planner = world.get("planner", {})

    depots = world.get("depots_xy")
    if depots is None:
        depots = planner.get("depots_xy")

    if depots is None:
        raise KeyError("Mission must contain depots_xy or planner.depots_xy")

    return np.asarray(depots, dtype=np.float64)


def wait_for_topic(topic: str, timeout_s: float = 90.0) -> None:
    print(f"Waiting for topic: {topic}")
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        result = run(["ros2", "topic", "list"])
        if result.returncode == 0:
            topics = set(line.strip() for line in result.stdout.splitlines())
            if topic in topics:
                print(f"Detected topic: {topic}")
                return

        time.sleep(1.0)

    raise RuntimeError(f"Timed out waiting for topic: {topic}")


def start_micro_xrce(log_dir: Path) -> subprocess.Popen:
    log_path = log_dir / "micro_xrce_agent.log"
    log_file = log_path.open("w", encoding="utf-8")

    print("\nStarting MicroXRCEAgent")
    print(f"Log: {log_path}")

    return subprocess.Popen(
        ["MicroXRCEAgent", "udp4", "-p", "8888"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def start_executor(
    args: argparse.Namespace,
    log_dir: Path,
    executor_lines: list[str],
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        args.executor_module,
        "--mission",
        str(args.mission.resolve()),
        "--mission-timeout-s",
        str(args.mission_timeout_s),
        "--waypoint-tolerance",
        str(args.waypoint_tolerance),
        "--waypoint-dwell-s",
        str(args.waypoint_dwell_s),
    ]

    if args.execute:
        cmd.append("--execute")
    if args.hover_only:
        cmd.append("--hover-only")

    log_path = log_dir / "dan_live_executor_serial.log"
    log_file = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print("\nStarting DAN live executor first")
    print("Command:")
    print(" ".join(cmd))
    print(f"Log: {log_path}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )

    def tee() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            executor_lines.append(line)

    threading.Thread(target=tee, daemon=True).start()
    return proc


def start_px4_instance(
    drone_index: int,
    pose_enu: np.ndarray,
    log_dir: Path,
) -> subprocess.Popen:
    log_path = log_dir / f"px4_instance_{drone_index}.log"
    log_file = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    env["PX4_SYS_AUTOSTART"] = "4001"
    env["PX4_SIM_MODEL"] = "gz_x500"
    env["PX4_GZ_MODEL_POSE"] = (
        f"{pose_enu[0]:.3f},{pose_enu[1]:.3f},0,0,0,0"
    )

    if drone_index != 1:
        env["PX4_GZ_STANDALONE"] = "1"

    cmd = [str(PX4_BIN), "-i", str(drone_index)]

    print(f"\nStarting PX4 drone instance {drone_index}")
    print("Command:")
    print(" ".join(cmd))
    print(f"PX4_GZ_MODEL_POSE={env['PX4_GZ_MODEL_POSE']}")
    print(f"Log: {log_path}")

    return subprocess.Popen(
        cmd,
        cwd=str(PX4_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )


def wait_for_hover(
    drone_index: int,
    executor_proc: subprocess.Popen,
    executor_lines: list[str],
    timeout_s: float,
) -> None:
    print(f"Waiting for Drone {drone_index} to reach HOVER...")
    deadline = time.time() + timeout_s

    patterns = [
        f"Drone {drone_index}: phase -> HOVER",
        f"Drone {drone_index}: phase -> Phase.HOVER",
        f"Drone {drone_index}: phase -> HOVER_READY",
        f"Drone {drone_index}: phase -> Phase.HOVER_READY",
    ]

    seen = 0

    while time.time() < deadline:
        if executor_proc.poll() is not None:
            raise RuntimeError(
                f"Executor exited while waiting for Drone {drone_index} HOVER."
            )

        new_lines = executor_lines[seen:]
        seen = len(executor_lines)

        for line in new_lines:
            if any(pattern in line for pattern in patterns):
                print(f"Drone {drone_index} reached HOVER.")
                return

        time.sleep(0.2)

    raise RuntimeError(f"Timed out waiting for Drone {drone_index} HOVER.")


def stop_process(name: str, proc: subprocess.Popen | None) -> None:
    if proc is None:
        return

    if proc.poll() is not None:
        return

    print(f"Stopping {name}...")

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serial PX4 spawn + serial executor takeoff pipeline."
    )

    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hover-only", action="store_true")
    parser.add_argument(
        "--executor-module",
        default="ros2.dan_live_executor",
    )

    parser.add_argument("--mission-timeout-s", type=float, default=900.0)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.6)
    parser.add_argument("--waypoint-dwell-s", type=float, default=1.5)

    parser.add_argument("--clean-start", action="store_true")
    parser.add_argument("--topic-timeout-s", type=float, default=120.0)
    parser.add_argument("--hover-timeout-s", type=float, default=180.0)
    parser.add_argument("--spawn-after-hover-gap-s", type=float, default=8.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.mission = args.mission.resolve()

    if args.clean_start:
        clean_old_processes()

    log_dir = PROJECT_ROOT / "outputs/sim_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    world = load_world(args.mission)
    depots_planner = get_depots_planner(world)
    num_drones = int(depots_planner.shape[0])

    print("\nMission:")
    print(f"  {args.mission}")
    print(f"Detected number of drones: {num_drones}")

    depot_poses_enu = np.stack(
        [np.asarray(planner_to_gazebo_enu(xy), dtype=np.float64)
         for xy in depots_planner],
        axis=0,
    )

    for i, pose in enumerate(depot_poses_enu, start=1):
        print(
            f"  Drone {i}: PX4_GZ_MODEL_POSE="
            f"{pose[0]:.3f},{pose[1]:.3f},0,0,0,0"
        )

    micro_proc = None
    executor_proc = None
    px4_procs: list[subprocess.Popen] = []
    executor_lines: list[str] = []

    try:
        micro_proc = start_micro_xrce(log_dir)
        time.sleep(2.0)

        executor_proc = start_executor(args, log_dir, executor_lines)
        time.sleep(3.0)

        for drone_index in range(1, num_drones + 1):
            pose = depot_poses_enu[drone_index - 1]
            px4_proc = start_px4_instance(drone_index, pose, log_dir)
            px4_procs.append(px4_proc)

            topic = f"/px4_{drone_index}/fmu/out/vehicle_local_position_v1"
            wait_for_topic(topic, timeout_s=args.topic_timeout_s)

            wait_for_hover(
                drone_index,
                executor_proc,
                executor_lines,
                timeout_s=args.hover_timeout_s,
            )

            if drone_index < num_drones:
                print(
                    f"Waiting {args.spawn_after_hover_gap_s:.1f} s "
                    "before spawning next PX4..."
                )
                time.sleep(args.spawn_after_hover_gap_s)

        print("\nAll drones spawned and reached hover.")
        print("Executor should now start DAN routing automatically.")

        return_code = executor_proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"DAN executor exited with nonzero code {return_code}"
            )

    finally:
        print("\nCleaning up started processes...")

        for i, proc in reversed(list(enumerate(px4_procs, start=1))):
            stop_process(f"PX4 drone instance {i}", proc)

        stop_process("DAN executor", executor_proc)
        stop_process("MicroXRCEAgent", micro_proc)

        print("Done.")


if __name__ == "__main__":
    main()
