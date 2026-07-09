from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.dan_async_mtsp_env import (
    DANAsyncMTSPEnv,
    compute_route_lengths,
    nearest_task_policy,
    random_policy,
)


def print_result(name: str, result) -> None:
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    for agent_id, route in enumerate(result.routes):
        readable_route = [task_id + 1 for task_id in route]
        print(
            f"Agent {agent_id + 1}: "
            f"tasks={readable_route}, "
            f"length={result.route_lengths[agent_id]:.4f}"
        )

    print(f"Makespan:       {result.makespan:.4f}")
    print(f"Total distance: {result.total_distance:.4f}")
    print(f"Reward:         {result.reward:.4f}")
    print(f"Decisions:      {len(result.decisions)}")
    print(f"Ticks:          {result.num_ticks}")
    print(f"Elapsed time:   {result.elapsed_time:.4f}")


def main() -> None:
    env = DANAsyncMTSPEnv(
        num_agents=3,
        num_tasks=10,
        delta_g=0.1,
        seed=7,
    )

    instance = env.generate_instance(seed=20260709)

    random_rng = np.random.default_rng(123)

    random_result = env.rollout(
        lambda obs: random_policy(obs, rng=random_rng),
        instance=instance,
    )

    nearest_result = env.rollout(
        nearest_task_policy,
        instance=instance,
    )

    print_result("Random asynchronous policy", random_result)
    print_result("Nearest-task asynchronous policy", nearest_result)

    # Check that the returned route lengths are internally consistent.
    checked_lengths = compute_route_lengths(
        instance,
        nearest_result.routes,
    )

    if not np.allclose(checked_lengths, nearest_result.route_lengths):
        raise RuntimeError(
            "Route length consistency check failed."
        )

    assigned_tasks = sorted(
        task_id
        for route in nearest_result.routes
        for task_id in route
    )

    expected_tasks = list(range(instance.num_tasks))

    if assigned_tasks != expected_tasks:
        raise RuntimeError(
            f"Task assignment invalid: {assigned_tasks} != {expected_tasks}"
        )

    print()
    print("DAN environment smoke test passed.")


if __name__ == "__main__":
    main()
