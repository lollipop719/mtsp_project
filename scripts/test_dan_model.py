from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from env.dan_async_mtsp_env import (
    DANAsyncMTSPEnv,
    compute_route_lengths,
)
from models.dan_mtsp import (
    DANMTSPPolicy,
    DANPolicyRunner,
    observation_to_tensors,
)


def check_rollout(name: str, env: DANAsyncMTSPEnv, result) -> None:
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
    print(f"Log probs:      {len(result.log_probs)}")
    print(f"Entropies:      {len(result.entropies)}")

    instance = env.instance

    if instance is None:
        raise RuntimeError("Environment has no active instance.")

    assigned_tasks = sorted(
        task_id
        for route in result.routes
        for task_id in route
    )

    expected_tasks = list(range(instance.num_tasks))

    if assigned_tasks != expected_tasks:
        raise RuntimeError(
            f"Invalid task assignment: {assigned_tasks} != {expected_tasks}"
        )

    checked_lengths = compute_route_lengths(
        instance,
        result.routes,
    )

    if not np.allclose(
        checked_lengths,
        result.route_lengths,
        atol=1e-6,
    ):
        raise RuntimeError("Route length check failed.")


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    env = DANAsyncMTSPEnv(
        num_agents=3,
        num_tasks=10,
        delta_g=0.1,
        seed=7,
    )

    instance = env.generate_instance(seed=20260709)

    model = DANMTSPPolicy(
        embedding_dim=128,
        num_heads=8,
        num_encoder_layers=1,
        dropout=0.0,
    ).to(device)

    model.train()

    # ------------------------------------------------------------
    # Forward pass and masking check.
    # ------------------------------------------------------------
    observation = env.reset(instance=instance)

    # Manually mark two tasks as visited to check masking.
    assert env.visited_mask is not None
    env.visited_mask[0] = True
    env.visited_mask[3] = True

    observation = env.get_observation(deciding_agent=0)

    cities_relative, agents_relative, action_mask = observation_to_tensors(
        observation,
        device,
    )

    logits = model(
        cities_relative=cities_relative,
        agents_relative=agents_relative,
        action_mask=action_mask,
    )

    print()
    print("Forward pass check")
    print(f"  logits shape: {tuple(logits.shape)}")

    if logits.shape != (1, env.num_tasks):
        raise RuntimeError(
            f"Unexpected logits shape: {logits.shape}"
        )

    masked_logits = logits[0, [0, 3]].detach().cpu().numpy()

    if not np.all(masked_logits < -1.0e8):
        raise RuntimeError(
            f"Visited tasks were not properly masked: {masked_logits}"
        )

    print("  masking check passed")

    # ------------------------------------------------------------
    # Greedy rollout.
    # ------------------------------------------------------------
    greedy_policy = DANPolicyRunner(
        model=model,
        device=device,
        mode="greedy",
    )

    greedy_result = env.rollout(
        greedy_policy,
        instance=instance,
    )

    check_rollout(
        "Untrained DAN greedy rollout",
        env,
        greedy_result,
    )

    # ------------------------------------------------------------
    # Sample rollout.
    # ------------------------------------------------------------
    sample_policy = DANPolicyRunner(
        model=model,
        device=device,
        mode="sample",
    )

    sample_result = env.rollout(
        sample_policy,
        instance=instance,
    )

    check_rollout(
        "Untrained DAN sampled rollout",
        env,
        sample_result,
    )

    if len(sample_result.log_probs) != env.num_tasks:
        raise RuntimeError(
            f"Expected {env.num_tasks} log_probs, "
            f"got {len(sample_result.log_probs)}"
        )

    loss_like_value = torch.stack(sample_result.log_probs).sum()

    if not torch.isfinite(loss_like_value):
        raise RuntimeError("Sampled log_probs are not finite.")

    print()
    print("DAN model smoke test passed.")


if __name__ == "__main__":
    main()
