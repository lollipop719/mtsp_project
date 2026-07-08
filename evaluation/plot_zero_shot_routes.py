from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines.greedy import greedy_minmax_insertion
from env.mtsp_2d_env import MTSPInstance, evaluate_routes
from evaluation.zero_shot_evaluate import (
    actions_to_routes,
    generate_zero_shot_instance,
    load_model,
    parse_int_list,
    solve_ortools_minmax_generic,
)


def route_lengths(
    instance: MTSPInstance,
    routes: list[list[int]],
) -> np.ndarray:
    depots = np.asarray(instance.depots, dtype=np.float64)
    tasks = np.asarray(instance.tasks, dtype=np.float64)

    lengths = np.zeros(len(routes), dtype=np.float64)

    for agent_id, route in enumerate(routes):
        depot = depots[agent_id]
        current = depot.copy()

        for task_id in route:
            task = tasks[task_id]
            lengths[agent_id] += np.linalg.norm(task - current)
            current = task

        lengths[agent_id] += np.linalg.norm(depot - current)

    return lengths


def make_agent_colors(num_agents: int) -> list[str]:
    base = [
        "tab:red",
        "tab:green",
        "tab:blue",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
    ]
    return [base[i % len(base)] for i in range(num_agents)]


def plot_routes_on_axis(
    axis,
    instance: MTSPInstance,
    routes: list[list[int]],
    title: str,
) -> None:
    depots = np.asarray(instance.depots, dtype=np.float64)
    tasks = np.asarray(instance.tasks, dtype=np.float64)

    num_agents = depots.shape[0]
    colors = make_agent_colors(num_agents)
    lengths = route_lengths(instance, routes)
    makespan = float(np.max(lengths))
    total = float(np.sum(lengths))

    # plot tasks
    axis.scatter(
        tasks[:, 0],
        tasks[:, 1],
        s=35,
        marker="o",
        label="Tasks",
    )

    for task_id, task_xy in enumerate(tasks):
        axis.text(
            task_xy[0] + 0.12,
            task_xy[1] + 0.12,
            f"T{task_id + 1}",
            fontsize=7,
        )

    # plot depots and routes
    for agent_id in range(num_agents):
        depot = depots[agent_id]
        route = routes[agent_id]
        color = colors[agent_id]

        axis.scatter(
            depot[0],
            depot[1],
            s=120,
            marker="s",
            label=f"Depot D{agent_id + 1}",
        )

        axis.text(
            depot[0] + 0.15,
            depot[1] + 0.15,
            f"D{agent_id + 1}",
            fontsize=9,
            fontweight="bold",
        )

        points = [depot]
        for task_id in route:
            points.append(tasks[task_id])
        points.append(depot)

        points = np.asarray(points, dtype=np.float64)

        axis.plot(
            points[:, 0],
            points[:, 1],
            "-o",
            markersize=4,
            linewidth=1.8,
            label=(
                f"A{agent_id + 1}: "
                f"{len(route)} tasks, "
                f"{lengths[agent_id]:.1f} m"
            ),
        )

    axis.set_title(
        f"{title}\n"
        f"Makespan={makespan:.2f} m, Total={total:.2f} m"
    )

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.legend(fontsize=7, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot zero-shot route visualizations."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/mixed_rl_variable_mtsp/"
            "mixed_rl_best_decode.pt"
        ),
    )

    parser.add_argument(
        "--agents",
        type=str,
        default="2,3,4,5",
        help="Comma-separated agent counts.",
    )

    parser.add_argument(
        "--tasks",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--workspace-size",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--ortools-time-limit",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/route_plots_20tasks"),
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    agent_counts = parse_int_list(args.agents)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Saving figures to: {args.output_dir}")
    print()

    for num_agents in agent_counts:
        instance = generate_zero_shot_instance(
            num_agents=num_agents,
            num_tasks=args.tasks,
            seed=args.seed,
            workspace_size=args.workspace_size,
        )

        depots_tensor = torch.from_numpy(
            instance.depots.astype(np.float32)[None, ...]
        ).to(device)

        tasks_tensor = torch.from_numpy(
            instance.tasks.astype(np.float32)[None, ...]
        ).to(device)

        with torch.inference_mode():
            learned_actions = model.decode(
                depots=depots_tensor,
                tasks=tasks_tensor,
            )[0].cpu().numpy()

        learned_routes = actions_to_routes(
            learned_actions,
            num_agents=num_agents,
            num_tasks=args.tasks,
        )

        greedy_routes = greedy_minmax_insertion(instance)

        ortools_routes = solve_ortools_minmax_generic(
            instance,
            time_limit_s=args.ortools_time_limit,
        )

        if ortools_routes is None:
            print(
                f"Skipping {num_agents} agents because OR-Tools failed."
            )
            continue

        learned_metrics = evaluate_routes(instance, learned_routes)
        greedy_metrics = evaluate_routes(instance, greedy_routes)
        ortools_metrics = evaluate_routes(instance, ortools_routes)

        figure = plt.figure(figsize=(18, 5))
        axes = [
            figure.add_subplot(1, 3, 1),
            figure.add_subplot(1, 3, 2),
            figure.add_subplot(1, 3, 3),
        ]

        plot_routes_on_axis(
            axes[0],
            instance,
            learned_routes,
            (
                f"Learned ({num_agents} agents, {args.tasks} tasks)\n"
                f"{learned_metrics.makespan:.2f} m"
            ),
        )

        plot_routes_on_axis(
            axes[1],
            instance,
            greedy_routes,
            (
                f"Greedy ({num_agents} agents, {args.tasks} tasks)\n"
                f"{greedy_metrics.makespan:.2f} m"
            ),
        )

        plot_routes_on_axis(
            axes[2],
            instance,
            ortools_routes,
            (
                f"OR-Tools ({num_agents} agents, {args.tasks} tasks)\n"
                f"{ortools_metrics.makespan:.2f} m"
            ),
        )

        figure.suptitle(
            f"Zero-shot route comparison | seed={args.seed}",
            fontsize=14,
        )

        figure.tight_layout()

        save_path = (
            args.output_dir
            / f"routes_{num_agents}agents_{args.tasks}tasks_seed{args.seed}.png"
        )

        figure.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(figure)

        print(
            f"Saved: {save_path}\n"
            f"  Learned makespan: {learned_metrics.makespan:.2f} m\n"
            f"  Greedy makespan:  {greedy_metrics.makespan:.2f} m\n"
            f"  OR-Tools makespan:{ortools_metrics.makespan:.2f} m"
        )
        print()


if __name__ == "__main__":
    main()
