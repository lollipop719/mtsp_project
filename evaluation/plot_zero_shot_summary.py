from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "num_agents": float(row["num_agents"]),
                    "num_tasks": float(row["num_tasks"]),
                    "learned_makespan_m": float(row["learned_makespan_m"]),
                    "greedy_makespan_m": float(row["greedy_makespan_m"]),
                    "ortools_makespan_m": float(row["ortools_makespan_m"]),
                    "learned_gap_to_ortools_pct": float(
                        row["learned_gap_to_ortools_pct"]
                    ),
                    "greedy_gap_to_ortools_pct": float(
                        row["greedy_gap_to_ortools_pct"]
                    ),
                }
            )

    return rows


def filter_rows(
    rows: list[dict[str, float]],
    *,
    num_agents: int | None = None,
    num_tasks: int | None = None,
) -> list[dict[str, float]]:
    selected = []

    for row in rows:
        if num_agents is not None and int(row["num_agents"]) != num_agents:
            continue

        if num_tasks is not None and int(row["num_tasks"]) != num_tasks:
            continue

        selected.append(row)

    return selected


def save_gap_vs_agents(
    rows: list[dict[str, float]],
    output_dir: Path,
    fixed_tasks: int,
) -> None:
    selected = filter_rows(
        rows,
        num_tasks=fixed_tasks,
    )

    selected = sorted(
        selected,
        key=lambda row: row["num_agents"],
    )

    agents = np.asarray(
        [row["num_agents"] for row in selected],
        dtype=np.float64,
    )

    learned_gap = np.asarray(
        [row["learned_gap_to_ortools_pct"] for row in selected],
        dtype=np.float64,
    )

    greedy_gap = np.asarray(
        [row["greedy_gap_to_ortools_pct"] for row in selected],
        dtype=np.float64,
    )

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(1, 1, 1)

    axis.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        label="OR-Tools reference",
    )

    axis.plot(
        agents,
        learned_gap,
        marker="o",
        linewidth=2.0,
        label="Learned mixed RL",
    )

    axis.plot(
        agents,
        greedy_gap,
        marker="s",
        linewidth=2.0,
        label="Greedy baseline",
    )

    axis.set_title(
        f"Makespan gap vs number of agents\n"
        f"{fixed_tasks} tasks"
    )

    axis.set_xlabel("Number of agents")
    axis.set_ylabel("Gap to OR-Tools makespan (%)")
    axis.set_xticks(agents)
    axis.grid(True)
    axis.legend()

    save_path = output_dir / f"gap_vs_agents_{fixed_tasks}tasks.png"

    figure.savefig(
        save_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {save_path}")


def save_gap_vs_tasks(
    rows: list[dict[str, float]],
    output_dir: Path,
    fixed_agents: int,
) -> None:
    selected = filter_rows(
        rows,
        num_agents=fixed_agents,
    )

    selected = sorted(
        selected,
        key=lambda row: row["num_tasks"],
    )

    tasks = np.asarray(
        [row["num_tasks"] for row in selected],
        dtype=np.float64,
    )

    learned_gap = np.asarray(
        [row["learned_gap_to_ortools_pct"] for row in selected],
        dtype=np.float64,
    )

    greedy_gap = np.asarray(
        [row["greedy_gap_to_ortools_pct"] for row in selected],
        dtype=np.float64,
    )

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(1, 1, 1)

    axis.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        label="OR-Tools reference",
    )

    axis.plot(
        tasks,
        learned_gap,
        marker="o",
        linewidth=2.0,
        label="Learned mixed RL",
    )

    axis.plot(
        tasks,
        greedy_gap,
        marker="s",
        linewidth=2.0,
        label="Greedy baseline",
    )

    axis.set_title(
        f"Makespan gap vs number of tasks\n"
        f"{fixed_agents} agents"
    )

    axis.set_xlabel("Number of tasks")
    axis.set_ylabel("Gap to OR-Tools makespan (%)")
    axis.set_xticks(tasks)
    axis.grid(True)
    axis.legend()

    save_path = output_dir / f"gap_vs_tasks_{fixed_agents}agents.png"

    figure.savefig(
        save_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {save_path}")


def save_makespan_comparison(
    rows: list[dict[str, float]],
    output_dir: Path,
) -> None:
    rows = sorted(
        rows,
        key=lambda row: (
            row["num_agents"],
            row["num_tasks"],
        ),
    )

    labels = [
        f"{int(row['num_agents'])}A-{int(row['num_tasks'])}T"
        for row in rows
    ]

    x_positions = np.arange(len(rows), dtype=np.float64)
    bar_width = 0.25

    learned = np.asarray(
        [row["learned_makespan_m"] for row in rows],
        dtype=np.float64,
    )

    greedy = np.asarray(
        [row["greedy_makespan_m"] for row in rows],
        dtype=np.float64,
    )

    ortools = np.asarray(
        [row["ortools_makespan_m"] for row in rows],
        dtype=np.float64,
    )

    figure = plt.figure(figsize=(14, 6))
    axis = figure.add_subplot(1, 1, 1)

    axis.bar(
        x_positions - bar_width,
        learned,
        width=bar_width,
        label="Learned mixed RL",
    )

    axis.bar(
        x_positions,
        greedy,
        width=bar_width,
        label="Greedy baseline",
    )

    axis.bar(
        x_positions + bar_width,
        ortools,
        width=bar_width,
        label="OR-Tools",
    )

    axis.set_title(
        "Makespan comparison across zero-shot settings"
    )

    axis.set_xlabel("Setting")
    axis.set_ylabel("Makespan (m)")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )
    axis.grid(True, axis="y")
    axis.legend()

    save_path = output_dir / "makespan_comparison_all_settings.png"

    figure.savefig(
        save_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {save_path}")


def save_gap_heatmap(
    rows: list[dict[str, float]],
    output_dir: Path,
) -> None:
    """
    Optional extra figure: learned gap heatmap.
    This is useful for reports because it shows the whole generalization table.
    """
    agents = sorted(
        {int(row["num_agents"]) for row in rows}
    )

    tasks = sorted(
        {int(row["num_tasks"]) for row in rows}
    )

    matrix = np.zeros(
        (len(agents), len(tasks)),
        dtype=np.float64,
    )

    lookup = {
        (int(row["num_agents"]), int(row["num_tasks"])): row
        for row in rows
    }

    for i, num_agents in enumerate(agents):
        for j, num_tasks in enumerate(tasks):
            matrix[i, j] = lookup[
                (num_agents, num_tasks)
            ]["learned_gap_to_ortools_pct"]

    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(1, 1, 1)

    image = axis.imshow(
        matrix,
        aspect="auto",
    )

    axis.set_title("Learned model gap to OR-Tools (%)")
    axis.set_xlabel("Number of tasks")
    axis.set_ylabel("Number of agents")

    axis.set_xticks(np.arange(len(tasks)))
    axis.set_xticklabels([str(task) for task in tasks])

    axis.set_yticks(np.arange(len(agents)))
    axis.set_yticklabels([str(agent) for agent in agents])

    for i in range(len(agents)):
        for j in range(len(tasks)):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:+.1f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Gap to OR-Tools (%)",
    )

    save_path = output_dir / "learned_gap_heatmap.png"

    figure.savefig(
        save_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot zero-shot summary figures."
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/zero_shot_mixed_rl/summary.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/zero_shot_mixed_rl/figures"),
    )

    parser.add_argument(
        "--fixed-tasks",
        type=int,
        default=20,
        help="Task count used for gap-vs-agents plot.",
    )

    parser.add_argument(
        "--fixed-agents",
        type=int,
        default=3,
        help="Agent count used for gap-vs-tasks plot.",
    )

    parser.add_argument(
        "--include-heatmap",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.summary.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {args.summary}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_summary_csv(args.summary)

    save_gap_vs_agents(
        rows=rows,
        output_dir=args.output_dir,
        fixed_tasks=args.fixed_tasks,
    )

    save_gap_vs_tasks(
        rows=rows,
        output_dir=args.output_dir,
        fixed_agents=args.fixed_agents,
    )

    save_makespan_comparison(
        rows=rows,
        output_dir=args.output_dir,
    )

    if args.include_heatmap:
        save_gap_heatmap(
            rows=rows,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
