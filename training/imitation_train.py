from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from env.mtsp_2d_env import MTSPInstance, evaluate_routes
from models.centralized_attention_mtsp import CentralizedAttentionMTSP


def augment_equivalent_teacher_actions(
    actions: np.ndarray,
    num_drones: int = 3,
) -> np.ndarray:
    """
    Create another equally valid teacher sequence.

    - Each drone's closed depot route may be reversed.
    - The three route sequences are randomly interleaved while preserving
      the internal order of every individual drone route.
    """
    routes: list[list[int]] = [[] for _ in range(num_drones)]

    for drone_id, task_id in actions:
        routes[int(drone_id)].append(int(task_id))

    for route in routes:
        if len(route) > 1 and np.random.rand() < 0.5:
            route.reverse()

    positions = [0] * num_drones
    augmented: list[tuple[int, int]] = []

    while len(augmented) < len(actions):
        active_drones = [
            drone_id
            for drone_id in range(num_drones)
            if positions[drone_id] < len(routes[drone_id])
        ]

        chosen_drone = int(np.random.choice(active_drones))
        chosen_task = routes[chosen_drone][positions[chosen_drone]]

        augmented.append((chosen_drone, chosen_task))
        positions[chosen_drone] += 1

    return np.asarray(augmented, dtype=np.int64)

class MTSPDataset(Dataset):
    """Loads one fixed-size 3-drone mTSP OR-Tools-label dataset."""

    def __init__(
        self,
        path: str | Path,
        augment_labels: bool = False,
    ) -> None:
        path = Path(path)

        with np.load(path, allow_pickle=False) as data:
            self.depots = data["depots"].astype(np.float32)
            self.tasks = data["tasks"].astype(np.float32)
            self.teacher_actions = data["teacher_actions"].astype(np.int64)
            self.makespans = data["makespans"].astype(np.float32)
            self.total_distances = data["total_distances"].astype(np.float32)

        if self.depots.ndim != 3 or self.depots.shape[1:] != (3, 2):
            raise ValueError("Expected depots with shape [N, 3, 2].")

        if self.tasks.ndim != 3 or self.tasks.shape[2] != 2:
            raise ValueError("Expected tasks with shape [N, T, 2].")

        if self.teacher_actions.shape[:2] != self.tasks.shape[:2]:
            raise ValueError(
                "teacher_actions must have one [drone, task] action per task."
            )

        if self.teacher_actions.shape[2] != 2:
            raise ValueError("teacher_actions must have shape [N, T, 2].")

        self.augment_labels = augment_labels

    @property
    def num_tasks(self) -> int:
        return int(self.tasks.shape[1])

    def __len__(self) -> int:
        return int(self.tasks.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        actions = self.teacher_actions[index].copy()

        if self.augment_labels:
            actions = augment_equivalent_teacher_actions(
                actions,
                num_drones=self.depots.shape[1],
            )

        return {
            "depots": torch.from_numpy(self.depots[index]),
            "tasks": torch.from_numpy(self.tasks[index]),
            "teacher_actions": torch.from_numpy(actions),
            "makespan": torch.tensor(self.makespans[index]),
            "total_distance": torch.tensor(
                self.total_distances[index]
            ),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def joint_action_targets(
    teacher_actions: Tensor,
    num_tasks: int,
) -> Tensor:
    """
    Convert [drone_id, task_id] to one class index.

    For 15 tasks:
      Drone 0, Task 0 -> class 0
      Drone 0, Task 1 -> class 1
      ...
      Drone 1, Task 0 -> class 15
      ...
    """
    drone_ids = teacher_actions[..., 0].long()
    task_ids = teacher_actions[..., 1].long()

    return drone_ids * num_tasks + task_ids


def teacher_forced_loss(
    model: CentralizedAttentionMTSP,
    batch: dict[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, float]:
    depots = batch["depots"].to(device, non_blocking=True)
    tasks = batch["tasks"].to(device, non_blocking=True)
    actions = batch["teacher_actions"].to(device, non_blocking=True)

    num_tasks = tasks.shape[1]
    num_drones = depots.shape[1]

    logits = model(
        depots=depots,
        tasks=tasks,
        teacher_actions=actions,
    )

    targets = joint_action_targets(
        actions,
        num_tasks=num_tasks,
    )

    flattened_logits = logits.reshape(
        -1,
        num_drones * num_tasks,
    )

    flattened_targets = targets.reshape(-1)

    loss = functional.cross_entropy(
        flattened_logits,
        flattened_targets,
    )

    predictions = flattened_logits.argmax(dim=1)

    accuracy = (
        predictions == flattened_targets
    ).float().mean().item()

    return loss, accuracy


def actions_to_routes(
    actions: np.ndarray,
    num_drones: int = 3,
) -> list[list[int]]:
    """Convert a decoded sequence of [drone_id, task_id] actions into routes."""
    routes: list[list[int]] = [[] for _ in range(num_drones)]

    for drone_id, task_id in actions:
        routes[int(drone_id)].append(int(task_id))

    return routes


@torch.no_grad()
def evaluate_teacher_forcing(
    model: CentralizedAttentionMTSP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    weighted_loss_sum = 0.0
    weighted_accuracy_sum = 0.0
    total_instances = 0

    for batch in loader:
        loss, accuracy = teacher_forced_loss(
            model=model,
            batch=batch,
            device=device,
        )

        batch_size = batch["tasks"].shape[0]

        weighted_loss_sum += loss.item() * batch_size
        weighted_accuracy_sum += accuracy * batch_size
        total_instances += batch_size

    return (
        weighted_loss_sum / total_instances,
        weighted_accuracy_sum / total_instances,
    )


@torch.no_grad()
def evaluate_greedy_decode(
    model: CentralizedAttentionMTSP,
    loader: DataLoader,
    device: torch.device,
    max_instances: int | None = None,
) -> dict[str, float]:
    """
    Evaluate actual decoded routes, not just teacher-forced token accuracy.

    This is the metric we care about for later PX4 execution:
    route makespan and total route distance.
    """
    model.eval()

    decoded_makespans: list[float] = []
    decoded_totals: list[float] = []
    expert_makespans: list[float] = []
    expert_totals: list[float] = []

    evaluated = 0

    for batch in loader:
        depots = batch["depots"].to(device, non_blocking=True)
        tasks = batch["tasks"].to(device, non_blocking=True)

        decoded_actions = model.decode(
            depots=depots,
            tasks=tasks,
        ).cpu().numpy()

        depots_np = batch["depots"].numpy()
        tasks_np = batch["tasks"].numpy()
        expert_makespan_np = batch["makespan"].numpy()
        expert_total_np = batch["total_distance"].numpy()

        for index in range(decoded_actions.shape[0]):
            instance = MTSPInstance(
                depots=depots_np[index].astype(np.float64),
                tasks=tasks_np[index].astype(np.float64),
            )

            decoded_routes = actions_to_routes(decoded_actions[index])
            decoded_metrics = evaluate_routes(
                instance,
                decoded_routes,
            )

            decoded_makespans.append(decoded_metrics.makespan)
            decoded_totals.append(decoded_metrics.total_distance)
            expert_makespans.append(float(expert_makespan_np[index]))
            expert_totals.append(float(expert_total_np[index]))

            evaluated += 1

            if max_instances is not None and evaluated >= max_instances:
                break

        if max_instances is not None and evaluated >= max_instances:
            break

    decoded_makespan_mean = float(np.mean(decoded_makespans))
    decoded_total_mean = float(np.mean(decoded_totals))
    expert_makespan_mean = float(np.mean(expert_makespans))
    expert_total_mean = float(np.mean(expert_totals))

    return {
        "decoded_makespan_m": decoded_makespan_mean,
        "decoded_total_distance_m": decoded_total_mean,
        "expert_makespan_m": expert_makespan_mean,
        "expert_total_distance_m": expert_total_mean,
        "makespan_gap_pct": 100.0
        * (decoded_makespan_mean - expert_makespan_mean)
        / expert_makespan_mean,
        "total_distance_gap_pct": 100.0
        * (decoded_total_mean - expert_total_mean)
        / expert_total_mean,
    }


def cpu_state_dict(model: torch.nn.Module) -> dict[str, Tensor]:
    """Save portable checkpoint weights, independent of GPU availability."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def save_checkpoint(
    path: Path,
    model: CentralizedAttentionMTSP,
    epoch: int,
    model_config: dict[str, int | float],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_config": model_config,
            "state_dict": cpu_state_dict(model),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train centralized 3-drone mTSP policy by imitation."
    )

    parser.add_argument(
        "--train",
        type=str,
        default="data/mtsp_3drones_15tasks_train.npz",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="data/mtsp_3drones_15tasks_val.npz",
    )

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decode-every", type=int, default=5)
    parser.add_argument("--decode-instances", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment-equivalent-labels", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    set_seed(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_dataset = MTSPDataset(
        args.train,
        augment_labels=args.augment_equivalent_labels,
    )

    val_dataset = MTSPDataset(
        args.val,
        augment_labels=False,
    )

    if train_dataset.num_tasks != val_dataset.num_tasks:
        raise ValueError("Train and validation task counts do not match.")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_config: dict[str, int | float] = {
        "embedding_dim": args.embedding_dim,
        "num_heads": args.heads,
        "num_encoder_layers": args.layers,
        "dropout": args.dropout,
        "coordinate_scale": 20.0,
    }

    model = CentralizedAttentionMTSP(
        **model_config,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = checkpoint_dir / "imitation_training_metrics.csv"

    best_val_loss = float("inf")
    best_decoded_makespan = float("inf")

    print(f"Device: {device}")
    print(f"Training instances: {len(train_dataset)}")
    print(f"Validation instances: {len(val_dataset)}")
    print(f"Tasks per instance: {train_dataset.num_tasks}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_action_accuracy",
            "val_loss",
            "val_action_accuracy",
            "decoded_makespan_m",
            "decoded_total_distance_m",
            "expert_makespan_m",
            "expert_total_distance_m",
            "makespan_gap_pct",
            "total_distance_gap_pct",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()

            weighted_loss_sum = 0.0
            weighted_accuracy_sum = 0.0
            total_instances = 0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)

                loss, accuracy = teacher_forced_loss(
                    model=model,
                    batch=batch,
                    device=device,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                batch_size = batch["tasks"].shape[0]

                weighted_loss_sum += loss.item() * batch_size
                weighted_accuracy_sum += accuracy * batch_size
                total_instances += batch_size

            scheduler.step()

            train_loss = weighted_loss_sum / total_instances
            train_accuracy = weighted_accuracy_sum / total_instances

            val_loss, val_accuracy = evaluate_teacher_forcing(
                model=model,
                loader=val_loader,
                device=device,
            )

            row: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_action_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_action_accuracy": val_accuracy,
                "decoded_makespan_m": float("nan"),
                "decoded_total_distance_m": float("nan"),
                "expert_makespan_m": float("nan"),
                "expert_total_distance_m": float("nan"),
                "makespan_gap_pct": float("nan"),
                "total_distance_gap_pct": float("nan"),
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    checkpoint_dir / "centralized_attention_best_val_loss.pt",
                    model=model,
                    epoch=epoch,
                    model_config=model_config,
                    metrics={
                        "val_loss": val_loss,
                        "val_action_accuracy": val_accuracy,
                    },
                )

            should_decode = (
                epoch % args.decode_every == 0
                or epoch == args.epochs
            )

            if should_decode:
                decoded_metrics = evaluate_greedy_decode(
                    model=model,
                    loader=val_loader,
                    device=device,
                    max_instances=args.decode_instances,
                )

                row.update(decoded_metrics)

                if (
                    decoded_metrics["decoded_makespan_m"]
                    < best_decoded_makespan
                ):
                    best_decoded_makespan = (
                        decoded_metrics["decoded_makespan_m"]
                    )

                    save_checkpoint(
                        checkpoint_dir / "centralized_attention_best_decode.pt",
                        model=model,
                        epoch=epoch,
                        model_config=model_config,
                        metrics=decoded_metrics,
                    )

            writer.writerow(row)
            file.flush()

            message = (
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss={train_loss:.4f}, "
                f"train acc={train_accuracy:.3f} | "
                f"val loss={val_loss:.4f}, "
                f"val acc={val_accuracy:.3f}"
            )

            if should_decode:
                message += (
                    f" | decoded makespan="
                    f"{row['decoded_makespan_m']:.2f} m "
                    f"({row['makespan_gap_pct']:+.1f}% vs OR-Tools)"
                )

            print(message)

    save_checkpoint(
        checkpoint_dir / "centralized_attention_last.pt",
        model=model,
        epoch=args.epochs,
        model_config=model_config,
        metrics={
            "best_val_loss": best_val_loss,
            "best_decoded_makespan_m": best_decoded_makespan,
        },
    )

    print("\nTraining complete.")
    print(
        "Best teacher-forced checkpoint: "
        "checkpoints/centralized_attention_best_val_loss.pt"
    )
    print(
        "Best decoded-route checkpoint: "
        "checkpoints/centralized_attention_best_decode.pt"
    )
    print(
        "Training metrics CSV: "
        "checkpoints/imitation_training_metrics.csv"
    )


if __name__ == "__main__":
    main()