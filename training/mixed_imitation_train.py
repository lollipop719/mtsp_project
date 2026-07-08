from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from env.mtsp_2d_env import MTSPInstance, evaluate_routes
from models.centralized_attention_mtsp import CentralizedAttentionMTSP
from training.imitation_train import (
    augment_equivalent_teacher_actions,
    cpu_state_dict,
    joint_action_targets,
    set_seed,
)


@dataclass(frozen=True)
class ShapeSpec:
    num_agents: int
    num_tasks: int
    split: str
    path: Path


class VariableMTSPDataset(Dataset):
    def __init__(
        self,
        path: Path,
        augment_labels: bool = False,
    ) -> None:
        self.path = path
        self.augment_labels = augment_labels

        with np.load(path, allow_pickle=False) as data:
            self.depots = data["depots"].astype(np.float32)
            self.tasks = data["tasks"].astype(np.float32)
            self.teacher_actions = data["teacher_actions"].astype(np.int64)
            self.makespans = data["makespans"].astype(np.float32)
            self.total_distances = data["total_distances"].astype(np.float32)

        if self.depots.ndim != 3 or self.depots.shape[-1] != 2:
            raise ValueError(f"Bad depots shape in {path}: {self.depots.shape}")

        if self.tasks.ndim != 3 or self.tasks.shape[-1] != 2:
            raise ValueError(f"Bad tasks shape in {path}: {self.tasks.shape}")

        if self.teacher_actions.shape[:2] != self.tasks.shape[:2]:
            raise ValueError(
                f"Bad teacher action shape in {path}: "
                f"{self.teacher_actions.shape}"
            )

        if self.teacher_actions.shape[-1] != 2:
            raise ValueError("teacher_actions must have final dimension 2.")

    @property
    def num_agents(self) -> int:
        return int(self.depots.shape[1])

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
                num_drones=self.num_agents,
            )

        return {
            "depots": torch.from_numpy(self.depots[index]),
            "tasks": torch.from_numpy(self.tasks[index]),
            "teacher_actions": torch.from_numpy(actions),
            "makespan": torch.tensor(self.makespans[index]),
            "total_distance": torch.tensor(self.total_distances[index]),
        }


def find_dataset_files(
    data_dir: Path,
    split: str,
) -> list[ShapeSpec]:
    pattern = f"mtsp_*agents_*tasks_{split}.npz"
    specs: list[ShapeSpec] = []

    for path in sorted(data_dir.glob(pattern)):
        name = path.stem
        # mtsp_5agents_25tasks_train
        parts = name.split("_")

        num_agents = int(parts[1].replace("agents", ""))
        num_tasks = int(parts[2].replace("tasks", ""))

        specs.append(
            ShapeSpec(
                num_agents=num_agents,
                num_tasks=num_tasks,
                split=split,
                path=path,
            )
        )

    if not specs:
        raise FileNotFoundError(
            f"No files matching {pattern} in {data_dir}"
        )

    return specs


def make_loaders(
    specs: list[ShapeSpec],
    batch_size: int,
    augment_labels: bool,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> dict[tuple[int, int], DataLoader]:
    loaders: dict[tuple[int, int], DataLoader] = {}

    for spec in specs:
        dataset = VariableMTSPDataset(
            spec.path,
            augment_labels=augment_labels,
        )

        loaders[(spec.num_agents, spec.num_tasks)] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return loaders


def teacher_forced_loss(
    model: CentralizedAttentionMTSP,
    batch: dict[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, float]:
    depots = batch["depots"].to(device, non_blocking=True)
    tasks = batch["tasks"].to(device, non_blocking=True)
    actions = batch["teacher_actions"].to(device, non_blocking=True)

    num_tasks = int(tasks.shape[1])
    num_agents = int(depots.shape[1])

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
        num_agents * num_tasks,
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
    num_agents: int,
) -> list[list[int]]:
    routes: list[list[int]] = [[] for _ in range(num_agents)]

    for agent_id, task_id in actions:
        routes[int(agent_id)].append(int(task_id))

    return routes


@torch.no_grad()
def evaluate_teacher_forcing_mixed(
    model: CentralizedAttentionMTSP,
    loaders: dict[tuple[int, int], DataLoader],
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss_sum = 0.0
    total_accuracy_sum = 0.0
    total_instances = 0

    per_shape: dict[str, float] = {}

    for (num_agents, num_tasks), loader in sorted(loaders.items()):
        shape_loss_sum = 0.0
        shape_accuracy_sum = 0.0
        shape_instances = 0

        for batch in loader:
            loss, accuracy = teacher_forced_loss(
                model=model,
                batch=batch,
                device=device,
            )

            batch_size = batch["tasks"].shape[0]

            shape_loss_sum += loss.item() * batch_size
            shape_accuracy_sum += accuracy * batch_size
            shape_instances += batch_size

        shape_loss = shape_loss_sum / shape_instances
        shape_accuracy = shape_accuracy_sum / shape_instances

        key = f"{num_agents}a_{num_tasks}t"
        per_shape[f"{key}_loss"] = shape_loss
        per_shape[f"{key}_accuracy"] = shape_accuracy

        total_loss_sum += shape_loss_sum
        total_accuracy_sum += shape_accuracy_sum
        total_instances += shape_instances

    per_shape["overall_loss"] = total_loss_sum / total_instances
    per_shape["overall_accuracy"] = total_accuracy_sum / total_instances

    return per_shape


@torch.no_grad()
def evaluate_decoded_mixed(
    model: CentralizedAttentionMTSP,
    loaders: dict[tuple[int, int], DataLoader],
    device: torch.device,
    max_instances_per_shape: int,
) -> dict[str, float]:
    model.eval()

    result: dict[str, float] = {}

    all_decoded_makespans: list[float] = []
    all_expert_makespans: list[float] = []
    all_gaps: list[float] = []

    for (num_agents, num_tasks), loader in sorted(loaders.items()):
        decoded_makespans: list[float] = []
        expert_makespans: list[float] = []
        gaps: list[float] = []

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
            expert_np = batch["makespan"].numpy()

            for index in range(decoded_actions.shape[0]):
                instance = MTSPInstance(
                    depots=depots_np[index].astype(np.float64),
                    tasks=tasks_np[index].astype(np.float64),
                )

                routes = actions_to_routes(
                    decoded_actions[index],
                    num_agents=num_agents,
                )

                metrics = evaluate_routes(
                    instance,
                    routes,
                )

                expert_makespan = float(expert_np[index])
                gap_pct = (
                    100.0
                    * (metrics.makespan - expert_makespan)
                    / expert_makespan
                )

                decoded_makespans.append(metrics.makespan)
                expert_makespans.append(expert_makespan)
                gaps.append(gap_pct)

                all_decoded_makespans.append(metrics.makespan)
                all_expert_makespans.append(expert_makespan)
                all_gaps.append(gap_pct)

                evaluated += 1

                if evaluated >= max_instances_per_shape:
                    break

            if evaluated >= max_instances_per_shape:
                break

        key = f"{num_agents}a_{num_tasks}t"

        result[f"{key}_decoded_makespan_m"] = float(
            np.mean(decoded_makespans)
        )

        result[f"{key}_expert_makespan_m"] = float(
            np.mean(expert_makespans)
        )

        result[f"{key}_gap_pct"] = float(np.mean(gaps))

    result["overall_decoded_makespan_m"] = float(
        np.mean(all_decoded_makespans)
    )

    result["overall_expert_makespan_m"] = float(
        np.mean(all_expert_makespans)
    )

    result["overall_gap_pct"] = float(np.mean(all_gaps))

    return result


def load_or_create_model(
    init_checkpoint: Path | None,
    device: torch.device,
    embedding_dim: int,
    heads: int,
    layers: int,
    dropout: float,
) -> tuple[CentralizedAttentionMTSP, dict[str, int | float]]:
    if init_checkpoint is not None:
        checkpoint = torch.load(
            init_checkpoint,
            map_location=device,
            weights_only=False,
        )

        model_config = checkpoint["model_config"]

        model = CentralizedAttentionMTSP(
            **model_config,
        ).to(device)

        model.load_state_dict(checkpoint["state_dict"])

        return model, model_config

    model_config: dict[str, int | float] = {
        "embedding_dim": embedding_dim,
        "num_heads": heads,
        "num_encoder_layers": layers,
        "dropout": dropout,
        "coordinate_scale": 20.0,
    }

    model = CentralizedAttentionMTSP(
        **model_config,
    ).to(device)

    return model, model_config


def save_checkpoint(
    path: Path,
    model: CentralizedAttentionMTSP,
    model_config: dict[str, int | float],
    epoch: int,
    metrics: dict[str, float],
    init_checkpoint: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_config": model_config,
            "state_dict": cpu_state_dict(model),
            "metrics": metrics,
            "init_checkpoint": (
                str(init_checkpoint) if init_checkpoint is not None else None
            ),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed-shape imitation training for variable-agent mTSP."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/variable_mtsp"),
    )

    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/final_3drone_15task/"
            "centralized_attention_mtsp_15task.pt"
        ),
    )

    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore --init-checkpoint and train a new model.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="mixed_imitation_variable_mtsp",
    )

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--decode-every", type=int, default=5)
    parser.add_argument("--decode-instances-per-shape", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    set_seed(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pin_memory = device.type == "cuda"

    train_specs = find_dataset_files(args.data_dir, "train")
    val_specs = find_dataset_files(args.data_dir, "val")

    train_loaders = make_loaders(
        specs=train_specs,
        batch_size=args.batch_size,
        augment_labels=True,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loaders = make_loaders(
        specs=val_specs,
        batch_size=args.batch_size,
        augment_labels=False,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    init_checkpoint = None if args.from_scratch else args.init_checkpoint

    if init_checkpoint is not None and not init_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {init_checkpoint}")

    model, model_config = load_or_create_model(
        init_checkpoint=init_checkpoint,
        device=device,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    checkpoint_dir = Path("checkpoints") / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = checkpoint_dir / "mixed_imitation_metrics.csv"

    shape_keys = sorted(train_loaders.keys())

    best_overall_gap = float("inf")
    best_val_loss = float("inf")

    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Run name: {args.run_name}")
    print(f"Init checkpoint: {init_checkpoint}")
    print("Training shapes:")
    for key in shape_keys:
        dataset_size = len(train_loaders[key].dataset)
        print(f"  {key[0]} agents, {key[1]} tasks: {dataset_size} train instances")

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_action_accuracy",
            "val_loss",
            "val_action_accuracy",
            "overall_gap_pct",
            "overall_decoded_makespan_m",
            "overall_expert_makespan_m",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()

            total_loss_sum = 0.0
            total_accuracy_sum = 0.0
            total_instances = 0

            # Each epoch sees all shape-specific loaders once.
            epoch_shape_keys = shape_keys.copy()
            random.shuffle(epoch_shape_keys)

            for shape_key in epoch_shape_keys:
                loader = train_loaders[shape_key]

                for batch in loader:
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
                    total_loss_sum += loss.item() * batch_size
                    total_accuracy_sum += accuracy * batch_size
                    total_instances += batch_size

            scheduler.step()

            train_loss = total_loss_sum / total_instances
            train_accuracy = total_accuracy_sum / total_instances

            val_metrics = evaluate_teacher_forcing_mixed(
                model=model,
                loaders=val_loaders,
                device=device,
            )

            val_loss = val_metrics["overall_loss"]
            val_accuracy = val_metrics["overall_accuracy"]

            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_action_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_action_accuracy": val_accuracy,
                "overall_gap_pct": float("nan"),
                "overall_decoded_makespan_m": float("nan"),
                "overall_expert_makespan_m": float("nan"),
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    checkpoint_dir / "mixed_imitation_best_val_loss.pt",
                    model=model,
                    model_config=model_config,
                    epoch=epoch,
                    metrics={
                        "val_loss": val_loss,
                        "val_action_accuracy": val_accuracy,
                    },
                    init_checkpoint=init_checkpoint,
                )

            should_decode = (
                epoch % args.decode_every == 0
                or epoch == args.epochs
            )

            if should_decode:
                decode_metrics = evaluate_decoded_mixed(
                    model=model,
                    loaders=val_loaders,
                    device=device,
                    max_instances_per_shape=args.decode_instances_per_shape,
                )

                row["overall_gap_pct"] = decode_metrics["overall_gap_pct"]
                row["overall_decoded_makespan_m"] = decode_metrics[
                    "overall_decoded_makespan_m"
                ]
                row["overall_expert_makespan_m"] = decode_metrics[
                    "overall_expert_makespan_m"
                ]

                if decode_metrics["overall_gap_pct"] < best_overall_gap:
                    best_overall_gap = decode_metrics["overall_gap_pct"]

                    save_checkpoint(
                        checkpoint_dir / "mixed_imitation_best_decode.pt",
                        model=model,
                        model_config=model_config,
                        epoch=epoch,
                        metrics=decode_metrics,
                        init_checkpoint=init_checkpoint,
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
                    f" | decoded gap="
                    f"{row['overall_gap_pct']:+.2f}%"
                )

            print(message)

    save_checkpoint(
        checkpoint_dir / "mixed_imitation_last.pt",
        model=model,
        model_config=model_config,
        epoch=args.epochs,
        metrics={
            "best_val_loss": best_val_loss,
            "best_overall_gap_pct": best_overall_gap,
        },
        init_checkpoint=init_checkpoint,
    )

    print()
    print("Mixed imitation training complete.")
    print(
        "Best decoded checkpoint: "
        f"{checkpoint_dir / 'mixed_imitation_best_decode.pt'}"
    )
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
