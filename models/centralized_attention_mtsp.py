from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as functional
from torch.distributions import Categorical


class CentralizedAttentionMTSP(nn.Module):
    """
    Centralized constructive policy for fixed-size 3-drone mTSP.

    Input:
        depots: [batch, 3, 2]
        tasks: [batch, num_tasks, 2]

    At each decoding step, the policy chooses one joint action:
        (drone_id, task_id)

    Output logits:
        [batch, num_tasks, 3, num_tasks]

    The policy sees all drones and all tasks at every step.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
        coordinate_scale: float = 20.0,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.coordinate_scale = coordinate_scale

        self.coordinate_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.depot_type_embedding = nn.Parameter(
            torch.zeros(1, 1, embedding_dim)
        )
        self.task_type_embedding = nn.Parameter(
            torch.zeros(1, 1, embedding_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.position_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Route length, number of assigned tasks, remaining-task fraction.
        self.scalar_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.drone_state_network = nn.Sequential(
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.drone_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        self.task_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        # Pairwise learned score:
        # drone state + task representation + outward distance + return distance.
        self.pair_scorer = nn.Sequential(
            nn.Linear(2 * embedding_dim + 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.depot_type_embedding, std=0.02)
        nn.init.normal_(self.task_type_embedding, std=0.02)

    def _encode_nodes(
        self,
        depots: Tensor,
        tasks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Encode all depot and task tokens together.

        This is centralized: every token can attend to every other token.
        """
        normalized_depots = depots / self.coordinate_scale
        normalized_tasks = tasks / self.coordinate_scale

        depot_tokens = (
            self.coordinate_encoder(normalized_depots)
            + self.depot_type_embedding
        )

        task_tokens = (
            self.coordinate_encoder(normalized_tasks)
            + self.task_type_embedding
        )

        all_tokens = torch.cat(
            [depot_tokens, task_tokens],
            dim=1,
        )

        encoded_tokens = self.encoder(all_tokens)

        num_drones = depots.shape[1]

        encoded_depots = encoded_tokens[:, :num_drones]
        encoded_tasks = encoded_tokens[:, num_drones:]

        return encoded_depots, encoded_tasks

    def _rollout(
        self,
        depots: Tensor,
        tasks: Tensor,
        teacher_actions: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """
        Teacher forcing if teacher_actions is provided.
        Greedy decoding otherwise.
        """
        batch_size, num_drones, _ = depots.shape
        num_tasks = tasks.shape[1]
        device = tasks.device

        encoded_depots, encoded_tasks = self._encode_nodes(
            depots,
            tasks,
        )

        current_positions = depots.clone()
        route_lengths = torch.zeros(
            batch_size,
            num_drones,
            device=device,
        )

        route_counts = torch.zeros(
            batch_size,
            num_drones,
            device=device,
        )

        visited = torch.zeros(
            batch_size,
            num_tasks,
            dtype=torch.bool,
            device=device,
        )

        all_logits: list[Tensor] = []
        chosen_actions: list[Tensor] = []

        batch_indices = torch.arange(
            batch_size,
            device=device,
        )

        for step in range(num_tasks):
            remaining_mask = (~visited).float()

            remaining_count = remaining_mask.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)

            remaining_context = (
                encoded_tasks * remaining_mask.unsqueeze(-1)
            ).sum(dim=1) / remaining_count

            normalized_positions = (
                current_positions / self.coordinate_scale
            )

            scalar_features = torch.stack(
                [
                    route_lengths / self.coordinate_scale,
                    route_counts / float(num_tasks),
                    remaining_count.expand(
                        -1,
                        num_drones,
                    ) / float(num_tasks),
                ],
                dim=-1,
            )

            drone_features = torch.cat(
                [
                    encoded_depots,
                    self.position_encoder(normalized_positions),
                    self.scalar_encoder(scalar_features),
                    remaining_context.unsqueeze(1).expand(
                        -1,
                        num_drones,
                        -1,
                    ),
                ],
                dim=-1,
            )

            drone_states = self.drone_state_network(
                drone_features
            )

            outward_distance = torch.linalg.vector_norm(
                current_positions.unsqueeze(2)
                - tasks.unsqueeze(1),
                dim=-1,
            ) / self.coordinate_scale

            return_distance = torch.linalg.vector_norm(
                tasks.unsqueeze(1)
                - depots.unsqueeze(2),
                dim=-1,
            ) / self.coordinate_scale

            dot_product_score = (
                self.drone_projection(drone_states).unsqueeze(2)
                * self.task_projection(encoded_tasks).unsqueeze(1)
            ).sum(dim=-1) / math.sqrt(self.embedding_dim)

            pair_features = torch.cat(
                [
                    drone_states.unsqueeze(2).expand(
                        -1,
                        -1,
                        num_tasks,
                        -1,
                    ),
                    encoded_tasks.unsqueeze(1).expand(
                        -1,
                        num_drones,
                        -1,
                        -1,
                    ),
                    outward_distance.unsqueeze(-1),
                    return_distance.unsqueeze(-1),
                ],
                dim=-1,
            )

            pair_score = self.pair_scorer(
                pair_features
            ).squeeze(-1)

            logits = dot_product_score + pair_score

            # A task may be assigned only once.
            logits = logits.masked_fill(
                visited.unsqueeze(1),
                torch.finfo(logits.dtype).min,
            )

            all_logits.append(logits)

            if teacher_actions is None:
                flat_choice = logits.flatten(start_dim=1).argmax(dim=1)

                chosen_drone = flat_choice // num_tasks
                chosen_task = flat_choice % num_tasks
            else:
                chosen_drone = teacher_actions[:, step, 0].long()
                chosen_task = teacher_actions[:, step, 1].long()

            chosen_actions.append(
                torch.stack(
                    [chosen_drone, chosen_task],
                    dim=-1,
                )
            )

            chosen_task_position = tasks[
                batch_indices,
                chosen_task,
            ]

            previous_position = current_positions[
                batch_indices,
                chosen_drone,
            ]

            travelled_distance = torch.linalg.vector_norm(
                chosen_task_position - previous_position,
                dim=-1,
            )

            drone_one_hot = functional.one_hot(
                chosen_drone,
                num_classes=num_drones,
            ).float()

            current_positions = (
                current_positions
                * (1.0 - drone_one_hot.unsqueeze(-1))
                + chosen_task_position.unsqueeze(1)
                * drone_one_hot.unsqueeze(-1)
            )

            route_lengths = (
                route_lengths
                + drone_one_hot * travelled_distance.unsqueeze(1)
            )

            route_counts = route_counts + drone_one_hot

            visited = visited.scatter(
                dim=1,
                index=chosen_task.unsqueeze(1),
                value=True,
            )

        return (
            torch.stack(all_logits, dim=1),
            torch.stack(chosen_actions, dim=1),
        )

    def forward(
        self,
        depots: Tensor,
        tasks: Tensor,
        teacher_actions: Tensor,
    ) -> Tensor:
        """
        Return logits under teacher forcing.

        Shape:
            [batch, num_tasks, num_drones, num_tasks]
        """
        logits, _ = self._rollout(
            depots,
            tasks,
            teacher_actions,
        )

        return logits

    @torch.no_grad()
    def decode(
        self,
        depots: Tensor,
        tasks: Tensor,
    ) -> Tensor:
        """
        Greedily decode one action sequence per instance.

        Shape:
            [batch, num_tasks, 2]

        Each action row is:
            [drone_id, task_id]
        """
        _, actions = self._rollout(
            depots,
            tasks,
            teacher_actions=None,
        )

        return actions

    def sample(
        self,
        depots: Tensor,
        tasks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Sample a complete action sequence for policy-gradient training.

        Returns:
            actions:    [batch, num_tasks, 2]
            log_probs:  [batch, num_tasks]
            entropies:  [batch, num_tasks]

        Each action is [drone_id, task_id].
        """
        batch_size, num_drones, _ = depots.shape
        num_tasks = tasks.shape[1]
        device = tasks.device

        encoded_depots, encoded_tasks = self._encode_nodes(
            depots,
            tasks,
        )

        current_positions = depots.clone()

        route_lengths = torch.zeros(
            batch_size,
            num_drones,
            device=device,
        )

        route_counts = torch.zeros(
            batch_size,
            num_drones,
            device=device,
        )

        visited = torch.zeros(
            batch_size,
            num_tasks,
            dtype=torch.bool,
            device=device,
        )

        batch_indices = torch.arange(
            batch_size,
            device=device,
        )

        chosen_actions: list[Tensor] = []
        action_log_probs: list[Tensor] = []
        action_entropies: list[Tensor] = []

        for _ in range(num_tasks):
            remaining_mask = (~visited).float()

            remaining_count = remaining_mask.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)

            remaining_context = (
                encoded_tasks * remaining_mask.unsqueeze(-1)
            ).sum(dim=1) / remaining_count

            normalized_positions = (
                current_positions / self.coordinate_scale
            )

            scalar_features = torch.stack(
                [
                    route_lengths / self.coordinate_scale,
                    route_counts / float(num_tasks),
                    remaining_count.expand(
                        -1,
                        num_drones,
                    ) / float(num_tasks),
                ],
                dim=-1,
            )

            drone_features = torch.cat(
                [
                    encoded_depots,
                    self.position_encoder(normalized_positions),
                    self.scalar_encoder(scalar_features),
                    remaining_context.unsqueeze(1).expand(
                        -1,
                        num_drones,
                        -1,
                    ),
                ],
                dim=-1,
            )

            drone_states = self.drone_state_network(
                drone_features
            )

            outward_distance = torch.linalg.vector_norm(
                current_positions.unsqueeze(2)
                - tasks.unsqueeze(1),
                dim=-1,
            ) / self.coordinate_scale

            return_distance = torch.linalg.vector_norm(
                tasks.unsqueeze(1)
                - depots.unsqueeze(2),
                dim=-1,
            ) / self.coordinate_scale

            dot_product_score = (
                self.drone_projection(drone_states).unsqueeze(2)
                * self.task_projection(encoded_tasks).unsqueeze(1)
            ).sum(dim=-1) / math.sqrt(self.embedding_dim)

            pair_features = torch.cat(
                [
                    drone_states.unsqueeze(2).expand(
                        -1,
                        -1,
                        num_tasks,
                        -1,
                    ),
                    encoded_tasks.unsqueeze(1).expand(
                        -1,
                        num_drones,
                        -1,
                        -1,
                    ),
                    outward_distance.unsqueeze(-1),
                    return_distance.unsqueeze(-1),
                ],
                dim=-1,
            )

            pair_score = self.pair_scorer(
                pair_features
            ).squeeze(-1)

            logits = dot_product_score + pair_score

            logits = logits.masked_fill(
                visited.unsqueeze(1),
                torch.finfo(logits.dtype).min,
            )

            flat_logits = logits.flatten(start_dim=1)

            distribution = Categorical(logits=flat_logits)
            flat_choice = distribution.sample()

            chosen_drone = flat_choice // num_tasks
            chosen_task = flat_choice % num_tasks

            chosen_actions.append(
                torch.stack(
                    [chosen_drone, chosen_task],
                    dim=-1,
                )
            )

            action_log_probs.append(
                distribution.log_prob(flat_choice)
            )

            action_entropies.append(
                distribution.entropy()
            )

            chosen_task_position = tasks[
                batch_indices,
                chosen_task,
            ]

            previous_position = current_positions[
                batch_indices,
                chosen_drone,
            ]

            travelled_distance = torch.linalg.vector_norm(
                chosen_task_position - previous_position,
                dim=-1,
            )

            drone_one_hot = functional.one_hot(
                chosen_drone,
                num_classes=num_drones,
            ).float()

            current_positions = (
                current_positions
                * (1.0 - drone_one_hot.unsqueeze(-1))
                + chosen_task_position.unsqueeze(1)
                * drone_one_hot.unsqueeze(-1)
            )

            route_lengths = (
                route_lengths
                + drone_one_hot * travelled_distance.unsqueeze(1)
            )

            route_counts = route_counts + drone_one_hot

            visited = visited.scatter(
                dim=1,
                index=chosen_task.unsqueeze(1),
                value=True,
            )

        return (
            torch.stack(chosen_actions, dim=1),
            torch.stack(action_log_probs, dim=1),
            torch.stack(action_entropies, dim=1),
        )


def _smoke_test() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CentralizedAttentionMTSP().to(device)

    depots = torch.tensor(
        [
            [
                [2.0, 2.0],
                [18.0, 2.0],
                [10.0, 17.0],
            ],
            [
                [2.0, 2.0],
                [18.0, 2.0],
                [10.0, 17.0],
            ],
        ],
        device=device,
    )

    tasks = torch.rand(
        2,
        15,
        2,
        device=device,
    ) * 18.0 + 1.0

    teacher_actions = torch.stack(
        [
            torch.stack(
                [
                    torch.arange(15, device=device) % 3,
                    torch.randperm(15, device=device),
                ],
                dim=-1,
            )
            for _ in range(2)
        ],
        dim=0,
    )

    logits = model(
        depots,
        tasks,
        teacher_actions,
    )

    decoded_actions = model.decode(
        depots,
        tasks,
    )

    print("Device:", device)
    print("Logits shape:", tuple(logits.shape))
    print("Decoded actions shape:", tuple(decoded_actions.shape))
    print("First decoded route actions:")
    print(decoded_actions[0].cpu())


if __name__ == "__main__":
    _smoke_test()