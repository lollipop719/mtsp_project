from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from env.dan_async_mtsp_env import (
    DANObservation,
    DANPolicyOutput,
)


class AttentionBlock(nn.Module):
    """
    Transformer-style attention block.

    Can be used as:
        self-attention: query = key_value
        cross-attention: query != key_value
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_attention = nn.LayerNorm(embedding_dim)
        self.norm_feedforward = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        attention_output, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        query = self.norm_attention(
            query + self.dropout(attention_output)
        )

        feedforward_output = self.feedforward(query)

        output = self.norm_feedforward(
            query + self.dropout(feedforward_output)
        )

        return output


class DANMTSPPolicy(nn.Module):
    """
    Decentralized Attention Network policy for asynchronous mTSP.

    Inputs:
        cities_relative:
            [batch, num_tasks, 2]

        agents_relative:
            [batch, num_agents, 3]
            dx, dy, remaining_travel_time

        action_mask:
            [batch, num_tasks]
            True for valid/unvisited tasks.

    Output:
        logits over tasks:
            [batch, num_tasks]
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.0,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.num_encoder_layers = int(num_encoder_layers)
        self.dropout = float(dropout)
        self.tanh_clipping = float(tanh_clipping)

        self.city_projection = nn.Linear(2, embedding_dim)
        self.agent_projection = nn.Linear(3, embedding_dim)

        self.city_encoder = nn.ModuleList(
            [
                AttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.agent_encoder = nn.ModuleList(
            [
                AttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.city_agent_encoder = nn.ModuleList(
            [
                AttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Decoder attention layers.
        self.state_agent_attention = AttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.glimpse_attention = AttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Final pointer-style attention.
        self.final_query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        self.final_key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

    def encode(
        self,
        *,
        cities_relative: Tensor,
        agents_relative: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        city_embedding = self.city_projection(cities_relative)
        agent_embedding = self.agent_projection(agents_relative)

        for layer in self.city_encoder:
            city_embedding = layer(
                query=city_embedding,
                key_value=city_embedding,
            )

        for layer in self.agent_encoder:
            agent_embedding = layer(
                query=agent_embedding,
                key_value=agent_embedding,
            )

        city_agent_embedding = city_embedding

        for layer in self.city_agent_encoder:
            city_agent_embedding = layer(
                query=city_agent_embedding,
                key_value=agent_embedding,
            )

        return city_embedding, agent_embedding, city_agent_embedding

    def forward(
        self,
        *,
        cities_relative: Tensor,
        agents_relative: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        if cities_relative.ndim != 3:
            raise ValueError(
                f"cities_relative must have shape [B, N, 2], "
                f"got {cities_relative.shape}"
            )

        if agents_relative.ndim != 3:
            raise ValueError(
                f"agents_relative must have shape [B, M, 3], "
                f"got {agents_relative.shape}"
            )

        if action_mask.ndim != 2:
            raise ValueError(
                f"action_mask must have shape [B, N], "
                f"got {action_mask.shape}"
            )

        if not torch.any(action_mask, dim=1).all():
            raise ValueError(
                "Every batch element must have at least one valid action."
            )

        city_embedding, agent_embedding, city_agent_embedding = self.encode(
            cities_relative=cities_relative,
            agents_relative=agents_relative,
        )

        # Current state embedding: mean city embedding, similar to graph embedding.
        state_embedding = city_embedding.mean(
            dim=1,
            keepdim=True,
        )

        # Decoder layer 1: relate deciding agent's state to all agents.
        current_state_embedding = self.state_agent_attention(
            query=state_embedding,
            key_value=agent_embedding,
        )

        # Decoder layer 2: glimpse over valid candidate cities.
        invalid_mask = ~action_mask.bool()

        final_candidate_embedding = self.glimpse_attention(
            query=current_state_embedding,
            key_value=city_agent_embedding,
            key_padding_mask=invalid_mask,
        )

        query = self.final_query_projection(
            final_candidate_embedding
        )

        keys = self.final_key_projection(
            city_agent_embedding
        )

        logits = torch.matmul(
            query,
            keys.transpose(1, 2),
        ).squeeze(1)

        logits = logits / math.sqrt(self.embedding_dim)

        if self.tanh_clipping > 0.0:
            logits = self.tanh_clipping * torch.tanh(logits)

        logits = logits.masked_fill(
            invalid_mask,
            -1.0e9,
        )

        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        *,
        cities_relative: Tensor,
        agents_relative: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        logits = self.forward(
            cities_relative=cities_relative,
            agents_relative=agents_relative,
            action_mask=action_mask,
        )

        return torch.argmax(logits, dim=-1)

    def sample_action(
        self,
        *,
        cities_relative: Tensor,
        agents_relative: Tensor,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.forward(
            cities_relative=cities_relative,
            agents_relative=agents_relative,
            action_mask=action_mask,
        )

        distribution = Categorical(logits=logits)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return action, log_prob, entropy


def observation_to_tensors(
    observation: DANObservation,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    cities_relative = torch.from_numpy(
        observation.cities_relative.astype(np.float32)
    ).unsqueeze(0).to(device)

    agents_relative = torch.from_numpy(
        observation.agents_relative.astype(np.float32)
    ).unsqueeze(0).to(device)

    action_mask = torch.from_numpy(
        observation.action_mask.astype(bool)
    ).unsqueeze(0).to(device)

    return cities_relative, agents_relative, action_mask


@dataclass
class DANPolicyRunner:
    """
    Small wrapper that converts DANObservation into DANPolicyOutput.

    This lets the PyTorch model plug directly into env.rollout().
    """

    model: DANMTSPPolicy
    device: torch.device
    mode: Literal["greedy", "sample"] = "sample"

    def __call__(
        self,
        observation: DANObservation,
    ) -> DANPolicyOutput:
        cities_relative, agents_relative, action_mask = observation_to_tensors(
            observation,
            self.device,
        )

        if self.mode == "greedy":
            with torch.no_grad():
                action = self.model.greedy_decode(
                    cities_relative=cities_relative,
                    agents_relative=agents_relative,
                    action_mask=action_mask,
                )

            return DANPolicyOutput(
                action=int(action.item()),
            )

        if self.mode == "sample":
            action, log_prob, entropy = self.model.sample_action(
                cities_relative=cities_relative,
                agents_relative=agents_relative,
                action_mask=action_mask,
            )

            return DANPolicyOutput(
                action=int(action.item()),
                log_prob=log_prob.squeeze(0),
                entropy=entropy.squeeze(0),
            )

        raise ValueError(f"Unknown mode: {self.mode}")
