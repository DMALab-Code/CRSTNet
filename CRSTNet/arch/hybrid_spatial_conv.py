from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficient_clustering import build_condensed_adjacency, sym_normalize_adj


class STGCNSpatialConv(nn.Module):
    """Hierarchical feature integration on the paper's two-layer H-Graph."""

    def __init__(self, in_channels: int = 64, out_channels: int = 64, gamma: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gamma = gamma

        self.lower_proj = nn.Linear(in_channels, out_channels)
        self.upper_proj = nn.Linear(in_channels, out_channels)
        self.query_proj = nn.Linear(out_channels, out_channels, bias=False)

    def _gcn(self, x: torch.Tensor, adj: np.ndarray, projection: nn.Linear) -> torch.Tensor:
        if x.numel() == 0:
            return x

        adj_norm = torch.tensor(sym_normalize_adj(adj), dtype=x.dtype, device=x.device)
        aggregated = torch.einsum("ij,btjf->btif", adj_norm, x)
        return torch.relu(projection(aggregated))

    def _attention_pool(
        self,
        member_features: torch.Tensor,
        prev_cluster_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if member_features.size(2) == 1:
            pooled = member_features[:, :, 0, :]
            return pooled, pooled

        if prev_cluster_state is None:
            query = member_features.mean(dim=2)
        else:
            query = prev_cluster_state

        query = F.normalize(self.query_proj(query), dim=-1)
        normalized_members = F.normalize(member_features, dim=-1)
        logits = torch.einsum("btnd,btd->btn", normalized_members, query)
        attn = torch.softmax(logits, dim=2).unsqueeze(-1)
        pooled = torch.sum(attn * member_features, dim=2)
        return pooled, pooled

    def _build_subgraph_adj(self, adj: np.ndarray, members: Sequence[int]) -> np.ndarray:
        members = [int(node) for node in members]
        if not members:
            return np.zeros((0, 0), dtype=np.float64)
        return np.asarray(adj[np.ix_(members, members)], dtype=np.float64)

    def forward(
        self,
        x: torch.Tensor,
        adj: np.ndarray,
        key_node_indices: Sequence[int],
        cluster_indices_list: Sequence[Sequence[int]],
        prev_cluster_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, num_nodes, _ = x.shape
        key_node_indices = [int(node) for node in key_node_indices if 0 <= int(node) < num_nodes]
        cluster_indices_list = [
            [int(node) for node in cluster if 0 <= int(node) < num_nodes]
            for cluster in cluster_indices_list
            if len(cluster) > 0
        ]

        full_output = torch.zeros(batch_size, time_steps, num_nodes, self.out_channels, device=x.device, dtype=x.dtype)
        condensed_inputs: List[torch.Tensor] = []
        cluster_member_features: List[Tuple[List[int], torch.Tensor]] = []
        updated_cluster_states: List[torch.Tensor] = []

        for key_node in key_node_indices:
            condensed_inputs.append(x[:, :, key_node, :])

        for cluster_idx, members in enumerate(cluster_indices_list):
            subgraph_adj = self._build_subgraph_adj(adj, members)
            subgraph_x = x[:, :, members, :]
            refined_members = self._gcn(subgraph_x, subgraph_adj, self.lower_proj)

            prev_state = None
            if prev_cluster_states is not None and cluster_idx < prev_cluster_states.shape[2]:
                prev_state = prev_cluster_states[:, :, cluster_idx, :]

            pooled, updated_state = self._attention_pool(refined_members, prev_state)
            condensed_inputs.append(pooled)
            cluster_member_features.append((members, refined_members))
            updated_cluster_states.append(updated_state)

        if condensed_inputs:
            condensed_x = torch.stack(condensed_inputs, dim=2)
            condensed_adj, _ = build_condensed_adjacency(key_node_indices, cluster_indices_list, adj, return_entities=False)
            condensed_out = self._gcn(condensed_x, condensed_adj, self.upper_proj)
        else:
            condensed_out = torch.zeros(batch_size, time_steps, 0, self.out_channels, device=x.device, dtype=x.dtype)

        condensed_idx = 0
        for key_node in key_node_indices:
            if condensed_idx < condensed_out.shape[2]:
                full_output[:, :, key_node, :] = condensed_out[:, :, condensed_idx, :]
            condensed_idx += 1

        for members, refined_members in cluster_member_features:
            if condensed_idx >= condensed_out.shape[2]:
                break
            cluster_feature = condensed_out[:, :, condensed_idx, :]
            fused = self.gamma * cluster_feature.unsqueeze(2) + (1.0 - self.gamma) * refined_members
            for member_offset, node_idx in enumerate(members):
                full_output[:, :, node_idx, :] = fused[:, :, member_offset, :]
            condensed_idx += 1

        if updated_cluster_states:
            cluster_states = torch.stack(updated_cluster_states, dim=2)
        else:
            cluster_states = torch.zeros(batch_size, time_steps, 0, self.out_channels, device=x.device, dtype=x.dtype)

        return full_output, cluster_states
