from pathlib import Path
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .adaptive_temporal_conv import MultiScaleTemporalConv
from .efficient_clustering import (
    OptimizedStructureManager,
    compute_importance_scores,
    detect_distribution_change,
    efficient_dtw_distance,
    select_dynamic_key_nodes,
)
from .hybrid_spatial_conv import STGCNSpatialConv
from .key_node_selector import KeyNodeSelector, QuotaSwapKeySelector


class CRSTNet(nn.Module):
    """
    Paper-aligned CRSTNet implementation.

    The model keeps the original repository interface, while the internal
    components are aligned with the paper:
    - key-node scoring uses normalized entropy x fluctuation,
    - remaining vertices are clustered with topology-constrained Ward updates,
    - temporal kernels are selected by FFT and variance-adaptive scaling,
    - H-Graph maintenance follows SMP/SANI and Case 1/2/3 updates.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        key_node_ratio: float = 0.2,
        theta: Optional[float] = None,
        distance_lambda: float = 0.5,
        gamma: float = 0.5,
        adj_mx: Optional[np.ndarray] = None,
        update_frequency: int = 5,
        maintenance_interval: Optional[int] = None,
        delta_s: Optional[float] = None,
        delta_e: Optional[float] = None,
        delta_d: Optional[float] = None,
        eta: Optional[float] = None,
        k_min: int = 2,
        k_max: Optional[int] = None,
        threshold_profile: Optional[str] = None,
        max_key_nodes: Optional[int] = None,
        **legacy_kwargs: Any,
    ):
        super().__init__()
        del legacy_kwargs

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.theta = key_node_ratio if theta is None else theta
        self.distance_lambda = distance_lambda
        self.gamma = gamma
        self.adj_mx = np.asarray(adj_mx, dtype=np.float64) if adj_mx is not None else np.eye(num_nodes, dtype=np.float64)
        self.maintenance_interval = maintenance_interval or update_frequency
        self.threshold_profile = threshold_profile

        self.temporal_convs = nn.ModuleList(
            [
                MultiScaleTemporalConv(
                    input_dim if layer_idx == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    k_min=k_min,
                    k_max=k_max,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.spatial_layers = nn.ModuleList(
            [STGCNSpatialConv(hidden_dim, hidden_dim, gamma=gamma) for _ in range(num_layers)]
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.key_selector = KeyNodeSelector(theta=self.theta, max_key_nodes=max_key_nodes)
        self.structure_manager = OptimizedStructureManager(
            graph_adj=self.adj_mx,
            theta=self.theta,
            lambda_value=distance_lambda,
            maintenance_interval=self.maintenance_interval,
            delta_s=delta_s,
            delta_e=delta_e,
            delta_d=delta_d,
            eta=eta,
            max_key_nodes=max_key_nodes,
            threshold_profile=threshold_profile,
        )

        self.key_nodes: List[int] = []
        self.non_key_nodes: List[int] = []
        self.cluster_indices_list: List[List[int]] = []
        self.prev_cluster_states: List[Optional[torch.Tensor]] = [None for _ in range(num_layers)]
        self.step_count = 0
        self.vldb_diagnostics: Dict[str, Any] = {}

        if threshold_profile:
            self._load_profile_metadata(threshold_profile)

    def _load_profile_metadata(self, profile_path: str) -> None:
        path = Path(profile_path)
        if not path.exists():
            return
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.vldb_diagnostics["threshold_profile"] = profile

    def _refresh_structure(self, x_np: np.ndarray) -> None:
        structure_info = self.structure_manager.update_structure(x_np)
        self.key_nodes = [int(node) for node in structure_info["key_nodes"]]
        self.non_key_nodes = [int(node) for node in structure_info["non_key_nodes"]]
        self.cluster_indices_list = [
            [int(node) for node in cluster]
            for cluster in structure_info["cluster_indices_list"]
        ]
        self.vldb_diagnostics = {
            "thresholds": structure_info.get("thresholds", {}),
            "theta": self.theta,
            "distance_lambda": self.distance_lambda,
            "maintenance_interval": self.maintenance_interval,
            "kernel_sizes": [conv.last_kernel_sizes for conv in self.temporal_convs],
        }

    def forward(self, history_data=None, future_data=None, *args, **kwargs):
        del future_data, args
        x = history_data if history_data is not None else kwargs.get("x", None)
        if x is None:
            raise ValueError("CRSTNet requires `history_data` as input.")
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor, adj_matrix: Optional[np.ndarray] = None) -> torch.Tensor:
        if adj_matrix is None:
            adj_matrix = self.adj_mx

        x_np = x[0].detach().cpu().numpy() if isinstance(x, torch.Tensor) else x[0]
        self.step_count += 1
        self._refresh_structure(x_np)

        h = x
        updated_cluster_states: List[Optional[torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            h = self.temporal_convs[layer_idx](h, self.key_nodes, self.cluster_indices_list)
            h = self.dropout_layer(h)

            prev_state = self.prev_cluster_states[layer_idx] if layer_idx < len(self.prev_cluster_states) else None
            h, cluster_state = self.spatial_layers[layer_idx](
                h,
                adj_matrix,
                self.key_nodes,
                self.cluster_indices_list,
                prev_cluster_states=prev_state,
            )
            h = self.dropout_layer(h)
            updated_cluster_states.append(cluster_state.detach())

        self.prev_cluster_states = updated_cluster_states
        return self.output_proj(h)

    def get_vldb_diagnostics(self) -> Dict[str, Any]:
        return self.vldb_diagnostics

    def get_key_node_info(self) -> Dict[str, Any]:
        return {
            "key_nodes": self.key_nodes,
            "non_key_nodes": self.non_key_nodes,
            "cluster_indices": self.cluster_indices_list,
            "vldb_diagnostics": self.vldb_diagnostics,
        }


__all__ = [
    "CRSTNet",
    "KeyNodeSelector",
    "QuotaSwapKeySelector",
    "MultiScaleTemporalConv",
    "STGCNSpatialConv",
    "OptimizedStructureManager",
    "compute_importance_scores",
    "select_dynamic_key_nodes",
    "efficient_dtw_distance",
    "detect_distribution_change",
]
