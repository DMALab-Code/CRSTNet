from .adaptive_temporal_conv import MultiScaleTemporalConv
from .hybrid_spatial_conv import STGCNSpatialConv
from .key_node_selector import KeyNodeSelector, QuotaSwapKeySelector
from .hierarchical_cluster import HierarchicalClusterTree, HTree, promote_to_key, demote_to_nonkey, reassign_nonkey, GraphOps
from .efficient_clustering import (select_dynamic_key_nodes, efficient_dtw_distance, detect_distribution_change,
                                 OptimizedStructureManager, HCIndex, TimeSeriesSummary,
                                 LocalOperationManager, SuperNodeManager)
from .select_top_k import KLLSketch, StreamingScore, select_key_nodes_bqh, select_key_nodes_base
from .non_k_clustering import TimeEnvelope, TwoLevelInvertedIndex, CF, SuperNode, lb_keogh, ward_cost_cf, ward_cost_merge, estimate_merge_gain
from .main import BQHState, BQHBudgeter, BQHEstimators, BQHIndexing, step_update, update_and_get_clusters, prepare_state, get_bqh_config

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CRSTNet(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, hidden_dim=64,
                 num_layers=2, dropout=0.1, key_node_ratio=0.3, adj_mx=None,
                 use_efficient_clustering=True, n_clusters=6, downsample_ratio=4, n_jobs=4,

                 use_advanced_selector=False, selection_strategy="fdr_diversity",
                 fdr_alpha=0.05, eps_stop=0.02, diversity_rho=None, stability_gamma=0.1,
                 max_key_nodes=100, budget=20, cache_size=1000, update_frequency=5, use_approximation=True,

                 use_optimized_structure=True, quota_ratio=0.1, paa_segments=16,
                 merge_cooldown=20, split_cooldown=20):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.key_node_ratio = key_node_ratio
        self.adj_mx = adj_mx

        self.use_efficient_clustering = use_efficient_clustering
        self.n_clusters = n_clusters
        self.downsample_ratio = downsample_ratio
        self.n_jobs = n_jobs

        self.optimized_mode = True
        self.use_optimized_structure = use_optimized_structure
        self.update_frequency = max(1, int(update_frequency))
        self.quota_ratio = quota_ratio
        self.paa_segments = paa_segments
        self.merge_cooldown = merge_cooldown
        self.split_cooldown = split_cooldown

        self.cluster_cache = {}
        self.cache_valid_steps = 5
        self.super_nodes = None
        self.super_node_features = None

        self.key_selector = KeyNodeSelector()

        self.cluster_tree = HierarchicalClusterTree()

        if self.use_optimized_structure and adj_mx is not None:
            self.structure_manager = OptimizedStructureManager(
                graph_adj=adj_mx,
                quota_ratio=quota_ratio,
                paa_segments=paa_segments,
                merge_cooldown=merge_cooldown,
                split_cooldown=split_cooldown
            )
        else:
            self.structure_manager = None

        self.temporal_convs = nn.ModuleList([
            MultiScaleTemporalConv(input_dim if i == 0 else hidden_dim, out_channels=hidden_dim, main_period=12)
            for i in range(num_layers)
        ])

        self.lower_spatial_convs = nn.ModuleList([
            STGCNSpatialConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.upper_spatial_convs = nn.ModuleList([
            STGCNSpatialConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.feature_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.dropout_layer = nn.Dropout(dropout)

        self.prev_state = None
        self.key_nodes = None
        self.non_key_nodes = None
        self.cluster_indices_list = None
        self.step_count = 0

        self.vldb_diagnostics = {}

    def fast_key_node_selection(self, x_np, percentile=95):

        T, N, F = x_np.shape

        mean_vals = np.mean(x_np[:, :, 0], axis=0)
        std_vals = np.std(x_np[:, :, 0], axis=0)
        recent_change = np.abs(x_np[-1, :, 0] - x_np[-2, :, 0])

        scores = mean_vals + std_vals + recent_change

        threshold = np.percentile(scores, percentile)
        key_nodes = np.where(scores >= threshold)[0]
        non_key_nodes = np.setdiff1d(np.arange(N), key_nodes)

        return key_nodes, non_key_nodes, scores, threshold

    def fast_clustering(self, key_nodes, non_key_nodes, x_np):

        if len(non_key_nodes) == 0:
            return [list(key_nodes)] if len(key_nodes) > 0 else []

        features = x_np[-1, :, :]

        if len(non_key_nodes) <= 3:
            clusters = [list(key_nodes)] + [[n] for n in non_key_nodes]
            return [c for c in clusters if len(c) > 0]

        from sklearn.cluster import KMeans
        non_key_features = features[non_key_nodes]
        n_clusters = min(self.n_clusters, len(non_key_nodes))

        if n_clusters <= 1:
            clusters = [list(key_nodes)] + [list(non_key_nodes)]
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(non_key_features)

            clusters = [list(key_nodes)]
            for i in range(n_clusters):
                cluster_nodes = non_key_nodes[cluster_labels == i]
                if len(cluster_nodes) > 0:
                    clusters.append(list(cluster_nodes))

        return clusters

    def create_super_nodes(self, cluster_indices_list, x_np):

        super_nodes = []
        super_node_features = []

        for cluster in cluster_indices_list:
            if len(cluster) == 0:
                continue

            cluster_features = x_np[-1, cluster, :]
            center_features = np.mean(cluster_features, axis=0)

            super_nodes.append(cluster)
            super_node_features.append(center_features)

        return super_nodes, np.array(super_node_features) if super_node_features else np.array([])

    def _create_super_node_features(self, h, super_nodes):

        batch_size, seq_len, num_nodes, hidden_dim = h.shape
        num_super_nodes = len(super_nodes)

        if num_super_nodes == 0:
            return torch.zeros(batch_size, seq_len, 0, hidden_dim, device=h.device)

        super_features = []
        for cluster in super_nodes:
            if len(cluster) > 0:

                cluster_features = h[:, :, cluster, :].mean(dim=2)
                super_features.append(cluster_features)

        if super_features:
            return torch.stack(super_features, dim=2)
        else:
            return torch.zeros(batch_size, seq_len, 0, hidden_dim, device=h.device)

    def _fuse_features(self, h_lower, h_upper, key_nodes, super_nodes):

        batch_size, seq_len, num_nodes, hidden_dim = h_lower.shape

        h_fused = h_lower.clone()

        if len(super_nodes) == 0:
            return h_fused

        for i, cluster in enumerate(super_nodes):
            if len(cluster) > 0:

                super_feature = h_upper[:, :, len(key_nodes) + i, :]

                for node_idx in cluster:
                    if node_idx < num_nodes:

                        lower_feature = h_lower[:, :, node_idx, :]
                        fused_feature = torch.cat([lower_feature, super_feature], dim=-1)
                        fused_feature = self.feature_fusion(fused_feature)
                        h_fused[:, :, node_idx, :] = fused_feature

        return h_fused

    def _traditional_structure_update(self, x_np, adj_matrix):

        T, N, F = x_np.shape
        num_nodes = N

        if self.training:

            if self.use_efficient_clustering and self.optimized_mode:

                self.optimized_clustering_with_dtw(x_np)
            elif self.use_efficient_clustering and not self.optimized_mode:

                changed_nodes, need_full_update, current_state = detect_distribution_change(
                    x_np, self.prev_state, threshold=2.0
                )

                if need_full_update or self.key_nodes is None:

                    importance = np.mean(x_np[:, :, 0], axis=0)
                    self.key_nodes, self.non_key_nodes, scores, threshold = select_dynamic_key_nodes(
                        x_np, percentile=95, importance=importance
                    )

                    all_nodes = np.concatenate([self.key_nodes, self.non_key_nodes])
                    if len(all_nodes) > 1:

                        D = efficient_dtw_distance(
                            x_np, all_nodes,
                            downsample=self.downsample_ratio,
                            n_jobs=self.n_jobs
                        )

                        self.cluster_tree.build_tree(D)
                        n_clusters = min(self.n_clusters, len(all_nodes))
                        clusters = self.cluster_tree.get_clusters(n_clusters=n_clusters)

                        self.cluster_indices_list = []
                        for i in range(1, n_clusters + 1):
                            cluster_mask = clusters == i
                            cluster_nodes = np.array(all_nodes)[cluster_mask]
                            self.cluster_indices_list.append(cluster_nodes.tolist())
                    else:
                        self.cluster_indices_list = []

                    self.prev_state = current_state
            elif not self.optimized_mode:

                cache_key = f"step_{self.step_count // self.cache_valid_steps}"
                if cache_key in self.cluster_cache and self.key_nodes is not None:

                    cached = self.cluster_cache[cache_key]
                    self.key_nodes = cached['key_nodes']
                    self.non_key_nodes = cached['non_key_nodes']
                    self.cluster_indices_list = cached['cluster_indices']
                else:

                    self.key_nodes, self.non_key_nodes, _, _ = self.fast_key_node_selection(x_np)

                    self.cluster_indices_list = self.fast_clustering(
                        self.key_nodes, self.non_key_nodes, x_np
                    )

                    self.cluster_cache[cache_key] = {
                        'key_nodes': self.key_nodes,
                        'non_key_nodes': self.non_key_nodes,
                        'cluster_indices': self.cluster_indices_list
                    }
            else:

                self.key_nodes, self.non_key_nodes, _, _ = self.key_selector.select_keys(x_np)

                all_nodes = np.concatenate([self.key_nodes, self.non_key_nodes])
                if len(all_nodes) > 1:

                    all_features = np.mean(x_np[-1, all_nodes, :], axis=0)
                    all_features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)
                    distances = torch.cdist(all_features_tensor, all_features_tensor)
                    self.cluster_tree.build_tree(distances.cpu().numpy())
                    n_clusters = max(1, len(all_nodes) // 10)
                    cluster_labels = self.cluster_tree.get_clusters(n_clusters=n_clusters)
                    self.cluster_indices_list = []
                    for i in range(1, n_clusters + 1):
                        cluster_mask = cluster_labels == i
                        cluster_nodes = np.array(all_nodes)[cluster_mask]
                        cluster_nodes = [int(idx) for idx in cluster_nodes if 0 <= idx < num_nodes]
                        self.cluster_indices_list.append(cluster_nodes)
                else:
                    self.cluster_indices_list = []
        else:

            if self.optimized_mode:

                if self.key_nodes is None:
                    self.optimized_clustering_with_dtw(x_np)
            elif not self.optimized_mode:

                if self.key_nodes is None:
                    self.key_nodes, self.non_key_nodes, _, _ = self.fast_key_node_selection(x_np)
                    self.cluster_indices_list = self.fast_clustering(
                        self.key_nodes, self.non_key_nodes, x_np
                    )
            else:

                scores = self.key_selector.compute_scores(x_np)
                threshold = np.quantile(scores, 1 - self.key_node_ratio)
                self.key_nodes = np.where(scores >= threshold)[0]
                self.non_key_nodes = np.setdiff1d(np.arange(num_nodes), self.key_nodes)

            all_nodes = np.concatenate([self.key_nodes, self.non_key_nodes])
            if len(all_nodes) > 1:

                all_features = np.mean(x_np[-1, all_nodes, :], axis=0)
                all_features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)
                distances = torch.cdist(all_features_tensor, all_features_tensor)
                self.cluster_tree.build_tree(distances.cpu().numpy())
                n_clusters = max(1, len(all_nodes) // 10)
                cluster_labels = self.cluster_tree.get_clusters(n_clusters=n_clusters)
                self.cluster_indices_list = []

                if len(cluster_labels) != len(all_nodes):

                    cluster_labels = np.arange(len(all_nodes)) % n_clusters

                for i in range(n_clusters):
                    cluster_mask = cluster_labels == i
                    cluster_nodes = np.array(all_nodes)[cluster_mask]
                    cluster_nodes = [int(idx) for idx in cluster_nodes if 0 <= idx < num_nodes]
                    self.cluster_indices_list.append(cluster_nodes)
            else:
                self.cluster_indices_list = []

        self.key_nodes = [int(i) for i in self.key_nodes if 0 <= i < num_nodes]
        self.non_key_nodes = [int(i) for i in self.non_key_nodes if 0 <= i < num_nodes]

    def optimized_clustering_with_dtw(self, x_np):

        T, N, F = x_np.shape

        cache_key = f"step_{self.step_count // self.cache_valid_steps}"
        if cache_key in self.cluster_cache:
            cached = self.cluster_cache[cache_key]
            self.key_nodes = cached['key_nodes']
            self.non_key_nodes = cached['non_key_nodes']
            self.cluster_indices_list = cached['cluster_indices']
            self.super_nodes = cached['super_nodes']
            self.super_node_features = cached['super_node_features']
            return

        importance = np.mean(x_np[:, :, 0], axis=0)
        self.key_nodes, self.non_key_nodes, scores, threshold = select_dynamic_key_nodes(
            x_np, percentile=95, importance=importance
        )

        if len(self.non_key_nodes) > 1:

            D = efficient_dtw_distance(
                x_np, self.non_key_nodes,
                downsample=self.downsample_ratio,
                n_jobs=self.n_jobs
            )

            self.cluster_tree.build_tree(D)
            n_clusters = min(self.n_clusters, len(self.non_key_nodes))
            clusters = self.cluster_tree.get_clusters(n_clusters=n_clusters)

            self.cluster_indices_list = []
            for i in range(1, n_clusters + 1):
                cluster_mask = clusters == i
                cluster_nodes = np.array(self.non_key_nodes)[cluster_mask]
                if len(cluster_nodes) > 0:
                    self.cluster_indices_list.append(cluster_nodes.tolist())
        else:
            self.cluster_indices_list = []

        self.super_nodes, self.super_node_features = self.create_super_nodes(
            self.cluster_indices_list, x_np
        )

        if len(self.super_nodes) == 0 and len(self.cluster_indices_list) > 0:
            self.super_nodes = self.cluster_indices_list
            self.super_node_features = np.array([np.mean(x_np[-1, cluster, :], axis=0) for cluster in self.cluster_indices_list])

        self.cluster_cache[cache_key] = {
            'key_nodes': self.key_nodes,
            'non_key_nodes': self.non_key_nodes,
            'cluster_indices': self.cluster_indices_list,
            'super_nodes': self.super_nodes,
            'super_node_features': self.super_node_features
        }

    def forward(self, history_data=None, future_data=None, *args, **kwargs):

        x = history_data if history_data is not None else kwargs.get('x', None)
        if x is None:
            raise ValueError('CRSTNet模型需要history_data作为输入')
        return self._forward_impl(x)

    def _forward_impl(self, x, adj_matrix=None):
        batch_size, seq_len, num_nodes, input_dim = x.shape
        if adj_matrix is None:
            if self.adj_mx is not None:
                adj_matrix = self.adj_mx
            else:
                adj_matrix = np.eye(num_nodes)

        x_np = x[0].detach().cpu().numpy() if isinstance(x, torch.Tensor) else x[0]
        self.step_count += 1

        if self.use_optimized_structure and self.structure_manager is not None:

            if (self.step_count == 1 or
                self.step_count % self.update_frequency == 0 or
                not hasattr(self, 'key_nodes') or len(self.key_nodes) == 0):
                structure_info = self.structure_manager.update_structure(x_np)
                self.key_nodes = structure_info['key_nodes']
                self.non_key_nodes = structure_info['non_key_nodes']
                self.cluster_indices_list = structure_info['cluster_indices_list']
                super_nodes = structure_info['super_nodes']
                if len(super_nodes) > 0 and isinstance(super_nodes[0], (int, np.integer)):
                    super_nodes = structure_info['cluster_indices_list']
                self.super_nodes = super_nodes
        else:

            self._traditional_structure_update(x_np, adj_matrix)

        self.key_nodes = [int(i) for i in self.key_nodes if 0 <= i < num_nodes]
        self.non_key_nodes = [int(i) for i in self.non_key_nodes if 0 <= i < num_nodes]

        h = x
        residual = None

        if self.optimized_mode:

            for i in range(self.num_layers):

                h = self.temporal_convs[i](h)
                h = self.dropout_layer(h)

                h_lower = self.lower_spatial_convs[i](h, adj_matrix, self.key_nodes, self.cluster_indices_list)

                if self.super_nodes is not None and len(self.super_nodes) > 0:

                    h_super = self._create_super_node_features(h, self.super_nodes)
                    h_upper = torch.cat([h[:, :, self.key_nodes, :], h_super], dim=2)

                    upper_adj = self.upper_spatial_convs[i]._build_block_sparse_adj(
                        adj_matrix, self.key_nodes, self.super_nodes
                    )
                    h_upper = self.upper_spatial_convs[i].basic_gcn(h_upper, upper_adj)

                    h_fused = self._fuse_features(h_lower, h_upper, self.key_nodes, self.super_nodes)
                    h = h_fused
                else:
                    h = h_lower

                h = self.dropout_layer(h)

                if i > 0 and residual is not None:
                    if h.shape == residual.shape:
                        h = h + residual
                residual = h
        else:

            for i in range(self.num_layers):
                h = self.temporal_convs[i](h)
                h = self.dropout_layer(h)
                h = self.spatial_convs[i](h, adj_matrix, self.key_nodes, self.cluster_indices_list)
                h = self.dropout_layer(h)
                if i > 0 and residual is not None:
                    if h.shape == residual.shape:
                        h = h + residual
                residual = h

        output = self.output_proj(h)

        if output.shape[-1] != self.output_dim:

            output = torch.nn.Linear(output.shape[-1], self.output_dim).to(output.device)(output)

        return output

    def get_vldb_diagnostics(self):

        return self.vldb_diagnostics

    def get_key_node_info(self):

        return {
            'key_nodes': self.key_nodes,
            'non_key_nodes': self.non_key_nodes,
            'cluster_indices': self.cluster_indices_list,
            'vldb_diagnostics': self.vldb_diagnostics
        }
