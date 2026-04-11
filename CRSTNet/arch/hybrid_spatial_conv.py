import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STGCNSpatialConv(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, feedback_weight=0.5, use_weighted_feedback=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feedback_weight = feedback_weight
        self.use_weighted_feedback = use_weighted_feedback

        self.intra_cluster_conv = nn.Linear(in_channels, out_channels)

        self.global_conv = nn.Linear(in_channels, out_channels)

        self.feedback_fusion = nn.Linear(out_channels * 2, out_channels)

        if self.use_weighted_feedback:

            self.intra_weight = nn.Parameter(torch.tensor(0.5))

            self.global_weight = nn.Parameter(torch.tensor(0.5))

            self.node_importance_conv = nn.Linear(in_channels, 1)

    def forward(self, X, adj, key_node_indices, cluster_indices_list):
        batch_size, seq_len, num_nodes, features = X.shape
        key_node_indices = [int(i) for i in key_node_indices if 0 <= i < num_nodes]

        full_output = torch.zeros(batch_size, seq_len, num_nodes, self.out_channels, device=X.device)

        sparse_adj = self._build_block_sparse_adj(adj, key_node_indices, cluster_indices_list)

        key_intra_out = None
        if len(key_node_indices) > 0:
            key_X = X[:, :, key_node_indices, :]
            key_A = adj[np.ix_(key_node_indices, key_node_indices)]
            key_intra_out = self.basic_gcn(key_X, key_A)

        cluster_intra_outputs = {}
        cluster_centers = {}

        for cluster_idx, c_idx in enumerate(cluster_indices_list):
            c_idx = [int(i) for i in c_idx if 0 <= i < num_nodes]
            if len(c_idx) == 0: continue

            cluster_X = X[:, :, c_idx, :]

            if self.use_weighted_feedback:

                node_importance = self.node_importance_conv(cluster_X)
                node_importance = torch.softmax(node_importance, dim=2)

                weighted_cluster_center = torch.sum(cluster_X * node_importance, dim=2)
            else:

                weighted_cluster_center = cluster_X.mean(dim=2)

            cluster_A = adj[np.ix_(c_idx, c_idx)]

            cluster_intra_features = self.basic_gcn(cluster_X, cluster_A)

            for i, node_idx in enumerate(c_idx):
                cluster_intra_outputs[node_idx] = cluster_intra_features[:, :, i, :]
                cluster_centers[node_idx] = weighted_cluster_center

        global_nodes = []
        global_features = []

        if len(key_node_indices) > 0 and key_intra_out is not None:
            global_nodes.extend(key_node_indices)
            global_features.append(key_intra_out.mean(dim=2))

        for cluster_idx, c_idx in enumerate(cluster_indices_list):
            c_idx = [int(i) for i in c_idx if 0 <= i < num_nodes]
            if len(c_idx) == 0: continue

            cluster_center = cluster_centers[c_idx[0]]
            global_features.append(cluster_center)

        if len(global_features) > 1:
            global_features = torch.stack(global_features, dim=2)
            N_global = global_features.shape[2]

            global_adj = np.eye(N_global)

            global_out = self.basic_gcn(global_features, global_adj)
        else:
            global_out = torch.stack(global_features, dim=2) if global_features else torch.zeros(batch_size, seq_len, 0, self.out_channels, device=X.device)

        global_idx = 0

        if len(key_node_indices) > 0:
            for i, node_idx in enumerate(key_node_indices):
                intra_feature = key_intra_out[:, :, i, :]
                global_feature = global_out[:, :, global_idx, :]

                if self.use_weighted_feedback:

                    weighted_intra = self.intra_weight * intra_feature
                    weighted_global = self.global_weight * global_feature
                    feedback_feature = torch.cat([weighted_intra, weighted_global], dim=-1)
                else:

                    feedback_feature = torch.cat([intra_feature, global_feature], dim=-1)

                final_feature = self.feedback_fusion(feedback_feature)
                full_output[:, :, node_idx, :] = final_feature
            global_idx += 1

        for cluster_idx, c_idx in enumerate(cluster_indices_list):
            c_idx = [int(i) for i in c_idx if 0 <= i < num_nodes]
            if len(c_idx) == 0: continue

            cluster_global_feature = global_out[:, :, global_idx, :]

            for node_idx in c_idx:
                if node_idx in cluster_intra_outputs:
                    intra_feature = cluster_intra_outputs[node_idx]

                    if self.use_weighted_feedback:

                        weighted_intra = self.intra_weight * intra_feature
                        weighted_global = self.global_weight * cluster_global_feature
                        feedback_feature = torch.cat([weighted_intra, weighted_global], dim=-1)
                    else:

                        feedback_feature = torch.cat([intra_feature, cluster_global_feature], dim=-1)

                    final_feature = self.feedback_fusion(feedback_feature)
                    full_output[:, :, node_idx, :] = final_feature

            global_idx += 1

        return full_output

    def basic_gcn(self, X, A_hat):

        batch_size, seq_len, num_nodes, features = X.shape

        norm = nn.LayerNorm(features).to(X.device)

        X_reshaped = X.view(-1, features)
        X_norm = norm(X_reshaped)
        X_norm = X_norm.view(batch_size, seq_len, num_nodes, features)

        A = torch.tensor(A_hat, dtype=X.dtype, device=X.device)

        assert A.shape[0] == num_nodes and A.shape[1] == num_nodes,\
            f"邻接矩阵维度 {A.shape} 与特征矩阵节点数 {num_nodes} 不匹配"

        out_list = []
        for t in range(seq_len):
            X_t = X_norm[:, t, :, :]
            out_t = torch.matmul(A, X_t)
            out_t = self.global_conv(out_t)
            out_list.append(out_t)

        out = torch.stack(out_list, dim=1)
        return out

    def _build_block_sparse_adj(self, adj, key_node_indices, cluster_indices_list):

        num_nodes = adj.shape[0]
        num_super_nodes = len(cluster_indices_list)
        total_nodes = len(key_node_indices) + num_super_nodes

        sparse_adj = np.zeros((total_nodes, total_nodes))

        for i, ki in enumerate(key_node_indices):
            for j, kj in enumerate(key_node_indices):
                if i != j and adj[ki, kj] > 0:
                    sparse_adj[i, j] = adj[ki, kj]

        for i, ki in enumerate(key_node_indices):
            for j, cluster in enumerate(cluster_indices_list):
                super_idx = len(key_node_indices) + j

                for cluster_node in cluster:
                    if adj[ki, cluster_node] > 0:
                        sparse_adj[i, super_idx] = max(sparse_adj[i, super_idx], adj[ki, cluster_node])
                        sparse_adj[super_idx, i] = sparse_adj[i, super_idx]

        for i, cluster_i in enumerate(cluster_indices_list):
            for j, cluster_j in enumerate(cluster_indices_list):
                if i != j:
                    super_i = len(key_node_indices) + i
                    super_j = len(key_node_indices) + j

                    max_weight = 0
                    for node_i in cluster_i:
                        for node_j in cluster_j:
                            if adj[node_i, node_j] > 0:
                                max_weight = max(max_weight, adj[node_i, node_j])
                    if max_weight > 0:
                        sparse_adj[super_i, super_j] = max_weight
                        sparse_adj[super_j, super_i] = max_weight

        return sparse_adj

    def _super_node_aggregation(self, X, cluster_indices_list):

        batch_size, seq_len, num_nodes, features = X.shape
        num_super_nodes = len(cluster_indices_list)

        if num_super_nodes == 0:
            return torch.zeros(batch_size, seq_len, 0, features, device=X.device)

        super_features = []
        for cluster in cluster_indices_list:
            if len(cluster) > 0:

                cluster_X = X[:, :, cluster, :]
                if self.use_weighted_feedback:

                    node_importance = self.node_importance_conv(cluster_X)
                    node_importance = torch.softmax(node_importance, dim=2)
                    aggregated = torch.sum(cluster_X * node_importance, dim=2)
                else:

                    aggregated = cluster_X.mean(dim=2)
                super_features.append(aggregated)

        return torch.stack(super_features, dim=2)

    def _distribute_super_features(self, super_features, cluster_indices_list, full_output):

        batch_size, seq_len, num_nodes, out_channels = full_output.shape

        for i, cluster in enumerate(cluster_indices_list):
            if i < super_features.shape[2]:
                super_feat = super_features[:, :, i, :]

                for node_idx in cluster:
                    if node_idx < num_nodes:

                        member_feat = full_output[:, :, node_idx, :]
                        fused_feat = member_feat + super_feat
                        full_output[:, :, node_idx, :] = fused_feat

        return full_output
