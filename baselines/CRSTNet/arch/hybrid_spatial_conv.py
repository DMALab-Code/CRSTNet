import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_cluster import normalize_adj


def dense_to_sparse_tensor(A, device=None, dtype=torch.float32):
    A = np.asarray(A, dtype=float)
    rows, cols = np.nonzero(np.abs(A) > 1e-12)
    if len(rows) == 0:
        idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        val = torch.zeros((0,), dtype=dtype, device=device)
    else:
        idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
        val = torch.tensor(A[rows, cols], dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=A.shape, device=device, dtype=dtype).coalesce()


def build_hgraph_matrices(num_nodes, key_nodes, clusters, adj):
    """
    Build cached route matrices for H-Graph.

    lower_adj:
        block-sparse adjacency for singleton key nodes and intra-cluster subgraphs.

    upper_adj:
        condensed graph adjacency over singleton key nodes and non-key clusters.

    full_adj:
        optional full graph branch.
    """
    A = np.asarray(adj, dtype=float)
    key_nodes = [int(i) for i in key_nodes if 0 <= int(i) < num_nodes]
    clusters = [sorted(set(int(v) for v in c if 0 <= int(v) < num_nodes)) for c in clusters]
    clusters = [c for c in clusters if len(c) > 0]

    groups = [[k] for k in key_nodes] + clusters

    lower = np.zeros((num_nodes, num_nodes), dtype=float)
    for k in key_nodes:
        lower[k, k] = 1.0
    for c in clusters:
        lower[np.ix_(c, c)] = A[np.ix_(c, c)]
        for v in c:
            lower[v, v] = 1.0
    lower = normalize_adj(lower, add_self_loop=False)

    M = len(groups)
    upper = np.zeros((M, M), dtype=float)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if i == j:
                upper[i, j] = 1.0
            else:
                block = A[np.ix_(gi, gj)]
                nz = block[block > 0]
                if nz.size:
                    upper[i, j] = float(nz.mean())
    upper = normalize_adj(upper, add_self_loop=True) if M > 0 else upper

    full = normalize_adj(A, add_self_loop=True)

    return {
        "lower_adj": lower,
        "upper_adj": upper,
        "full_adj": full,
        "groups": groups,
    }


class STGCNSpatialConv(nn.Module):
    """
    Lightweight GCN layer.
    Supports both dense and sparse normalized adjacency.
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def basic_gcn(self, X, A_hat):
        """
        X: [B,T,N,F]
        A_hat: [N,N], dense or sparse COO
        """
        Xn = self.input_norm(X)
        B, T, N, F_in = Xn.shape

        if torch.is_tensor(A_hat) and A_hat.is_sparse:
            A = A_hat.to(dtype=Xn.dtype, device=Xn.device)
            xr = Xn.permute(2, 0, 1, 3).reshape(N, -1)
            yr = torch.sparse.mm(A, xr)
            out = yr.reshape(N, B, T, F_in).permute(1, 2, 0, 3).contiguous()
        else:
            A = A_hat.to(dtype=Xn.dtype, device=Xn.device) if torch.is_tensor(A_hat) else torch.as_tensor(
                A_hat, dtype=Xn.dtype, device=Xn.device
            )
            xr = Xn.permute(2, 0, 1, 3).reshape(N, -1)
            yr = torch.matmul(A, xr)
            out = yr.reshape(N, B, T, F_in).permute(1, 2, 0, 3).contiguous()

        out = self.linear(out)
        out = self.dropout(out)
        return F.relu(out)

    def forward(self, X, A_hat):
        return self.basic_gcn(X, A_hat)
