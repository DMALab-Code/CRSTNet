import hashlib
from collections import OrderedDict

import numpy as np


class SmallLRUCache:
    def __init__(self, max_size=4096):
        self.max_size = max_size
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def put(self, key, value):
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        while len(self.data) > self.max_size:
            self.data.popitem(last=False)

    def clear(self):
        self.data.clear()


def _hash_array(a):
    a = np.asarray(a)
    h = hashlib.md5()
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


class DistanceCache:
    """
    Caches:
    1. all-pairs shortest path matrix for fixed adjacency;
    2. one-hop DTW terms within a maintenance interval.
    """
    def __init__(self, max_dtw_cache=20000):
        self.shortest_path_cache = {}
        self.dtw_cache = SmallLRUCache(max_dtw_cache)

    def shortest_path(self, adj):
        key = _hash_array((np.asarray(adj) > 0).astype(np.uint8))
        if key in self.shortest_path_cache:
            return self.shortest_path_cache[key]

        A = np.asarray(adj)
        N = A.shape[0]
        D = np.full((N, N), np.inf, dtype=float)

        for s in range(N):
            D[s, s] = 0.0
            q = [s]
            for u in q:
                for v in np.where(A[u] > 0)[0]:
                    if np.isinf(D[s, v]):
                        D[s, v] = D[s, u] + 1.0
                        q.append(int(v))

        finite = D[np.isfinite(D)]
        D[~np.isfinite(D)] = finite.max() + 1.0 if finite.size else 1.0
        self.shortest_path_cache[key] = D
        return D

    def dtw(self, x, y, u, v, window_sig, radius=6):
        a, b = sorted((int(u), int(v)))
        key = (a, b, window_sig, int(radius))
        cached = self.dtw_cache.get(key)
        if cached is not None:
            return cached
        value = dtw_distance(x, y, radius=radius)
        self.dtw_cache.put(key, value)
        return value


def normalize_adj(adj, add_self_loop=True):
    A = np.asarray(adj, dtype=float).copy()
    if add_self_loop:
        A = A + np.eye(A.shape[0])
    deg = A.sum(axis=1)
    inv = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv[mask] = 1.0 / np.sqrt(deg[mask])
    return inv[:, None] * A * inv[None, :]


def dtw_distance(x, y, radius=6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    radius = max(int(radius), abs(n - m))

    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        lo = max(1, i - radius)
        hi = min(m, i + radius)
        for j in range(lo, hi + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _window_signature(data):
    x = np.asarray(data)
    if x.ndim == 4:
        x = x[0]
    sig = x[:, :, 0]
    return (
        sig.shape[0],
        sig.shape[1],
        round(float(sig.mean()), 3),
        round(float(sig.std()), 3),
        round(float(sig[-1].mean()), 3),
    )


def composite_distance_matrix(
    data,
    node_indices,
    adj,
    lambda_spatial=0.5,
    d0=None,
    tau0=None,
    one_hop_only=True,
    dtw_radius=6,
    cache=None,
):
    data = np.asarray(data)
    if data.ndim == 4:
        data = data[0]

    x = data[:, :, 0].astype(float)
    node_indices = [int(i) for i in node_indices]
    n = len(node_indices)

    if cache is None:
        cache = DistanceCache()

    D_sp_all = cache.shortest_path(adj)
    D_sp = D_sp_all[np.ix_(node_indices, node_indices)]

    finite = D_sp[(D_sp > 0) & np.isfinite(D_sp)]
    d0 = np.median(finite) if d0 is None and finite.size else (1.0 if d0 is None else d0)

    D_dtw = np.zeros((n, n), dtype=float)
    large = 1e6
    window_sig = _window_signature(data)

    A = np.asarray(adj)
    for i in range(n):
        for j in range(i + 1, n):
            u, v = node_indices[i], node_indices[j]
            if one_hop_only and A[u, v] <= 0 and A[v, u] <= 0:
                val = large
            else:
                val = cache.dtw(x[:, u], x[:, v], u, v, window_sig, radius=dtw_radius)
            D_dtw[i, j] = D_dtw[j, i] = val

    valid = D_dtw[(D_dtw > 0) & (D_dtw < large)]
    tau0 = np.median(valid) if tau0 is None and valid.size else (1.0 if tau0 is None else tau0)

    D = lambda_spatial * (D_sp / (d0 + 1e-12)) + (1.0 - lambda_spatial) * (D_dtw / (tau0 + 1e-12))
    np.fill_diagonal(D, 0.0)
    return D


def _connected(c1, c2, adj):
    A = np.asarray(adj)
    for u in c1:
        for v in c2:
            if A[u, v] > 0 or A[v, u] > 0:
                return True
    return False


def _ward_cost(c1, c2, D):
    block = D[np.ix_(list(c1), list(c2))]
    return float((len(c1) * len(c2)) / max(1, len(c1) + len(c2)) * np.mean(block))


def topology_constrained_ward(
    data,
    node_indices,
    adj,
    n_clusters,
    lambda_spatial=0.5,
    d0=None,
    tau0=None,
    one_hop_only=True,
    dtw_radius=6,
    cache=None,
):
    node_indices = [int(i) for i in node_indices]
    if not node_indices:
        return []

    n_clusters = max(1, min(int(n_clusters), len(node_indices)))
    if len(node_indices) <= n_clusters:
        return [[i] for i in node_indices]

    D = composite_distance_matrix(
        data,
        node_indices,
        adj,
        lambda_spatial=lambda_spatial,
        d0=d0,
        tau0=tau0,
        one_hop_only=one_hop_only,
        dtw_radius=dtw_radius,
        cache=cache,
    )

    clusters = [[i] for i in range(len(node_indices))]

    while len(clusters) > n_clusters:
        best = None
        best_cost = float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ci = [node_indices[k] for k in clusters[i]]
                cj = [node_indices[k] for k in clusters[j]]
                if not _connected(ci, cj, adj):
                    continue
                cost = _ward_cost(clusters[i], clusters[j], D)
                if cost < best_cost:
                    best = (i, j)
                    best_cost = cost

        if best is None:
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cost = _ward_cost(clusters[i], clusters[j], D)
                    if cost < best_cost:
                        best = (i, j)
                        best_cost = cost

        i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    return [sorted(node_indices[k] for k in c) for c in clusters]


class HierarchicalClusterTree:
    def __init__(self, n_clusters=6, lambda_spatial=0.5):
        self.n_clusters = n_clusters
        self.lambda_spatial = lambda_spatial
        self.labels_ = None
        self.clusters_ = []

    def fit(self, data, node_indices, adj, n_clusters=None, cache=None):
        k = self.n_clusters if n_clusters is None else n_clusters
        self.clusters_ = topology_constrained_ward(
            data,
            node_indices,
            adj,
            k,
            lambda_spatial=self.lambda_spatial,
            cache=cache,
        )
        pos = {n: i for i, n in enumerate(node_indices)}
        labels = np.zeros(len(node_indices), dtype=int)
        for cid, c in enumerate(self.clusters_):
            for n in c:
                labels[pos[n]] = cid
        self.labels_ = labels
        return self

    def build_tree(self, D):
        n = D.shape[0]
        self.labels_ = np.arange(n)

    def get_clusters(self, n_clusters=None):
        return np.asarray(self.labels_) + 1 if self.labels_ is not None else np.array([])


HTree = HierarchicalClusterTree


class GraphOps:
    @staticmethod
    def promote_to_key(node, key_nodes, clusters):
        key = sorted(set(map(int, key_nodes)) | {int(node)})
        clusters = [[v for v in c if v != node] for c in clusters]
        return key, [c for c in clusters if c]

    @staticmethod
    def demote_to_nonkey(node, key_nodes, clusters):
        key = [v for v in key_nodes if v != node]
        return key, list(clusters) + [[int(node)]]

    @staticmethod
    def reassign_nonkey(node, from_cluster, to_cluster):
        return [v for v in from_cluster if v != node], sorted(set(to_cluster) | {int(node)})


def promote_to_key(*args, **kwargs):
    return GraphOps.promote_to_key(*args, **kwargs)


def demote_to_nonkey(*args, **kwargs):
    return GraphOps.demote_to_nonkey(*args, **kwargs)


def reassign_nonkey(*args, **kwargs):
    return GraphOps.reassign_nonkey(*args, **kwargs)
