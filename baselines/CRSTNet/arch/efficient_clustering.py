from dataclasses import dataclass, field

import numpy as np

from .hierarchical_cluster import (
    DistanceCache,
    topology_constrained_ward,
    dtw_distance,
)
from .key_node_selector import KeyNodeSelector


def _as_tnf(data):
    data = np.asarray(data)
    return data[0] if data.ndim == 4 else data


def select_dynamic_key_nodes(data, percentile=95, importance=None, theta=None, bins=None):
    data = _as_tnf(data)
    N = data.shape[1]
    theta = theta if theta is not None else max(1.0 / max(1, N), (100.0 - percentile) / 100.0)
    return KeyNodeSelector(bins=bins, theta=theta).select(data, theta=theta)


def fastdtw(x, y, radius=6):
    return dtw_distance(x, y, radius), None


def efficient_dtw_distance(data, node_indices, top_k=5, downsample=1, n_jobs=1):
    data = _as_tnf(data)
    if downsample and downsample > 1:
        data = data[:: int(downsample)]
    x = data[:, :, 0]
    node_indices = [int(i) for i in node_indices]
    D = np.zeros((len(node_indices), len(node_indices)), dtype=float)
    for i in range(len(node_indices)):
        for j in range(i + 1, len(node_indices)):
            d = dtw_distance(x[:, node_indices[i]], x[:, node_indices[j]], radius=6)
            D[i, j] = D[j, i] = d
    return D


def detect_distribution_change(data, prev_state=None, threshold=2.0):
    data = _as_tnf(data)
    x = data[:, :, 0]
    cur = {"mean": x.mean(axis=0), "std": x.std(axis=0), "last": x[-1].copy()}

    if prev_state is None:
        return list(range(x.shape[1])), True, cur

    score = (
        np.abs(cur["mean"] - prev_state["mean"]) / (np.abs(prev_state["mean"]) + 1e-5)
        + np.abs(cur["std"] - prev_state["std"]) / (np.abs(prev_state["std"]) + 1e-5)
        + np.abs(cur["last"] - prev_state["last"]) / (np.abs(prev_state["last"]) + 1e-5)
    )
    changed = np.where(score > threshold)[0].astype(int).tolist()
    return changed, len(changed) / max(1, x.shape[1]) > 0.3, cur


class SpatioTemporalSketch:
    """
    Per-vertex sufficient-statistics sketch for MDL maintenance.

    The sketch stores O(1) statistics per vertex and lets the maintainer
    estimate split/extract/merge description-length changes without repeatedly
    rescanning the raw temporal window for every candidate.
    """
    def __init__(self, eps=1e-5, ema_beta=0.90):
        self.eps = float(eps)
        self.ema_beta = min(max(float(ema_beta), 0.0), 0.999)
        self.count = None
        self.sum = None
        self.sq_sum = None
        self.last = None
        self.slope = None
        self.updates = 0
        self.queries = 0

    def update(self, data):
        data = _as_tnf(data)
        x = data[:, :, 0].astype(float)
        count = np.full(x.shape[1], max(1, x.shape[0]), dtype=float)
        cur_sum = x.sum(axis=0)
        cur_sq_sum = np.sum(x * x, axis=0)
        cur_last = x[-1].copy()
        cur_slope = np.abs(x[-1] - x[-2]) if x.shape[0] > 1 else np.zeros(x.shape[1], dtype=float)

        if self.sum is None or self.sum.shape != cur_sum.shape:
            self.count = count
            self.sum = cur_sum
            self.sq_sum = cur_sq_sum
            self.last = cur_last
            self.slope = cur_slope
        else:
            beta = self.ema_beta
            self.count = beta * self.count + (1.0 - beta) * count
            self.sum = beta * self.sum + (1.0 - beta) * cur_sum
            self.sq_sum = beta * self.sq_sum + (1.0 - beta) * cur_sq_sum
            self.last = beta * self.last + (1.0 - beta) * cur_last
            self.slope = beta * self.slope + (1.0 - beta) * cur_slope
        self.updates += 1

    def _nodes(self, nodes):
        return np.asarray(sorted(set(int(v) for v in nodes)), dtype=int)

    def data_cost(self, nodes):
        if self.sum is None:
            return None
        idx = self._nodes(nodes)
        if idx.size <= 1:
            self.queries += 1
            return 0.0
        cnt = float(np.sum(self.count[idx]))
        s = float(np.sum(self.sum[idx]))
        ss = float(np.sum(self.sq_sum[idx]))
        sse = max(0.0, ss - (s * s) / max(cnt, self.eps))
        self.queries += 1
        return 0.5 * max(1.0, cnt) * float(np.log2(1.0 + sse / (max(1.0, cnt) + self.eps)))

    def vertex_deviation(self, node, nodes):
        if self.sum is None:
            return 0.0
        node = int(node)
        idx = self._nodes(nodes)
        if idx.size == 0:
            return 0.0
        means = self.sum[idx] / np.maximum(self.count[idx], self.eps)
        mean = float(np.mean(means))
        node_mean = float(self.sum[node] / max(self.count[node], self.eps))
        return abs(node_mean - mean)

    def compact_state(self):
        return {
            "enabled": self.sum is not None,
            "updates": int(self.updates),
            "queries": int(self.queries),
        }


class ProbabilisticErrorSketch:
    """
    Count-Min-style probabilistic screening sketch for affected vertices.

    The sketch is not allowed to commit H-Graph changes by itself. It only
    produces a small candidate tail for the exact MDL/IPAC maintainer. This
    keeps the original maintenance narrative intact while avoiding exact
    split/extract checks for obviously stable vertices.
    """
    def __init__(self, width=64, depth=4, ema_beta=0.90, tail_ratio=0.40, seed=13, eps=1e-5):
        self.width = max(8, int(width))
        self.depth = max(2, int(depth))
        self.ema_beta = min(max(float(ema_beta), 0.0), 0.999)
        self.tail_ratio = min(max(float(tail_ratio), 0.05), 1.0)
        self.seed = int(seed)
        self.eps = float(eps)
        rng = np.random.RandomState(self.seed)
        self.hash_a = rng.randint(1, 2**31 - 1, size=self.depth, dtype=np.int64)
        self.hash_b = rng.randint(0, 2**31 - 1, size=self.depth, dtype=np.int64)
        self.table = np.zeros((self.depth, self.width), dtype=float)
        self.last_estimate = None
        self.updates = 0
        self.queries = 0

    def _normalize(self, x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(np.min(x)) if x.size else 0.0
        hi = float(np.max(x)) if x.size else 0.0
        if hi - lo < self.eps:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo + self.eps)

    def _buckets(self, n):
        idx = np.arange(int(n), dtype=np.int64)
        return [((self.hash_a[d] * idx + self.hash_b[d]) % self.width).astype(np.int64) for d in range(self.depth)]

    def update_from_state(self, st_sketch, scores=None, protected=None):
        if st_sketch.sum is None:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)

        count = np.maximum(np.asarray(st_sketch.count, dtype=float), self.eps)
        mean = np.asarray(st_sketch.sum, dtype=float) / count
        var = np.maximum(0.0, np.asarray(st_sketch.sq_sum, dtype=float) / count - mean * mean)
        slope = np.asarray(st_sketch.slope, dtype=float)
        mass = 0.55 * self._normalize(var) + 0.35 * self._normalize(slope)
        if scores is not None:
            mass += 0.10 * self._normalize(scores)
        if protected:
            protected_idx = np.asarray(sorted(set(int(v) for v in protected)), dtype=int)
            protected_idx = protected_idx[(protected_idx >= 0) & (protected_idx < mass.size)]
            mass[protected_idx] = np.maximum(mass[protected_idx], 1.0)

        self.table *= self.ema_beta
        for d, buckets in enumerate(self._buckets(mass.size)):
            np.add.at(self.table[d], buckets, (1.0 - self.ema_beta) * mass)

        estimates = np.zeros((self.depth, mass.size), dtype=float)
        for d, buckets in enumerate(self._buckets(mass.size)):
            estimates[d] = self.table[d, buckets]
        self.last_estimate = np.min(estimates, axis=0)
        self.updates += 1

        k = max(1, int(np.ceil(self.tail_ratio * mass.size)))
        order = np.argsort(-self.last_estimate, kind="mergesort")
        candidates = np.sort(order[:k].astype(int))
        return candidates, self.last_estimate

    def compact_state(self):
        # Standard Count-Min bound: with width w and depth d, additive
        # over-estimation is O(e / w) with failure probability O(e^-d).
        return {
            "enabled": True,
            "updates": int(self.updates),
            "queries": int(self.queries),
            "width": int(self.width),
            "depth": int(self.depth),
            "tail_ratio": float(self.tail_ratio),
            "epsilon_bound": float(np.e / max(1, self.width)),
            "delta_bound": float(np.exp(-max(1, self.depth))),
        }


class DynamicConnectivityForest:
    """
    Runtime dynamic-connectivity view for IPAC.

    For the current small traffic benchmarks this is implemented as a cached
    local forest over the road graph. The public operations mirror a dynamic
    forest interface (component query after deletions), so CA-ANIM can reuse
    stable cores and only materialize the affected fringe.
    """
    def __init__(self):
        self.adj = None
        self.cache = {}
        self.queries = 0
        self.cache_hits = 0
        self.splits = 0

    def set_graph(self, adj):
        A = (np.asarray(adj) > 0).astype(np.uint8)
        if self.adj is None or self.adj.shape != A.shape or np.any(self.adj != A):
            self.adj = A
            self.cache.clear()

    def components(self, nodes, removed=None):
        nodes = tuple(sorted(set(int(v) for v in nodes)))
        removed = tuple(sorted(set(int(v) for v in (removed or []))))
        key = (nodes, removed)
        self.queries += 1
        if key in self.cache:
            self.cache_hits += 1
            return [list(c) for c in self.cache[key]]

        if self.adj is None:
            active = [v for v in nodes if v not in set(removed)]
            comps = [[v] for v in active]
            self.cache[key] = tuple(tuple(c) for c in comps)
            return comps

        removed_set = set(removed)
        active = [v for v in nodes if v not in removed_set]
        active_set = set(active)
        seen = set()
        comps = []
        for start in active:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                neigh = np.where((self.adj[u] > 0) | (self.adj[:, u] > 0))[0]
                for v in neigh:
                    v = int(v)
                    if v in active_set and v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(sorted(comp))
        if len(comps) > 1:
            self.splits += len(comps) - 1
        self.cache[key] = tuple(tuple(c) for c in comps)
        if len(self.cache) > 4096:
            self.cache.clear()
        return [list(c) for c in comps]

    def compact_state(self):
        return {
            "enabled": self.adj is not None,
            "queries": int(self.queries),
            "cache_hits": int(self.cache_hits),
            "splits": int(self.splits),
            "amortized_target": "O(log^2 V) dynamic-forest interface; cached local forest runtime",
        }


class StatefulBoundaryIndex:
    """
    CPU canonical boundary index for incumbent-plan-aware maintenance.

    The index tags each vertex as internal/boundary under the current H-Graph
    snapshot and stores the physical cut-set between condensed groups. It lets
    IPAC query candidate hosts and route-coupled neighbors without repeatedly
    scanning all clusters.
    """
    def __init__(self):
        self.version = None
        self.key_count = 0
        self.node_to_group = {}
        self.group_members = []
        self.group_neighbors = {}
        self.node_external_groups = {}
        self.node_external_clusters = {}
        self.internal_vertices = set()
        self.boundary_vertices = set()
        self.intra_degree = {}
        self.cut_edges = []
        self.queries = 0
        self.candidate_hits = 0
        self.zero_traversal_skips = 0

    def rebuild(self, snapshot, adj):
        A = (np.asarray(adj) > 0).astype(np.uint8)
        groups = [list(map(int, g)) for g in snapshot.groups]
        node_to_group = {}
        for gid, group in enumerate(groups):
            for node in group:
                node_to_group[int(node)] = int(gid)

        self.version = int(snapshot.version)
        self.key_count = len(snapshot.key_nodes)
        self.node_to_group = node_to_group
        self.group_members = groups
        self.group_neighbors = {gid: set() for gid in range(len(groups))}
        self.node_external_groups = {}
        self.node_external_clusters = {}
        self.internal_vertices = set()
        self.boundary_vertices = set()
        self.intra_degree = {}
        self.cut_edges = []

        for u, gid in node_to_group.items():
            neigh = np.where((A[u] > 0) | (A[:, u] > 0))[0]
            external_groups = set()
            intra = 0
            for v in neigh:
                v = int(v)
                ngid = node_to_group.get(v)
                if ngid is None:
                    continue
                if ngid == gid:
                    intra += 1
                else:
                    external_groups.add(int(ngid))
                    self.group_neighbors.setdefault(gid, set()).add(int(ngid))
                    self.group_neighbors.setdefault(int(ngid), set()).add(int(gid))
                    if u < v:
                        self.cut_edges.append((int(u), int(v), int(gid), int(ngid)))

            self.intra_degree[int(u)] = int(intra)
            self.node_external_groups[int(u)] = sorted(external_groups)
            external_clusters = [
                int(g - self.key_count)
                for g in external_groups
                if int(g) >= self.key_count
            ]
            self.node_external_clusters[int(u)] = sorted(set(external_clusters))
            if external_groups:
                self.boundary_vertices.add(int(u))
            else:
                self.internal_vertices.add(int(u))

        return self

    def compact_state(self):
        return {
            "enabled": self.version is not None,
            "version": self.version,
            "key_count": int(self.key_count),
            "boundary_vertices": int(len(self.boundary_vertices)),
            "internal_vertices": int(len(self.internal_vertices)),
            "cut_edges": int(len(self.cut_edges)),
            "queries": int(self.queries),
            "candidate_hits": int(self.candidate_hits),
            "zero_traversal_skips": int(self.zero_traversal_skips),
            "mode": "stateful_boundary_index",
        }

    def is_internal(self, node):
        return int(node) in self.internal_vertices

    def candidate_clusters_for_node(self, node):
        self.queries += 1
        clusters = list(self.node_external_clusters.get(int(node), []))
        if clusters:
            self.candidate_hits += 1
        return clusters

    def condensed_neighbors(self, group_count):
        self.queries += 1
        return {
            int(gid): set(int(v) for v in self.group_neighbors.get(int(gid), set()) if 0 <= int(v) < group_count)
            for gid in range(group_count)
        }

    def zero_traversal_block(self, vertex, parent_cluster):
        vertex = int(vertex)
        parent = set(int(v) for v in parent_cluster)
        if vertex not in parent or len(parent) <= 2:
            return None
        if not self.is_internal(vertex):
            return None
        # Local non-articulation proxy: an internal vertex with at least two
        # intra-cluster physical neighbors usually does not invalidate any
        # condensed boundary entry when isolated as an unstable fringe.
        if self.intra_degree.get(vertex, 0) >= 2:
            self.zero_traversal_skips += 1
            return [vertex]
        return None


@dataclass
class HGraphSnapshot:
    key_nodes: list
    clusters: list
    scores: np.ndarray
    key_threshold: float
    version: int = 0
    node_to_group: dict = field(default_factory=dict)

    @property
    def groups(self):
        return [[int(k)] for k in self.key_nodes] + [list(map(int, c)) for c in self.clusters]

    def rebuild_index(self):
        self.node_to_group = {}
        for gid, g in enumerate(self.groups):
            for v in g:
                self.node_to_group[int(v)] = int(gid)


@dataclass
class MaintenanceStats:
    stable_ranges: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    last_smp: dict = field(default_factory=dict)
    last_affected: list = field(default_factory=list)
    delta_s: float = None
    delta_e: float = None
    delta_d: float = None
    delta_c: float = None
    changed_last_update: bool = False
    promoted: list = field(default_factory=list)
    demoted: list = field(default_factory=list)
    hysteresis_kept: list = field(default_factory=list)
    error_protected: list = field(default_factory=list)
    proxy_protected: list = field(default_factory=list)
    model_error_protected: list = field(default_factory=list)
    key_protection_mode: str = "hxr_only"
    update_components: list = field(default_factory=list)
    update_component_count: int = 0
    internal_update_blocks: list = field(default_factory=list)
    connectivity_repairs: int = 0
    joint_demotions: list = field(default_factory=list)
    singleton_demotions: list = field(default_factory=list)
    component_update_mode: str = "nodewise_anim"
    protected_vertices: list = field(default_factory=list)
    dry_run_admitted: bool = False
    dry_run_fit_gain: float = 0.0
    dry_run_old_fit: float = 0.0
    dry_run_new_fit: float = 0.0
    dry_run_reason: str = "not_planned"
    dry_run_invalidation_cost: dict = field(default_factory=dict)
    merged_route_patch: dict = field(default_factory=dict)
    route_patch_mode: str = "none"
    maintenance_objective: str = "mdl_ipac_parameter_free"
    mdl_affected: dict = field(default_factory=dict)
    mdl_patch_delta: float = 0.0
    mdl_old_cost: float = 0.0
    mdl_new_cost: float = 0.0
    mdl_split_units: list = field(default_factory=list)
    mdl_extract_vertices: list = field(default_factory=list)
    mdl_merge_keys: list = field(default_factory=list)
    ipac_core_fringe: list = field(default_factory=list)
    ipac_bridge_vertices: list = field(default_factory=list)
    ipac_patch_operators: list = field(default_factory=list)
    sketch_mdl_enabled: bool = False
    sketch_mdl_state: dict = field(default_factory=dict)
    probabilistic_sketch_enabled: bool = False
    probabilistic_sketch_state: dict = field(default_factory=dict)
    probabilistic_candidate_nodes: list = field(default_factory=list)
    probabilistic_candidate_groups: list = field(default_factory=list)
    probabilistic_error_bound: dict = field(default_factory=dict)
    dynamic_forest_enabled: bool = False
    dynamic_forest_state: dict = field(default_factory=dict)
    boundary_index_enabled: bool = False
    boundary_index_state: dict = field(default_factory=dict)
    boundary_index_candidate_hits: int = 0
    boundary_index_zero_traversal_skips: int = 0
    repair_unit_index_enabled: bool = False
    repair_unit_index_state: dict = field(default_factory=dict)
    repair_unit_index_entries: list = field(default_factory=list)
    patch_saving_coalescing_enabled: bool = False
    patch_saving_coalescing_state: dict = field(default_factory=dict)
    patch_saving_merges: list = field(default_factory=list)
    patch_saving_rejections: list = field(default_factory=list)
    zcbs_enabled: bool = False
    zcbs_state: dict = field(default_factory=dict)
    zcbs_moves: list = field(default_factory=list)
    zcbs_singletons: list = field(default_factory=list)
    zcbs_fallback_to_ward: bool = False
    sublinear_maintenance_mode: str = "disabled"


class HGraphMaintainer:
    """
    Paper-aligned H-Graph incremental maintenance.

    Cached:
        - shortest path matrix
        - one-hop DTW terms
        - stable ranges
        - residual errors
        - previous structure snapshot

    Incremental update:
        - identify affected key/non-key nodes with SMP and RE
        - add SANI secondary affected one-hop neighbors
        - locally re-cluster only affected region
    """
    def __init__(
        self,
        theta=0.3,
        n_clusters=6,
        lambda_spatial=0.5,
        eta=0.05,
        eps=1e-5,
        dtw_radius=6,
        affected_ratio_full_rebuild=0.35,
        max_dtw_cache=20000,
        promote_margin=1.05,
        demote_margin=0.90,
        demote_patience=2,
        min_cluster_size=3,
        max_cluster_size=8,
        error_key_ratio=0.30,
        model_error_warmup_epochs=10,
        model_error_full_epochs=30,
        protected_patience=2,
        maintenance_objective="mdl_ipac_parameter_free",
        sketch_mdl_enabled=True,
        sketch_ema_beta=0.90,
        probabilistic_sketch_enabled=False,
        probabilistic_sketch_width=64,
        probabilistic_sketch_depth=4,
        probabilistic_sketch_tail_ratio=0.40,
        probabilistic_sketch_seed=13,
        dynamic_forest_enabled=True,
        boundary_index_enabled=False,
        boundary_index_skip_internal_connectivity=True,
        boundary_index_candidate_limit=8,
        boundary_index_restrict_candidate_hosts=True,
        repair_unit_index_enabled=False,
        patch_saving_coalescing_enabled=False,
        zero_clustering_boundary_shift_enabled=False,
        zcbs_candidate_limit=8,
        zcbs_promote_unresolved=True,
        zcbs_min_gain=0.0,
        zcbs_fallback_to_ward=False,
    ):
        self.theta = theta
        self.n_clusters = n_clusters
        self.lambda_spatial = lambda_spatial
        self.eta = eta
        self.eps = eps
        self.dtw_radius = dtw_radius
        self.affected_ratio_full_rebuild = affected_ratio_full_rebuild
        self.promote_margin = float(promote_margin)
        self.demote_margin = float(demote_margin)
        self.demote_patience = max(1, int(demote_patience))
        self.min_cluster_size = max(1, int(min_cluster_size))
        self.max_cluster_size = max(self.min_cluster_size, int(max_cluster_size))
        self.error_key_ratio = min(max(float(error_key_ratio), 0.0), 0.9)
        self.model_error_warmup_epochs = max(0, int(model_error_warmup_epochs))
        self.model_error_full_epochs = max(self.model_error_warmup_epochs + 1, int(model_error_full_epochs))
        self.protected_patience = max(1, int(protected_patience))
        self.maintenance_objective = str(maintenance_objective)
        self.sketch_mdl_enabled = bool(sketch_mdl_enabled)
        self.probabilistic_sketch_enabled = bool(probabilistic_sketch_enabled)
        self.dynamic_forest_enabled = bool(dynamic_forest_enabled)
        self.boundary_index_enabled = bool(boundary_index_enabled)
        self.boundary_index_skip_internal_connectivity = bool(boundary_index_skip_internal_connectivity)
        self.boundary_index_candidate_limit = max(1, int(boundary_index_candidate_limit))
        self.boundary_index_restrict_candidate_hosts = bool(boundary_index_restrict_candidate_hosts)
        self.repair_unit_index_enabled = bool(repair_unit_index_enabled)
        self.patch_saving_coalescing_enabled = bool(patch_saving_coalescing_enabled)
        self.zero_clustering_boundary_shift_enabled = bool(zero_clustering_boundary_shift_enabled)
        self.zcbs_candidate_limit = max(1, int(zcbs_candidate_limit))
        self.zcbs_promote_unresolved = bool(zcbs_promote_unresolved)
        self.zcbs_min_gain = max(0.0, float(zcbs_min_gain))
        self.zcbs_fallback_to_ward = bool(zcbs_fallback_to_ward)

        self.selector = KeyNodeSelector(theta=theta, eps=eps)
        self.distance_cache = DistanceCache(max_dtw_cache=max_dtw_cache)
        self.st_sketch = SpatioTemporalSketch(eps=eps, ema_beta=sketch_ema_beta)
        self.prob_sketch = ProbabilisticErrorSketch(
            width=probabilistic_sketch_width,
            depth=probabilistic_sketch_depth,
            ema_beta=sketch_ema_beta,
            tail_ratio=probabilistic_sketch_tail_ratio,
            seed=probabilistic_sketch_seed,
            eps=eps,
        )
        self.dynamic_forest = DynamicConnectivityForest()
        self.boundary_index = StatefulBoundaryIndex()
        self.stats = MaintenanceStats()
        self.stats.maintenance_objective = self.maintenance_objective
        self.stats.sketch_mdl_enabled = self.sketch_mdl_enabled
        self.stats.probabilistic_sketch_enabled = self.probabilistic_sketch_enabled
        self.stats.probabilistic_sketch_state = self.prob_sketch.compact_state()
        self.stats.probabilistic_error_bound = {
            "epsilon_bound": self.stats.probabilistic_sketch_state["epsilon_bound"],
            "delta_bound": self.stats.probabilistic_sketch_state["delta_bound"],
        }
        self.stats.dynamic_forest_enabled = self.dynamic_forest_enabled
        self.stats.boundary_index_enabled = self.boundary_index_enabled
        self.stats.repair_unit_index_enabled = self.repair_unit_index_enabled
        self.stats.patch_saving_coalescing_enabled = self.patch_saving_coalescing_enabled
        self.stats.zcbs_enabled = self.zero_clustering_boundary_shift_enabled
        self.stats.sublinear_maintenance_mode = (
            "repair_unit_index_patch_saving_ipac"
            if self.repair_unit_index_enabled and self.patch_saving_coalescing_enabled
            else
            "probabilistic_sketch_mdl_plus_dynamic_forest_sbi_ipac"
            if self.probabilistic_sketch_enabled and self.sketch_mdl_enabled and self.dynamic_forest_enabled and self.boundary_index_enabled
            else "sketch_mdl_zcbs_ipac"
            if self.sketch_mdl_enabled and self.zero_clustering_boundary_shift_enabled
            else "sketch_mdl_plus_dynamic_forest_sbi_ipac"
            if self.sketch_mdl_enabled and self.dynamic_forest_enabled and self.boundary_index_enabled
            else "sketch_mdl_plus_dynamic_forest_ipac"
            if self.sketch_mdl_enabled and self.dynamic_forest_enabled
            else "component_mdl_ipac"
        )
        self.demote_counts = {}
        self.protected_counts = {}
        self.error_ema = None
        self.current_epoch = None
        self.external_probabilistic_candidate_nodes = None
        self.external_probabilistic_sketch_state = None

    def set_probabilistic_candidates(self, candidate_nodes=None, sketch_state=None):
        if candidate_nodes is None:
            self.external_probabilistic_candidate_nodes = None
        else:
            self.external_probabilistic_candidate_nodes = sorted(set(int(v) for v in candidate_nodes))
        self.external_probabilistic_sketch_state = dict(sketch_state or {})

    def copy_for_worker(self):
        other = HGraphMaintainer(
            theta=self.theta,
            n_clusters=self.n_clusters,
            lambda_spatial=self.lambda_spatial,
            eta=self.eta,
            eps=self.eps,
            dtw_radius=self.dtw_radius,
            affected_ratio_full_rebuild=self.affected_ratio_full_rebuild,
            promote_margin=self.promote_margin,
            demote_margin=self.demote_margin,
            demote_patience=self.demote_patience,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
            error_key_ratio=self.error_key_ratio,
            model_error_warmup_epochs=self.model_error_warmup_epochs,
            model_error_full_epochs=self.model_error_full_epochs,
            protected_patience=self.protected_patience,
            maintenance_objective=self.maintenance_objective,
            sketch_mdl_enabled=self.sketch_mdl_enabled,
            sketch_ema_beta=self.st_sketch.ema_beta,
            probabilistic_sketch_enabled=self.probabilistic_sketch_enabled,
            probabilistic_sketch_width=self.prob_sketch.width,
            probabilistic_sketch_depth=self.prob_sketch.depth,
            probabilistic_sketch_tail_ratio=self.prob_sketch.tail_ratio,
            probabilistic_sketch_seed=self.prob_sketch.seed,
            dynamic_forest_enabled=self.dynamic_forest_enabled,
            boundary_index_enabled=self.boundary_index_enabled,
            boundary_index_skip_internal_connectivity=self.boundary_index_skip_internal_connectivity,
            boundary_index_candidate_limit=self.boundary_index_candidate_limit,
            boundary_index_restrict_candidate_hosts=self.boundary_index_restrict_candidate_hosts,
            repair_unit_index_enabled=self.repair_unit_index_enabled,
            patch_saving_coalescing_enabled=self.patch_saving_coalescing_enabled,
            zero_clustering_boundary_shift_enabled=self.zero_clustering_boundary_shift_enabled,
            zcbs_candidate_limit=self.zcbs_candidate_limit,
            zcbs_promote_unresolved=self.zcbs_promote_unresolved,
            zcbs_min_gain=self.zcbs_min_gain,
            zcbs_fallback_to_ward=self.zcbs_fallback_to_ward,
        )
        other.stats.stable_ranges = dict(self.stats.stable_ranges)
        other.stats.residuals = dict(self.stats.residuals)
        other.stats.last_smp = dict(self.stats.last_smp)
        other.stats.delta_s = self.stats.delta_s
        other.stats.delta_e = self.stats.delta_e
        other.stats.delta_d = self.stats.delta_d
        other.stats.delta_c = self.stats.delta_c
        other.demote_counts = dict(self.demote_counts)
        other.protected_counts = dict(self.protected_counts)
        other.current_epoch = self.current_epoch
        other.error_ema = None if self.error_ema is None else np.asarray(self.error_ema, dtype=float).copy()
        other.external_probabilistic_candidate_nodes = (
            None if self.external_probabilistic_candidate_nodes is None
            else list(self.external_probabilistic_candidate_nodes)
        )
        other.external_probabilistic_sketch_state = (
            None if self.external_probabilistic_sketch_state is None
            else dict(self.external_probabilistic_sketch_state)
        )
        other.selector.bin_edges = None if self.selector.bin_edges is None else np.asarray(self.selector.bin_edges).copy()
        other.distance_cache.shortest_path_cache = dict(self.distance_cache.shortest_path_cache)
        return other

    def set_error_ema(self, error_ema):
        if error_ema is None:
            self.error_ema = None
            return
        arr = np.asarray(error_ema, dtype=float).reshape(-1)
        self.error_ema = arr

    def set_runtime_context(self, epoch=None):
        self.current_epoch = epoch

    def _use_mdl_objective(self):
        objective = self.maintenance_objective.lower()
        return "mdl" in objective or "ipac" in objective or "sublinear" in objective

    def _refresh_mdl_sketch(self, data, adj=None):
        if self.sketch_mdl_enabled:
            self.st_sketch.update(data)
            self.stats.sketch_mdl_state = self.st_sketch.compact_state()
        if self.dynamic_forest_enabled and adj is not None:
            self.dynamic_forest.set_graph(adj)
            self.stats.dynamic_forest_state = self.dynamic_forest.compact_state()

    def _refresh_boundary_index(self, snapshot, adj):
        if not self.boundary_index_enabled or snapshot is None or adj is None:
            self.stats.boundary_index_enabled = False
            self.stats.boundary_index_state = {}
            return
        self.boundary_index.rebuild(snapshot, adj)
        self.stats.boundary_index_enabled = True
        self.stats.boundary_index_state = self.boundary_index.compact_state()
        self.stats.boundary_index_candidate_hits = int(self.boundary_index.candidate_hits)
        self.stats.boundary_index_zero_traversal_skips = int(self.boundary_index.zero_traversal_skips)

    def _sync_boundary_index_stats(self):
        if not self.boundary_index_enabled:
            return
        self.stats.boundary_index_state = self.boundary_index.compact_state()
        self.stats.boundary_index_candidate_hits = int(self.boundary_index.candidate_hits)
        self.stats.boundary_index_zero_traversal_skips = int(self.boundary_index.zero_traversal_skips)

    def _indexed_candidate_clusters_for_node(self, node, clusters):
        if not self.boundary_index_enabled or not self.boundary_index_restrict_candidate_hosts:
            return None
        candidates = self.boundary_index.candidate_clusters_for_node(node)
        candidates = [cid for cid in candidates if 0 <= int(cid) < len(clusters)]
        if self.boundary_index_candidate_limit is not None:
            candidates = candidates[:self.boundary_index_candidate_limit]
        self._sync_boundary_index_stats()
        return candidates

    def _probabilistic_candidates(self, snapshot, scores, protected_vertex_set):
        if not self.probabilistic_sketch_enabled or not self.sketch_mdl_enabled:
            self.stats.probabilistic_candidate_nodes = []
            self.stats.probabilistic_candidate_groups = []
            return None, None

        if self.external_probabilistic_candidate_nodes is not None:
            candidate_nodes = set(int(v) for v in self.external_probabilistic_candidate_nodes)
            state = dict(self.external_probabilistic_sketch_state or {})
            state.setdefault("enabled", True)
            state.setdefault("source", "external_tensorized_zero_copy")
            state.setdefault("candidate_count", len(candidate_nodes))
        else:
            candidates, estimates = self.prob_sketch.update_from_state(
                self.st_sketch,
                scores=scores,
                protected=protected_vertex_set,
            )
            candidate_nodes = set(int(v) for v in candidates)
            state = self.prob_sketch.compact_state()

        for v in protected_vertex_set:
            candidate_nodes.add(int(v))

        candidate_groups = set()
        for gid, unit in enumerate(snapshot.groups):
            if any(int(v) in candidate_nodes for v in unit):
                candidate_groups.add(int(gid))

        self.stats.probabilistic_sketch_state = state
        self.stats.probabilistic_error_bound = {
            "epsilon_bound": state.get("epsilon_bound", 0.0),
            "delta_bound": state.get("delta_bound", 0.0),
        }
        self.stats.probabilistic_candidate_nodes = sorted(candidate_nodes)
        self.stats.probabilistic_candidate_groups = sorted(candidate_groups)
        return candidate_nodes, candidate_groups

    def _normalize_score(self, score):
        score = np.asarray(score, dtype=float)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(np.min(score)) if score.size else 0.0
        hi = float(np.max(score)) if score.size else 0.0
        if hi - lo < self.eps:
            return np.zeros_like(score, dtype=float)
        return (score - lo) / (hi - lo + self.eps)

    def _proxy_error_score(self, data, hx_score):
        data = _as_tnf(data)
        x = data[:, :, 0].astype(float)
        if x.shape[0] <= 1:
            persistence = np.zeros(x.shape[1], dtype=float)
        else:
            persistence = np.abs(x[-1] - x[-2]) / (np.abs(x[-2]) + self.eps)
        variance = np.var(x, axis=0)
        std = np.std(x, axis=0)
        deviation = np.abs(x[-1] - np.mean(x, axis=0)) / (std + self.eps)
        hx = self._normalize_score(hx_score)

        proxy = (
            0.35 * self._normalize_score(persistence)
            + 0.30 * self._normalize_score(variance)
            + 0.20 * self._normalize_score(deviation)
            + 0.15 * hx
        )
        return proxy

    def _split_protected_quota(self, key_budget):
        protected_k = int(round(key_budget * self.error_key_ratio))
        protected_k = max(0, min(key_budget - 1 if key_budget > 1 else 0, protected_k))
        if protected_k <= 0:
            return 0, 0

        epoch = self.current_epoch
        if epoch is None or epoch <= self.model_error_warmup_epochs or self.error_ema is None:
            model_ratio = 0.0
        elif epoch <= self.model_error_full_epochs:
            model_ratio = min(0.10, self.error_key_ratio)
        else:
            model_ratio = min(0.20, self.error_key_ratio)

        model_k = int(round(key_budget * model_ratio))
        model_k = max(0, min(protected_k, model_k))
        proxy_k = protected_k - model_k
        return proxy_k, model_k

    def _select_key_nodes(self, data, include_error_protected=False):
        data = _as_tnf(data)
        _, N, _ = data.shape
        k = max(1, min(N, int(np.floor(self.theta * N))))
        scores = self.selector.compute_scores(data)
        score_order = np.argsort(-scores, kind="mergesort")
        threshold = float(scores[score_order[k - 1]])

        proxy_k, model_k = self._split_protected_quota(k)
        if (not include_error_protected) or self.error_key_ratio <= 0 or (proxy_k + model_k) <= 0:
            key_nodes = sorted(int(i) for i in score_order[:k])
            self.stats.error_protected = []
            self.stats.proxy_protected = []
            self.stats.model_error_protected = []
            self.stats.key_protection_mode = "hxr_only_keys"
        else:
            base_k = max(1, k - proxy_k - model_k)
            base = [int(i) for i in score_order[:base_k]]
            used = set(base)

            proxy = self._proxy_error_score(data, scores)
            proxy_order = np.argsort(-proxy, kind="mergesort")
            proxy_protected = []
            for node in proxy_order:
                node = int(node)
                if node in used:
                    continue
                proxy_protected.append(node)
                used.add(node)
                if len(proxy_protected) >= proxy_k:
                    break

            model_protected = []
            if model_k > 0 and self.error_ema is not None and len(self.error_ema) == N:
                err = np.asarray(self.error_ema, dtype=float)
                err = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
                err_order = np.argsort(-err, kind="mergesort")
                for node in err_order:
                    node = int(node)
                    if node in used:
                        continue
                    model_protected.append(node)
                    used.add(node)
                    if len(model_protected) >= model_k:
                        break

            keys = base + proxy_protected + model_protected
            for node in score_order:
                node = int(node)
                if len(keys) >= k:
                    break
                if node not in used:
                    keys.append(node)
                    used.add(node)
            key_nodes = sorted(keys[:k])
            self.stats.proxy_protected = sorted(proxy_protected)
            self.stats.model_error_protected = sorted(model_protected)
            self.stats.error_protected = sorted(set(proxy_protected) | set(model_protected))
            self.stats.key_protection_mode = f"hxr_proxy_model:{base_k}/{proxy_k}/{model_k}"

        key_set = set(key_nodes)
        non_key_nodes = [i for i in range(N) if i not in key_set]
        return key_nodes, non_key_nodes, scores, threshold

    def _select_protected_vertices(self, data, scores, key_nodes):
        data = _as_tnf(data)
        _, N, _ = data.shape
        key_budget = max(1, min(N, int(np.floor(self.theta * N))))
        protected_k = int(round(key_budget * self.error_key_ratio))
        protected_k = max(0, min(N - len(key_nodes), protected_k))
        if protected_k <= 0:
            self.stats.protected_vertices = []
            self.stats.error_protected = []
            self.stats.proxy_protected = []
            self.stats.model_error_protected = []
            self.stats.key_protection_mode = "hxr_only_keys_no_protected_vertices"
            return set()

        proxy_k, model_k = self._split_protected_quota(key_budget)
        protected_k = min(protected_k, proxy_k + model_k)
        key_set = set(map(int, key_nodes))
        used = set(key_set)

        proxy = self._proxy_error_score(data, scores)
        proxy_order = np.argsort(-proxy, kind="mergesort")
        proxy_protected = []
        for node in proxy_order:
            node = int(node)
            if node in used:
                continue
            proxy_protected.append(node)
            used.add(node)
            if len(proxy_protected) >= proxy_k:
                break

        model_protected = []
        if model_k > 0 and self.error_ema is not None and len(self.error_ema) == N:
            err = np.nan_to_num(np.asarray(self.error_ema, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            err_order = np.argsort(-err, kind="mergesort")
            for node in err_order:
                node = int(node)
                if node in used:
                    continue
                model_protected.append(node)
                used.add(node)
                if len(model_protected) >= model_k:
                    break

        protected = sorted(set(proxy_protected) | set(model_protected))
        for node in protected:
            self.protected_counts[node] = self.protected_counts.get(node, 0) + 1
        for node in list(self.protected_counts):
            if node not in protected:
                self.protected_counts[node] -= 1
                if self.protected_counts[node] <= 0:
                    del self.protected_counts[node]

        self.stats.protected_vertices = protected
        self.stats.error_protected = protected
        self.stats.proxy_protected = sorted(proxy_protected)
        self.stats.model_error_protected = sorted(model_protected)
        self.stats.key_protection_mode = f"hxr_keys_with_protected_vertices:{len(key_nodes)}/{len(proxy_protected)}/{len(model_protected)}"
        return set(protected)

    def _unit_key(self, unit):
        return tuple(sorted(int(v) for v in unit))

    def _smp(self, data, unit):
        unit = list(unit)
        return float(np.mean(data[:, unit, 0]))

    def _vertex_smp(self, data, v):
        return float(np.mean(data[:, int(v), 0]))

    def _residual(self, data, unit):
        unit = list(unit)
        if len(unit) <= 1:
            return 0.0
        x = data[:, unit, 0]
        mu = x.mean(axis=1, keepdims=True)
        return float(np.mean((x - mu) ** 2))

    def _mdl_code_len(self, value):
        return float(np.log2(1.0 + max(1.0, float(value))))

    def _mdl_data_cost(self, data, nodes):
        nodes = sorted(set(int(v) for v in nodes))
        if len(nodes) <= 1:
            return 0.0
        if self.sketch_mdl_enabled:
            sketch_cost = self.st_sketch.data_cost(nodes)
            self.stats.sketch_mdl_state = self.st_sketch.compact_state()
            if sketch_cost is not None:
                return float(sketch_cost)
        data = _as_tnf(data)
        x = data[:, nodes, 0].astype(float)
        mu = x.mean(axis=1, keepdims=True)
        sse = float(np.sum((x - mu) ** 2))
        denom = max(1, x.shape[0] * x.shape[1])
        return 0.5 * denom * float(np.log2(1.0 + sse / (denom + self.eps)))

    def _mdl_edge_count(self, groups, adj):
        A = np.asarray(adj)
        edges = 0
        cleaned = [sorted(set(int(v) for v in g)) for g in groups if g]
        for i, gi in enumerate(cleaned):
            for j in range(i + 1, len(cleaned)):
                gj = cleaned[j]
                if np.any(A[np.ix_(gi, gj)] > 0) or np.any(A[np.ix_(gj, gi)] > 0):
                    edges += 1
        return edges

    def _mdl_structure_cost(self, data, groups, adj):
        groups = [sorted(set(int(v) for v in g)) for g in groups if g]
        if not groups:
            return 0.0
        data = _as_tnf(data)
        num_nodes = max(1, data.shape[1])
        token_bits = len(groups) * self._mdl_code_len(num_nodes)
        membership_bits = sum(len(g) for g in groups) * self._mdl_code_len(len(groups)) / max(1.0, self._mdl_code_len(num_nodes))
        edge_bits = self._mdl_edge_count(groups, adj) * self._mdl_code_len(len(groups))
        return float(token_bits + membership_bits + edge_bits)

    def _mdl_plan_cost(self, data, groups, adj):
        groups = [sorted(set(int(v) for v in g)) for g in groups if g]
        return float(
            sum(self._mdl_data_cost(data, g) for g in groups)
            + self._mdl_structure_cost(data, groups, adj)
        )

    def _mdl_split_candidate(self, data, cluster, adj):
        cluster = sorted(set(int(v) for v in cluster))
        if len(cluster) <= 1:
            return [], float("inf")
        connected = self._connected_components(cluster, adj)
        if len(connected) > 1:
            candidate = connected
        else:
            if len(cluster) < 2 * self.min_cluster_size:
                return [cluster], float("inf")
            candidate = topology_constrained_ward(
                data,
                cluster,
                adj,
                2,
                lambda_spatial=self.lambda_spatial,
                one_hop_only=True,
                dtw_radius=self.dtw_radius,
                cache=self.distance_cache,
            )
            repaired = []
            for c in candidate:
                repaired.extend(self._connected_components(c, adj))
            candidate = repaired
        old_cost = self._mdl_plan_cost(data, [cluster], adj)
        new_cost = self._mdl_plan_cost(data, candidate, adj)
        return candidate, float(new_cost - old_cost)

    def _mdl_extract_delta(self, data, vertex, parent_cluster, adj):
        vertex = int(vertex)
        parent = sorted(set(int(v) for v in parent_cluster))
        if vertex not in parent or len(parent) <= 1:
            return float("inf"), [parent]
        remaining = [v for v in parent if v != vertex]
        candidate = self._connected_components(remaining, adj) + [[vertex]]
        old_cost = self._mdl_plan_cost(data, [parent], adj)
        new_cost = self._mdl_plan_cost(data, candidate, adj)
        return float(new_cost - old_cost), candidate

    def _mdl_group_merge_delta(self, data, a, b, adj):
        a = sorted(set(int(v) for v in a))
        b = sorted(set(int(v) for v in b))
        if not a or not b or not self._has_boundary_edge(a, b, adj):
            return float("inf")
        merged = sorted(set(a) | set(b))
        if not self._is_connected(merged, adj):
            return float("inf")
        old_cost = self._mdl_plan_cost(data, [a, b], adj)
        new_cost = self._mdl_plan_cost(data, [merged], adj)
        return float(new_cost - old_cost)

    def _best_mdl_merge_host(self, data, node, clusters, adj):
        best = (float("inf"), None)
        indexed_cids = self._indexed_candidate_clusters_for_node(node, clusters)
        if indexed_cids is None:
            search_cids = list(range(len(clusters)))
        else:
            search_cids = indexed_cids
        for cid in search_cids:
            cluster = clusters[cid]
            delta = self._mdl_group_merge_delta(data, [node], cluster, adj)
            if delta < best[0]:
                best = (delta, cid)
        if indexed_cids is not None and best[1] is None:
            indexed_set = set(indexed_cids)
            for cid, cluster in enumerate(clusters):
                if cid in indexed_set:
                    continue
                delta = self._mdl_group_merge_delta(data, [node], cluster, adj)
                if delta < best[0]:
                    best = (delta, cid)
        return best

    def _enforce_key_anchor_budget(self, data, key_nodes, clusters, scores, adj):
        key_nodes = sorted(set(int(v) for v in key_nodes))
        clusters = [sorted(set(int(v) for v in c)) for c in clusters if c]
        target = max(1, min(data.shape[1], int(np.floor(self.theta * data.shape[1]))))
        if len(key_nodes) >= target:
            return key_nodes, self._split_connected_clusters(clusters, adj)

        key_set = set(key_nodes)
        order = [int(v) for v in np.argsort(-np.asarray(scores), kind="mergesort")]
        promoted = []
        for node in order:
            if len(key_set) >= target:
                break
            if node in key_set:
                continue
            key_set.add(node)
            promoted.append(node)

        if not promoted:
            return sorted(key_set), self._split_connected_clusters(clusters, adj)

        promoted_set = set(promoted)
        repaired = []
        for cluster in clusters:
            remain = [v for v in cluster if v not in promoted_set]
            repaired.extend(self._connected_components(remain, adj))
        return sorted(key_set), self._split_connected_clusters(repaired, adj)

    def _enforce_cluster_budget(self, data, clusters, adj):
        clusters = self._split_connected_clusters(clusters, adj)
        budget = max(1, int(self.n_clusters))
        if len(clusters) <= budget:
            return clusters
        nodes = sorted(set(v for c in clusters for v in c))
        if len(nodes) <= budget:
            return [[v] for v in nodes]
        return self._split_connected_clusters(topology_constrained_ward(
            data,
            nodes,
            adj,
            budget,
            lambda_spatial=self.lambda_spatial,
            one_hop_only=True,
            dtw_radius=self.dtw_radius,
            cache=self.distance_cache,
        ), adj)

    def _local_hgraph_mdl_cost(self, data, key_nodes, clusters, local_nodes, adj):
        local_set = set(int(v) for v in local_nodes)
        groups = [[int(k)] for k in key_nodes if int(k) in local_set]
        groups.extend([list(c) for c in clusters if local_set & set(map(int, c))])
        return self._mdl_plan_cost(data, groups, adj)

    def _calibrate(self, data, snapshot):
        if self.stats.delta_s is None:
            self.stats.delta_s = float(np.quantile(snapshot.scores, 0.50))

        residuals = [self._residual(data, c) for c in snapshot.clusters if len(c) > 1]
        if self.stats.delta_e is None:
            self.stats.delta_e = float(np.quantile(residuals, 0.90)) if residuals else 0.0

        values = []
        for c in snapshot.clusters:
            if len(c) <= 1:
                continue
            mu = self._smp(data, c)
            values.extend([abs(self._vertex_smp(data, v) - mu) for v in c])
        if self.stats.delta_d is None:
            self.stats.delta_d = float(np.quantile(values, 0.90)) if values else self.eta

        if self.stats.delta_c is None:
            self.stats.delta_c = max(self.stats.delta_d, self.eta)

    def _update_aux(self, data, snapshot):
        for unit in snapshot.groups:
            key = self._unit_key(unit)
            mu = self._smp(data, unit)

            if key not in self.stats.stable_ranges:
                self.stats.stable_ranges[key] = (mu - self.eta, mu + self.eta)
            else:
                lo, hi = self.stats.stable_ranges[key]
                self.stats.stable_ranges[key] = (
                    min(lo, (lo + mu) / 2.0) - self.eta,
                    max(hi, (hi + mu) / 2.0) + self.eta,
                )

            self.stats.last_smp[key] = mu
            self.stats.residuals[key] = self._residual(data, unit)

    def _compact_clusters(self, data, clusters, adj):
        clusters = self._split_connected_clusters(clusters, adj)
        if self.min_cluster_size <= 1 or len(clusters) <= 1:
            return clusters

        A = np.asarray(adj)
        changed = True
        while changed:
            changed = False
            small_ids = [i for i, c in enumerate(clusters) if len(c) < self.min_cluster_size]
            if not small_ids or len(clusters) <= 1:
                break
            i = small_ids[0]
            source = clusters[i]
            best_j = None
            best_score = float("inf")
            for j, target in enumerate(clusters):
                if i == j:
                    continue
                merged = sorted(set(source) | set(target))
                if len(merged) > self.max_cluster_size:
                    continue
                connected = np.any(A[np.ix_(source, target)] > 0) or np.any(A[np.ix_(target, source)] > 0)
                if not connected:
                    continue
                score = self._residual(data, merged)
                if self.stats.delta_e is not None and score > self.stats.delta_e * 1.1:
                    continue
                if score < best_score:
                    best_j = j
                    best_score = score

            if best_j is None:
                break

            clusters[best_j] = sorted(set(clusters[best_j]) | set(source))
            del clusters[i]
            changed = True
        return self._split_connected_clusters(clusters, adj)

    def _connected_components(self, nodes, adj):
        nodes = sorted(set(int(v) for v in nodes))
        if not nodes:
            return []
        if self.dynamic_forest_enabled:
            self.dynamic_forest.set_graph(adj)
            comps = self.dynamic_forest.components(nodes)
            self.stats.dynamic_forest_state = self.dynamic_forest.compact_state()
            return comps
        node_set = set(nodes)
        A = np.asarray(adj)
        seen = set()
        comps = []
        for start in nodes:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                neigh = np.where((A[u] > 0) | (A[:, u] > 0))[0]
                for v in neigh:
                    v = int(v)
                    if v in node_set and v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(sorted(comp))
        return comps

    def _is_connected(self, nodes, adj):
        nodes = sorted(set(int(v) for v in nodes))
        return len(nodes) <= 1 or len(self._connected_components(nodes, adj)) == 1

    def _has_boundary_edge(self, a, b, adj):
        a = sorted(set(int(v) for v in a))
        b = sorted(set(int(v) for v in b))
        if not a or not b:
            return False
        A = np.asarray(adj)
        return bool(np.any(A[np.ix_(a, b)] > 0) or np.any(A[np.ix_(b, a)] > 0))

    def _split_connected_clusters(self, clusters, adj):
        repaired = []
        splits = 0
        for cluster in clusters:
            cluster = sorted(set(int(v) for v in cluster))
            if not cluster:
                continue
            comps = self._connected_components(cluster, adj)
            if len(comps) > 1:
                splits += len(comps) - 1
            repaired.extend(comps)
        self.stats.connectivity_repairs += splits
        return repaired

    def _incompatible_vertices(self, data, snapshot, scores, protected_key_set, adj=None, candidate_nodes=None):
        data = _as_tnf(data)
        promote_threshold = float(snapshot.key_threshold) * self.promote_margin
        bad = set()
        candidate_nodes = None if candidate_nodes is None else set(int(v) for v in candidate_nodes)
        for cluster in snapshot.clusters:
            if len(cluster) <= 1:
                continue
            mu = self._smp(data, cluster)
            deviations = [(int(v), abs(self._vertex_smp(data, v) - mu)) for v in cluster]
            for v, dev in deviations:
                if candidate_nodes is not None and v not in candidate_nodes and v not in protected_key_set:
                    continue
                mdl_extract = False
                if self._use_mdl_objective() and adj is not None:
                    delta_l, _ = self._mdl_extract_delta(data, v, cluster, adj)
                    mdl_extract = delta_l < 0
                if mdl_extract or dev > self.stats.delta_d or scores[v] >= promote_threshold or v in protected_key_set:
                    bad.add(v)
        return bad

    def _vertex_local_block(self, data, vertex, parent_cluster, adj):
        data = _as_tnf(data)
        vertex = int(vertex)
        parent = sorted(set(int(v) for v in parent_cluster))
        if self.boundary_index_enabled and self.boundary_index_skip_internal_connectivity:
            indexed = self.boundary_index.zero_traversal_block(vertex, parent)
            if indexed is not None:
                self._sync_boundary_index_stats()
                return indexed
        A = np.asarray(adj)
        v_smp = self._vertex_smp(data, vertex)
        parent_mu = self._smp(data, parent)
        block = {vertex}
        for u in parent:
            if u == vertex:
                continue
            if A[vertex, u] <= 0 and A[u, vertex] <= 0:
                continue
            u_smp = self._vertex_smp(data, u)
            coupled = abs(u_smp - v_smp) <= max(self.stats.delta_d, self.eta)
            locally_unstable = abs(u_smp - parent_mu) > 0.5 * max(self.stats.delta_d, self.eta)
            if coupled or locally_unstable:
                block.add(int(u))
        return sorted(block)

    def _repair_unit_patch_entry(self, gids, groups, group_repair_nodes, group_blocks, condensed_neigh, adj):
        gids = sorted(set(int(g) for g in gids))
        repair_nodes = sorted(set().union(*(set(group_repair_nodes.get(g, set())) for g in gids)))
        old_members = sorted(set().union(*(set(int(v) for v in groups[g]) for g in gids)))
        boundary_neighbors = sorted(
            set().union(*(set(int(n) for n in condensed_neigh.get(g, set())) for g in gids)) - set(gids)
        )
        internal_route_edges = 0
        for i, gid in enumerate(gids):
            for nid in gids[i + 1:]:
                if int(nid) in condensed_neigh.get(int(gid), set()):
                    internal_route_edges += 1

        blocks = []
        for gid in gids:
            blocks.extend(group_blocks.get(gid, []))
        if not blocks and repair_nodes:
            blocks = [repair_nodes]
        ward_required = any(len(set(int(v) for v in block)) > self.max_cluster_size for block in blocks)
        membership_rows = len(repair_nodes)
        pooling_rows = len(repair_nodes)
        expansion_rows = len(repair_nodes)
        condensed_edges = len(boundary_neighbors)
        changed_route_entries = membership_rows + pooling_rows + expansion_rows + 2 * condensed_edges
        if ward_required:
            changed_route_entries += max(1, len(repair_nodes))

        return {
            "gids": gids,
            "old_members": old_members,
            "repair_nodes": repair_nodes,
            "stable_core": sorted(set(old_members) - set(repair_nodes)),
            "boundary_neighbors": boundary_neighbors,
            "internal_route_edges": int(internal_route_edges),
            "membership_rows": int(membership_rows),
            "condensed_edges": int(condensed_edges),
            "pooling_rows": int(pooling_rows),
            "expansion_rows": int(expansion_rows),
            "changed_route_entries": int(changed_route_entries),
            "ward_required": bool(ward_required),
            "blocks": [sorted(set(int(v) for v in b)) for b in blocks],
        }

    def _build_repair_unit_index(self, groups, key_count, local_gids, group_repair_nodes, group_blocks, condensed_neigh, adj, affected, secondary):
        entries = []
        for gid in sorted(set(int(g) for g in local_gids)):
            unit = list(map(int, groups[gid]))
            repair_nodes = sorted(set(int(v) for v in group_repair_nodes.get(gid, set())))
            boundary_vertices = []
            for node in unit:
                external = False
                for nid in condensed_neigh.get(gid, set()):
                    if self._has_boundary_edge([node], groups[int(nid)], adj):
                        external = True
                        break
                if external:
                    boundary_vertices.append(int(node))
            patch_entry = self._repair_unit_patch_entry(
                [gid],
                groups,
                group_repair_nodes,
                group_blocks,
                condensed_neigh,
                adj,
            )
            entries.append({
                "gid": int(gid),
                "role": "key" if gid < key_count else "non_key",
                "membership": {
                    "nodes": unit,
                    "stable_core": sorted(set(unit) - set(repair_nodes)),
                    "unstable_fringe": repair_nodes,
                    "affected": bool(gid in affected),
                    "secondary": bool(gid in secondary),
                },
                "boundary": {
                    "neighbors": sorted(int(n) for n in condensed_neigh.get(gid, set())),
                    "boundary_vertices": boundary_vertices,
                    "connected": bool(self._is_connected(unit, adj)),
                    "zero_traversal_candidate": bool(
                        self.boundary_index_enabled
                        and all(self.boundary_index.is_internal(v) for v in repair_nodes)
                    ) if repair_nodes else False,
                },
                "patch": patch_entry,
            })

        return entries

    def _patch_saving_decision(self, data, groups, root_a, root_b, group_repair_nodes, group_blocks, condensed_neigh, adj):
        gids_a = sorted(set(int(g) for g in root_a))
        gids_b = sorted(set(int(g) for g in root_b))
        if not gids_a or not gids_b or set(gids_a) & set(gids_b):
            return False, {"reason": "same_repair_unit"}

        nodes_a = sorted(set().union(*(set(groups[g]) for g in gids_a)))
        nodes_b = sorted(set().union(*(set(groups[g]) for g in gids_b)))
        route_coupled = self._has_boundary_edge(nodes_a, nodes_b, adj)
        if not route_coupled:
            return False, {"reason": "not_route_coupled"}

        merged_old_members = sorted(set(nodes_a) | set(nodes_b))
        if not self._is_connected(merged_old_members, adj):
            return False, {"reason": "merged_candidate_disconnected"}

        behavior_delta = self._mdl_group_merge_delta(data, nodes_a, nodes_b, adj) if self._use_mdl_objective() else -1.0
        behavior_compatible = behavior_delta < 0
        if not behavior_compatible:
            return False, {
                "reason": "mdl_incompatible",
                "delta_l": float(behavior_delta),
            }

        patch_a = self._repair_unit_patch_entry(gids_a, groups, group_repair_nodes, group_blocks, condensed_neigh, adj)
        patch_b = self._repair_unit_patch_entry(gids_b, groups, group_repair_nodes, group_blocks, condensed_neigh, adj)
        patch_joint = self._repair_unit_patch_entry(
            sorted(set(gids_a) | set(gids_b)),
            groups,
            group_repair_nodes,
            group_blocks,
            condensed_neigh,
            adj,
        )
        separate_cost = int(patch_a["changed_route_entries"] + patch_b["changed_route_entries"])
        joint_cost = int(patch_joint["changed_route_entries"])
        if patch_joint["ward_required"] and not (patch_a["ward_required"] or patch_b["ward_required"]):
            return False, {
                "reason": "joint_requires_extra_ward",
                "separate_cost": separate_cost,
                "joint_cost": joint_cost,
                "delta_l": float(behavior_delta),
            }
        if joint_cost >= separate_cost:
            return False, {
                "reason": "no_patch_saving",
                "separate_cost": separate_cost,
                "joint_cost": joint_cost,
                "delta_l": float(behavior_delta),
            }

        return True, {
            "reason": "patch_saving_admit",
            "separate_cost": separate_cost,
            "joint_cost": joint_cost,
            "saved_route_entries": int(separate_cost - joint_cost),
            "delta_l": float(behavior_delta),
            "joint_patch": patch_joint,
        }

    def _coalesce_update_components(self, data, snapshot, adj, affected, secondary, protected_key_set, scores):
        groups = snapshot.groups
        key_count = len(snapshot.key_nodes)
        local_gids = sorted(set(affected) | set(secondary))
        if not local_gids:
            self.stats.repair_unit_index_enabled = bool(self.repair_unit_index_enabled)
            self.stats.repair_unit_index_entries = []
            self.stats.repair_unit_index_state = {"enabled": bool(self.repair_unit_index_enabled), "entries": 0}
            self.stats.patch_saving_coalescing_enabled = bool(self.patch_saving_coalescing_enabled)
            self.stats.patch_saving_merges = []
            self.stats.patch_saving_rejections = []
            self.stats.patch_saving_coalescing_state = {
                "enabled": bool(self.patch_saving_coalescing_enabled),
                "merge_count": 0,
                "rejection_count": 0,
            }
            return []

        condensed_neigh = self._condensed_neighbors(groups, adj)
        incompatible = self._incompatible_vertices(data, snapshot, scores, protected_key_set, adj=adj)
        group_repair_nodes = {gid: set() for gid in local_gids}
        group_blocks = {gid: [] for gid in local_gids}

        for gid in local_gids:
            unit = list(map(int, groups[gid]))
            if gid < key_count:
                if gid in affected:
                    group_repair_nodes[gid].update(unit)
                    group_blocks[gid].append(unit)
                continue

            bad = sorted(v for v in unit if v in incompatible)
            if bad:
                for v in bad:
                    block = self._vertex_local_block(data, v, unit, adj)
                    group_repair_nodes[gid].update(block)
                    group_blocks[gid].append(block)
            elif gid in affected:
                # Group-level instability without a single dominant vertex: repair the whole connected token.
                comps = self._connected_components(unit, adj)
                for comp in comps:
                    group_repair_nodes[gid].update(comp)
                    group_blocks[gid].append(comp)

        if self.repair_unit_index_enabled:
            self.stats.repair_unit_index_enabled = True
            self.stats.repair_unit_index_entries = self._build_repair_unit_index(
                groups,
                key_count,
                local_gids,
                group_repair_nodes,
                group_blocks,
                condensed_neigh,
                adj,
                affected,
                secondary,
            )
            self.stats.repair_unit_index_state = {
                "enabled": True,
                "mode": "repair_unit_index",
                "entries": int(len(self.stats.repair_unit_index_entries)),
                "membership_entry": "key/non-key role + stable-core/unstable-fringe membership",
                "boundary_entry": "one-hop raw/condensed boundary and connectivity metadata",
                "patch_entry": "membership + condensed-edge + pooling + expansion invalidation estimate",
                "old_membership_first": True,
            }
        else:
            self.stats.repair_unit_index_enabled = False
            self.stats.repair_unit_index_entries = []
            self.stats.repair_unit_index_state = {"enabled": False}

        parent = {gid: gid for gid in local_gids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        patch_merges = []
        patch_rejections = []
        for gid in local_gids:
            for nid in condensed_neigh.get(gid, set()):
                if nid not in parent:
                    continue
                route_coupled = self._has_boundary_edge(groups[gid], groups[nid], adj)
                sani_coupled = (gid in affected and nid in secondary) or (nid in affected and gid in secondary)
                both_need_repair = bool(group_repair_nodes[gid]) and bool(group_repair_nodes[nid])
                if route_coupled and (sani_coupled or both_need_repair):
                    if self.patch_saving_coalescing_enabled:
                        if find(gid) == find(nid):
                            continue
                        root_a = [g for g in local_gids if find(g) == find(gid)]
                        root_b = [g for g in local_gids if find(g) == find(nid)]
                        admit, decision = self._patch_saving_decision(
                            data,
                            groups,
                            root_a,
                            root_b,
                            group_repair_nodes,
                            group_blocks,
                            condensed_neigh,
                            adj,
                        )
                        decision.update({
                            "left_gids": sorted(set(int(g) for g in root_a)),
                            "right_gids": sorted(set(int(g) for g in root_b)),
                            "edge": [int(gid), int(nid)],
                            "sani_coupled": bool(sani_coupled),
                            "both_need_repair": bool(both_need_repair),
                        })
                        if admit:
                            union(gid, nid)
                            patch_merges.append(decision)
                        else:
                            patch_rejections.append(decision)
                    else:
                        union(gid, nid)

        self.stats.patch_saving_coalescing_enabled = bool(self.patch_saving_coalescing_enabled)
        self.stats.patch_saving_merges = patch_merges
        self.stats.patch_saving_rejections = patch_rejections[:64]
        self.stats.patch_saving_coalescing_state = {
            "enabled": bool(self.patch_saving_coalescing_enabled),
            "mode": "patch_saving_coalescing" if self.patch_saving_coalescing_enabled else "legacy_route_coupled_union",
            "merge_count": int(len(patch_merges)),
            "rejection_count": int(len(patch_rejections)),
            "rule": (
                "merge iff joint patch invalidates fewer route entries than separate patches, "
                "preserves connected old members, is MDL-compatible, and avoids extra Ward"
            ),
            "old_membership_first": True,
            "cross_boundary_second": True,
        }

        by_root = {}
        for gid in local_gids:
            by_root.setdefault(find(gid), []).append(gid)

        components = []
        for gids in by_root.values():
            gids = sorted(gids)
            repair_nodes = sorted(set().union(*(group_repair_nodes[g] for g in gids)))
            if not repair_nodes:
                continue
            blocks = []
            for g in gids:
                blocks.extend(group_blocks[g])
            components.append({
                "gids": gids,
                "repair_nodes": repair_nodes,
                "candidate_gids": sorted(set(gids) | set().union(*(condensed_neigh.get(g, set()) for g in gids))),
                "internal_blocks": [sorted(set(b)) for b in blocks],
            })

        self.stats.update_components = [
            {
                "gids": c["gids"],
                "repair_nodes": c["repair_nodes"],
                "candidate_gids": c["candidate_gids"],
            }
            for c in components
        ]
        self.stats.update_component_count = len(components)
        self.stats.internal_update_blocks = [b for c in components for b in c["internal_blocks"]]
        self.stats.component_update_mode = (
            "repair_unit_index_patch_saving_coalescing"
            if self.patch_saving_coalescing_enabled
            else "connectivity_preserving_sani_closed_coalescing"
        )
        self.stats.ipac_core_fringe = [
            {
                "candidate_gids": c["candidate_gids"],
                "unstable_fringe": c["repair_nodes"],
                "stable_core_reused": sorted(
                    set().union(*(set(groups[g]) for g in c["gids"])) - set(c["repair_nodes"])
                ),
            }
            for c in components
        ]
        return components

    def _recluster_connected_nodes(self, data, nodes, adj, cluster_budget=None):
        nodes = sorted(set(int(v) for v in nodes))
        if not nodes:
            return []
        out = []
        comps = self._connected_components(nodes, adj)
        for comp in comps:
            if len(comp) <= self.max_cluster_size:
                out.append(comp)
                continue
            by_size = max(1, int(np.ceil(len(comp) / max(1, self.max_cluster_size))))
            if cluster_budget is not None:
                by_size = min(by_size, max(1, int(cluster_budget)))
            k = max(1, min(len(comp), by_size))
            out.extend(topology_constrained_ward(
                data,
                comp,
                adj,
                k,
                lambda_spatial=self.lambda_spatial,
                one_hop_only=True,
                dtw_radius=self.dtw_radius,
                cache=self.distance_cache,
            ))
        return self._split_connected_clusters(out, adj)

    def _zero_clustering_boundary_shift(self, data, nodes, clusters, adj):
        nodes = sorted(set(int(v) for v in nodes))
        clusters = [sorted(set(int(v) for v in c)) for c in clusters if c]
        if not nodes:
            return clusters, []

        A = (np.asarray(adj) > 0).astype(np.uint8)
        cluster_of = {}
        for cid, cluster in enumerate(clusters):
            for node in cluster:
                cluster_of[int(node)] = int(cid)

        moves = []
        unresolved = []
        for node in nodes:
            neigh = np.where((A[node] > 0) | (A[:, node] > 0))[0]
            candidate_cids = []
            source_cid = cluster_of.get(int(node))
            for nb in neigh:
                cid = cluster_of.get(int(nb))
                if cid is not None and cid not in candidate_cids:
                    candidate_cids.append(int(cid))
            candidate_cids = candidate_cids[:self.zcbs_candidate_limit]

            best = (float("inf"), None, None)
            for cid in candidate_cids:
                cluster = clusters[cid]
                if node in cluster:
                    continue
                merged = sorted(set(cluster) | {node})
                if not self._has_boundary_edge([node], cluster, adj):
                    continue
                if not self._is_connected(merged, adj):
                    continue
                delta = self._mdl_group_merge_delta(data, [node], cluster, adj)
                if delta < best[0]:
                    best = (delta, cid, merged)

            delta, cid, merged = best
            if cid is not None and delta < -self.zcbs_min_gain:
                if source_cid is not None and source_cid != cid:
                    clusters[source_cid] = [v for v in clusters[source_cid] if int(v) != int(node)]
                clusters[cid] = merged
                cluster_of[node] = int(cid)
                moves.append({
                    "node": int(node),
                    "source": None if source_cid is None else int(source_cid),
                    "target": int(cid),
                    "delta_l": float(delta),
                })
            else:
                unresolved.append(int(node))

        self.stats.zcbs_moves = moves
        self.stats.zcbs_singletons = sorted(unresolved)
        self.stats.zcbs_state = {
            "enabled": True,
            "mode": "zero_clustering_boundary_shifting",
            "candidate_limit": int(self.zcbs_candidate_limit),
            "promote_unresolved": bool(self.zcbs_promote_unresolved),
            "min_gain": float(self.zcbs_min_gain),
            "fallback_to_ward": bool(self.zcbs_fallback_to_ward),
            "moved": int(len(moves)),
            "unresolved": int(len(unresolved)),
            "complexity": (
                "O(d_v) per shifted vertex; unresolved components fallback to local Ward"
                if self.zcbs_fallback_to_ward
                else "O(d_v) per affected vertex; no online Ward in Case2"
            ),
        }
        return self._split_connected_clusters([c for c in clusters if c], adj), unresolved

    def _apply_joint_demotions(self, data, demoted, clusters, adj):
        demoted = sorted(set(int(v) for v in demoted))
        clusters = self._split_connected_clusters(clusters, adj)
        if not demoted:
            self.stats.joint_demotions = []
            self.stats.singleton_demotions = []
            return clusters

        assignments = {}
        singleton = []
        for node in demoted:
            candidates = []
            indexed_cids = self._indexed_candidate_clusters_for_node(node, clusters)
            if indexed_cids is None:
                search_cids = list(range(len(clusters)))
            else:
                search_cids = indexed_cids
            for cid in search_cids:
                cluster = clusters[cid]
                merged = sorted(set(cluster) | {node})
                if not self._has_boundary_edge([node], cluster, adj):
                    continue
                if not self._is_connected(merged, adj):
                    continue
                residual = self._residual(data, merged)
                candidates.append((residual, cid))
            if indexed_cids is not None and not candidates:
                indexed_set = set(indexed_cids)
                for cid, cluster in enumerate(clusters):
                    if cid in indexed_set:
                        continue
                    merged = sorted(set(cluster) | {node})
                    if not self._has_boundary_edge([node], cluster, adj):
                        continue
                    if not self._is_connected(merged, adj):
                        continue
                    residual = self._residual(data, merged)
                    candidates.append((residual, cid))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                assignments.setdefault(candidates[0][1], []).append(node)
            else:
                singleton.append(node)

        accepted = []
        for cid, nodes in assignments.items():
            merged = sorted(set(clusters[cid]) | set(nodes))
            residual = self._residual(data, merged)
            valid_residual = self.stats.delta_e is None or residual <= self.stats.delta_e * 1.1
            if valid_residual and self._is_connected(merged, adj):
                clusters[cid] = merged
                accepted.append({"target": cid, "nodes": sorted(nodes), "residual": residual})
            else:
                singleton.extend(nodes)

        for node in sorted(set(singleton)):
            clusters.append([node])

        self.stats.joint_demotions = accepted
        self.stats.singleton_demotions = sorted(set(singleton))
        return self._split_connected_clusters(clusters, adj)

    def _route_edges(self, key_nodes, clusters, adj):
        groups = [[int(k)] for k in key_nodes] + [list(map(int, c)) for c in clusters]
        edges = set()
        for i, gi in enumerate(groups):
            for j in range(i + 1, len(groups)):
                if self._has_boundary_edge(gi, groups[j], adj):
                    edges.add((i, j))
        return edges

    def _membership_map(self, key_nodes, clusters):
        mapping = {}
        for k in key_nodes:
            mapping[int(k)] = ("key", int(k))
        for cid, cluster in enumerate(clusters):
            token = ("cluster", tuple(sorted(int(v) for v in cluster)))
            for node in cluster:
                mapping[int(node)] = token
        return mapping

    def _route_invalidation_cost(self, snapshot, new_keys, new_clusters, adj):
        old_membership = self._membership_map(snapshot.key_nodes, snapshot.clusters)
        new_membership = self._membership_map(new_keys, new_clusters)
        all_nodes = sorted(set(old_membership) | set(new_membership))
        changed_members = [n for n in all_nodes if old_membership.get(n) != new_membership.get(n)]

        old_edges = self._route_edges(snapshot.key_nodes, snapshot.clusters, adj)
        new_edges = self._route_edges(new_keys, new_clusters, adj)
        changed_edges = sorted(old_edges ^ new_edges)

        return {
            "changed_membership_rows": len(changed_members),
            "changed_condensed_edges": len(changed_edges),
            "changed_route_entries": len(changed_members) + 2 * len(changed_edges),
            "changed_nodes": changed_members,
            "changed_edges": changed_edges,
        }

    def _local_fit_score(self, data, clusters, nodes):
        node_set = set(int(v) for v in nodes)
        score = 0.0
        for cluster in clusters:
            cluster = sorted(set(int(v) for v in cluster))
            if not cluster or not (node_set & set(cluster)):
                continue
            score += self._residual(data, cluster)
        return float(score)

    def _dry_run_admission(self, data, snapshot, new_keys, new_clusters, local_nodes, adj, promoted, demoted):
        old_clusters = [list(c) for c in snapshot.clusters]
        old_fit = self._local_fit_score(data, old_clusters, local_nodes)
        new_fit = self._local_fit_score(data, new_clusters, local_nodes)
        fit_gain = old_fit - new_fit
        invalidation = self._route_invalidation_cost(snapshot, new_keys, new_clusters, adj)

        relevant_new = [c for c in new_clusters if set(c) & set(local_nodes)]
        all_connected = all(self._is_connected(c, adj) for c in relevant_new)
        residual_ok = True
        if self.stats.delta_e is not None:
            residual_ok = all(len(c) <= 1 or self._residual(data, c) <= self.stats.delta_e * 1.1 for c in relevant_new)

        old_violation = False
        if self.stats.delta_e is not None:
            old_violation = any(len(c) > 1 and self._residual(data, c) > self.stats.delta_e for c in old_clusters if set(c) & set(local_nodes))
        old_disconnected = any(not self._is_connected(c, adj) for c in old_clusters if set(c) & set(local_nodes))
        structural_role_change = bool(promoted or demoted or self.stats.connectivity_repairs > 0)
        changed = invalidation["changed_route_entries"] > 0

        if not changed:
            admitted = False
            reason = "dry_run_no_route_change"
        elif not all_connected:
            admitted = False
            reason = "dry_run_reject_disconnected_nonkey"
        elif not residual_ok:
            admitted = False
            reason = "dry_run_reject_residual_invalid"
        elif fit_gain > 0:
            admitted = True
            reason = "dry_run_admit_residual_gain"
        elif (old_violation or old_disconnected) and structural_role_change:
            admitted = True
            reason = "dry_run_admit_structural_repair"
        else:
            admitted = False
            reason = "dry_run_reject_cost_dominates_gain"

        self.stats.dry_run_admitted = admitted
        self.stats.dry_run_fit_gain = float(fit_gain)
        self.stats.dry_run_old_fit = float(old_fit)
        self.stats.dry_run_new_fit = float(new_fit)
        self.stats.dry_run_reason = reason
        self.stats.dry_run_invalidation_cost = invalidation
        self.stats.merged_route_patch = {
            "changed_membership_rows": invalidation["changed_membership_rows"],
            "changed_condensed_edges": invalidation["changed_condensed_edges"],
            "changed_route_entries": invalidation["changed_route_entries"],
            "component_count": self.stats.update_component_count,
            "mode": "merged_one_shot_route_patch",
        }
        self.stats.route_patch_mode = "copy_on_write_delta_patch"
        return admitted, reason

    def build_initial(self, data, adj):
        data = _as_tnf(data)
        self._refresh_mdl_sketch(data, adj)
        key_nodes, non_key_nodes, scores, threshold = self._select_key_nodes(data)

        if non_key_nodes:
            k = min(max(1, self.n_clusters), len(non_key_nodes))
            clusters = topology_constrained_ward(
                data,
                non_key_nodes,
                adj,
                k,
                lambda_spatial=self.lambda_spatial,
                one_hop_only=True,
                dtw_radius=self.dtw_radius,
                cache=self.distance_cache,
            )
        else:
            clusters = []

        clusters = self._compact_clusters(data, clusters, adj)

        snapshot = HGraphSnapshot(key_nodes, clusters, scores, threshold, version=0)
        snapshot.rebuild_index()
        self._refresh_boundary_index(snapshot, adj)
        self._update_aux(data, snapshot)
        self._calibrate(data, snapshot)
        self.stats.changed_last_update = True
        return snapshot

    def _condensed_neighbors(self, groups, adj):
        if self.boundary_index_enabled and self.boundary_index.version is not None:
            indexed = self.boundary_index.condensed_neighbors(len(groups))
            self._sync_boundary_index_stats()
            return indexed
        A = np.asarray(adj)
        neigh = {i: set() for i in range(len(groups))}
        for i, gi in enumerate(groups):
            for j, gj in enumerate(groups):
                if i == j:
                    continue
                if np.any(A[np.ix_(gi, gj)] > 0) or np.any(A[np.ix_(gj, gi)] > 0):
                    neigh[i].add(j)
        return neigh

    def identify_affected(self, data, snapshot, adj):
        data = _as_tnf(data)
        candidate_key_nodes, _, scores, _ = self._select_key_nodes(data)
        protected_vertex_set = self._select_protected_vertices(data, scores, candidate_key_nodes)
        probabilistic_nodes, probabilistic_groups = self._probabilistic_candidates(
            snapshot,
            scores,
            protected_vertex_set,
        )

        affected = set()
        key_set = set(snapshot.key_nodes)
        groups = snapshot.groups
        candidate_key_set = set(map(int, candidate_key_nodes))
        incompatible = self._incompatible_vertices(
            data,
            snapshot,
            scores,
            protected_vertex_set,
            adj=adj,
            candidate_nodes=probabilistic_nodes,
        )
        mdl_split_units = []
        mdl_extract_vertices = []
        mdl_merge_keys = []

        for gid, unit in enumerate(groups):
            unit_key = self._unit_key(unit)
            mu = self._smp(data, unit)
            lo, hi = self.stats.stable_ranges.get(unit_key, (mu - self.eta, mu + self.eta))
            in_stable = lo <= mu <= hi

            if any((int(v) in protected_vertex_set and int(v) not in key_set) for v in unit):
                affected.add(gid)

            if len(unit) == 1 and unit[0] in key_set:
                v = int(unit[0])
                self.demote_counts[v] = 0
                if self._use_mdl_objective():
                    if probabilistic_groups is not None and gid not in probabilistic_groups:
                        continue
                    merge_delta, host = self._best_mdl_merge_host(data, v, snapshot.clusters, adj)
                    if merge_delta < 0:
                        affected.add(gid)
                        mdl_merge_keys.append({"node": v, "delta_l": float(merge_delta), "host": host})
            else:
                re = self._residual(data, unit)
                member_dev = 0.0
                if len(unit) > 1:
                    member_dev = max(abs(self._vertex_smp(data, v) - mu) for v in unit)
                if self._use_mdl_objective():
                    if probabilistic_groups is not None and gid not in probabilistic_groups:
                        continue
                    split_candidate, split_delta = self._mdl_split_candidate(data, unit, adj)
                    extracted = []
                    for v in unit:
                        extract_delta, _ = self._mdl_extract_delta(data, v, unit, adj)
                        if extract_delta < 0:
                            extracted.append({"node": int(v), "delta_l": float(extract_delta)})
                    if split_delta < 0:
                        affected.add(gid)
                        mdl_split_units.append({
                            "gid": int(gid),
                            "nodes": sorted(int(v) for v in unit),
                            "delta_l": float(split_delta),
                            "parts": [sorted(int(v) for v in c) for c in split_candidate],
                        })
                    if extracted or any(int(v) in incompatible for v in unit):
                        affected.add(gid)
                        mdl_extract_vertices.extend(extracted)
                elif (
                    re > self.stats.delta_e
                    or member_dev > self.stats.delta_d
                    or not in_stable
                    or any(int(v) in incompatible for v in unit)
                ):
                    affected.add(gid)

        self.stats.mdl_affected = {
            "split_units": mdl_split_units,
            "extract_vertices": mdl_extract_vertices,
            "merge_keys": mdl_merge_keys,
            "protected_vertices": sorted(int(v) for v in protected_vertex_set),
        }
        self.stats.mdl_split_units = mdl_split_units
        self.stats.mdl_extract_vertices = mdl_extract_vertices
        self.stats.mdl_merge_keys = mdl_merge_keys
        return affected, scores, protected_vertex_set

    def sani_secondary(self, data, snapshot, adj, affected):
        data = _as_tnf(data)
        groups = snapshot.groups
        neigh = self._condensed_neighbors(groups, adj)
        secondary = set()

        for gid in affected:
            u = groups[gid]
            u_key = self._unit_key(u)
            u_mu = self._smp(data, u)
            lo, hi = self.stats.stable_ranges.get(u_key, (u_mu - self.eta, u_mu + self.eta))

            for vid in neigh.get(gid, set()):
                v = groups[vid]
                route_coupled = self._has_boundary_edge(u, v, adj)
                if not route_coupled:
                    continue

                if self._use_mdl_objective():
                    if self._mdl_group_merge_delta(data, u, v, adj) < 0:
                        secondary.add(vid)
                    continue

                v_mu = self._smp(data, v)
                cd = max(0.0, v_mu - hi) + max(0.0, lo - v_mu)
                if cd > self.stats.delta_d:
                    continue

                xu = data[:, u, 0].mean(axis=1)
                xv = data[:, v, 0].mean(axis=1)
                trend_dist = dtw_distance(xu, xv, radius=self.dtw_radius)
                if trend_dist <= max(self.stats.delta_c, self.eta):
                    secondary.add(vid)

        return secondary

    def update(self, data, adj, snapshot):
        data = _as_tnf(data)
        self._refresh_mdl_sketch(data, adj)
        self.stats.zcbs_enabled = self.zero_clustering_boundary_shift_enabled
        self.stats.zcbs_fallback_to_ward = self.zcbs_fallback_to_ward
        self.stats.zcbs_moves = []
        self.stats.zcbs_singletons = []
        self.stats.zcbs_state = {
            "enabled": bool(self.zero_clustering_boundary_shift_enabled),
            "mode": "zero_clustering_boundary_shifting" if self.zero_clustering_boundary_shift_enabled else "disabled",
            "fallback_to_ward": bool(self.zcbs_fallback_to_ward),
        }
        if snapshot is None:
            return self.build_initial(data, adj)
        self._refresh_boundary_index(snapshot, adj)

        affected, scores, protected_vertex_set = self.identify_affected(data, snapshot, adj)
        secondary = self.sani_secondary(data, snapshot, adj, affected)
        update_components = self._coalesce_update_components(
            data,
            snapshot,
            adj,
            affected,
            secondary,
            protected_vertex_set,
            scores,
        )

        if not update_components:
            self._update_aux(data, snapshot)
            self.stats.last_affected = []
            self.stats.changed_last_update = False
            self.stats.update_components = []
            self.stats.update_component_count = 0
            self.stats.internal_update_blocks = []
            self.stats.component_update_mode = "reuse_cached_route"
            return HGraphSnapshot(snapshot.key_nodes, snapshot.clusters, scores, snapshot.key_threshold, snapshot.version, snapshot.node_to_group)

        groups = snapshot.groups
        local_gids = sorted(set().union(*(set(c["gids"]) for c in update_components)))
        local_nodes = set().union(*(set(c["repair_nodes"]) for c in update_components))

        self.stats.last_affected = [self._unit_key(groups[gid]) for gid in local_gids]

        if len(local_nodes) / max(1, data.shape[1]) >= self.affected_ratio_full_rebuild:
            self._update_aux(data, snapshot)
            self.stats.changed_last_update = False
            self.stats.dry_run_admitted = False
            self.stats.dry_run_reason = "dry_run_reject_component_overflow_cost_guard"
            self.stats.component_update_mode = "component_overflow_reuse_cached_route"
            self.stats.merged_route_patch = {
                "changed_membership_rows": 0,
                "changed_condensed_edges": 0,
                "changed_route_entries": 0,
                "component_count": self.stats.update_component_count,
                "mode": "overflow_guard_no_patch",
            }
            self.stats.route_patch_mode = "copy_on_write_delta_patch_rejected"
            return HGraphSnapshot(snapshot.key_nodes, snapshot.clusters, scores, snapshot.key_threshold, snapshot.version, snapshot.node_to_group)

        max_keys = max(1, int(np.floor(self.theta * data.shape[1])))
        old_key_set = set(snapshot.key_nodes)
        candidate_key_nodes, _, _, _ = self._select_key_nodes(data)
        candidate_key_set = set(map(int, candidate_key_nodes))

        kept_keys = [k for k in snapshot.key_nodes if k not in local_nodes]
        kept_clusters = [[v for v in c if v not in local_nodes] for c in snapshot.clusters]
        kept_clusters = self._split_connected_clusters([c for c in kept_clusters if c], adj)

        local_sorted = sorted(local_nodes, key=lambda v: scores[v], reverse=True)
        local_keys = []
        demoted = []
        hysteresis_kept = []

        # Case 3 starts first for unstable internal vertices: keep/promote key-like nodes
        # and cut them out of their parent non-key components before any repair merge.
        for v in local_sorted:
            if v not in old_key_set:
                continue
            self.demote_counts[v] = 0
            local_keys.append(int(v))
            hysteresis_kept.append(int(v))

        for v in local_sorted:
            if len(kept_keys) + len(local_keys) >= max_keys:
                break
            if v in old_key_set:
                continue
            protected_ready = (
                v in protected_vertex_set
                and self.protected_counts.get(int(v), 0) >= self.protected_patience
                and scores[v] >= snapshot.key_threshold
            )
            promote_score = snapshot.key_threshold * self.promote_margin
            if v in candidate_key_set or protected_ready or scores[v] >= promote_score:
                local_keys.append(int(v))

        local_key_set = set(local_keys)
        demoted_set = set(demoted)
        local_nonkey = [int(v) for v in sorted(local_nodes) if v not in local_key_set and v not in demoted_set]

        # Case 2 repairs only the unresolved connected blocks inside each coalesced
        # component. Blocks from different components are never reclustered together.
        local_clusters = []
        if self.zero_clustering_boundary_shift_enabled:
            zcbs_pool = list(kept_clusters)
            zcbs_unresolved = []
            for component in update_components:
                component_nodes = sorted(set(component["repair_nodes"]) & set(local_nonkey))
                if not component_nodes:
                    continue
                zcbs_pool, unresolved = self._zero_clustering_boundary_shift(
                    data,
                    component_nodes,
                    zcbs_pool,
                    adj,
                )
                zcbs_unresolved.extend(unresolved)
            kept_clusters = zcbs_pool
            zcbs_unresolved = sorted(set(zcbs_unresolved))
            if self.zcbs_fallback_to_ward and zcbs_unresolved:
                for component in update_components:
                    unresolved_nodes = sorted(set(component["repair_nodes"]) & set(zcbs_unresolved))
                    if not unresolved_nodes:
                        continue
                    remain_cluster_slots = max(1, self.n_clusters - len(kept_clusters) - len(local_clusters))
                    local_clusters.extend(self._recluster_connected_nodes(
                        data,
                        unresolved_nodes,
                        adj,
                        cluster_budget=remain_cluster_slots,
                    ))
                self.stats.component_update_mode = "ipac_zcbs_with_ward_fallback"
            else:
                for v in sorted(zcbs_unresolved, key=lambda node: scores[node], reverse=True):
                    can_promote = (
                        self.zcbs_promote_unresolved
                        and len(kept_keys) + len(local_key_set) < max_keys
                        and scores[v] >= snapshot.key_threshold
                    )
                    if can_promote:
                        local_key_set.add(int(v))
                        local_keys.append(int(v))
                    else:
                        local_clusters.append([int(v)])
                self.stats.component_update_mode = "ipac_zero_clustering_boundary_shifting"
        else:
            for component in update_components:
                component_nodes = sorted(set(component["repair_nodes"]) & set(local_nonkey))
                if not component_nodes:
                    continue
                remain_cluster_slots = max(1, self.n_clusters - len(kept_clusters) - len(local_clusters))
                local_clusters.extend(self._recluster_connected_nodes(
                    data,
                    component_nodes,
                    adj,
                    cluster_budget=remain_cluster_slots,
                ))

        new_keys = sorted(set(kept_keys) | local_key_set)
        repaired_clusters = self._compact_clusters(data, kept_clusters + local_clusters, adj)

        # Case 1 runs last: demoted keys are inserted jointly into already repaired
        # non-key nodes only when joint residual and connectivity constraints hold.
        repaired_clusters = self._apply_joint_demotions(data, demoted, repaired_clusters, adj)
        new_clusters = self._compact_clusters(data, repaired_clusters, adj)
        promoted = sorted(set(new_keys) - old_key_set)
        self.stats.ipac_patch_operators = [
            {"case": "case3_key_anchor_extraction", "promoted": promoted},
            {
                "case": (
                    "case2_zero_clustering_boundary_shift_with_ward_fallback"
                    if self.zero_clustering_boundary_shift_enabled and self.zcbs_fallback_to_ward
                    else "case2_zero_clustering_boundary_shift"
                    if self.zero_clustering_boundary_shift_enabled
                    else "case2_fallback_connected_recluster"
                ),
                "components": self.stats.update_component_count,
                "zcbs_moves": self.stats.zcbs_moves,
                "zcbs_singletons": self.stats.zcbs_singletons,
                "zcbs_fallback_to_ward": self.stats.zcbs_fallback_to_ward,
            },
            {"case": "case1_key_demotion_reuse", "demoted": sorted(demoted)},
        ]

        changed = (set(new_keys) != old_key_set) or (sorted(map(sorted, new_clusters)) != sorted(map(sorted, snapshot.clusters)))

        admitted, reason = self._dry_run_admission(
            data,
            snapshot,
            new_keys,
            new_clusters,
            local_nodes,
            adj,
            promoted,
            demoted,
        )
        if not admitted:
            self._update_aux(data, snapshot)
            self.stats.changed_last_update = False
            self.stats.promoted = []
            self.stats.demoted = []
            self.stats.hysteresis_kept = hysteresis_kept
            self.stats.component_update_mode = reason
            return HGraphSnapshot(snapshot.key_nodes, snapshot.clusters, scores, snapshot.key_threshold, snapshot.version, snapshot.node_to_group)

        new_snapshot = HGraphSnapshot(
            key_nodes=new_keys,
            clusters=new_clusters,
            scores=scores,
            key_threshold=snapshot.key_threshold,
            version=snapshot.version + 1 if changed else snapshot.version,
        )
        new_snapshot.rebuild_index()
        self._refresh_boundary_index(new_snapshot, adj)
        self._update_aux(data, new_snapshot)
        self._calibrate(data, new_snapshot)
        self.stats.changed_last_update = changed
        self.stats.promoted = promoted
        self.stats.demoted = demoted
        self.stats.hysteresis_kept = hysteresis_kept
        return new_snapshot


class TimeSeriesSummary:
    pass


class HCIndex:
    pass


class LocalOperationManager:
    pass


class SuperNodeManager:
    pass


class OptimizedStructureManager:
    pass
