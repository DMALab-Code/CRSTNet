import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .key_node_selector import compute_importance_scores, select_top_theta


def dtw_distance(ts1: np.ndarray, ts2: np.ndarray, window: Optional[int] = None) -> float:
    """A lightweight DTW implementation for short traffic windows."""

    n, m = len(ts1), len(ts2)
    if n == 0 or m == 0:
        return 0.0

    if window is None:
        window = max(1, int(max(n, m) * 0.25))

    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        for j in range(j_start, j_end):
            cost = abs(ts1[i - 1] - ts2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def infer_eta(values: np.ndarray) -> float:
    """Infer the SMP tolerance from training statistics."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 1e-3

    median = np.median(values)
    mad = np.median(np.abs(values - median)) + 1e-8
    robust_std = 1.4826 * mad
    return float(max(0.1 * robust_std, 1e-4))


def split_slices(total_steps: int, slice_size: int) -> List[Tuple[int, int]]:
    slice_size = max(1, int(slice_size))
    return [(start, min(total_steps, start + slice_size)) for start in range(0, total_steps, slice_size)]


def compute_member_slice_profiles(
    data: np.ndarray,
    members: Sequence[int],
    slice_size: int,
    feature_index: int = 0,
) -> np.ndarray:
    if data.ndim == 4:
        data = data[0]

    members = list(members)
    if not members:
        return np.zeros((0, 0), dtype=np.float64)

    slices = split_slices(data.shape[0], slice_size)
    profiles = np.zeros((len(members), len(slices)), dtype=np.float64)
    for slice_idx, (start, end) in enumerate(slices):
        segment = data[start:end, members, feature_index]
        profiles[:, slice_idx] = np.mean(segment, axis=0)
    return profiles


def compute_cluster_slice_mean(member_profiles: np.ndarray) -> np.ndarray:
    if member_profiles.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.mean(member_profiles, axis=0)


def compute_residual_error(member_profiles: np.ndarray, cluster_profile: np.ndarray) -> float:
    if member_profiles.size == 0:
        return 0.0
    return float(np.mean((member_profiles - cluster_profile[None, :]) ** 2))


def compute_correlation_deviation(
    candidate_profile: np.ndarray,
    stable_low: np.ndarray,
    stable_high: np.ndarray,
) -> float:
    candidate_profile = np.asarray(candidate_profile, dtype=np.float64)
    stable_low = np.asarray(stable_low, dtype=np.float64)
    stable_high = np.asarray(stable_high, dtype=np.float64)
    above = np.maximum(0.0, candidate_profile - stable_high)
    below = np.maximum(0.0, stable_low - candidate_profile)
    return float(np.mean(above + below))


def compute_correlation_closeness(
    profile_u: np.ndarray,
    profile_v: np.ndarray,
    max_backward_shift: Optional[int] = None,
    epsilon: float = 1e-5,
) -> Tuple[float, float]:
    profile_u = np.asarray(profile_u, dtype=np.float64)
    profile_v = np.asarray(profile_v, dtype=np.float64)

    if profile_u.size == 0 or profile_v.size == 0:
        return 0.0, float("inf")

    if max_backward_shift is None:
        max_backward_shift = 2 * len(profile_u)

    best_mse = float("inf")
    for shift in range(0, max_backward_shift + 1):
        shifted = np.roll(profile_v, -shift)
        if shift > 0:
            shifted[-shift:] = profile_v[-1]
        mse = float(np.mean((profile_u - shifted[: len(profile_u)]) ** 2))
        best_mse = min(best_mse, mse)

    closeness = 1.0 / (best_mse + epsilon)
    return closeness, best_mse


def compute_shortest_path_distances(adj: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adj, dtype=np.float64)
    matrix = np.where(matrix > 0, matrix, 0.0)
    return shortest_path(csgraph=csr_matrix(matrix), directed=False, unweighted=True)


def build_one_hop_dtw_cache(data: np.ndarray, adj: np.ndarray, feature_index: int = 0) -> Dict[Tuple[int, int], float]:
    if data.ndim == 4:
        data = data[0]

    cache: Dict[Tuple[int, int], float] = {}
    num_nodes = data.shape[1]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] > 0 or adj[j, i] > 0:
                cache[(i, j)] = dtw_distance(data[:, i, feature_index], data[:, j, feature_index])
    return cache


def aggregate_boundary_weight(cluster_a: Sequence[int], cluster_b: Sequence[int], adj: np.ndarray) -> float:
    if not cluster_a or not cluster_b:
        return 0.0

    boundary = np.asarray(adj[np.ix_(cluster_a, cluster_b)], dtype=np.float64)
    weight = float(np.sum(boundary))
    if weight <= 0.0:
        return 0.0
    return weight / math.sqrt(max(1.0, len(cluster_a) * len(cluster_b)))


def build_condensed_adjacency(
    key_nodes: Sequence[int],
    clusters: Sequence[Sequence[int]],
    adj: np.ndarray,
    return_entities: bool = False,
) -> Tuple[np.ndarray, Optional[List[Tuple[str, object, List[int]]]]]:
    entities: List[Tuple[str, object, List[int]]] = [("key", int(node), [int(node)]) for node in key_nodes]
    entities.extend(("cluster", tuple(sorted(int(v) for v in cluster)), [int(v) for v in cluster]) for cluster in clusters if len(cluster) > 0)

    num_entities = len(entities)
    condensed = np.zeros((num_entities, num_entities), dtype=np.float64)
    for i in range(num_entities):
        condensed[i, i] = 1.0
        for j in range(i + 1, num_entities):
            members_i = entities[i][2]
            members_j = entities[j][2]
            weight = aggregate_boundary_weight(members_i, members_j, adj)
            if weight > 0.0:
                condensed[i, j] = weight
                condensed[j, i] = weight

    return condensed, entities if return_entities else None


def sym_normalize_adj(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float64)
    degree = np.sum(adj, axis=1)
    degree = np.where(degree > 0.0, degree, 1.0)
    inv_sqrt = np.power(degree, -0.5)
    return inv_sqrt[:, None] * adj * inv_sqrt[None, :]


def composite_distance(
    node_i: int,
    node_j: int,
    shortest_distances: np.ndarray,
    dtw_cache: Dict[Tuple[int, int], float],
    lambda_value: float,
    d0: float,
    tau0: float,
) -> float:
    pair = (min(node_i, node_j), max(node_i, node_j))
    spatial = float(shortest_distances[node_i, node_j])
    if not np.isfinite(spatial):
        return float("inf")

    spatial = spatial / max(d0, 1e-5)
    temporal = float(dtw_cache.get(pair, 0.0)) / max(tau0, 1e-5)
    return float(lambda_value * spatial + (1.0 - lambda_value) * temporal)


def _clusters_are_adjacent(cluster_a: Sequence[int], cluster_b: Sequence[int], adj: np.ndarray) -> bool:
    for node_i in cluster_a:
        for node_j in cluster_b:
            if adj[node_i, node_j] > 0 or adj[node_j, node_i] > 0:
                return True
    return False


def _merge_cost(
    cluster_a: Sequence[int],
    cluster_b: Sequence[int],
    data: np.ndarray,
    adj: np.ndarray,
    shortest_distances: np.ndarray,
    dtw_cache: Dict[Tuple[int, int], float],
    lambda_value: float,
    d0: float,
    tau0: float,
    feature_index: int = 0,
) -> float:
    if not _clusters_are_adjacent(cluster_a, cluster_b, adj):
        return float("inf")

    boundary_distances: List[float] = []
    for node_i in cluster_a:
        for node_j in cluster_b:
            if adj[node_i, node_j] > 0 or adj[node_j, node_i] > 0:
                boundary_distances.append(
                    composite_distance(node_i, node_j, shortest_distances, dtw_cache, lambda_value, d0, tau0)
                )

    if not boundary_distances:
        return float("inf")

    series_a = np.mean(data[:, list(cluster_a), feature_index], axis=1)
    series_b = np.mean(data[:, list(cluster_b), feature_index], axis=1)
    centroid_gap = np.mean((series_a - series_b) ** 2)
    ward_term = (len(cluster_a) * len(cluster_b) / max(1, len(cluster_a) + len(cluster_b))) * centroid_gap
    return float(ward_term + np.mean(boundary_distances))


def estimate_target_cluster_count(num_nodes: int) -> int:
    if num_nodes <= 1:
        return num_nodes
    return max(1, int(math.ceil(math.sqrt(num_nodes))))


def topology_constrained_ward(
    nodes: Sequence[int],
    adj: np.ndarray,
    data: np.ndarray,
    shortest_distances: np.ndarray,
    lambda_value: float,
    dtw_cache: Dict[Tuple[int, int], float],
    target_clusters: Optional[int] = None,
    feature_index: int = 0,
) -> List[List[int]]:
    nodes = sorted(int(node) for node in nodes)
    if not nodes:
        return []
    if len(nodes) == 1:
        return [nodes]

    if target_clusters is None:
        target_clusters = estimate_target_cluster_count(len(nodes))
    target_clusters = max(1, min(int(target_clusters), len(nodes)))

    dtw_values = np.array(list(dtw_cache.values()), dtype=np.float64) if dtw_cache else np.array([1.0], dtype=np.float64)
    tau0 = float(np.median(dtw_values)) if dtw_values.size > 0 else 1.0
    d0 = float(np.median(shortest_distances[np.isfinite(shortest_distances) & (shortest_distances > 0)]))
    if not np.isfinite(d0):
        d0 = 1.0

    clusters: Dict[int, List[int]] = {idx: [node] for idx, node in enumerate(nodes)}
    next_cluster_id = len(clusters)
    while len(clusters) > target_clusters:
        best_pair: Optional[Tuple[int, int]] = None
        best_cost = float("inf")
        cluster_items = list(clusters.items())
        for i, (cluster_id_i, cluster_i) in enumerate(cluster_items):
            for cluster_id_j, cluster_j in cluster_items[i + 1 :]:
                cost = _merge_cost(
                    cluster_i,
                    cluster_j,
                    data=data,
                    adj=adj,
                    shortest_distances=shortest_distances,
                    dtw_cache=dtw_cache,
                    lambda_value=lambda_value,
                    d0=d0,
                    tau0=tau0,
                    feature_index=feature_index,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (cluster_id_i, cluster_id_j)

        if best_pair is None or not np.isfinite(best_cost):
            break

        first_id, second_id = best_pair
        merged = sorted(clusters.pop(first_id) + clusters.pop(second_id))
        clusters[next_cluster_id] = merged
        next_cluster_id += 1

    return [members for _, members in sorted(clusters.items(), key=lambda item: (len(item[1]), item[1][0]))]


def efficient_dtw_distance(data: np.ndarray, node_indices: Sequence[int], top_k: int = 5, downsample: int = 1, n_jobs: int = 1) -> np.ndarray:
    del top_k, n_jobs

    if data.ndim == 4:
        data = data[0]
    node_indices = list(node_indices)

    sampled = data[:: max(1, downsample)]
    size = len(node_indices)
    matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i + 1, size):
            matrix[i, j] = dtw_distance(sampled[:, node_indices[i], 0], sampled[:, node_indices[j], 0])
            matrix[j, i] = matrix[i, j]
    return matrix


def select_dynamic_key_nodes(
    data: np.ndarray,
    percentile: float = 95.0,
    importance: Optional[np.ndarray] = None,
    theta: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    del importance

    scores, _, _ = compute_importance_scores(data)
    if theta is None:
        theta = max(0.0, 1.0 - percentile / 100.0)
    key_nodes = select_top_theta(scores, theta)
    non_key_nodes = np.setdiff1d(np.arange(scores.shape[0]), key_nodes)
    threshold = float(scores[key_nodes[-1]]) if len(key_nodes) > 0 else float(np.max(scores))
    return key_nodes, non_key_nodes, scores, threshold


def detect_distribution_change(data: np.ndarray, prev_state=None, threshold: float = 2.0) -> Tuple[np.ndarray, bool, Dict[str, np.ndarray]]:
    scores, _, _ = compute_importance_scores(data)
    current_state = {"scores": scores}
    if prev_state is None:
        return np.arange(scores.shape[0]), True, current_state

    prev_scores = np.asarray(prev_state.get("scores", np.zeros_like(scores)))
    delta = np.abs(scores - prev_scores)
    changed = np.where(delta > threshold * np.mean(delta + 1e-5))[0]
    need_full_update = len(changed) > 0.3 * len(scores)
    return changed, bool(need_full_update), current_state


@dataclass
class StableRangeState:
    low: np.ndarray
    high: np.ndarray


@dataclass
class EntityProfile:
    kind: str
    members: List[int]
    state_key: Tuple[str, object]
    smp: np.ndarray
    member_profiles: np.ndarray
    stable_low: np.ndarray
    stable_high: np.ndarray
    residual: float
    out_of_range: bool
    member_cd: Dict[int, float]


class PaperAlignedStructureManager:
    """Maintain the H-Graph using the paper's SMP/SANI rules."""

    def __init__(
        self,
        graph_adj: np.ndarray,
        theta: float = 0.2,
        lambda_value: float = 0.5,
        maintenance_interval: int = 5,
        delta_s: Optional[float] = None,
        delta_e: Optional[float] = None,
        delta_d: Optional[float] = None,
        eta: Optional[float] = None,
        max_key_nodes: Optional[int] = None,
        target_clusters: Optional[int] = None,
        feature_index: int = 0,
        threshold_profile: Optional[str] = None,
    ):
        self.adj = np.asarray(graph_adj, dtype=np.float64)
        self.theta = theta
        self.lambda_value = lambda_value
        self.maintenance_interval = max(1, int(maintenance_interval))
        self.feature_index = feature_index
        self.max_key_nodes = max_key_nodes
        self.target_clusters = target_clusters

        self.shortest_distances = compute_shortest_path_distances(self.adj)
        finite_distances = self.shortest_distances[np.isfinite(self.shortest_distances) & (self.shortest_distances > 0)]
        self.d0 = float(np.median(finite_distances)) if finite_distances.size > 0 else 1.0

        self.delta_s = delta_s
        self.delta_e = delta_e
        self.delta_d = delta_d
        self.eta = eta
        if threshold_profile:
            self.load_threshold_profile(threshold_profile)

        self.step_count = 0
        self.key_nodes: List[int] = []
        self.non_key_clusters: List[List[int]] = []
        self.max_key_budget: Optional[int] = None
        self.dtw_cache: Dict[Tuple[int, int], float] = {}
        self.profile_states: Dict[Tuple[str, object], StableRangeState] = {}
        self.last_scores = np.zeros(self.adj.shape[0], dtype=np.float64)
        self.last_profiles: Dict[Tuple[str, object], EntityProfile] = {}

    def load_threshold_profile(self, path: str) -> None:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
        self.delta_s = profile.get("delta_s", self.delta_s)
        self.delta_e = profile.get("delta_e", self.delta_e)
        self.delta_d = profile.get("delta_d", self.delta_d)
        self.eta = profile.get("eta", self.eta)

    def _state_key(self, kind: str, members: Sequence[int]) -> Tuple[str, object]:
        if kind == "key":
            return kind, int(members[0])
        return kind, tuple(sorted(int(member) for member in members))

    def _update_stable_range(self, state_key: Tuple[str, object], smp: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        if state_key not in self.profile_states:
            low = smp - eta
            high = smp + eta
        else:
            prev = self.profile_states[state_key]
            low = np.minimum(prev.low, (prev.low + smp) / 2.0 - eta)
            high = np.maximum(prev.high, (prev.high + smp) / 2.0 + eta)
        self.profile_states[state_key] = StableRangeState(low=low, high=high)
        return low, high

    def _compute_entity_profile(self, kind: str, members: Sequence[int], data: np.ndarray) -> EntityProfile:
        member_profiles = compute_member_slice_profiles(data, members, self.maintenance_interval, self.feature_index)
        smp = compute_cluster_slice_mean(member_profiles)

        eta = self.eta if self.eta is not None else infer_eta(member_profiles)
        state_key = self._state_key(kind, members)
        stable_low, stable_high = self._update_stable_range(state_key, smp, eta)

        residual = compute_residual_error(member_profiles, smp) if kind == "cluster" else 0.0
        out_of_range = bool(np.any((smp < stable_low) | (smp > stable_high)))
        member_cd = {
            int(member): compute_correlation_deviation(member_profiles[idx], stable_low, stable_high)
            for idx, member in enumerate(members)
        }

        return EntityProfile(
            kind=kind,
            members=list(int(member) for member in members),
            state_key=state_key,
            smp=smp,
            member_profiles=member_profiles,
            stable_low=stable_low,
            stable_high=stable_high,
            residual=residual,
            out_of_range=out_of_range,
            member_cd=member_cd,
        )

    def _bootstrap_thresholds(self, data: np.ndarray, scores: np.ndarray) -> None:
        if self.delta_s is None:
            key_nodes, non_key_nodes, _, _ = select_dynamic_key_nodes(data, theta=self.theta)
            if len(non_key_nodes) > 0:
                self.delta_s = float(np.quantile(scores[non_key_nodes], 0.9))
            else:
                self.delta_s = float(np.quantile(scores, 0.5))

        if self.delta_e is None or self.delta_d is None:
            residuals: List[float] = []
            deviations: List[float] = []
            for cluster in self.non_key_clusters:
                profile = self._compute_entity_profile("cluster", cluster, data)
                residuals.append(profile.residual)
                deviations.extend(profile.member_cd.values())

            if self.delta_e is None:
                self.delta_e = float(np.quantile(residuals, 0.9)) if residuals else 0.0
            if self.delta_d is None:
                self.delta_d = float(np.quantile(deviations, 0.9)) if deviations else infer_eta(data[..., self.feature_index])

    def _build_initial_structure(self, data: np.ndarray) -> None:
        self.dtw_cache = build_one_hop_dtw_cache(data, self.adj, self.feature_index)
        scores, _, _ = compute_importance_scores(data, feature_index=self.feature_index)
        self.last_scores = scores

        key_nodes = select_top_theta(scores, self.theta, self.max_key_nodes)
        self.max_key_budget = len(key_nodes) if self.max_key_nodes is None else int(self.max_key_nodes)
        self.key_nodes = sorted(int(node) for node in key_nodes.tolist())

        non_key_nodes = [node for node in range(self.adj.shape[0]) if node not in set(self.key_nodes)]
        self.non_key_clusters = topology_constrained_ward(
            non_key_nodes,
            self.adj,
            data,
            shortest_distances=self.shortest_distances,
            lambda_value=self.lambda_value,
            dtw_cache=self.dtw_cache,
            target_clusters=self.target_clusters,
            feature_index=self.feature_index,
        )

        self._bootstrap_thresholds(data, scores)
        self._update_profiles(data)

    def _update_profiles(self, data: np.ndarray) -> Dict[Tuple[str, object], EntityProfile]:
        current_profiles: Dict[Tuple[str, object], EntityProfile] = {}
        active_keys: Set[Tuple[str, object]] = set()

        for node in self.key_nodes:
            profile = self._compute_entity_profile("key", [node], data)
            current_profiles[profile.state_key] = profile
            active_keys.add(profile.state_key)

        for cluster in self.non_key_clusters:
            if not cluster:
                continue
            profile = self._compute_entity_profile("cluster", cluster, data)
            current_profiles[profile.state_key] = profile
            active_keys.add(profile.state_key)

        self.profile_states = {key: state for key, state in self.profile_states.items() if key in active_keys}
        self.last_profiles = current_profiles
        return current_profiles

    def _sani_neighbors(
        self,
        state_key: Tuple[str, object],
        profiles: Dict[Tuple[str, object], EntityProfile],
        entity_labels: List[Tuple[str, object, List[int]]],
        condensed_adj: np.ndarray,
    ) -> List[Tuple[Tuple[str, object], float, float]]:
        if self.delta_d is None or state_key not in profiles:
            return []

        label_to_idx = {(kind, identity): idx for idx, (kind, identity, _) in enumerate(entity_labels)}
        idx = label_to_idx.get(state_key)
        if idx is None:
            return []

        target = profiles[state_key]
        neighbors: List[Tuple[Tuple[str, object], float, float]] = []
        closeness_threshold = 1.0 / (self.delta_d + 1e-5)
        for neighbor_idx, edge_weight in enumerate(condensed_adj[idx]):
            if neighbor_idx == idx or edge_weight <= 0.0:
                continue

            kind, identity, _ = entity_labels[neighbor_idx]
            neighbor_key = (kind, identity)
            if neighbor_key not in profiles:
                continue

            neighbor_profile = profiles[neighbor_key]
            cd = compute_correlation_deviation(neighbor_profile.smp, target.stable_low, target.stable_high)
            if cd > self.delta_d:
                continue

            closeness, _ = compute_correlation_closeness(target.smp, neighbor_profile.smp, 2 * max(1, len(target.smp)))
            if closeness >= closeness_threshold:
                neighbors.append((neighbor_key, cd, closeness))
        return neighbors

    def _affected_entities(
        self,
        scores: np.ndarray,
        profiles: Dict[Tuple[str, object], EntityProfile],
        entity_labels: List[Tuple[str, object, List[int]]],
        condensed_adj: np.ndarray,
    ) -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
        demoted_keys: Set[int] = set()
        promoted_vertices: Set[int] = set()
        local_vertices: Set[int] = set()
        directly_affected_clusters: Set[int] = set()

        key_set = set(self.key_nodes)
        min_key_score = min((scores[node] for node in key_set), default=-np.inf)

        for cluster_index, cluster in enumerate(self.non_key_clusters):
            profile_key = self._state_key("cluster", cluster)
            profile = profiles.get(profile_key)
            if profile is None:
                continue

            violates_cohesion = self.delta_e is not None and profile.residual > self.delta_e
            incompatible_vertices = [
                vertex
                for vertex, deviation in profile.member_cd.items()
                if self.delta_d is not None and deviation > self.delta_d
            ]

            if violates_cohesion or profile.out_of_range or incompatible_vertices:
                directly_affected_clusters.add(cluster_index)
                local_vertices.update(cluster)
                for neighbor_key, _, _ in self._sani_neighbors(profile_key, profiles, entity_labels, condensed_adj):
                    neighbor_profile = profiles.get(neighbor_key)
                    if neighbor_profile is not None:
                        local_vertices.update(neighbor_profile.members)

            for vertex in incompatible_vertices:
                if (len(key_set) < (self.max_key_budget or len(key_set) + 1)) or scores[vertex] > min_key_score:
                    if len(key_set) >= (self.max_key_budget or len(key_set) + 1) and key_set:
                        victim = min(key_set, key=lambda node: scores[node])
                        if scores[vertex] > scores[victim]:
                            key_set.discard(victim)
                            demoted_keys.add(victim)
                            local_vertices.add(victim)
                    key_set.add(vertex)
                    promoted_vertices.add(vertex)
                    min_key_score = min((scores[node] for node in key_set), default=-np.inf)

        for node in self.key_nodes:
            profile_key = self._state_key("key", [node])
            profile = profiles.get(profile_key)
            if profile is None:
                continue

            if self.delta_s is not None and scores[node] < self.delta_s and profile.out_of_range:
                demoted_keys.add(node)
                local_vertices.add(node)
                for neighbor_key, _, _ in self._sani_neighbors(profile_key, profiles, entity_labels, condensed_adj):
                    neighbor_profile = profiles.get(neighbor_key)
                    if neighbor_profile is not None:
                        local_vertices.update(neighbor_profile.members)

        return demoted_keys, promoted_vertices, local_vertices, directly_affected_clusters

    def _maintain_structure(self, data: np.ndarray) -> None:
        self.dtw_cache = build_one_hop_dtw_cache(data, self.adj, self.feature_index)
        scores, _, _ = compute_importance_scores(data, feature_index=self.feature_index)
        self.last_scores = scores

        profiles = self._update_profiles(data)
        condensed_adj, entity_labels = build_condensed_adjacency(self.key_nodes, self.non_key_clusters, self.adj, return_entities=True)
        assert entity_labels is not None

        demoted_keys, promoted_vertices, local_vertices, affected_cluster_ids = self._affected_entities(
            scores,
            profiles,
            entity_labels,
            condensed_adj,
        )

        if not demoted_keys and not promoted_vertices and not local_vertices:
            return

        updated_key_nodes = set(self.key_nodes)
        updated_key_nodes -= demoted_keys
        updated_key_nodes |= promoted_vertices
        updated_key_nodes = {int(node) for node in updated_key_nodes}

        local_vertices -= updated_key_nodes
        unaffected_clusters = [
            [node for node in cluster if node not in updated_key_nodes]
            for idx, cluster in enumerate(self.non_key_clusters)
            if idx not in affected_cluster_ids and not any(node in local_vertices for node in cluster)
        ]
        unaffected_clusters = [cluster for cluster in unaffected_clusters if cluster]

        for node in demoted_keys:
            local_vertices.add(int(node))

        recluster_pool = sorted(int(node) for node in local_vertices if node not in updated_key_nodes)
        local_clusters = topology_constrained_ward(
            recluster_pool,
            self.adj,
            data,
            shortest_distances=self.shortest_distances,
            lambda_value=self.lambda_value,
            dtw_cache=self.dtw_cache,
            target_clusters=min(
                estimate_target_cluster_count(len(recluster_pool)),
                len(recluster_pool),
            )
            if recluster_pool
            else 0,
            feature_index=self.feature_index,
        )

        covered_non_key_nodes = {node for cluster in unaffected_clusters for node in cluster}
        covered_non_key_nodes.update(node for cluster in local_clusters for node in cluster)
        remaining_non_key_nodes = [
            node
            for node in range(self.adj.shape[0])
            if node not in updated_key_nodes and node not in covered_non_key_nodes
        ]
        local_clusters.extend([[node] for node in remaining_non_key_nodes])

        self.key_nodes = sorted(updated_key_nodes)
        self.non_key_clusters = [sorted(cluster) for cluster in unaffected_clusters + local_clusters if cluster]
        self._update_profiles(data)

    def update_structure(self, data: np.ndarray, performance_metric=None) -> Dict[str, object]:
        del performance_metric

        self.step_count += 1
        if not self.key_nodes and not self.non_key_clusters:
            self._build_initial_structure(data)
        elif self.step_count % self.maintenance_interval == 0:
            self._maintain_structure(data)

        return self._export_structure_info()

    def _export_structure_info(self) -> Dict[str, object]:
        non_key_nodes = sorted(node for cluster in self.non_key_clusters for node in cluster)
        condensed_adj, _ = build_condensed_adjacency(self.key_nodes, self.non_key_clusters, self.adj, return_entities=False)
        return {
            "key_nodes": list(self.key_nodes),
            "non_key_nodes": non_key_nodes,
            "cluster_indices_list": [list(cluster) for cluster in self.non_key_clusters],
            "super_nodes": [list(cluster) for cluster in self.non_key_clusters],
            "condensed_adj": condensed_adj,
            "thresholds": {
                "delta_s": self.delta_s,
                "delta_e": self.delta_e,
                "delta_d": self.delta_d,
                "eta": self.eta,
            },
        }

    def get_comprehensive_statistics(self) -> Dict[str, object]:
        return {
            "step_count": self.step_count,
            "theta": self.theta,
            "lambda_value": self.lambda_value,
            "maintenance_interval": self.maintenance_interval,
            "delta_s": self.delta_s,
            "delta_e": self.delta_e,
            "delta_d": self.delta_d,
            "num_key_nodes": len(self.key_nodes),
            "num_non_key_clusters": len(self.non_key_clusters),
        }


# Backward-compatible alias kept for the rest of the repository.
OptimizedStructureManager = PaperAlignedStructureManager
