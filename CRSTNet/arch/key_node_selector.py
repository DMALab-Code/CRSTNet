import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict, Set, Optional, Union
import warnings
import heapq
from collections import defaultdict, deque

class KeyNodeSelector:

    def __init__(self,
                 entropy_weight: float = 1.0,
                 change_weight: float = 1.0,
                 threshold_quantile: float = 0.95,
                 fdr_alpha: float = 0.05,
                 eps_stop: float = 0.02,
                 diversity_rho: Optional[float] = None,
                 stability_gamma: float = 0.1,
                 max_key_nodes: int = 100):

        self.entropy_weight = entropy_weight
        self.change_weight = change_weight
        self.threshold_quantile = threshold_quantile
        self.fdr_alpha = fdr_alpha
        self.eps_stop = eps_stop
        self.diversity_rho = diversity_rho
        self.stability_gamma = stability_gamma
        self.max_key_nodes = max_key_nodes

        self.prev_key_nodes: Set[int] = set()
        self.prev_scores: Dict[int, float] = {}
        self.drift_history: Dict[int, List[float]] = {}

    def compute_dynamic_entropy(self, data: np.ndarray, bins: int = 20) -> np.ndarray:

        T, N, _ = data.shape
        entropies = np.zeros(N)
        for i in range(N):
            ts = data[:, i, 0]
            hist, _ = np.histogram(ts, bins=bins, density=True)
            hist = hist + 1e-10
            entropies[i] = -np.sum(hist * np.log(hist))
        return entropies

    def compute_change_rate(self, data: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:

        latest = data[-1, :, 0]
        prev = data[-2, :, 0]
        return np.abs(latest - prev) / (np.abs(prev) + epsilon)

    def compute_scores(self, data: np.ndarray) -> np.ndarray:

        H = self.compute_dynamic_entropy(data)
        R = self.compute_change_rate(data)
        H_norm = H / (np.max(H) + 1e-10)
        R_norm = R / (np.max(R) + 1e-10)
        scores = self.entropy_weight * H_norm + self.change_weight * R_norm
        return scores

    def detect_drift_cusum(self, node_data: np.ndarray, window_size: int = 10) -> float:

        if len(node_data) < window_size * 2:
            return 0.5

        half_window = window_size // 2
        first_half = node_data[:half_window]
        second_half = node_data[half_window:]

        if len(first_half) == 0 or len(second_half) == 0:
            return 0.5

        try:
            t_stat, p_value = stats.ttest_ind(first_half, second_half)
            return p_value
        except:
            return 0.5

    def bh_fdr_control(self, p_values: List[Tuple[int, float]], alpha: float) -> Set[int]:

        if not p_values:
            return set()

        sorted_pvals = sorted(p_values, key=lambda x: x[1])
        significant_nodes = set()

        for i, (node_idx, p_val) in enumerate(sorted_pvals):

            threshold = alpha * (i + 1) / len(sorted_pvals)
            if p_val <= threshold:
                significant_nodes.add(node_idx)
            else:
                break

        return significant_nodes

    def facility_location_objective(self, selected: Set[int], candidate: int,
                                  distances: np.ndarray, node_indices: List[int]) -> float:

        if not selected:

            return np.sum(np.exp(-distances[candidate, :]))

        current_coverage = 0
        new_coverage = 0

        for i, node_idx in enumerate(node_indices):
            if node_idx in selected:
                continue

            min_dist_current = min(distances[s, i] for s in selected) if selected else float('inf')
            current_coverage += np.exp(-min_dist_current)

            min_dist_new = min(min_dist_current, distances[candidate, i])
            new_coverage += np.exp(-min_dist_new)

        return new_coverage - current_coverage

    def lazy_greedy_facility_location(self, candidates: Set[int],
                                    distances: np.ndarray,
                                    node_indices: List[int],
                                    K: int,
                                    eps: float = 0.02,
                                    rho: Optional[float] = None) -> List[int]:

        if not candidates:
            return []

        selected = []
        remaining = candidates.copy()

        candidate_indices = [node_indices.index(c) for c in candidates if c in node_indices]
        if not candidate_indices:
            return []

        D_candidates = distances[np.ix_(candidate_indices, candidate_indices)]

        while len(selected) < K and remaining:
            best_gain = -float('inf')
            best_candidate = None

            for candidate in remaining:

                if rho is not None and selected:
                    min_dist_to_selected = min(
                        distances[node_indices.index(candidate), node_indices.index(s)]
                        for s in selected if s in node_indices
                    )
                    if min_dist_to_selected < rho:
                        continue

                gain = self.facility_location_objective(
                    set(selected), candidate, distances, node_indices
                )

                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate

            if best_candidate is None:
                break

            if len(selected) > 0:
                current_coverage = sum(
                    self.facility_location_objective(set(selected), s, distances, node_indices)
                    for s in selected
                )
                if current_coverage > 0 and best_gain / current_coverage <= eps:
                    break

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected

    def select_keys_streaming(self,
                            data: np.ndarray,
                            prev_keys: Set[int],
                            K_cap: int,
                            budget: int,
                            distances: Optional[np.ndarray] = None,
                            node_indices: Optional[List[int]] = None) -> Tuple[Set[int], Dict]:

        if data.ndim == 4:
            data = data[0]

        T, N, _ = data.shape

        scores = self.compute_scores(data)

        drift_pvalues = []
        for i in range(N):
            node_data = data[:, i, 0]
            p_val = self.detect_drift_cusum(node_data)
            drift_pvalues.append((i, p_val))

        significant_nodes = self.bh_fdr_control(drift_pvalues, self.fdr_alpha)

        median_score = np.median(scores)
        degraded_prev_keys = {
            k for k in prev_keys
            if k < N and scores[k] < median_score
        }

        candidates = significant_nodes | degraded_prev_keys

        if distances is None or node_indices is None:

            features = data[-1, :, :]
            distances = cdist(features, features)
            node_indices = list(range(N))

        selected = self.lazy_greedy_facility_location(
            candidates, distances, node_indices,
            K_cap, self.eps_stop, self.diversity_rho
        )

        selected = selected[:min(len(selected), budget)]
        selected_keys = set(selected)

        if self.prev_key_nodes and self.stability_gamma > 0:

            symmetric_diff = len(selected_keys ^ self.prev_key_nodes)
            if symmetric_diff > budget * 0.5:

                prev_scores = {k: scores[k] if k < N else 0 for k in self.prev_key_nodes}
                top_prev = sorted(prev_scores.items(), key=lambda x: x[1], reverse=True)
                selected_keys = {k for k, _ in top_prev[:K_cap]}

        self.prev_key_nodes = selected_keys.copy()
        self.prev_scores = {i: scores[i] for i in range(N)}

        diagnostics = {
            'num_candidates': len(candidates),
            'num_significant': len(significant_nodes),
            'num_degraded_prev': len(degraded_prev_keys),
            'fdr_threshold': self.fdr_alpha,
            'eps_stop_triggered': len(selected) < K_cap,
            'budget_used': len(selected_keys),
            'stability_penalty': len(selected_keys ^ self.prev_key_nodes) if self.prev_key_nodes else 0
        }

        return selected_keys, diagnostics

    def select_keys(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:

        if data.ndim == 4:
            data = data[0]

        scores = self.compute_scores(data)
        threshold = np.quantile(scores, self.threshold_quantile)
        key_nodes = np.where(scores >= threshold)[0]
        non_key_nodes = np.setdiff1d(np.arange(data.shape[1]), key_nodes)

        return key_nodes, non_key_nodes, threshold, scores

    def get_optimal_k_analysis(self, data: np.ndarray,
                              k_range: List[int] = None) -> Dict:

        if k_range is None:
            k_range = [5, 10, 15, 20, 25, 30, 40, 50]

        if data.ndim == 4:
            data = data[0]

        scores = self.compute_scores(data)
        features = data[-1, :, :]
        distances = cdist(features, features)
        node_indices = list(range(data.shape[1]))

        results = {}
        for k in k_range:

            top_k_indices = np.argsort(scores)[-k:]
            top_k_set = set(top_k_indices)

            coverage = self.facility_location_objective(
                set(), top_k_indices[0], distances, node_indices
            )

            if k > 1:
                cluster_diameter = np.max(distances[np.ix_(top_k_indices, top_k_indices)])
            else:
                cluster_diameter = 0

            results[k] = {
                'coverage': coverage,
                'cluster_diameter': cluster_diameter,
                'key_nodes': top_k_indices.tolist()
            }

        return results

class QuotaSwapKeySelector:

    def __init__(self, quota_ratio=0.1, tau_add=0.0, tau_swap=0.01,
                 lambda_cov=0.5, alpha_sim=0.5, window_size=10):

        self.quota_ratio = quota_ratio
        self.Q = None
        self.K = set()
        self.min_heap = []
        self.tau_add = tau_add
        self.tau_swap = tau_swap
        self.lambda_cov = lambda_cov
        self.alpha_sim = alpha_sim
        self.window_size = window_size

        self.hc_index = None

        self.score_history = defaultdict(lambda: deque(maxlen=window_size))
        self.performance_history = deque(maxlen=20)

    def prepare(self, num_nodes, hc_index):

        self.Q = int(self.quota_ratio * num_nodes)
        self.hc_index = hc_index
        self.K = set()
        self.min_heap = []

    def _compute_importance_score(self, u, data):

        if u >= data.shape[1]:
            return 0.0

        ts = data[:, u, 0]

        hist, _ = np.histogram(ts, bins=20, density=True)
        hist = hist + 1e-10
        entropy = -np.sum(hist * np.log(hist))

        if len(ts) > 1:
            change_rate = np.abs(ts[-1] - ts[-2]) / (np.abs(ts[-2]) + 1e-5)
        else:
            change_rate = 0.0

        entropy_norm = entropy / (np.max(entropy) + 1e-10) if np.max(entropy) > 0 else 0
        change_norm = change_rate / (np.max(change_rate) + 1e-10) if np.max(change_rate) > 0 else 0

        return entropy_norm + change_norm

    def _marginal_gain(self, u, data):

        if self.hc_index is None:

            return self._compute_importance_score(u, data)

        try:
            return self.hc_index.estimate_marginal_gain(u, self.K, data,
                                                      lambda_cov=self.lambda_cov,
                                                      alpha_sim=self.alpha_sim)
        except:

            return self._compute_importance_score(u, data)

    def _retain_gain(self, w, data):

        if self.hc_index is None:

            return self._compute_importance_score(w, data)

        try:
            return self.hc_index.estimate_retain_gain(w, self.K, data)
        except:
            return self._compute_importance_score(w, data)

    def consider(self, u, data):

        if u in self.K:
            return

        gain_u = self._marginal_gain(u, data)

        if len(self.K) < self.Q:
            if gain_u > self.tau_add:
                self.K.add(u)
                rg = self._retain_gain(u, data)
                heapq.heappush(self.min_heap, (rg, u))
            return

        if not self.min_heap:
            return

        rg_w, w_star = self.min_heap[0]

        swap_gain = gain_u - rg_w

        if swap_gain > self.tau_swap:

            heapq.heapreplace(self.min_heap, (self._retain_gain(u, data), u))
            self.K.remove(w_star)
            self.K.add(u)

    def batch_update_optimized(self, data, top_m_ratio=2.0):

        T, N, F = data.shape
        M = min(int(top_m_ratio * self.Q), N)

        scores = []
        for u in range(N):
            if u not in self.K:
                score = self._compute_importance_score(u, data)
                scores.append((score, u))

        scores.sort(reverse=True)
        top_candidates = [u for _, u in scores[:M]]

        for u in top_candidates:
            self.consider(u, data)

    def update_cover_cache(self, data):

        if not hasattr(self, 'cover_of'):
            self.cover_of = {}

        self.cover_of.clear()

        for u in range(data.shape[1]):
            if u not in self.K:
                best_cover = None
                best_gain = 0.0

                for k in self.K:
                    if self.hc_index:

                        try:
                            gain = self.hc_index.estimate_marginal_gain(u, {k}, data)
                            if gain > best_gain:
                                best_gain = gain
                                best_cover = k
                        except:
                            pass

                if best_cover is not None:
                    self.cover_of[u] = best_cover

    def batch_update(self, data, top_k_candidates=None):

        T, N, F = data.shape

        if top_k_candidates is None:

            scores = []
            for u in range(N):
                score = self._compute_importance_score(u, data)
                scores.append((score, u))
            scores.sort(reverse=True)
            top_k_candidates = [u for _, u in scores[:min(2*self.Q, N)]]

        for u in top_k_candidates:
            self.consider(u, data)

    def get_key_nodes(self):

        return self.K.copy()

    def get_non_key_nodes(self, total_nodes):

        return set(range(total_nodes)) - self.K

    def update_performance(self, performance_metric):

        self.performance_history.append(performance_metric)

        if len(self.performance_history) >= 5:
            recent_perf = np.mean(list(self.performance_history)[-5:])
            if recent_perf < 0.5:
                self.tau_add = max(0.0, self.tau_add - 0.01)
                self.tau_swap = max(0.005, self.tau_swap - 0.005)
            elif recent_perf > 0.8:
                self.tau_add = min(0.1, self.tau_add + 0.01)
                self.tau_swap = min(0.05, self.tau_swap + 0.005)

    def get_statistics(self):

        return {
            'quota_ratio': self.quota_ratio,
            'quota_limit': self.Q,
            'current_key_count': len(self.K),
            'utilization_rate': len(self.K) / self.Q if self.Q > 0 else 0,
            'tau_add': self.tau_add,
            'tau_swap': self.tau_swap,
            'heap_size': len(self.min_heap)
        }

