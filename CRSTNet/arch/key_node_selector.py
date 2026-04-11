import heapq
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def compute_dynamic_entropy(data: np.ndarray, bins: int = 20, feature_index: int = 0) -> np.ndarray:
    """Compute per-vertex temporal entropy on the recent window."""

    if data.ndim == 4:
        data = data[0]

    _, num_nodes, _ = data.shape
    entropies = np.zeros(num_nodes, dtype=np.float64)
    for node_idx in range(num_nodes):
        ts = data[:, node_idx, feature_index]
        hist, _ = np.histogram(ts, bins=bins, density=True)
        hist = hist + 1e-10
        entropies[node_idx] = -np.sum(hist * np.log(hist))
    return entropies


def compute_fluctuation_rate(data: np.ndarray, epsilon: float = 1e-5, feature_index: int = 0) -> np.ndarray:
    """Compute the fluctuation rate in Eq. (2)."""

    if data.ndim == 4:
        data = data[0]

    latest = data[-1, :, feature_index]
    prev = data[-2, :, feature_index]
    return np.abs(latest - prev) / (np.abs(prev) + epsilon)


def compute_importance_scores(
    data: np.ndarray,
    bins: int = 20,
    epsilon: float = 1e-5,
    feature_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the normalized entropy, fluctuation rate, and the paper score."""

    entropy = compute_dynamic_entropy(data, bins=bins, feature_index=feature_index)
    fluctuation = compute_fluctuation_rate(data, epsilon=epsilon, feature_index=feature_index)

    entropy_norm = entropy / (np.max(entropy) + epsilon)
    fluctuation_norm = fluctuation / (np.max(fluctuation) + epsilon)
    scores = entropy_norm * fluctuation_norm
    return scores, entropy_norm, fluctuation_norm


def select_top_theta(scores: np.ndarray, theta: float, max_key_nodes: Optional[int] = None) -> np.ndarray:
    """Select the top floor(theta * |V|) vertices as key nodes."""

    num_nodes = int(scores.shape[0])
    theta = float(np.clip(theta, 0.0, 1.0))
    count = max(1, int(np.floor(theta * num_nodes)))
    if max_key_nodes is not None:
        count = min(count, int(max_key_nodes))

    ranked = np.argsort(scores)[::-1]
    return np.sort(ranked[:count])


class KeyNodeSelector:
    """Paper-aligned key-node selector using normalized entropy x fluctuation."""

    def __init__(
        self,
        theta: float = 0.2,
        max_key_nodes: Optional[int] = None,
        bins: int = 20,
        epsilon: float = 1e-5,
        threshold_quantile: Optional[float] = None,
    ):
        if threshold_quantile is not None:
            theta = max(0.0, 1.0 - float(threshold_quantile))

        self.theta = theta
        self.max_key_nodes = max_key_nodes
        self.bins = bins
        self.epsilon = epsilon
        self.prev_key_nodes: Set[int] = set()
        self.prev_scores: Dict[int, float] = {}
        self.score_history: Dict[int, List[float]] = {}

    def compute_scores(self, data: np.ndarray) -> np.ndarray:
        scores, _, _ = compute_importance_scores(
            data,
            bins=self.bins,
            epsilon=self.epsilon,
        )
        return scores

    def select_keys(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        scores = self.compute_scores(data)
        key_nodes = select_top_theta(scores, self.theta, self.max_key_nodes)
        non_key_nodes = np.setdiff1d(np.arange(scores.shape[0]), key_nodes)
        threshold = float(scores[key_nodes[-1]]) if len(key_nodes) > 0 else float(np.max(scores))

        self.prev_key_nodes = set(int(i) for i in key_nodes)
        self.prev_scores = {int(i): float(scores[i]) for i in range(scores.shape[0])}
        return key_nodes, non_key_nodes, threshold, scores

    def select_keys_streaming(
        self,
        data: np.ndarray,
        prev_keys: Optional[Set[int]] = None,
        K_cap: Optional[int] = None,
        budget: Optional[int] = None,
        distances: Optional[np.ndarray] = None,
        node_indices: Optional[List[int]] = None,
    ) -> Tuple[Set[int], Dict]:
        del distances, node_indices  # Not needed for the paper-aligned score rule.

        scores = self.compute_scores(data)
        key_nodes = select_top_theta(
            scores,
            self.theta,
            max_key_nodes=budget if budget is not None else K_cap if K_cap is not None else self.max_key_nodes,
        )
        selected = set(int(idx) for idx in key_nodes.tolist())
        previous = prev_keys if prev_keys is not None else self.prev_key_nodes

        diagnostics = {
            "num_selected": len(selected),
            "theta": self.theta,
            "budget_used": len(selected),
            "symmetric_difference": len(selected ^ set(previous)),
        }

        self.prev_key_nodes = selected.copy()
        self.prev_scores = {int(i): float(scores[i]) for i in range(scores.shape[0])}
        return selected, diagnostics


class QuotaSwapKeySelector:
    """
    Maintain a bounded set of key nodes.

    The legacy class name is preserved for compatibility, but the
    importance score now follows Eq. (2) in the paper.
    """

    def __init__(
        self,
        quota_ratio: float = 0.1,
        tau_add: float = 0.0,
        tau_swap: float = 0.0,
        epsilon: float = 1e-5,
    ):
        self.quota_ratio = quota_ratio
        self.tau_add = tau_add
        self.tau_swap = tau_swap
        self.epsilon = epsilon

        self.Q: Optional[int] = None
        self.K: Set[int] = set()
        self.min_heap: List[Tuple[float, int]] = []
        self.performance_history = deque(maxlen=20)
        self.score_history = defaultdict(lambda: deque(maxlen=10))

    def prepare(self, num_nodes: int, hc_index=None) -> None:
        del hc_index
        self.Q = max(1, int(np.floor(self.quota_ratio * num_nodes)))
        self.K.clear()
        self.min_heap.clear()

    def _score_vertex(self, u: int, data: np.ndarray) -> float:
        scores, _, _ = compute_importance_scores(data, epsilon=self.epsilon)
        return float(scores[u]) if 0 <= u < scores.shape[0] else 0.0

    def consider(self, u: int, data: np.ndarray) -> None:
        if self.Q is None:
            self.prepare(data.shape[1])

        if u in self.K:
            return

        gain_u = self._score_vertex(u, data)
        self.score_history[u].append(gain_u)

        if len(self.K) < self.Q:
            if gain_u >= self.tau_add:
                self.K.add(u)
                heapq.heappush(self.min_heap, (gain_u, u))
            return

        if not self.min_heap:
            return

        min_gain, victim = self.min_heap[0]
        if gain_u - min_gain <= self.tau_swap:
            return

        heapq.heapreplace(self.min_heap, (gain_u, u))
        self.K.discard(victim)
        self.K.add(u)

    def batch_update(self, data: np.ndarray, top_k_candidates: Optional[Sequence[int]] = None) -> None:
        if top_k_candidates is None:
            scores, _, _ = compute_importance_scores(data, epsilon=self.epsilon)
            ranking = np.argsort(scores)[::-1]
            top_k_candidates = ranking[: max(1, 2 * (self.Q or 1))]

        for u in top_k_candidates:
            self.consider(int(u), data)

    def get_key_nodes(self) -> Set[int]:
        return set(self.K)

    def get_non_key_nodes(self, total_nodes: int) -> Set[int]:
        return set(range(total_nodes)) - self.K

    def update_performance(self, performance_metric: float) -> None:
        self.performance_history.append(float(performance_metric))

    def get_statistics(self) -> Dict[str, float]:
        quota_limit = self.Q if self.Q is not None else 0
        return {
            "quota_ratio": self.quota_ratio,
            "quota_limit": quota_limit,
            "current_key_count": len(self.K),
            "utilization_rate": (len(self.K) / quota_limit) if quota_limit > 0 else 0.0,
            "tau_add": self.tau_add,
            "tau_swap": self.tau_swap,
        }
