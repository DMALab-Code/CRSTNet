from collections import deque

import numpy as np

from .key_node_selector import KeyNodeSelector


class KLLSketch:
    def __init__(self, k=200):
        self.k = int(k)
        self.samples = []

    def update(self, value):
        self.samples.append(float(value))
        if len(self.samples) > self.k:
            self.samples = self.samples[::2]

    def get_quantile(self, q):
        if not self.samples:
            return 0.0
        arr = sorted(self.samples)
        idx = int(min(max(q, 0), 1) * (len(arr) - 1))
        return float(arr[idx])


class StreamingScore:
    def __init__(self, num_nodes=None, window_size=100, theta=0.3):
        self.window_size = window_size
        self.theta = theta
        self.history = deque(maxlen=window_size)
        self.score = np.zeros(num_nodes if num_nodes is not None else 0)
        self.selector = KeyNodeSelector(theta=theta)

    def update(self, data_or_score):
        arr = np.asarray(data_or_score)
        scores = self.selector.compute_scores(arr) if arr.ndim >= 3 else arr.astype(float)
        self.score = scores
        self.history.append(scores)
        return scores

    def get_current_score(self):
        return self.score if self.score.size else 0.0


def select_key_nodes_bqh(
    streamer,
    p_t=None,
    alpha=0.1,
    k_min_ratio=0.05,
    k_max_ratio=0.3,
    theta_min=0.5,
):
    scores = np.asarray(streamer.get_current_score(), dtype=float)
    if scores.ndim == 0 or scores.size == 0:
        return [], 0.0, 0.0

    N = scores.size
    k = max(1, int(np.floor(k_max_ratio * N)))
    order = np.argsort(-scores)
    keys = order[:k].astype(int).tolist()
    return keys, float(scores[order[k - 1]]), float(k / max(1, N))


def select_key_nodes_base(*args, **kwargs):
    return select_key_nodes_bqh(*args, **kwargs)
