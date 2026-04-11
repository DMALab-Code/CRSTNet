import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque

class KLLSketch:

    def __init__(self, k=200):
        self.k = k
        self.samples = []

    def update(self, value):

        self.samples.append(value)
        if len(self.samples) > self.k:
            self.samples = self.samples[::2]

    def get_quantile(self, q):

        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(q * len(sorted_samples))
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

class StreamingScore:

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.scores = deque(maxlen=window_size)

    def update(self, score):

        self.scores.append(score)

    def get_current_score(self):

        if not self.scores:
            return 0.0
        return self.scores[-1]

def select_key_nodes_bqh(streamer, p_t, alpha=0.1, k_min_ratio=0.05, k_max_ratio=0.3, theta_min=0.5):

    return [], 0.5, 0.1

def select_key_nodes_base(streamer, p_t, alpha=0.1, k_min_ratio=0.05, k_max_ratio=0.3, theta_min=0.5):

    return [], 0.5, 0.1
