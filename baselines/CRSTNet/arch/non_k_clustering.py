from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np


class TimeEnvelope:
    def __init__(self, ts_or_segments=16):
        if np.isscalar(ts_or_segments):
            self.segments = int(ts_or_segments)
            self.upper = None
            self.lower = None
        else:
            x = np.asarray(ts_or_segments, dtype=float)
            self.segments = len(x)
            self.upper = x.copy()
            self.lower = x.copy()

    @classmethod
    def from_series(cls, series, segments=16):
        obj = cls(segments)
        obj.update_series(series)
        return obj

    def _paa(self, x):
        x = np.asarray(x, dtype=float)
        if len(x) <= self.segments:
            return x
        cuts = np.linspace(0, len(x), self.segments + 1, dtype=int)
        return np.asarray([x[cuts[i]:cuts[i + 1]].mean() for i in range(self.segments)])

    def update_series(self, series):
        z = self._paa(series)
        if self.upper is None:
            self.upper = z.copy()
            self.lower = z.copy()
        else:
            m = min(len(z), len(self.upper))
            self.upper[:m] = np.maximum(self.upper[:m], z[:m])
            self.lower[:m] = np.minimum(self.lower[:m], z[:m])

    def update_incremental(self, x_t, decay=0.9):
        if self.upper is None:
            self.upper = np.full(self.segments, float(x_t))
            self.lower = np.full(self.segments, float(x_t))
        else:
            self.upper = np.maximum(self.upper * decay, float(x_t))
            self.lower = np.minimum(self.lower * decay, float(x_t))


def lb_keogh(a, b, lower=None):
    if isinstance(a, TimeEnvelope) and isinstance(b, TimeEnvelope):
        if a.upper is None or b.upper is None:
            return 0.0
        m = min(len(a.upper), len(b.upper))
        return float(
            np.mean(
                np.maximum(0, a.lower[:m] - b.upper[:m])
                + np.maximum(0, b.lower[:m] - a.upper[:m])
            )
        )

    x = np.asarray(a, dtype=float)
    upper = np.asarray(b, dtype=float)
    lower = np.asarray(lower, dtype=float)
    m = min(len(x), len(upper), len(lower))
    return float(np.sum(np.maximum(0, x[:m] - upper[:m]) + np.maximum(0, lower[:m] - x[:m])))


class TwoLevelInvertedIndex:
    def __init__(self, sp_bucket=10):
        self.sp_bucket = int(sp_bucket)
        self.index = defaultdict(list)
        self.envs = {}

    def _bucket(self, d):
        return int(float(d) // max(1, self.sp_bucket))

    def update(self, node_id, sp_dist, env):
        b = self._bucket(sp_dist)
        node_id = int(node_id)
        if node_id not in self.index[b]:
            self.index[b].append(node_id)
        self.envs[node_id] = env

    def add(self, key, value):
        self.index[key].append(value)

    def get(self, key):
        return self.index.get(key, [])

    def query(self, sp_dist, env, topK=10):
        b = self._bucket(sp_dist)
        ans = []
        for bb in [b - 1, b, b + 1]:
            for node in self.index.get(bb, []):
                ans.append((node, lb_keogh(env, self.envs.get(node, env))))
        ans.sort(key=lambda x: x[1])
        return ans[:topK]


@dataclass
class CF:
    n: int = 0
    linear_sum: np.ndarray = None
    squared_sum: np.ndarray = None

    @classmethod
    def from_series(cls, ts):
        x = np.asarray(ts, dtype=float)
        return cls(1, x.copy(), x ** 2)

    def add_series(self, ts):
        x = np.asarray(ts, dtype=float)
        if self.n == 0:
            self.n = 1
            self.linear_sum = x.copy()
            self.squared_sum = x ** 2
        else:
            m = min(len(x), len(self.linear_sum))
            self.linear_sum[:m] += x[:m]
            self.squared_sum[:m] += x[:m] ** 2
            self.n += 1

    def merge(self, other):
        if other.n == 0:
            return self
        if self.n == 0:
            self.n = other.n
            self.linear_sum = other.linear_sum.copy()
            self.squared_sum = other.squared_sum.copy()
            return self
        m = min(len(self.linear_sum), len(other.linear_sum))
        self.linear_sum[:m] += other.linear_sum[:m]
        self.squared_sum[:m] += other.squared_sum[:m]
        self.n += other.n
        return self

    @property
    def sse(self):
        if self.n == 0:
            return 0.0
        return float(np.sum(self.squared_sum - (self.linear_sum ** 2) / max(1, self.n)))


@dataclass
class SuperNode:
    id: int
    members: set = field(default_factory=set)
    cf: CF = field(default_factory=CF)
    is_key: bool = False

    def __post_init__(self):
        if not self.members:
            self.members.add(int(self.id))

    def add_member(self, node_id, series=None):
        self.members.add(int(node_id))
        if series is not None:
            self.cf.add_series(series)

    def remove_member(self, node_id):
        self.members.discard(int(node_id))


def ward_cost_cf(cf1, cf2):
    merged = CF()
    merged.merge(cf1)
    merged.merge(cf2)
    return merged.sse - cf1.sse - cf2.sse


def ward_cost_merge(cf1, cf2):
    return ward_cost_cf(cf1, cf2)


def estimate_merge_gain(cf1, cf2):
    return -ward_cost_merge(cf1, cf2)
