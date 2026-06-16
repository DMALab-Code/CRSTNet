import numpy as np


class KeyNodeSelector:
    """
    Paper-aligned key-node selector.

    Score_y = normalized temporal entropy H_y * normalized fluctuation rate R_y.
    The number of key nodes never exceeds floor(theta * |V|).
    """
    def __init__(self, bins=None, eps=1e-5, theta=0.3):
        self.bins = bins
        self.eps = eps
        self.theta = theta
        self.bin_edges = None
        self.last_details = {}

    def _as_tnf(self, data):
        data = np.asarray(data)
        if data.ndim == 4:
            data = data[0]
        if data.ndim != 3:
            raise ValueError(f"Expected [T,N,F] or [B,T,N,F], got {data.shape}")
        return data

    def _get_edges(self, x, T):
        B = self.bins if self.bins is not None else max(2, min(20, T // 2))
        if self.bin_edges is not None and len(self.bin_edges) == B + 1:
            return self.bin_edges

        lo, hi = np.percentile(x, [1, 99])
        if abs(hi - lo) < self.eps:
            hi = lo + self.eps
        self.bin_edges = np.linspace(lo, hi + self.eps, B + 1)
        return self.bin_edges

    def compute_components(self, data):
        data = self._as_tnf(data)
        T, N, _ = data.shape
        x = data[:, :, 0].astype(float)
        edges = self._get_edges(x, T)

        H = np.zeros(N, dtype=float)
        for i in range(N):
            hist, _ = np.histogram(np.clip(x[:, i], edges[0], edges[-1]), bins=edges)
            q = hist.astype(float) / max(1, T)
            q = q[q > 0]
            H[i] = -np.sum(q * np.log(np.maximum(q, 1e-12)))

        if T > 1:
            R = np.abs(x[-1] - x[-2]) / (np.abs(x[-2]) + self.eps)
        else:
            R = np.zeros(N, dtype=float)

        Hn = H / (np.max(H) + self.eps)
        Rn = R / (np.max(R) + self.eps)
        score = Hn * Rn

        self.last_details = {
            "entropy": H,
            "fluctuation": R,
            "entropy_norm": Hn,
            "fluctuation_norm": Rn,
            "score": score,
        }
        return Hn, Rn, score

    def compute_scores(self, data):
        return self.compute_components(data)[-1]

    def select(self, data, theta=None, max_keys=None):
        data = self._as_tnf(data)
        _, N, _ = data.shape
        theta = self.theta if theta is None else theta

        k = max(1, min(N, int(np.floor(theta * N))))
        if max_keys is not None:
            k = min(k, int(max_keys))

        scores = self.compute_scores(data)
        order = np.argsort(-scores, kind="mergesort")
        key_nodes = sorted(int(i) for i in order[:k])
        key_set = set(key_nodes)
        non_key_nodes = [i for i in range(N) if i not in key_set]
        threshold = float(scores[order[k - 1]])
        return key_nodes, non_key_nodes, scores, threshold


class QuotaSwapKeySelector:
    """
    Compatibility wrapper.
    """
    def __init__(self, quota_ratio=0.3, eps=1e-5, bins=None, **kwargs):
        self.quota_ratio = quota_ratio
        self.selector = KeyNodeSelector(bins=bins, eps=eps, theta=quota_ratio)
        self.K = set()
        self.Q = None

    def prepare(self, num_nodes, hc_index=None):
        self.Q = max(1, int(np.floor(self.quota_ratio * num_nodes)))
        self.K = set()

    def update(self, data):
        key_nodes, _, scores, threshold = self.selector.select(data, max_keys=self.Q)
        self.K = set(key_nodes)
        return key_nodes, scores, threshold

    def consider(self, u, data):
        return self.update(data)
