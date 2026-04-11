import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Union
import heapq
from .key_node_selector import QuotaSwapKeySelector

def select_dynamic_key_nodes(data, percentile=95, importance=None):

    T, N, F = data.shape

    entropies = np.zeros(N)
    for i in range(N):
        ts = data[:, i, 0]
        hist, _ = np.histogram(ts, bins=20, density=True)
        hist = hist + 1e-10
        entropies[i] = -np.sum(hist * np.log(hist))

    latest = data[-1, :, 0]
    prev = data[-2, :, 0]
    change_rate = np.abs(latest - prev) / (np.abs(prev) + 1e-5)

    entropies_norm = entropies / (np.max(entropies) + 1e-10)
    change_rate_norm = change_rate / (np.max(change_rate) + 1e-10)

    scores = entropies_norm + change_rate_norm

    if importance is not None:
        importance_norm = importance / (np.max(importance) + 1e-10)
        scores = scores + 0.5 * importance_norm

    threshold = np.percentile(scores, percentile)
    key_nodes = np.where(scores >= threshold)[0]
    non_key_nodes = np.setdiff1d(np.arange(N), key_nodes)

    return key_nodes, non_key_nodes, scores, threshold

def efficient_dtw_distance(data, node_indices, top_k=5, downsample=4, n_jobs=4):

    T, N, F = data.shape

    if downsample > 1:
        data_down = data[::downsample]
    else:
        data_down = data

    def compute_dtw_pair(i, j):
        if i >= j:
            return 0.0
        ts1 = data_down[:, node_indices[i], 0]
        ts2 = data_down[:, node_indices[j], 0]
        return fastdtw(ts1, ts2, radius=6)[0]

    n_nodes = len(node_indices)
    distances = Parallel(n_jobs=n_jobs)(
        delayed(compute_dtw_pair)(i, j)
        for i in range(n_nodes)
        for j in range(i+1, n_nodes)
    )

    D = np.zeros((n_nodes, n_nodes))
    idx = 0
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            D[i, j] = distances[idx]
            D[j, i] = distances[idx]
            idx += 1

    return D

def detect_distribution_change(data, prev_state=None, threshold=2.0):

    T, N, F = data.shape

    current_state = {
        'mean': np.mean(data[:, :, 0], axis=0),
        'std': np.std(data[:, :, 0], axis=0),
        'trend': np.polyfit(np.arange(T), np.mean(data[:, :, 0], axis=1), 1)[0]
    }

    if prev_state is None:
        return [], True, current_state

    mean_change = np.abs(current_state['mean'] - prev_state['mean']) / (np.abs(prev_state['mean']) + 1e-5)
    std_change = np.abs(current_state['std'] - prev_state['std']) / (np.abs(prev_state['std']) + 1e-5)
    trend_change = np.abs(current_state['trend'] - prev_state['trend'])

    change_score = mean_change + std_change + 0.1 * trend_change
    changed_nodes = np.where(change_score > threshold)[0]

    change_ratio = len(changed_nodes) / N
    need_full_update = change_ratio > 0.3

    return changed_nodes, need_full_update, current_state

def fastdtw(x, y, radius=6):

    n, m = len(x), len(y)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - radius), min(m + 1, i + radius + 1)):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

    return dtw_matrix[n, m], None

class TimeSeriesSummary:

    def __init__(self, paa_segments=16):
        self.paa_segments = paa_segments
        self.count = 0
        self.mu = None
        self.env_upper = None
        self.env_lower = None
        self.last_update_ts = 0
        self.members = set()

    def update(self, ts, timestamp=0):

        self.count += 1
        self.last_update_ts = timestamp

        paa_ts = self._paa_compress(ts)

        if self.mu is None:
            self.mu = paa_ts
            self.env_upper = paa_ts.copy()
            self.env_lower = paa_ts.copy()
        else:

            alpha = 1.0 / self.count
            self.mu = (1 - alpha) * self.mu + alpha * paa_ts

            self.env_upper = np.maximum(self.env_upper, paa_ts)
            self.env_lower = np.minimum(self.env_lower, paa_ts)

    def _paa_compress(self, ts):

        if len(ts) <= self.paa_segments:
            return ts

        segment_size = len(ts) // self.paa_segments
        paa_ts = []

        for i in range(self.paa_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, len(ts))
            segment = ts[start:end]
            paa_ts.append(np.mean(segment))

        return np.array(paa_ts)

    def merge_with(self, other):

        if other.count == 0:
            return

        total_count = self.count + other.count
        if total_count == 0:
            return

        if self.mu is None:
            self.mu = other.mu.copy()
            self.env_upper = other.env_upper.copy()
            self.env_lower = other.env_lower.copy()
        else:

            self.mu = (self.count * self.mu + other.count * other.mu) / total_count

            self.env_upper = np.maximum(self.env_upper, other.env_upper)
            self.env_lower = np.minimum(self.env_lower, other.env_lower)

        self.count = total_count
        self.members.update(other.members)

class HCIndex:

    def __init__(self, graph_adj, paa_segments=16, window_ratio=0.1,
                 eta_env=0.6, eta_mu=0.15, theta_merge=0.8, theta_var=0.5):

        self.adj = graph_adj
        self.paa_segments = paa_segments
        self.window_ratio = window_ratio
        self.eta_env = eta_env
        self.eta_mu = eta_mu
        self.theta_merge = theta_merge
        self.theta_var = theta_var

        self.super_summaries = {}
        self.key_summaries = {}
        self.node_to_super = {}

        self.super_adjacency = defaultdict(set)
        self._build_super_adjacency()

        self.update_count = 0
        self.merge_count = 0
        self.split_count = 0

    def _build_super_adjacency(self):

        n = self.adj.shape[0]
        for i in range(n):
            for j in range(n):
                if self.adj[i, j] > 0:

                    self.super_adjacency[i].add(j)

    def update_summary(self, node_id, ts, timestamp=0, is_key=False):

        if is_key:
            if node_id not in self.key_summaries:
                self.key_summaries[node_id] = TimeSeriesSummary(self.paa_segments)
            self.key_summaries[node_id].update(ts, timestamp)
        else:

            super_id = self.node_to_super.get(node_id, node_id)
            if super_id not in self.super_summaries:
                self.super_summaries[super_id] = TimeSeriesSummary(self.paa_segments)
            self.super_summaries[super_id].update(ts, timestamp)
            self.super_summaries[super_id].members.add(node_id)

    def neighbors_of(self, node_or_cluster_id):

        if node_or_cluster_id in self.super_adjacency:
            return list(self.super_adjacency[node_or_cluster_id])
        return []

    def lb_kim(self, ts1, ts2):

        if len(ts1) == 0 or len(ts2) == 0:
            return float('inf')

        endpoint_dist = (ts1[0] - ts2[0]) ** 2 + (ts1[-1] - ts2[-1]) ** 2

        min1, max1 = np.min(ts1), np.max(ts1)
        min2, max2 = np.min(ts2), np.max(ts2)
        extremum_dist = max(0, max1 - min2) ** 2 + max(0, max2 - min1) ** 2

        return np.sqrt(endpoint_dist + extremum_dist)

    def lb_keogh(self, ts_paa, env_lower, env_upper):

        if env_lower is None or env_upper is None:
            return float('inf')

        min_len = min(len(ts_paa), len(env_lower), len(env_upper))
        ts_paa = ts_paa[:min_len]
        env_lower = env_lower[:min_len]
        env_upper = env_upper[:min_len]

        lb_sum = 0.0
        for i in range(min_len):
            if ts_paa[i] > env_upper[i]:
                lb_sum += (ts_paa[i] - env_upper[i]) ** 2
            elif ts_paa[i] < env_lower[i]:
                lb_sum += (env_lower[i] - ts_paa[i]) ** 2

        return np.sqrt(lb_sum)

    def multi_level_pruning(self, ts1, ts2, threshold=float('inf')):

        lb_kim_dist = self.lb_kim(ts1, ts2)
        if lb_kim_dist >= threshold:
            return lb_kim_dist, False

        ts1_paa = self._paa_compress(ts1)
        ts2_paa = self._paa_compress(ts2)

        env_lower = np.minimum(ts1_paa, ts2_paa)
        env_upper = np.maximum(ts1_paa, ts2_paa)

        lb_keogh_dist = self.lb_keogh(ts1_paa, env_lower, env_upper)
        if lb_keogh_dist >= threshold:
            return lb_keogh_dist, False

        return self._compute_dtw_with_early_stop(ts1, ts2, threshold), True

    def _compute_dtw_with_early_stop(self, ts1, ts2, threshold, window_ratio=0.1):

        n, m = len(ts1), len(ts2)
        w = max(1, int(window_ratio * max(n, m)))

        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(max(1, i - w), min(m + 1, i + w + 1)):
                cost = abs(ts1[i-1] - ts2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )

                if dtw_matrix[i, j] >= threshold:
                    return threshold

        return dtw_matrix[n, m]

    def get_center(self, super_id):

        if super_id in self.super_summaries:
            summary = self.super_summaries[super_id]
            return summary.mu if summary.mu is not None else None
        return None

    def estimate_marginal_gain(self, u, K, data, lambda_cov=0.5, alpha_sim=0.5):

        if u in K:
            return 0.0

        if u >= data.shape[1]:
            return 0.0
        ts = data[:, u, 0]
        ts_paa = self._paa_compress(ts)

        importance_score = np.var(ts)

        coverage_gain = 0.0
        neighbors = self.neighbors_of(u)

        max_neighbors = min(10, len(neighbors))
        for neighbor_id in neighbors[:max_neighbors]:
            if neighbor_id in self.super_summaries:
                summary = self.super_summaries[neighbor_id]
                if summary.env_upper is not None and summary.env_lower is not None:

                    lb_dist = self.lb_keogh(ts_paa, summary.env_lower, summary.env_upper)
                    if lb_dist < self.theta_merge:
                        coverage_gain += 1.0 / (1.0 + lb_dist)

        total_gain = lambda_cov * importance_score + (1 - lambda_cov) * coverage_gain
        return total_gain

    def estimate_retain_gain(self, w, K, data):

        if w not in K:
            return 0.0

        if w >= data.shape[1]:
            return 0.0
        ts = data[:, w, 0]
        return self._compute_importance_score(ts)

    def _paa_compress(self, ts):

        if len(ts) <= self.paa_segments:
            return ts

        segment_size = len(ts) // self.paa_segments
        paa_ts = []

        for i in range(self.paa_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, len(ts))
            segment = ts[start:end]
            paa_ts.append(np.mean(segment))

        return np.array(paa_ts)

    def _compute_importance_score(self, ts):

        hist, _ = np.histogram(ts, bins=20, density=True)
        hist = hist + 1e-10
        entropy = -np.sum(hist * np.log(hist))

        if len(ts) > 1:
            change_rate = np.abs(ts[-1] - ts[-2]) / (np.abs(ts[-2]) + 1e-5)
        else:
            change_rate = 0.0

        return entropy + change_rate

    def try_merge(self, ci, cj):

        if ci not in self.super_summaries or cj not in self.super_summaries:
            return False

        summary_i = self.super_summaries[ci]
        summary_j = self.super_summaries[cj]

        if not self._envelope_overlap(summary_i, summary_j):
            return False

        if not self._mean_close(summary_i, summary_j):
            return False

        if summary_i.mu is not None and summary_j.mu is not None:
            dtw_dist = self._compute_dtw(summary_i.mu, summary_j.mu)
            if dtw_dist <= self.theta_merge:

                summary_i.merge_with(summary_j)
                del self.super_summaries[cj]
                self.merge_count += 1
                return True

        return False

    def try_split(self, c):

        if c not in self.super_summaries:
            return False

        summary = self.super_summaries[c]

        if not (self._var_large(summary) or self._env_wide(summary)):
            return False

        if len(summary.members) < 2:
            return False

        members = list(summary.members)
        if len(members) < 2:
            return False

        mid = len(members) // 2
        sub1_members = set(members[:mid])
        sub2_members = set(members[mid:])

        new_c1 = max(self.super_summaries.keys()) + 1 if self.super_summaries else 0
        new_c2 = new_c1 + 1

        self.super_summaries[new_c1] = TimeSeriesSummary(self.paa_segments)
        self.super_summaries[new_c2] = TimeSeriesSummary(self.paa_segments)

        for member in sub1_members:
            self.node_to_super[member] = new_c1
            self.super_summaries[new_c1].members.add(member)
        for member in sub2_members:
            self.node_to_super[member] = new_c2
            self.super_summaries[new_c2].members.add(member)

        del self.super_summaries[c]
        self.split_count += 1
        return True

    def _envelope_overlap(self, summary_i, summary_j):

        if (summary_i.env_upper is None or summary_i.env_lower is None or
            summary_j.env_upper is None or summary_j.env_lower is None):
            return False

        overlap = 0
        total = 0
        min_len = min(len(summary_i.env_upper), len(summary_j.env_upper))

        for i in range(min_len):
            if (summary_i.env_lower[i] <= summary_j.env_upper[i] and
                summary_j.env_lower[i] <= summary_i.env_upper[i]):
                overlap += 1
            total += 1

        return overlap / total >= self.eta_env if total > 0 else False

    def _mean_close(self, summary_i, summary_j):

        if summary_i.mu is None or summary_j.mu is None:
            return False

        mean_diff = np.linalg.norm(summary_i.mu - summary_j.mu)
        return mean_diff <= self.eta_mu

    def _var_large(self, summary):

        if summary.mu is None:
            return False

        var_estimate = np.var(summary.mu)
        return var_estimate >= self.theta_var

    def _env_wide(self, summary):

        if summary.env_upper is None or summary.env_lower is None:
            return False

        width = np.mean(summary.env_upper - summary.env_lower)
        return width >= self.theta_var

    def _compute_dtw(self, ts1, ts2):

        return fastdtw(ts1, ts2, radius=int(self.window_ratio * len(ts1)))[0]

    def local_insert_or_swap(self, u, as_key=False, data=None):

        if as_key:

            if u not in self.key_summaries:
                self.key_summaries[u] = TimeSeriesSummary(self.paa_segments)
            if data is not None and u < data.shape[1]:
                ts = data[:, u, 0]
                self.key_summaries[u].update(ts)
        else:

            if data is not None and u < data.shape[1]:
                ts = data[:, u, 0]
                ts_paa = self._paa_compress(ts)

                best_super = None
                best_dist = float('inf')

                neighbors = self.neighbors_of(u)
                for neighbor_id in neighbors:
                    if neighbor_id in self.super_summaries:
                        summary = self.super_summaries[neighbor_id]
                        if summary.env_upper is not None and summary.env_lower is not None:
                            lb_dist = self.lb_keogh(ts_paa, summary.env_lower, summary.env_upper)
                            if lb_dist < best_dist:
                                best_dist = lb_dist
                                best_super = neighbor_id

                if best_super is not None and best_dist <= self.theta_merge:

                    self.node_to_super[u] = best_super
                    self.super_summaries[best_super].members.add(u)
                    self.super_summaries[best_super].update(ts)
                else:

                    new_super_id = max(self.super_summaries.keys()) + 1 if self.super_summaries else 0
                    self.super_summaries[new_super_id] = TimeSeriesSummary(self.paa_segments)
                    self.super_summaries[new_super_id].members.add(u)
                    self.super_summaries[new_super_id].update(ts)
                    self.node_to_super[u] = new_super_id

    def get_statistics(self):

        return {
            'num_super_nodes': len(self.super_summaries),
            'num_key_nodes': len(self.key_summaries),
            'update_count': self.update_count,
            'merge_count': self.merge_count,
            'split_count': self.split_count,
            'total_members': sum(len(s.members) for s in self.super_summaries.values())
        }

class LocalOperationManager:

    def __init__(self, hc_index, quota_selector, theta_join=0.9, theta_outlier=1.5):

        self.hc_index = hc_index
        self.quota_selector = quota_selector
        self.theta_join = theta_join
        self.theta_outlier = theta_outlier

        self.promotion_count = 0
        self.demotion_count = 0
        self.reassignment_count = 0
        self.outlier_count = 0

    def promote_node(self, u, data):

        if u in self.quota_selector.K:
            return True

        if len(self.quota_selector.K) < self.quota_selector.Q:

            self.quota_selector.K.add(u)
            if u in self.hc_index.node_to_super:

                super_id = self.hc_index.node_to_super[u]
                if super_id in self.hc_index.super_summaries:
                    self.hc_index.super_summaries[super_id].members.discard(u)
                del self.hc_index.node_to_super[u]

            if u < data.shape[1]:
                ts = data[:, u, 0]
                self.hc_index.update_summary(u, ts, is_key=True)

            self.promotion_count += 1
            return True
        else:

            return self._try_swap_promotion(u, data)

    def demote_node(self, u, data):

        if u not in self.quota_selector.K:
            return True

        self.quota_selector.K.discard(u)

        best_super = self._find_best_super_for_node(u, data)

        if best_super is not None:

            self.hc_index.node_to_super[u] = best_super
            self.hc_index.super_summaries[best_super].members.add(u)
            if u < data.shape[1]:
                ts = data[:, u, 0]
                self.hc_index.super_summaries[best_super].update(ts)
        else:

            self._create_new_super_for_node(u, data)

        self.demotion_count += 1
        return True

    def reassign_outlier(self, u, data):

        if u in self.quota_selector.K:
            return True

        if not self._is_outlier(u, data):
            return True

        best_super = self._find_best_super_for_node(u, data)

        if best_super is not None:

            old_super = self.hc_index.node_to_super.get(u)
            if old_super and old_super in self.hc_index.super_summaries:
                self.hc_index.super_summaries[old_super].members.discard(u)

            self.hc_index.node_to_super[u] = best_super
            self.hc_index.super_summaries[best_super].members.add(u)
            if u < data.shape[1]:
                ts = data[:, u, 0]
                self.hc_index.super_summaries[best_super].update(ts)

            self.reassignment_count += 1
        else:

            self._create_new_super_for_node(u, data)
            self.outlier_count += 1

        return True

    def _try_swap_promotion(self, u, data):

        gain_u = self.hc_index.estimate_marginal_gain(u, self.quota_selector.K, data)

        if not self.quota_selector.min_heap:
            return False

        rg_w, w_star = self.quota_selector.min_heap[0]

        swap_gain = gain_u - rg_w

        if swap_gain > self.quota_selector.tau_swap:

            self.quota_selector.K.remove(w_star)
            self.quota_selector.K.add(u)

            heapq.heapreplace(self.quota_selector.min_heap,
                            (self.hc_index.estimate_retain_gain(u, self.quota_selector.K, data), u))

            self.demote_node(w_star, data)

            if u < data.shape[1]:
                ts = data[:, u, 0]
                self.hc_index.update_summary(u, ts, is_key=True)

            self.promotion_count += 1
            return True

        return False

    def _find_best_super_for_node(self, u, data):

        if u >= data.shape[1]:
            return None

        ts = data[:, u, 0]
        ts_paa = self.hc_index._paa_compress(ts)

        best_super = None
        best_dist = float('inf')

        neighbors = self.hc_index.neighbors_of(u)
        for neighbor_id in neighbors:
            if neighbor_id in self.hc_index.super_summaries:
                summary = self.hc_index.super_summaries[neighbor_id]
                if summary.env_upper is not None and summary.env_lower is not None:
                    lb_dist = self.hc_index.lb_keogh(ts_paa, summary.env_lower, summary.env_upper)
                    if lb_dist < best_dist and lb_dist <= self.theta_join:
                        best_dist = lb_dist
                        best_super = neighbor_id

        return best_super

    def _create_new_super_for_node(self, u, data):

        new_super_id = max(self.hc_index.super_summaries.keys()) + 1 if self.hc_index.super_summaries else 0

        self.hc_index.super_summaries[new_super_id] = TimeSeriesSummary(self.hc_index.paa_segments)
        self.hc_index.super_summaries[new_super_id].members.add(u)
        self.hc_index.node_to_super[u] = new_super_id

        if u < data.shape[1]:
            ts = data[:, u, 0]
            self.hc_index.super_summaries[new_super_id].update(ts)

    def _is_outlier(self, u, data):

        if u >= data.shape[1]:
            return False

        ts = data[:, u, 0]
        ts_paa = self.hc_index._paa_compress(ts)

        neighbors = self.hc_index.neighbors_of(u)
        min_dist = float('inf')

        for neighbor_id in neighbors:
            if neighbor_id in self.hc_index.super_summaries:
                summary = self.hc_index.super_summaries[neighbor_id]
                if summary.env_upper is not None and summary.env_lower is not None:
                    lb_dist = self.hc_index.lb_keogh(ts_paa, summary.env_lower, summary.env_upper)
                    min_dist = min(min_dist, lb_dist)

        return min_dist > self.theta_outlier

    def batch_operations(self, data, operation_list):

        results = []

        for op_type, u in operation_list:
            if op_type == 'promote':
                success = self.promote_node(u, data)
            elif op_type == 'demote':
                success = self.demote_node(u, data)
            elif op_type == 'reassign':
                success = self.reassign_outlier(u, data)
            else:
                success = False

            results.append((op_type, u, success))

        return results

    def get_statistics(self):

        return {
            'promotion_count': self.promotion_count,
            'demotion_count': self.demotion_count,
            'reassignment_count': self.reassignment_count,
            'outlier_count': self.outlier_count,
            'total_operations': (self.promotion_count + self.demotion_count +
                               self.reassignment_count + self.outlier_count)
        }

class SuperNodeManager:

    def __init__(self, hc_index, merge_cooldown=20, split_cooldown=20):

        self.hc_index = hc_index
        self.merge_cooldown = merge_cooldown
        self.split_cooldown = split_cooldown

        self.last_merge_time = 0
        self.last_split_time = 0

        self.merge_attempts = 0
        self.merge_successes = 0
        self.split_attempts = 0
        self.split_successes = 0

    def try_merge_operations(self, current_step, data):

        if current_step - self.last_merge_time < self.merge_cooldown:
            return []

        merge_results = []
        super_ids = list(self.hc_index.super_summaries.keys())

        for i, ci in enumerate(super_ids):
            for j, cj in enumerate(super_ids[i+1:], i+1):
                if self._are_neighbors(ci, cj):
                    self.merge_attempts += 1
                    success = self.hc_index.try_merge(ci, cj)
                    if success:
                        self.merge_successes += 1
                        merge_results.append((ci, cj, True))
                    else:
                        merge_results.append((ci, cj, False))

        self.last_merge_time = current_step
        return merge_results

    def try_split_operations(self, current_step, data):

        if current_step - self.last_split_time < self.split_cooldown:
            return []

        split_results = []
        super_ids = list(self.hc_index.super_summaries.keys())

        for ci in super_ids:
            if self._should_split(ci):
                self.split_attempts += 1
                success = self.hc_index.try_split(ci)
                if success:
                    self.split_successes += 1
                    split_results.append((ci, True))
                else:
                    split_results.append((ci, False))

        self.last_split_time = current_step
        return split_results

    def _are_neighbors(self, ci, cj):

        members_i = self.hc_index.super_summaries[ci].members
        members_j = self.hc_index.super_summaries[cj].members

        for mi in members_i:
            for mj in members_j:
                if (mi < self.hc_index.adj.shape[0] and
                    mj < self.hc_index.adj.shape[1] and
                    self.hc_index.adj[mi, mj] > 0):
                    return True
        return False

    def _should_split(self, ci):

        if ci not in self.hc_index.super_summaries:
            return False

        summary = self.hc_index.super_summaries[ci]

        if len(summary.members) < 2:
            return False

        if summary.mu is not None:
            var_estimate = np.var(summary.mu)
            if var_estimate >= self.hc_index.theta_var:
                return True

        if (summary.env_upper is not None and summary.env_lower is not None):
            width = np.mean(summary.env_upper - summary.env_lower)
            if width >= self.hc_index.theta_var:
                return True

        return False

    def batch_merge_split(self, current_step, data):

        merge_results = self.try_merge_operations(current_step, data)
        split_results = self.try_split_operations(current_step, data)

        return {
            'merge_results': merge_results,
            'split_results': split_results,
            'merge_attempts': self.merge_attempts,
            'merge_successes': self.merge_successes,
            'split_attempts': self.split_attempts,
            'split_successes': self.split_successes
        }

    def get_statistics(self):

        return {
            'merge_attempts': self.merge_attempts,
            'merge_successes': self.merge_successes,
            'merge_success_rate': self.merge_successes / max(1, self.merge_attempts),
            'split_attempts': self.split_attempts,
            'split_successes': self.split_successes,
            'split_success_rate': self.split_successes / max(1, self.split_attempts),
            'last_merge_time': self.last_merge_time,
            'last_split_time': self.last_split_time
        }

class OptimizedStructureManager:

    def __init__(self, graph_adj, quota_ratio=0.1, paa_segments=16,
                 merge_cooldown=20, split_cooldown=20, dtw_budget=100, swap_budget=50, fast_mode=True):

        self.fast_mode = fast_mode
        self.hc_index = HCIndex(graph_adj, paa_segments=paa_segments)
        self.quota_selector = QuotaSwapKeySelector(quota_ratio=quota_ratio)
        self.local_ops = LocalOperationManager(self.hc_index, self.quota_selector)
        self.super_manager = SuperNodeManager(self.hc_index, merge_cooldown, split_cooldown)

        self.quota_selector.prepare(graph_adj.shape[0], self.hc_index)

        self.step_count = 0
        self.last_full_update = 0
        self.update_frequency = 20

        self.dtw_budget = dtw_budget
        self.swap_budget = swap_budget
        self.current_dtw_count = 0
        self.current_swap_count = 0
        self.drift_threshold = 0.1
        self.last_performance = None

    def update_structure(self, data, performance_metric=None):

        self.step_count += 1
        T, N, F = data.shape

        if not self._should_update_structure(performance_metric):
            return self._export_structure_info()

        self.current_dtw_count = 0
        self.current_swap_count = 0

        self._update_summaries(data)

        self._update_key_nodes(data, performance_metric)

        self._perform_local_operations(data)

        if self.step_count % self.update_frequency == 0:
            self._perform_merge_split_operations(data)

        return self._export_structure_info()

    def _should_update_structure(self, performance_metric):

        if self.fast_mode:
            if self.step_count % (self.update_frequency * 2) == 0:
                return True
            if self.step_count < 20:
                return True
            return False

        if self.step_count % self.update_frequency == 0:
            return True

        if performance_metric is not None and self.last_performance is not None:
            drift = abs(performance_metric - self.last_performance) / self.last_performance
            if drift > self.drift_threshold * 2:
                return True

        if self.step_count < 100:
            return True

        return False

    def _check_dtw_budget(self):

        return self.current_dtw_count < self.dtw_budget

    def _check_swap_budget(self):

        return self.current_swap_count < self.swap_budget

    def _record_dtw_usage(self, count=1):

        self.current_dtw_count += count

    def _record_swap_usage(self, count=1):

        self.current_swap_count += count

    def _update_summaries(self, data):

        T, N, F = data.shape

        for u in range(N):
            ts = data[:, u, 0]
            is_key = u in self.quota_selector.K
            self.hc_index.update_summary(u, ts, timestamp=self.step_count, is_key=is_key)

    def _update_key_nodes(self, data, performance_metric):

        if performance_metric is not None:
            self.quota_selector.update_performance(performance_metric)

        self.quota_selector.batch_update(data)

    def _perform_local_operations(self, data):

        operation_list = []

        for u in range(data.shape[1]):
            if u not in self.quota_selector.K:
                if self.local_ops._is_outlier(u, data):
                    operation_list.append(('reassign', u))

        if operation_list:
            self.local_ops.batch_operations(data, operation_list)

    def _perform_merge_split_operations(self, data):

        results = self.super_manager.batch_merge_split(self.step_count, data)
        return results

    def _export_structure_info(self):

        key_nodes = list(self.quota_selector.K)

        total_nodes = int(self.hc_index.adj.shape[0]) if hasattr(self.hc_index, 'adj') else len(self.hc_index.node_to_super)
        non_key_nodes = list(self.quota_selector.get_non_key_nodes(total_nodes))

        cluster_indices_list = []
        for super_id, summary in self.hc_index.super_summaries.items():
            if len(summary.members) > 0:
                cluster_indices_list.append(list(summary.members))

        return {
            'key_nodes': key_nodes,
            'non_key_nodes': non_key_nodes,
            'cluster_indices_list': cluster_indices_list,
            'super_nodes': list(self.hc_index.super_summaries.keys()),
            'node_to_super': self.hc_index.node_to_super.copy()
        }

    def get_comprehensive_statistics(self):

        return {
            'step_count': self.step_count,
            'quota_stats': self.quota_selector.get_statistics(),
            'hc_stats': self.hc_index.get_statistics(),
            'local_ops_stats': self.local_ops.get_statistics(),
            'super_manager_stats': self.super_manager.get_statistics()
        }
