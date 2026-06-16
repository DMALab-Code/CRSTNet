from dataclasses import dataclass
from collections import deque

import numpy as np

from .bqh_config import get_bqh_config, validate_bqh_config
from .efficient_clustering import HGraphMaintainer, HGraphSnapshot
from .select_top_k import StreamingScore


class BQHBudgeter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.used_tree = 0
        self.used_eval = 0
        self.performance_history = deque(maxlen=10)

    def decide_p_base(self, performance_metric=None):
        return float(self.cfg.get("keynode", {}).get("p_base", 0.1))

    def reset_budget(self):
        self.used_tree = 0
        self.used_eval = 0

    def can_afford(self, tree_cost, eval_cost):
        b = self.cfg.get("budgeter", {})
        return (
            self.used_tree + tree_cost <= b.get("B_tree_ops", 2048)
            and self.used_eval + eval_cost <= b.get("B_eval_ops", 4096)
        )

    def consume_budget(self, tree_cost, eval_cost):
        self.used_tree += int(tree_cost)
        self.used_eval += int(eval_cost)


class BQHEstimators:
    def __init__(self, streamer, indexing=None):
        self.streamer = streamer
        self.indexing = indexing

    def gain(self, op):
        _, u = op
        scores = np.asarray(self.streamer.get_current_score())
        return float(scores[u]) if scores.ndim == 1 and 0 <= u < len(scores) else 0.0

    def cost(self, op):
        return {"promote": 2.0, "demote": 3.0, "reassign": 2.0}.get(op[0], 1.0)

    def cost_breakdown(self, op):
        c = int(self.cost(op))
        return c, max(1, c // 2)


class BQHIndexing:
    def __init__(self, n, cfg):
        self.n = int(n)
        self.cfg = cfg
        self.node_to_super = {}
        self.super_nodes = {}
        self.residuals = {}

    def best_super_for(self, u):
        return self.node_to_super.get(int(u), int(u))

    def update_residuals(self, residuals):
        self.residuals.update(residuals)


@dataclass
class BQHState:
    cfg: dict
    maintainer: HGraphMaintainer
    snapshot: HGraphSnapshot
    streamer: StreamingScore
    budgeter: BQHBudgeter
    indexing: BQHIndexing
    step: int = 0


def prepare_state(num_nodes, cfg=None):
    cfg = get_bqh_config() if cfg is None else cfg
    validate_bqh_config(cfg)

    maintainer = HGraphMaintainer(
        theta=cfg["keynode"]["theta"],
        n_clusters=cfg["clustering"]["n_clusters"],
        lambda_spatial=cfg["clustering"]["lambda_spatial"],
        eta=cfg["maintenance"]["eta"],
        eps=cfg["keynode"].get("eps", 1e-5),
        dtw_radius=cfg["clustering"].get("dtw_radius", 6),
        affected_ratio_full_rebuild=cfg["maintenance"].get("affected_ratio_full_rebuild", 0.35),
        max_dtw_cache=cfg["clustering"].get("max_dtw_cache", 20000),
    )

    return BQHState(
        cfg=cfg,
        maintainer=maintainer,
        snapshot=None,
        streamer=StreamingScore(num_nodes=num_nodes, theta=cfg["keynode"]["theta"]),
        budgeter=BQHBudgeter(cfg),
        indexing=BQHIndexing(num_nodes, cfg),
    )


def step_update(state, data, adj):
    state.step += 1
    state.streamer.update(data)

    C = int(state.cfg.get("maintenance", {}).get("C", 12))
    if state.snapshot is None:
        state.snapshot = state.maintainer.build_initial(data, adj)
    elif state.step % max(1, C) == 0:
        state.snapshot = state.maintainer.update(data, adj, state.snapshot)

    return state.snapshot


def update_and_get_clusters(state, data, adj):
    snap = step_update(state, data, adj)
    return snap.key_nodes, snap.clusters
