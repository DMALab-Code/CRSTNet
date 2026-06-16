import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../../.."))

from baselines.CRSTNet.arch.efficient_clustering import HGraphMaintainer


def _load_cfg(path):
    module_path = path[:-3].replace("/", ".").replace("\\", ".") if path.endswith(".py") else path
    return __import__(module_path, fromlist=["CFG"]).CFG


def _load_data(dataset_name):
    root = os.path.join("datasets", dataset_name)
    npz_path = os.path.join(root, "data.npz")
    dat_path = os.path.join(root, "data.dat")
    desc_path = os.path.join(root, "desc.json")
    if os.path.exists(npz_path):
        raw = np.load(npz_path)
        data = raw["data"] if "data" in raw else raw[raw.files[0]]
    elif os.path.exists(dat_path) and os.path.exists(desc_path):
        with open(desc_path, "r") as file:
            desc = json.load(file)
        data = np.memmap(dat_path, dtype="float32", mode="r", shape=tuple(desc["shape"]))
    else:
        raise FileNotFoundError(f"Cannot find data.npz or data.dat+desc.json under {root}")
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"Expected dataset data [T,N,F], got {data.shape}")
    return np.nan_to_num(data)


def _to_numpy_adj(adj, num_nodes):
    if isinstance(adj, (list, tuple)):
        adj = adj[0]
    adj = np.asarray(adj, dtype=float)
    if adj.ndim == 3:
        adj = adj[0]
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"adj shape {adj.shape} does not match num_nodes={num_nodes}")
    return adj


def build_cache(cfg_path, max_profile_steps=2016, force=True):
    cfg = _load_cfg(cfg_path)
    dataset = cfg.DATASET.NAME
    params = cfg.MODEL.PARAM
    num_nodes = int(params["num_nodes"])
    cache_path = params.get("initial_hgraph_cache_path")
    if not cache_path:
        raise ValueError("MODEL.PARAM.initial_hgraph_cache_path is not set")
    if os.path.exists(cache_path) and not force:
        return cache_path

    data = _load_data(dataset)
    train_ratio = float(cfg.DATASET.PARAM["train_val_test_ratio"][0])
    train_end = max(1, int(len(data) * train_ratio))
    train = data[:train_end, :, : int(params["input_dim"])]
    if max_profile_steps and train.shape[0] > max_profile_steps:
        train = train[-int(max_profile_steps) :]

    adj = _to_numpy_adj(params["adj_mx"], num_nodes)
    maintainer = HGraphMaintainer(
        theta=float(params.get("key_node_ratio", 0.25)),
        n_clusters=int(params.get("n_clusters", 6)),
        lambda_spatial=float(params.get("lambda_spatial", 0.5)),
        eta=float(params.get("eta", 0.05)),
        eps=float(params.get("eps", 1e-5)),
        dtw_radius=int(params.get("dtw_radius", 6)),
        affected_ratio_full_rebuild=float(params.get("affected_ratio_full_rebuild", 0.35)),
        promote_margin=float(params.get("promote_margin", 1.05)),
        demote_margin=float(params.get("demote_margin", 0.90)),
        demote_patience=int(params.get("demote_patience", 2)),
        min_cluster_size=int(params.get("min_cluster_size", 3)),
        max_cluster_size=int(params.get("max_cluster_size", 8)),
        error_key_ratio=float(params.get("error_key_ratio", 0.30)),
        model_error_warmup_epochs=int(params.get("model_error_warmup_epochs", 10)),
        model_error_full_epochs=int(params.get("model_error_full_epochs", 30)),
    )
    snapshot = maintainer.build_initial(train, adj)

    payload = {
        "dataset": dataset,
        "profile_shape": tuple(train.shape),
        "key_nodes": snapshot.key_nodes,
        "clusters": snapshot.clusters,
        "scores": snapshot.scores,
        "key_threshold": snapshot.key_threshold,
        "version": snapshot.version,
        "profile_ema": train,
        "stats": {
            "stable_ranges": maintainer.stats.stable_ranges,
            "residuals": maintainer.stats.residuals,
            "last_smp": maintainer.stats.last_smp,
            "delta_s": maintainer.stats.delta_s,
            "delta_e": maintainer.stats.delta_e,
            "delta_d": maintainer.stats.delta_d,
            "delta_c": maintainer.stats.delta_c,
        },
        "distance_cache_state": {
            "shortest_path_cache": maintainer.distance_cache.shortest_path_cache,
            "dtw_cache_data": list(maintainer.distance_cache.dtw_cache.data.items()),
        },
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        "built_hgraph_cache",
        f"path={cache_path}",
        f"profile_shape={train.shape}",
        f"keys={len(snapshot.key_nodes)}",
        f"clusters={len(snapshot.clusters)}",
        f"dtw_cache={len(maintainer.distance_cache.dtw_cache.data)}",
    )
    return cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", required=True)
    parser.add_argument("--max-profile-steps", type=int, default=2016)
    parser.add_argument("--no-force", action="store_true")
    args = parser.parse_args()
    build_cache(args.cfg, max_profile_steps=args.max_profile_steps, force=not args.no_force)


if __name__ == "__main__":
    main()
