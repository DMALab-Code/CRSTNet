#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from CRSTNet.arch.efficient_clustering import (
    compute_correlation_deviation,
    compute_importance_scores,
    compute_member_slice_profiles,
    compute_residual_error,
    compute_shortest_path_distances,
    compute_cluster_slice_mean,
    build_one_hop_dtw_cache,
    infer_eta,
    select_dynamic_key_nodes,
    topology_constrained_ward,
)


def load_data(path: str) -> np.ndarray:
    data_path = Path(path)
    if data_path.suffix == ".npy":
        array = np.load(data_path, allow_pickle=True)
    else:
        archive = np.load(data_path, allow_pickle=True)
        if isinstance(archive, np.lib.npyio.NpzFile):
            if "data" in archive:
                array = archive["data"]
            else:
                first_key = archive.files[0]
                array = archive[first_key]
        else:
            array = archive

    if array.ndim == 2:
        array = array[..., None]
    return np.asarray(array, dtype=np.float64)


def load_adj(path: str, num_nodes: int) -> np.ndarray:
    if not path:
        return np.eye(num_nodes, dtype=np.float64)

    with open(path, "rb") as file:
        obj = pickle.load(file)

    if isinstance(obj, tuple):
        obj = obj[-1]
    return np.asarray(obj, dtype=np.float64)


def profile_thresholds(
    data: np.ndarray,
    adj: np.ndarray,
    train_ratio: float,
    window: int,
    theta: float,
    lambda_value: float,
    maintenance_interval: int,
) -> Dict[str, Any]:
    train_end = max(window, int(len(data) * train_ratio))
    train = data[:train_end]

    shortest_distances = compute_shortest_path_distances(adj)

    non_key_score_samples = []
    residual_samples = []
    deviation_samples = []
    eta_samples = []

    for start in range(0, max(1, train.shape[0] - window + 1), maintenance_interval):
        end = min(train.shape[0], start + window)
        if end - start < max(4, maintenance_interval):
            continue

        window_data = train[start:end]
        scores, _, _ = compute_importance_scores(window_data)
        key_nodes, non_key_nodes, _, _ = select_dynamic_key_nodes(window_data, theta=theta)
        if len(non_key_nodes) > 0:
            non_key_score_samples.extend(scores[non_key_nodes].tolist())

        dtw_cache = build_one_hop_dtw_cache(window_data, adj)
        clusters = topology_constrained_ward(
            non_key_nodes,
            adj,
            window_data,
            shortest_distances=shortest_distances,
            lambda_value=lambda_value,
            dtw_cache=dtw_cache,
            feature_index=0,
        )

        for cluster in clusters:
            if not cluster:
                continue
            member_profiles = compute_member_slice_profiles(window_data, cluster, maintenance_interval, feature_index=0)
            cluster_profile = compute_cluster_slice_mean(member_profiles)
            residual_samples.append(compute_residual_error(member_profiles, cluster_profile))

            eta = infer_eta(member_profiles)
            eta_samples.append(eta)
            stable_low = cluster_profile - eta
            stable_high = cluster_profile + eta
            deviation_samples.extend(
                compute_correlation_deviation(member_profiles[idx], stable_low, stable_high)
                for idx in range(member_profiles.shape[0])
            )

    if not non_key_score_samples:
        fallback_scores, _, _ = compute_importance_scores(train[-window:])
        non_key_score_samples = fallback_scores.tolist()
    if not residual_samples:
        residual_samples = [0.0]
    if not deviation_samples:
        deviation_samples = [infer_eta(train[..., 0])]
    if not eta_samples:
        eta_samples = [infer_eta(train[..., 0])]

    return {
        "delta_s": float(np.quantile(non_key_score_samples, 0.9)),
        "delta_e": float(np.quantile(residual_samples, 0.9)),
        "delta_d": float(np.quantile(deviation_samples, 0.9)),
        "eta": float(np.quantile(eta_samples, 0.9)),
        "theta": float(theta),
        "lambda_value": float(lambda_value),
        "maintenance_interval": int(maintenance_interval),
        "window": int(window),
        "train_ratio": float(train_ratio),
        "profiling_quantile": 0.9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile delta_s, delta_e, and delta_d from the training set using fixed high-quantile cutoffs."
    )
    parser.add_argument("--data", required=True, help="Path to the dataset array (.npz or .npy).")
    parser.add_argument("--adj", default="", help="Path to the adjacency pickle file.")
    parser.add_argument("--dataset", default="dataset", help="Dataset name for the output metadata.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--window", type=int, default=12, help="Historical window length P.")
    parser.add_argument("--theta", type=float, default=0.2, help="Key-node ratio theta.")
    parser.add_argument("--lambda_value", type=float, default=0.5, help="Distance balance lambda.")
    parser.add_argument("--maintenance_interval", type=int, default=5, help="Maintenance interval C.")
    parser.add_argument(
        "--output",
        default="CRSTNet/thresholds/profile.json",
        help="Output JSON path for the profiled thresholds.",
    )
    args = parser.parse_args()

    data = load_data(args.data)
    adj = load_adj(args.adj, data.shape[1])
    profile = profile_thresholds(
        data=data,
        adj=adj,
        train_ratio=args.train_ratio,
        window=args.window,
        theta=args.theta,
        lambda_value=args.lambda_value,
        maintenance_interval=args.maintenance_interval,
    )
    profile["dataset"] = args.dataset

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
