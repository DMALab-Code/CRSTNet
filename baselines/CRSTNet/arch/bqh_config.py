def get_bqh_config():
    return {
        "keynode": {
            "theta": 0.3,
            "bins": None,
            "eps": 1e-5,
            "p_base": 0.1,
            "k_min_ratio": 0.05,
            "k_max_ratio": 0.3,
        },
        "clustering": {
            "n_clusters": 6,
            "lambda_spatial": 0.5,
            "one_hop_dtw": True,
            "dtw_radius": 6,
            "max_dtw_cache": 20000,
        },
        "temporal": {
            "main_period": 12,
            "k_min": 2,
            "k_max": 6,
            "kernel_cache_size": 256,
            "eps": 1e-5,
        },
        "spatial": {
            "gamma": 0.5,
            "use_full_graph_fusion": False,
            "route_cache_size": 8,
        },
        "maintenance": {
            "C": 12,
            "eta": 0.05,
            "affected_ratio_full_rebuild": 0.35,
            "freeze_eval_structure": True,
        },
        "budgeter": {
            "B_tree_ops": 2048,
            "B_eval_ops": 4096,
        },
        "indexing": {
            "env_segments": 16,
            "sp_bucket": 10,
            "topK_candidates": 16,
            "env_decay": 0.9,
        },
    }


def validate_bqh_config(cfg):
    required = ["keynode", "clustering", "temporal", "spatial", "maintenance"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    theta = cfg["keynode"].get("theta", 0.3)
    if not (0 < theta <= 1):
        raise ValueError("keynode.theta must be in (0,1].")

    C = cfg["maintenance"].get("C", 12)
    if int(C) <= 0:
        raise ValueError("maintenance.C must be positive.")

    return True
