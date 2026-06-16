import os
import sys

from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import CRSTNet
from .runner import CRSTNetRunner


DATASET_NUM_NODES = {
    "METR-LA": 207,
    "PEMS-BAY": 325,
    "PEMS04": 307,
    "PEMS08": 170,
}
FORWARD_FEATURES = [0, 1, 2]
MAX_GROUPS = {
    "METR-LA": 74,
    "PEMS-BAY": 96,
    "PEMS04": 104,
    "PEMS08": 60,
}


def _cluster_count(data_name: str) -> int:
    num_nodes = DATASET_NUM_NODES[data_name]
    key_ratio = _key_ratio(data_name)
    target_cluster_size = {
        "METR-LA": 5,
        "PEMS-BAY": 6,
        "PEMS04": 5,
        "PEMS08": 4,
    }[data_name]
    non_key_nodes = max(1, int(round(num_nodes * (1.0 - key_ratio))))
    key_nodes = max(1, int(round(num_nodes * key_ratio)))
    by_size = max(4, int(round(non_key_nodes / target_cluster_size)))
    by_budget = max(4, MAX_GROUPS[data_name] - key_nodes)
    return min(by_size, by_budget)


def _key_ratio(data_name: str) -> float:
    return {
        "METR-LA": 0.20,
        "PEMS-BAY": 0.18,
        "PEMS04": 0.22,
        "PEMS08": 0.20,
    }[data_name]


def build_crstnet_cfg(data_name: str, num_epochs: int = 100):
    regular_settings = get_regular_settings(data_name)
    input_len = regular_settings["INPUT_LEN"]
    output_len = regular_settings["OUTPUT_LEN"]
    screen_interval = int(input_len + output_len)
    min_repair_interval = int(3 * screen_interval)
    train_val_test_ratio = regular_settings["TRAIN_VAL_TEST_RATIO"]
    norm_each_channel = regular_settings["NORM_EACH_CHANNEL"]
    rescale = regular_settings["RESCALE"]
    null_val = regular_settings["NULL_VAL"]
    _, adj_mx = load_adj(os.path.join("datasets", data_name, "adj_mx.pkl"), "doubletransition")

    model_param = {
        "adj_mx": adj_mx,
        "num_nodes": DATASET_NUM_NODES[data_name],
        "input_dim": len(FORWARD_FEATURES),
        "output_dim": 1,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "key_node_ratio": _key_ratio(data_name),
        "n_clusters": _cluster_count(data_name),
        "lambda_spatial": 0.5,
        "update_frequency": screen_interval,
        "gamma": 0.5,
        "eta": 0.05,
        "use_persistence_anchor": True,
        "persistence_anchor_scale": 1.0,
        "use_full_graph_fusion": False,
        "main_period": 12,
        "k_min": 2,
        "k_max": 6,
        "eps": 1e-5,
        "dtw_radius": 6,
        "affected_ratio_full_rebuild": 0.35,
        "freeze_eval_structure": True,
        "route_cache_size": 8,
        "temporal_kernel_cache_size": 256,
        "temporal_soft_top_k": 2,
        "soft_warmup_epochs": 10,
        "soft_until_epoch": 30,
        "soft_interval": 1,
        "sparse_soft_margin": 0.25,
        "temporal_plan_ema_beta": 0.85,
        "temporal_basis_kernels": (3, 5, 11),
        "temporal_gate_ema_beta": 0.90,
        "temporal_norm_groups": 8,
        "temporal_diagnostics_interval": 50,
        "input_window": input_len,
        "output_window": output_len,
        "profile_ema_beta": 0.90,
        "anchor_gate_start": 0.95,
        "anchor_gate_end": 0.50,
        "horizon_head_gate_start": 0.20,
        "horizon_head_gate_end": 0.55,
        "initial_hgraph_cache_path": os.path.join(
            "baselines",
            "CRSTNet",
            "cache",
            f"{data_name}_{input_len}_{output_len}_stable_hgraph.pkl",
        ),
        "promote_margin": 1.05,
        "demote_margin": 0.90,
        "demote_patience": 2,
        "min_cluster_size": 3,
        "max_cluster_size": 8,
        "error_key_ratio": 0.30,
        "error_ema_beta": 0.90,
        "model_error_warmup_epochs": 10,
        "model_error_full_epochs": 30,
        "protected_patience": 2,
        "protected_attention_boost": 0.25,
        "protected_residual_multiplier": 1.0,
        "nonkey_residual_scale": 0.10,
        "spatial_dim": 32,
        "nonkey_adapter_rank": 16,
        "diagnostics_interval": 100,
        "maintenance_poll_interval": 1,
        "async_structure_update": True,
        "drift_check_interval": screen_interval,
        "min_recluster_interval": min_repair_interval,
        "drift_noop_backoff_after": 1,
        "drift_noop_backoff_max_interval": min_repair_interval,
        "drift_threshold": None,
        "drift_screening_mode": "self_normalized_tail",
        "spatial_backend": "auto",
        "dense_spatial_node_threshold": 256,
        "attention_cache_start_epoch": 10,
        "attention_cache_interval": 4,
        "maintenance_objective": "sublinear_mdl_ipac",
        "mdl_ipac_enabled": True,
        "sketch_mdl_enabled": True,
        "sketch_ema_beta": 0.90,
        "probabilistic_sketch_enabled": False,
        "probabilistic_sketch_width": 64,
        "probabilistic_sketch_depth": 4,
        "probabilistic_sketch_tail_ratio": 0.40,
        "probabilistic_sketch_seed": 13,
        "tensorized_sketch_enabled": False,
        "tensorized_sketch_top_k": None,
        "dynamic_forest_enabled": True,
    }

    cfg = EasyDict()
    cfg.DESCRIPTION = f"CRSTNet paper-aligned BasicTS run on {data_name}"
    cfg.GPU_NUM = 1
    cfg.RUNNER = CRSTNetRunner

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = data_name
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict({
        "dataset_name": data_name,
        "train_val_test_ratio": train_val_test_ratio,
        "input_len": input_len,
        "output_len": output_len,
    })

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ZScoreScaler
    cfg.SCALER.PARAM = EasyDict({
        "dataset_name": data_name,
        "train_ratio": train_val_test_ratio[0],
        "norm_each_channel": norm_each_channel,
        "rescale": rescale,
    })

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = CRSTNet.__name__
    cfg.MODEL.ARCH = CRSTNet
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = FORWARD_FEATURES
    cfg.MODEL.TARGET_FEATURES = [0]
    cfg.MODEL.SETUP_GRAPH = True

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict({
        "MAE": masked_mae,
        "RMSE": masked_rmse,
        "WAPE": masked_wape,
        "MAPE": masked_mape,
    })
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.NULL_VAL = null_val

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        CRSTNet.__name__,
        "_".join([
            data_name,
            str(cfg.TRAIN.NUM_EPOCHS),
            str(input_len),
            str(output_len),
            "crstnet",
        ]),
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.MEASURE_GPU_MEMORY = True

    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.001, "eps": 1e-3, "foreach": True}
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [80], "gamma": 0.3}

    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = 64
    cfg.TRAIN.DATA.SHUFFLE = True
    cfg.TRAIN.DATA.NUM_WORKERS = 4
    cfg.TRAIN.DATA.PIN_MEMORY = True

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()
    cfg.VAL.DATA.BATCH_SIZE = 64
    cfg.VAL.DATA.NUM_WORKERS = 4
    cfg.VAL.DATA.PIN_MEMORY = True

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = 1
    cfg.TEST.DATA = EasyDict()
    cfg.TEST.DATA.BATCH_SIZE = 64
    cfg.TEST.DATA.NUM_WORKERS = 4
    cfg.TEST.DATA.PIN_MEMORY = True

    cfg.EVAL = EasyDict()
    cfg.EVAL.HORIZONS = [3, 6, 12]
    cfg.EVAL.USE_GPU = True
    cfg.EVAL.MEASURE_EFFICIENCY = True
    return cfg


def build_crstnet_paper_cfg(
    data_name: str,
    num_epochs: int = 100,
):
    cfg = build_crstnet_cfg(data_name, num_epochs=num_epochs)
    input_len = cfg.DATASET.PARAM.input_len
    output_len = cfg.DATASET.PARAM.output_len
    suffix = "_".join([
        data_name,
        str(num_epochs),
        str(input_len),
        str(output_len),
        "crstnet",
    ])

    cfg.DESCRIPTION = f"CRSTNet paper-aligned 100-epoch BasicTS run on {data_name}"
    cfg.MODEL.PARAM.update({
        "maintenance_objective": "tensorized_probabilistic_sketch_mdl_ipac_cached_loop_plan",
        "sketch_mdl_enabled": True,
        "sketch_ema_beta": 0.90,
        "probabilistic_sketch_enabled": True,
        "probabilistic_sketch_width": 64,
        "probabilistic_sketch_depth": 4,
        "probabilistic_sketch_tail_ratio": 0.40,
        "probabilistic_sketch_seed": 13,
        "tensorized_sketch_enabled": True,
        "tensorized_sketch_top_k": None,
        "dynamic_forest_enabled": True,
        "boundary_index_enabled": True,
        "boundary_index_skip_internal_connectivity": True,
        "boundary_index_candidate_limit": 8,
        "boundary_index_restrict_candidate_hosts": True,
        "repair_unit_index_enabled": True,
        "patch_saving_coalescing_enabled": True,
        "zero_clustering_boundary_shift_enabled": False,
        "route_double_buffer_enabled": False,
        "route_tensor_no_grad_materialization": True,
        "hardware_tf32_enabled": False,
        "jit_profile_fusion_enabled": False,
        "eval_fast_diagnostics": True,
        "eval_disable_diagnostics_record": True,
        "temporal_eval_diagnostics_enabled": False,
        "cluster_routed_temporal": True,
        "temporal_tensorized_route_pooling": False,
        "temporal_loop_plan_cache": True,
        "temporal_loop_scatter_back": False,
        "temporal_residual_window": 3,
        "temporal_residual_beta": 0.30,
        "temporal_residual_gate": True,
        "temporal_residual_gate_bias": 2.0,
        "temporal_refinement_ratio": 0.08,
        "temporal_refinement_max_nodes": 5,
        "temporal_refinement_beta": 0.30,
        "temporal_refinement_start_step": 0,
        "temporal_refinement_interval": 1,
        "temporal_refinement_layers": 1,
        "temporal_refinement_eval_enabled": False,
        "temporal_selected_residual_beta": 0.05,
        "temporal_selected_residual_gate": True,
        "temporal_selected_residual_gate_bias": 1.5,
    })
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", cfg.MODEL.NAME, suffix)
    cfg.TRAIN.pop("FINETUNE_FROM", None)
    cfg.TRAIN.pop("FINETUNE_STRICT_LOAD", None)
    return cfg
