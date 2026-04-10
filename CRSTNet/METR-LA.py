import os
import sys
import random
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import CRSTNet

DATA_NAME = 'METR-LA'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_ARCH = CRSTNet
_, adj_mx = load_adj("datasets/" + DATA_NAME +
                     "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "adj_mx": adj_mx,
    "num_nodes": 207,
    "input_dim": 2,
    "output_dim": 1,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "key_node_ratio": 0.3,
    "use_efficient_clustering": True,
    "n_clusters": 6,
    "downsample_ratio": 4,
    "n_jobs": 4,

    "use_advanced_selector": True,
    "selection_strategy": "fdr_diversity",
    "fdr_alpha": 0.05,
    "eps_stop": 0.02,
    "diversity_rho": None,
    "stability_gamma": 0.1,
    "max_key_nodes": 100,
    "budget": 30,
    "cache_size": 1000,
    "update_frequency": 5,
    "use_approximation": True,
    "use_optimized_structure": True,
    "quota_ratio": 0.1,
    "paa_segments": 16,
    "merge_cooldown": 20,
    "split_cooldown": 20
}
NUM_EPOCHS = 100

CFG = EasyDict()

CFG.DESCRIPTION = 'CRSTNet paper configuration'
CFG.GPU_NUM = 1

CFG.RUNNER = SimpleTimeSeriesForecastingRunner

CFG._ = random.randint(-10**6, 10**6)

CFG.DATASET = EasyDict()

CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,

})

CFG.SCALER = EasyDict()

CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

CFG.MODEL = EasyDict()

CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.SETUP_GRAPH = True

CFG.METRICS = EasyDict()

CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                                'WAPE': masked_wape,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.01,
    "eps": 1e-3
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [20, 30, 40, 50],
    "gamma": 0.1
}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True

CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()

CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True

