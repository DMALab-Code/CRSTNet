# CRSTNet

This is an implementation of "CRSTNet: Dynamic Cluster Routing for Adaptive Spatio-Temporal Traffic Prediction".

# Table of Contents

```text
basicts   --> The BasicTS framework, which provides standard pipelines for training MTS forecasting models.

datasets  --> Raw datasets and preprocessed data.

experiments  --> Training, evaluation, and inference scripts.

baselines/CRSTNet  --> The main implementation of CRSTNet.

baselines/CRSTNet/${DATASET_NAME}.py  --> Training configs.

scripts/parameter_search/profile_thresholds.py  --> Script for determining threshold values such as delta_s, delta_e, delta_d, and eta.
```

Replace `${DATASET_NAME}` with one of `PEMS04`, `PEMS08`, `METR-LA`, or `PEMS-BAY`.

# Implementation Location

The CRSTNet implementation is located in `baselines/CRSTNet/arch/`. The dataset-specific configurations are in `baselines/CRSTNet/${DATASET_NAME}.py`, and the unified training and evaluation pipeline is organized under `experiments/` and `basicts/`.

The threshold profiling script is located at `scripts/parameter_search/profile_thresholds.py`. It profiles threshold-related values from the training data and writes the result to `baselines/CRSTNet/thresholds/` by default.

# Environment

Python 3.11 + PyTorch 2.5.1 + CUDA 12.4 (Recommended)

```bash
# Install Python
conda create -n BasicTS python=3.11
conda activate BasicTS
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# Install other dependencies
pip install -r requirements.txt
```

# Dataset

To facilitate the assessment of the experimental results and to ensure fairness in comparison with the baseline, we used the dataset from BasicTS.

You can download the `all_data.zip` file from [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1shA2scuMdZHlx6pj35Dl7A?pwd=s2xe). Unzip the files to the `datasets/` directory:

```bash
cd /path/to/CRSTNet
unzip /path/to/all_data.zip -d datasets/
```

These datasets have been preprocessed and are ready for use.

# Train CRSTNet

```bash
python experiments/train.py --cfg baselines/CRSTNet/${DATASET_NAME}.py --gpus '0'
```

Replace `${DATASET_NAME}` with one of `PEMS04`, `PEMS08`, `METR-LA`, or `PEMS-BAY`.

# Cross-Domain Evaluation on Weather2K

We evaluate CRSTNet on Weather2K, a meteorological sensor network containing 1,866 ground weather stations. We follow the hourly 12-step input/output protocol and the distance-based station graph construction used in Weather2K [1].

| Method | Air Temperature MAE/RMSE | Relative Humidity MAE/RMSE | Wind Speed MAE/RMSE | Air Pressure MAE/RMSE | Avg. MAE/RMSE | Training (s/epoch) |
|:--|--:|--:|--:|--:|--:|--:|
| CDPNet [2] | **1.39 / 1.86** | **7.26 / 9.75** | **0.87 / 1.23** | **1.43 / 1.98** | **2.74 / 3.70** | 153.71 |
| **ours** | 1.43 / 1.90 | 7.40 / 9.90 | 0.88 / 1.26 | 1.54 / 2.11 | 2.81 / 3.79 | **19.31** |

Lower values indicate better performance. `Avg.` is the arithmetic mean over air temperature, relative humidity, wind speed, and air pressure. Training time was measured on the same hardware and excludes validation and testing. The
CDPNet training time was obtained from our same-hardware rerun because it is not reported in the original paper.

[1] X. Zhu et al., “Weather2K: A Multivariate Spatio-Temporal Benchmark Dataset for Meteorological Forecasting Based on Real-Time Observation Data from Ground Weather Stations,” AISTATS, 2023. Dataset: https://github.com/bycnfz/weather2k

[2] C. Xu et al., “Continuous Diffusive Prediction Network for Multi-Station Weather Prediction,” IJCAI, 2025.

# Acknowledgement

The baseline implementations in this repository are derived from BasicTS.

We thank the authors of the following repository for code reference: [BasicTS](https://github.com/zezhishao/BasicTS)
