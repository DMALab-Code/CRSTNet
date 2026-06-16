# CRSTNet Baseline

This is the main CRSTNet implementation used by this repository.

Important files:

```text
common.py                   Shared BasicTS config builder.
PEMS04.py                   PEMS04 100-epoch run file.
PEMS08.py                   PEMS08 100-epoch run file.
METR-LA.py                  METR-LA 100-epoch run file.
PEMS-BAY.py                 PEMS-BAY 100-epoch run file.
arch/                       CRSTNet model, H-Graph, clustering, spatial, and temporal modules.
runner/                     CRSTNet runner hooks.
tools/build_hgraph_cache.py Optional H-Graph cache builder.
cache/                      Generated H-Graph cache files.
thresholds/                 Generated parameter/threshold profile files.
```

Run from the repository root:

```bash
python experiments/train.py --cfg baselines/CRSTNet/PEMS04.py --gpus "0"
python experiments/train.py --cfg baselines/CRSTNet/PEMS08.py --gpus "0"
python experiments/train.py --cfg baselines/CRSTNet/METR-LA.py --gpus "0"
python experiments/train.py --cfg baselines/CRSTNet/PEMS-BAY.py --gpus "0"
```

All four configs use `100` epochs, `12 -> 12` forecasting, Adam, Masked MAE, batch size `64`, and BasicTS evaluation horizons `[3, 6, 12]`.

Automatic parameter scripts are outside this directory:

```text
scripts/parameter_search/tune_paper_params.py
scripts/parameter_search/profile_thresholds.py
```
