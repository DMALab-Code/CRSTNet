#!/usr/bin/env python3
import argparse
import importlib.util
import runpy
import time
import copy
from typing import List, Tuple

import torch

from easydict import EasyDict
from basicts.runners import SimpleTimeSeriesForecastingRunner


def load_cfg_from_py(py_path: str) -> EasyDict:
    # Execute the config file and fetch CFG
    glb = runpy.run_path(py_path)
    cfg = glb.get('CFG', None)
    if cfg is None:
        raise RuntimeError(f'CFG not found in {py_path}')
    return cfg


def set_bqh_params(cfg: EasyDict, update_every: int, env_update_every: int, topk: int):
    cfg = copy.deepcopy(cfg)
    bqh = cfg['MODEL']['PARAM']['bqh_config']
    bqh['keynode']['update_every'] = int(update_every)
    bqh['indexing']['env_update_every'] = int(env_update_every)
    bqh['indexing']['topK_candidates'] = int(topk)
    return cfg


@torch.no_grad()
def quick_val_loss(runner: SimpleTimeSeriesForecastingRunner, max_batches: int = 20) -> float:
    runner.model.eval()
    losses = []
    it = iter(runner.val_data_loader)
    for i in range(max_batches):
        try:
            data = next(it)
        except StopIteration:
            break
        out = runner.forward(data=data, epoch=None, iter_num=i, train=False)
        loss = runner.metric_forward(runner.loss, out)
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def measure_train_speed(runner: SimpleTimeSeriesForecastingRunner, warmup: int = 5, measure: int = 30) -> float:
    runner.model.train()
    it = iter(runner.train_data_loader)
    # warmup
    for i in range(warmup):
        data = next(it)
        loss = runner.train_iters(epoch=1, iter_index=i, data=data)
        runner.backward(loss)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    # measure
    t0 = time.time()
    for i in range(measure):
        data = next(it)
        loss = runner.train_iters(epoch=1, iter_index=i, data=data)
        runner.backward(loss)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - t0
    return elapsed / max(1, measure)


def main():
    parser = argparse.ArgumentParser(description='Grid search for BQH parameters')
    parser.add_argument('-c', '--config', default='baselines/CASTNet/PEMS-BAY.py', help='config py path')
    parser.add_argument('--update_every', nargs='+', type=int, default=[6, 8, 12])
    parser.add_argument('--env_update_every', nargs='+', type=int, default=[40, 60, 80])
    parser.add_argument('--topk', nargs='+', type=int, default=[8, 12, 16])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--measure', type=int, default=30)
    parser.add_argument('--val_batches', type=int, default=20)
    args = parser.parse_args()

    base_cfg = load_cfg_from_py(args.config)
    results: List[Tuple[Tuple[int, int, int], float, float]] = []

    for ue in args.update_every:
        for ee in args.env_update_every:
            for tk in args.topk:
                cfg = set_bqh_params(base_cfg, ue, ee, tk)
                # Reduce epochs to speed up init only
                cfg['TRAIN']['NUM_EPOCHS'] = 1
                runner = SimpleTimeSeriesForecastingRunner(cfg)
                runner.init_training(cfg)
                try:
                    avg_iter = measure_train_speed(runner, warmup=args.warmup, measure=args.measure)
                    val_loss = quick_val_loss(runner, max_batches=args.val_batches)
                    results.append(((ue, ee, tk), avg_iter, val_loss))
                    print(f'[ue={ue}, ee={ee}, tk={tk}] iter={avg_iter:.3f}s, val_loss={val_loss:.4f}')
                finally:
                    del runner
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Summaries
    print('\nTop-3 fastest:')
    for comb, avg_iter, val_loss in sorted(results, key=lambda x: x[1])[:3]:
        print(f'  ue={comb[0]} ee={comb[1]} tk={comb[2]}  iter={avg_iter:.3f}s  val={val_loss:.4f}')

    print('\nTop-3 best val:')
    for comb, avg_iter, val_loss in sorted(results, key=lambda x: x[2])[:3]:
        print(f'  ue={comb[0]} ee={comb[1]} tk={comb[2]}  val={val_loss:.4f}  iter={avg_iter:.3f}s')


if __name__ == '__main__':
    main()


