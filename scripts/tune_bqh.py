#!/usr/bin/env python3
import argparse
import copy
import runpy
import time
from typing import List, Tuple

import torch
from easydict import EasyDict

from basicts.runners import SimpleTimeSeriesForecastingRunner


def load_cfg_from_py(py_path: str) -> EasyDict:
    namespace = runpy.run_path(py_path)
    cfg = namespace.get("CFG")
    if cfg is None:
        raise RuntimeError(f"CFG not found in {py_path}")
    return cfg


def set_paper_params(cfg: EasyDict, theta: float, lambda_value: float, gamma: float) -> EasyDict:
    cfg = copy.deepcopy(cfg)
    model_param = cfg["MODEL"]["PARAM"]
    model_param["theta"] = float(theta)
    model_param["distance_lambda"] = float(lambda_value)
    model_param["gamma"] = float(gamma)
    return cfg


@torch.no_grad()
def quick_val_loss(runner: SimpleTimeSeriesForecastingRunner, max_batches: int = 20) -> float:
    runner.model.eval()
    losses = []
    iterator = iter(runner.val_data_loader)
    for batch_idx in range(max_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        output = runner.forward(data=batch, epoch=None, iter_num=batch_idx, train=False)
        loss = runner.metric_forward(runner.loss, output)
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def measure_train_speed(
    runner: SimpleTimeSeriesForecastingRunner,
    warmup: int = 5,
    measure: int = 30,
) -> float:
    runner.model.train()
    iterator = iter(runner.train_data_loader)
    for step in range(warmup):
        batch = next(iterator)
        loss = runner.train_iters(epoch=1, iter_index=step, data=batch)
        runner.backward(loss)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for step in range(measure):
        batch = next(iterator)
        loss = runner.train_iters(epoch=1, iter_index=step, data=batch)
        runner.backward(loss)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start
    return elapsed / max(1, measure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search for paper-aligned CRSTNet parameters (legacy filename retained)."
    )
    parser.add_argument("-c", "--config", default="CRSTNet/PEMS-BAY.py", help="config py path")
    parser.add_argument("--theta", nargs="+", type=float, default=[0.15, 0.2, 0.25])
    parser.add_argument("--lambda_value", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--gamma", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=30)
    parser.add_argument("--val_batches", type=int, default=20)
    args = parser.parse_args()

    base_cfg = load_cfg_from_py(args.config)
    results: List[Tuple[Tuple[float, float, float], float, float]] = []

    for theta in args.theta:
        for lambda_value in args.lambda_value:
            for gamma in args.gamma:
                cfg = set_paper_params(base_cfg, theta, lambda_value, gamma)
                cfg["TRAIN"]["NUM_EPOCHS"] = 1
                runner = SimpleTimeSeriesForecastingRunner(cfg)
                runner.init_training(cfg)
                try:
                    avg_iter = measure_train_speed(runner, warmup=args.warmup, measure=args.measure)
                    val_loss = quick_val_loss(runner, max_batches=args.val_batches)
                    results.append(((theta, lambda_value, gamma), avg_iter, val_loss))
                    print(
                        f"[theta={theta:.3f}, lambda={lambda_value:.3f}, gamma={gamma:.3f}] "
                        f"iter={avg_iter:.3f}s, val_loss={val_loss:.4f}"
                    )
                finally:
                    del runner
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print("\nTop-3 fastest:")
    for params, avg_iter, val_loss in sorted(results, key=lambda item: item[1])[:3]:
        print(
            f"  theta={params[0]:.3f} lambda={params[1]:.3f} gamma={params[2]:.3f} "
            f"iter={avg_iter:.3f}s val={val_loss:.4f}"
        )

    print("\nTop-3 best val:")
    for params, avg_iter, val_loss in sorted(results, key=lambda item: item[2])[:3]:
        print(
            f"  theta={params[0]:.3f} lambda={params[1]:.3f} gamma={params[2]:.3f} "
            f"val={val_loss:.4f} iter={avg_iter:.3f}s"
        )


if __name__ == "__main__":
    main()
