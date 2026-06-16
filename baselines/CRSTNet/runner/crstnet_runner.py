import json
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
from easytorch.core.checkpoint import backup_last_ckpt, clear_ckpt, load_ckpt, save_ckpt
from easytorch.utils import master_only
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from basicts.runners import SimpleTimeSeriesForecastingRunner


class CRSTNetRunner(SimpleTimeSeriesForecastingRunner):
    TRAIN_GPU_MEMORY_METER = "train/max_gpu_allocated_memory"
    TEST_GPU_MEMORY_METER = "test/max_gpu_allocated_memory"
    TEST_INFERENCE_TIME_METER = "test/inference_time"
    TEST_INFERENCE_TIME_PER_BATCH_METER = "test/inference_time_per_batch"
    TEST_INFERENCE_TIME_PER_SAMPLE_METER = "test/inference_time_per_sample"

    def _model_obj(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _random_state_payload(self):
        payload = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["torch_cuda"] = torch.cuda.get_rng_state_all()
        return payload

    def _restore_random_state(self, payload):
        if not payload:
            return
        try:
            if "python" in payload:
                random.setstate(payload["python"])
            if "numpy" in payload:
                np.random.set_state(payload["numpy"])
            if "torch_cpu" in payload:
                torch.set_rng_state(payload["torch_cpu"])
            if torch.cuda.is_available() and "torch_cuda" in payload:
                torch.cuda.set_rng_state_all(payload["torch_cuda"])
        except Exception as exc:
            self.logger.warning("Failed to restore random state from CRSTNet checkpoint: %s", exc)

    def _stateful_payload(self):
        model = self._model_obj()
        prepare = getattr(model, "prepare_stateful_checkpoint", None)
        if not callable(prepare):
            return None
        payload = prepare()
        fingerprint = payload.get("hgraph_fingerprint", {}) if payload else {}
        self.logger.info("CRSTNet stateful checkpoint fingerprint(save): %s", fingerprint)
        return payload

    def _light_runtime_payload(self):
        model = self._model_obj()
        return {
            "step_count": int(getattr(model, "step_count", 0)),
            "last_recluster_step": int(getattr(model, "last_recluster_step", 0)),
            "last_profile_drift": getattr(model, "last_profile_drift", None),
            "stateful": False,
        }

    def _ckpt_dict(self, epoch: int, full_stateful: bool = False):
        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": self._model_obj().state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "best_metrics": self.best_metrics,
            "crstnet_light_runtime": self._light_runtime_payload(),
            "random_state": self._random_state_payload(),
        }
        if full_stateful:
            ckpt_dict["crstnet_stateful_hgraph"] = self._stateful_payload()
        if getattr(self, "scheduler", None) is not None:
            ckpt_dict["scheduler_state_dict"] = self.scheduler.state_dict()
        return ckpt_dict

    def _load_stateful_payload(self, checkpoint_dict, strict=True):
        payload = checkpoint_dict.get("crstnet_stateful_hgraph")
        model = self._model_obj()
        load_fn = getattr(model, "load_stateful_hgraph_checkpoint", None)
        if payload is None:
            self.logger.warning("CRSTNet checkpoint has no stateful H-Graph payload; using runtime-initialized structure.")
            return None
        if not callable(load_fn):
            self.logger.warning("CRSTNet model does not expose load_stateful_hgraph_checkpoint.")
            return None
        result = load_fn(payload, strict=strict)
        self.logger.info("CRSTNet stateful checkpoint fingerprint(load): %s", result)
        return result

    def load_model(self, ckpt_path: str = None, strict: bool = True) -> None:
        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger)
            self._model_obj().load_state_dict(checkpoint_dict["model_state_dict"], strict=strict)
            self._load_stateful_payload(checkpoint_dict, strict=strict)
        except (IndexError, OSError) as e:
            raise OSError("Ckpt file does not exist") from e

    def load_model_resume(self, strict: bool = True):
        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, logger=self.logger)
            self._model_obj().load_state_dict(checkpoint_dict["model_state_dict"], strict=strict)
            self._load_stateful_payload(checkpoint_dict, strict=strict)
            self.optim.load_state_dict(checkpoint_dict["optim_state_dict"])
            self.start_epoch = checkpoint_dict["epoch"]
            if checkpoint_dict.get("best_metrics") is not None:
                self.best_metrics = checkpoint_dict["best_metrics"]
            if self.scheduler is not None:
                if checkpoint_dict.get("scheduler_state_dict") is not None:
                    self.scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
                else:
                    self.scheduler.last_epoch = checkpoint_dict["epoch"]
            self._restore_random_state(checkpoint_dict.get("random_state"))
            self.logger.info("Resume CRSTNet training from stateful checkpoint")
        except (IndexError, OSError, KeyError):
            pass

    @master_only
    def save_model(self, epoch: int):
        full_stateful = epoch == self.num_epochs
        ckpt_dict = self._ckpt_dict(epoch, full_stateful=full_stateful)
        last_ckpt_path = self.get_ckpt_path(epoch - 1)
        backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)
        ckpt_path = self.get_ckpt_path(epoch)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)
        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)

    @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):
        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            ckpt_dict = self._ckpt_dict(epoch, full_stateful=True)
            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                "{}_best_{}.pt".format(self.model_name, metric_name.replace("/", "_")),
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)
            self.current_patience = self.early_stopping_patience
        else:
            if self.early_stopping_patience is not None:
                self.current_patience -= 1

    def _reset_model_transient_state(self, reason: str):
        model = self._model_obj()
        reset_fn = getattr(model, "reset_transient_state", None)
        if callable(reset_fn):
            reset_fn(reason=reason, reset_route_cache=True, reset_temporal=True)
            self.logger.info("CRSTNet transient cache reset: %s", reason)

    def init_training(self, cfg: Dict):
        super().init_training(cfg)
        self.measure_gpu_memory = cfg.get("TRAIN", {}).get("MEASURE_GPU_MEMORY", True)
        if self.measure_gpu_memory:
            self.register_epoch_meter(self.TRAIN_GPU_MEMORY_METER, "train", "{:.2f} (MB)")

    def init_test(self, cfg: Dict):
        super().init_test(cfg)
        self.measure_eval_efficiency = cfg.get("EVAL", {}).get("MEASURE_EFFICIENCY", True)
        if self.measure_eval_efficiency:
            self.register_epoch_meter(self.TEST_GPU_MEMORY_METER, "test", "{:.2f} (MB)")
            self.register_epoch_meter(self.TEST_INFERENCE_TIME_METER, "test", "{:.4f} (s)")
            self.register_epoch_meter(self.TEST_INFERENCE_TIME_PER_BATCH_METER, "test", "{:.4f} (ms)")
            self.register_epoch_meter(self.TEST_INFERENCE_TIME_PER_SAMPLE_METER, "test", "{:.4f} (ms)")

    def on_epoch_start(self, epoch: int):
        super().on_epoch_start(epoch)
        if getattr(self, "measure_gpu_memory", False):
            self._reset_cuda_peak_memory()

    @master_only
    def on_validating_start(self, train_epoch: Optional[int]):
        super().on_validating_start(train_epoch)
        self._reset_model_transient_state("frozen_validation_start")

    def on_epoch_end(self, epoch: int) -> None:
        if getattr(self, "measure_gpu_memory", False):
            self.update_epoch_meter(self.TRAIN_GPU_MEMORY_METER, self._max_cuda_memory_mb())
        super().on_epoch_end(epoch)
        if getattr(self, "num_epochs", None) == epoch:
            self._export_runtime_hgraph()

    def on_training_end(self, *args, **kwargs) -> None:
        self._export_runtime_hgraph()
        try:
            super().on_training_end(*args, **kwargs)
        except TypeError:
            super().on_training_end()

    def _export_runtime_hgraph(self) -> None:
        model = getattr(self.model, "module", self.model)
        export_fn = getattr(model, "export_hgraph_cache", None)
        if callable(export_fn):
            exported = export_fn()
            if exported:
                self.logger.info("Export final CRSTNet H-Graph cache to %s", exported)

    @master_only
    def on_test_start(self) -> None:
        super().on_test_start()
        self._reset_model_transient_state("frozen_test_start")
        self._reset_cuda_peak_memory()

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        prediction, target, inputs = [], [], []
        inference_time = 0.0
        batch_count = 0
        sample_count = 0

        for data in tqdm(self.test_data_loader):
            self._sync_cuda()
            start_time = time.perf_counter()
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            self._sync_cuda()
            inference_time += time.perf_counter() - start_time
            batch_count += 1
            sample_count += int(forward_return["target"].shape[0])

            loss = self.metric_forward(self.loss, forward_return)
            self.update_epoch_meter("test/loss", loss.item())

            if not self.if_evaluate_on_gpu:
                forward_return["prediction"] = forward_return["prediction"].detach().cpu()
                forward_return["target"] = forward_return["target"].detach().cpu()
                forward_return["inputs"] = forward_return["inputs"].detach().cpu()

            prediction.append(forward_return["prediction"])
            target.append(forward_return["target"])
            inputs.append(forward_return["inputs"])

        returns_all = {
            "prediction": torch.cat(prediction, dim=0),
            "target": torch.cat(target, dim=0),
            "inputs": torch.cat(inputs, dim=0),
        }
        metrics_results = self.compute_evaluation_metrics(returns_all)
        self._update_test_efficiency(metrics_results, inference_time, batch_count, sample_count)

        if save_results:
            test_results = {key: value.cpu().numpy() for key, value in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, "test_results.npz"), **test_results)
        if save_metrics:
            with open(os.path.join(self.ckpt_save_dir, "test_metrics.json"), "w") as file:
                json.dump(metrics_results, file, indent=4)
        return returns_all

    def _update_test_efficiency(self, metrics_results: Dict, inference_time: float, batch_count: int, sample_count: int):
        if not getattr(self, "measure_eval_efficiency", True):
            return

        max_gpu_allocated_memory_mb = self._max_cuda_memory_mb()
        inference_time_per_batch_ms = inference_time / max(batch_count, 1) * 1000.0
        inference_time_per_sample_ms = inference_time / max(sample_count, 1) * 1000.0
        self.update_epoch_meter(self.TEST_GPU_MEMORY_METER, max_gpu_allocated_memory_mb)
        self.update_epoch_meter(self.TEST_INFERENCE_TIME_METER, inference_time)
        self.update_epoch_meter(self.TEST_INFERENCE_TIME_PER_BATCH_METER, inference_time_per_batch_ms)
        self.update_epoch_meter(self.TEST_INFERENCE_TIME_PER_SAMPLE_METER, inference_time_per_sample_ms)
        metrics_results["efficiency"] = {
            "max_gpu_allocated_memory_mb": max_gpu_allocated_memory_mb,
            "inference_time_s": inference_time,
            "inference_time_per_batch_ms": inference_time_per_batch_ms,
            "inference_time_per_sample_ms": inference_time_per_sample_ms,
            "num_batches": batch_count,
            "num_samples": sample_count,
        }

    @staticmethod
    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @classmethod
    def _reset_cuda_peak_memory(cls):
        if torch.cuda.is_available():
            cls._sync_cuda()
            torch.cuda.reset_peak_memory_stats()

    @classmethod
    def _max_cuda_memory_mb(cls) -> float:
        if not torch.cuda.is_available():
            return 0.0
        cls._sync_cuda()
        return torch.cuda.max_memory_allocated() / 1024**2
