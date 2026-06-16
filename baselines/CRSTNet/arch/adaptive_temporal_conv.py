import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallLRUCache:
    def __init__(self, max_size=256):
        self.max_size = max_size
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def put(self, key, value):
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        while len(self.data) > self.max_size:
            self.data.popitem(last=False)


class MultiScaleTemporalConv(nn.Module):
    """
    Fused multi-scale temporal bank.

    The CRSTNet routing structure is still key/cluster based, but temporal
    selection is no longer executed as many small node-bucket convolutions.
    A fixed 3-basis bank is computed once per layer, then a continuous
    node-level scalar gate mixes short/mid/long basis outputs.

    v9 efficiency path:
        input projection -> 3 depthwise temporal basis -> gate fusion
        -> one shared pointwise projection -> one shared GroupNorm.
    This keeps the CRSTNet temporal idea but avoids repeating pointwise
    projection and normalization for every basis branch.
    """

    def __init__(
        self,
        in_channels,
        out_channels=64,
        main_period=12,
        k_min=2,
        k_max=None,
        eps=1e-5,
        cache_size=256,
        soft_top_k=2,
        soft_warmup_epochs=5,
        soft_until_epoch=20,
        soft_interval=4,
        sparse_soft_margin=0.25,
        plan_ema_beta=0.85,
        basis_kernels=(3, 5, 11),
        gate_ema_beta=0.90,
        norm_groups=8,
        diagnostics_interval=50,
        eval_diagnostics_enabled=False,
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.main_period = int(main_period)
        self.k_min = int(k_min)
        self.k_max = k_max
        self.eps = float(eps)
        self.plan_ema_beta = min(max(float(plan_ema_beta), 0.0), 0.999)
        self.gate_ema_beta = min(max(float(gate_ema_beta), 0.0), 0.999)
        self.diagnostics_interval = max(1, int(diagnostics_interval))
        self.eval_diagnostics_enabled = bool(eval_diagnostics_enabled)
        self.current_epoch = None
        self.current_global_step = None

        kernels = [int(k) for k in basis_kernels]
        if len(kernels) != 3:
            raise ValueError("basis_kernels must contain exactly three kernels: short, mid, long.")
        self.basis_kernels = tuple(max(1, k) for k in kernels)

        groups = max(1, min(int(norm_groups), out_channels))
        while out_channels % groups != 0 and groups > 1:
            groups -= 1

        self.kernel_cache = SmallLRUCache(cache_size)
        self.group_plan_cache = SmallLRUCache(cache_size)

        self.input_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.input_norm = nn.GroupNorm(groups, out_channels)

        self.depthwise_basis = nn.ModuleList()
        for k in self.basis_kernels:
            self.depthwise_basis.append(
                nn.Conv2d(out_channels, out_channels, (k, 1), padding=(k // 2, 0), groups=out_channels)
            )
        self.shared_pointwise = nn.Conv2d(out_channels, out_channels, 1)
        self.shared_norm = nn.GroupNorm(groups, out_channels)

        self.gate_refine = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        with torch.no_grad():
            self.gate_refine[-1].weight.zero_()
            self.gate_refine[-1].bias.zero_()

        self.residual_gate = nn.Conv2d(out_channels * 2, out_channels, 1)
        self.register_buffer("gate_ema", torch.empty(0), persistent=False)
        self.last_adaptive_diagnostics = {}

    def set_runtime_context(self, epoch=None, global_step=None):
        self.current_epoch = epoch
        self.current_global_step = global_step

    @staticmethod
    def _crop_or_pad(y, target_t):
        if y.size(2) > target_t:
            return y[:, :, :target_t, :]
        if y.size(2) < target_t:
            return F.pad(y, (0, 0, 0, target_t - y.size(2)))
        return y

    def _normalize01(self, value):
        lo = value.amin(dim=1, keepdim=True)
        hi = value.amax(dim=1, keepdim=True)
        return (value - lo) / (hi - lo + self.eps)

    def _node_basis_gate(self, x):
        """
        x: [B,T,N,F]
        returns [B,N,3] for short/mid/long basis.
        """
        with torch.no_grad():
            sig = x[..., 0].detach().float()
            B, T, N = sig.shape
            centered = sig - sig.mean(dim=1, keepdim=True)
            variance = centered.var(dim=1, unbiased=False)
            if T > 1:
                fluctuation = (sig[:, -1, :] - sig[:, -2, :]).abs() / (sig[:, -2, :].abs() + self.eps)
            else:
                fluctuation = torch.zeros_like(variance)

            if T > 2:
                amp = torch.fft.rfft(centered, dim=1).abs()
                amp = amp[:, 1:, :]
                if amp.numel() == 0:
                    periodicity = torch.zeros_like(variance)
                else:
                    periodicity = amp.amax(dim=1) / (amp.sum(dim=1) + self.eps)
            else:
                periodicity = torch.zeros_like(variance)

            var_n = self._normalize01(variance)
            fluc_n = self._normalize01(fluctuation)
            period_n = self._normalize01(periodicity)

            short_logit = 1.25 * var_n + 0.75 * fluc_n
            long_logit = 1.25 * period_n + 0.50 * (1.0 - var_n)
            mid_logit = 1.0 - (short_logit - long_logit).abs()
            stats = torch.stack([short_logit, mid_logit, long_logit], dim=-1)

        refined = stats + self.gate_refine(stats)
        gate = torch.softmax(refined, dim=-1)

        if self.gate_ema.numel() == gate.size(1) * gate.size(2):
            ema = self.gate_ema.view(1, gate.size(1), gate.size(2)).to(device=gate.device, dtype=gate.dtype)
            gate = 0.7 * gate + 0.3 * ema
            gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        if self.training:
            with torch.no_grad():
                mean_gate = gate.detach().mean(dim=0)
                if self.gate_ema.numel() != mean_gate.numel():
                    self.gate_ema = mean_gate.reshape(-1).detach()
                else:
                    self.gate_ema.mul_(self.gate_ema_beta).add_(mean_gate.reshape(-1), alpha=1.0 - self.gate_ema_beta)

        return gate

    def forward(self, x, node_groups=None, plan_cache_key=None, refresh_plan=False):
        if x.dim() != 4:
            raise ValueError(f"Expected [B,T,N,F], got {tuple(x.shape)}")

        B, T, N, _ = x.shape
        z = x.permute(0, 3, 1, 2).contiguous()
        res = F.relu(self.input_norm(self.input_proj(z)))

        basis_outputs = []
        for conv in self.depthwise_basis:
            basis_outputs.append(self._crop_or_pad(conv(res), T))
        bank = torch.stack(basis_outputs, dim=-1)

        gate = self._node_basis_gate(x)
        fused_dw = (bank * gate[:, None, None, :, :]).sum(dim=-1)
        fused = F.relu(self.shared_norm(self.shared_pointwise(fused_dw)))
        res = self._crop_or_pad(res, T)
        r_gate = torch.sigmoid(self.residual_gate(torch.cat([fused, res], dim=1)))
        out = r_gate * fused + (1.0 - r_gate) * res

        should_record = (
            (
                self.current_global_step is None
                or int(self.current_global_step) % self.diagnostics_interval == 0
            )
            if self.training
            else self.eval_diagnostics_enabled
        )
        gate_mean = None
        gate_entropy = None
        if should_record:
            with torch.no_grad():
                gate_detached = gate.detach()
                gate_mean = gate_detached.mean(dim=(0, 1)).cpu().tolist()
                gate_entropy = float((-(gate_detached * (gate_detached + self.eps).log()).sum(dim=-1)).mean().cpu())

        self.last_adaptive_diagnostics = {
            "mode": "light_shared_fused_temporal_bank",
            "fused_temporal_bank": True,
            "light_shared_temporal_bank": True,
            "shared_pointwise": True,
            "shared_norm": True,
            "per_basis_pointwise": False,
            "node_bucket_batched": False,
            "node_level_gate": True,
            "scalar_basis_gate": True,
            "low_rank_gate": True,
            "hard_top1": False,
            "basis_kernels": list(self.basis_kernels),
            "bank_conv_calls": len(self.basis_kernels),
            "pointwise_conv_calls": 1,
            "norm_calls": 1,
            "groups": len(node_groups) if node_groups is not None else 1,
            "plan_cache_hit": True,
            "plan_cache_refreshed": bool(refresh_plan),
            "diagnostics_throttled": not should_record,
            "gate_mean": gate_mean,
            "gate_entropy": gate_entropy,
        }
        return out.permute(0, 2, 3, 1).contiguous()
