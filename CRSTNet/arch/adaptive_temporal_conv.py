from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalConv(nn.Module):
    """
    Paper-aligned adaptive temporal module.

    The long branch uses FFT-based period estimation, while the short branch
    uses the variance-adaptive window in Eq. (9). Features are produced on the
    condensed graph entities and then broadcast back to vertices.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 48,
        k_min: int = 2,
        k_max: Optional[int] = None,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_min = max(1, int(k_min))
        self.k_max = k_max
        self.epsilon = epsilon

        max_kernel = max(3, int(k_max) if k_max is not None else 12)
        self.max_kernel = max_kernel
        self.long_weight = nn.Parameter(torch.randn(out_channels, in_channels, max_kernel) * 0.02)
        self.short_weight = nn.Parameter(torch.randn(out_channels, in_channels, max_kernel) * 0.02)
        self.long_bias = nn.Parameter(torch.zeros(out_channels))
        self.short_bias = nn.Parameter(torch.zeros(out_channels))
        self.gate = nn.Parameter(torch.zeros(out_channels))

        self.last_kernel_sizes: List[Tuple[int, int]] = []

    def _estimate_kernel_sizes(self, seq: torch.Tensor) -> Tuple[int, int]:
        """
        Estimate K_L and K_S from the condensed series.

        seq shape: [batch, time, features]
        """

        time_steps = seq.shape[1]
        if time_steps <= 1:
            return 1, 1

        pooled = seq.mean(dim=(0, 2))
        centered = pooled - pooled.mean()
        spectrum = torch.fft.rfft(centered)
        amplitudes = torch.abs(spectrum)
        if amplitudes.numel() > 0:
            amplitudes[0] = 0.0

        dominant_idx = int(torch.argmax(amplitudes).item()) if amplitudes.numel() > 1 else 0
        if dominant_idx <= 0:
            k_long = min(time_steps, max(self.k_min, time_steps // 2))
        else:
            k_long = max(self.k_min, int(time_steps / dominant_idx))

        k_max = min(time_steps, self.k_max or max(self.k_min, time_steps // 2))
        k_long = int(max(self.k_min, min(k_long, k_max)))

        variance = float(torch.var(pooled, unbiased=False).item())
        scale = 1.0 + torch.log1p(torch.tensor(variance / self.epsilon)).item()
        k_short = int(round(k_long / max(scale, 1.0)))
        k_short = int(max(self.k_min, min(k_short, k_max)))
        return k_long, k_short

    def _dynamic_conv(self, seq: torch.Tensor, kernel_size: int, weight_bank: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        seq = seq.transpose(1, 2)
        kernel_size = max(1, min(int(kernel_size), self.max_kernel))
        weight = weight_bank[:, :, :kernel_size]
        padding = kernel_size // 2
        out = F.conv1d(seq, weight, bias=bias, padding=padding)
        if out.size(-1) > seq.size(-1):
            out = out[..., : seq.size(-1)]
        elif out.size(-1) < seq.size(-1):
            out = F.pad(out, (0, seq.size(-1) - out.size(-1)))
        return out.transpose(1, 2)

    def _build_condensed_sequences(
        self,
        x: torch.Tensor,
        key_node_indices: Sequence[int],
        cluster_indices_list: Sequence[Sequence[int]],
    ) -> Tuple[torch.Tensor, List[Tuple[str, List[int]]]]:
        sequences = []
        owners: List[Tuple[str, List[int]]] = []

        for key_node in key_node_indices:
            sequences.append(x[:, :, int(key_node), :])
            owners.append(("key", [int(key_node)]))

        for cluster in cluster_indices_list:
            members = [int(node) for node in cluster]
            if not members:
                continue
            sequences.append(x[:, :, members, :].mean(dim=2))
            owners.append(("cluster", members))

        if not sequences:
            sequences.append(x.mean(dim=2))
            owners.append(("cluster", list(range(x.shape[2]))))

        condensed = torch.stack(sequences, dim=2)
        return condensed, owners

    def _broadcast_to_vertices(
        self,
        condensed: torch.Tensor,
        owners: Sequence[Tuple[str, List[int]]],
        num_nodes: int,
    ) -> torch.Tensor:
        batch_size, time_steps, _, channels = condensed.shape
        output = torch.zeros(batch_size, time_steps, num_nodes, channels, device=condensed.device, dtype=condensed.dtype)
        for entity_idx, (_, members) in enumerate(owners):
            entity_feature = condensed[:, :, entity_idx, :]
            for node_idx in members:
                output[:, :, node_idx, :] = entity_feature
        return output

    def forward(
        self,
        x: torch.Tensor,
        key_node_indices: Optional[Sequence[int]] = None,
        cluster_indices_list: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        batch_size, time_steps, num_nodes, _ = x.shape
        key_node_indices = list(key_node_indices or [])
        cluster_indices_list = list(cluster_indices_list or [])

        condensed, owners = self._build_condensed_sequences(x, key_node_indices, cluster_indices_list)
        fused_entities = []
        kernel_sizes: List[Tuple[int, int]] = []
        gate = torch.sigmoid(self.gate).view(1, 1, 1, -1)

        for entity_idx in range(condensed.shape[2]):
            seq = condensed[:, :, entity_idx, :]
            k_long, k_short = self._estimate_kernel_sizes(seq)
            kernel_sizes.append((k_long, k_short))
            z_long = self._dynamic_conv(seq, k_long, self.long_weight, self.long_bias)
            z_short = self._dynamic_conv(seq, k_short, self.short_weight, self.short_bias)
            fused = gate[:, :, :, :] * z_long.unsqueeze(2) + (1.0 - gate[:, :, :, :]) * z_short.unsqueeze(2)
            fused_entities.append(fused.squeeze(2))

        self.last_kernel_sizes = kernel_sizes
        condensed_fused = torch.stack(fused_entities, dim=2)
        return self._broadcast_to_vertices(condensed_fused, owners, num_nodes)
