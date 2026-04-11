import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """Simple moving average over the temporal dimension."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N]
        if self.kernel_size <= 1:
            return x
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class Temporal_Decomposition(nn.Module):
    """Decompose a series into seasonal and trend components."""

    def __init__(self, kernel_size: int):
        super(Temporal_Decomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend
