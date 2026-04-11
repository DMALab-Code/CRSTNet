import torch
import torch.nn as nn


class MultiScaleTrendMixing(nn.Module):
    """Top-down trend mixing used for multi-scale residual aggregation."""

    def __init__(self, history_seq_len: int, future_seq_len: int, num_channels: int,
                 down_sampling_layers: int, down_sampling_window: int):
        super(MultiScaleTrendMixing, self).__init__()
        self.seq_len = history_seq_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window

        # Upsampling layers mirror the down-sampling hierarchy.
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                        self.seq_len // (self.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.seq_len // (self.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(self.down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        if len(trend_list) <= 1:
            return trend_list

        # Work in (B, N, T) for linear projections over the temporal axis.
        trend_list_n = [trend.permute(0, 2, 1) for trend in trend_list]
        trend_list_n.reverse()

        out_low = trend_list_n[0]
        out_high = trend_list_n[1] if len(trend_list_n) > 1 else trend_list_n[0]
        out_trend_list = [out_low]

        for i in range(len(trend_list_n) - 1):
            upsampled = self.up_sampling_layers[i](out_low)
            out_high = out_high + upsampled
            out_low = out_high
            if i + 2 <= len(trend_list_n) - 1:
                out_high = trend_list_n[i + 2]
            out_trend_list.append(out_low)

        out_trend_list.reverse()
        return [trend.permute(0, 2, 1) for trend in out_trend_list]
