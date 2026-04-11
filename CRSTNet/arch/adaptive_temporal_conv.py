import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels=48, main_period=12):
        super().__init__()
        self.conv_long = nn.Conv2d(in_channels, out_channels, kernel_size=(main_period, 1), padding=(main_period // 2, 0))
        self.conv_short = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn_long = nn.BatchNorm2d(out_channels)
        self.bn_short = nn.BatchNorm2d(out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)
        res = self.residual_conv(x)
        out_long = F.relu(self.bn_long(self.conv_long(x)))
        out_short = F.relu(self.bn_short(self.conv_short(x)))
        min_len = min(out_long.shape[2], out_short.shape[2], res.shape[2])
        out_long = out_long[:, :, :min_len, :]
        out_short = out_short[:, :, :min_len, :]
        res = res[:, :, :min_len, :]
        out = (out_long + out_short + res) / 3
        return out.permute(0, 2, 3, 1)

