"""Vendored OSNet-AIN architecture (torch model definition only).

Adapted from `torchreid` (KaiyangZhou/deep-person-reid, MIT license):
https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet_ain.py

    Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV 2019.
    Zhou et al. Learning Generalisable Omni-Scale Representations for Person
    Re-Identification. TPAMI 2021.

Why vendor instead of `pip install torchreid`: the PyPI package is stale and the upstream repo
installs from source with extra build steps; all we need is ~300 lines of plain PyTorch to load the
published checkpoints. Only the **AIN** ("all instance-norm") family is kept — its instance
normalisation is what makes the embeddings generalise across camera domains, which is exactly our
situation (dek-hockey fisheye footage is far from the Market/Duke/MSMT training domains).

Trimmed relative to upstream: no training heads beyond the checkpoint-compatible `fc`/`classifier`
layers, no gdown download logic (that lives in ``reid_embedder``), no non-AIN variants.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["OSNet", "osnet_ain_x1_0", "osnet_ain_x0_25"]


# --------------------------------------------------------------------------------------------
# Basic layers
# --------------------------------------------------------------------------------------------

class ConvLayer(nn.Module):
    """Convolution + (batch|instance) norm + relu."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if IN else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (no non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return x if self.bn is None else self.bn(x)


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution: 1x1 (linear) + depthwise 3x3 (nonlinear)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False,
                               groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    """A stream of ``depth`` LightConv3x3 layers (the multi-scale branches of an OS block)."""

    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        assert depth >= 1
        layers = [LightConv3x3(in_channels, out_channels)]
        layers += [LightConv3x3(out_channels, out_channels) for _ in range(depth - 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# --------------------------------------------------------------------------------------------
# Omni-scale blocks
# --------------------------------------------------------------------------------------------

class ChannelGate(nn.Module):
    """Mini-network generating channel-wise gates conditioned on the input (the AG gate)."""

    def __init__(self, in_channels, num_gates=None, reduction=16):
        super().__init__()
        num_gates = num_gates or in_channels
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True)
        self.norm1 = None
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        inp = x
        x = self.global_avgpool(x)
        x = self.relu(self.fc1(x))
        x = self.gate_activation(self.fc2(x))
        return inp * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4):
        super().__init__()
        assert T >= 1 and out_channels >= reduction and out_channels % reduction == 0
        mid = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for stream in self.conv2:
            x2 = x2 + self.gate(stream(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSBlockINin(nn.Module):
    """Omni-scale block with instance normalisation inside the residual (the AIN part)."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4):
        super().__init__()
        assert T >= 1 and out_channels >= reduction and out_channels % reduction == 0
        mid = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels, bn=False)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for stream in self.conv2:
            x2 = x2 + self.gate(stream(x1))
        x3 = self.IN(self.conv3(x2))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


# --------------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------------

class OSNet(nn.Module):
    """Omni-Scale Network. In eval mode ``forward`` returns the ``feature_dim`` embedding."""

    def __init__(self, blocks, layers, channels, feature_dim=512, num_classes=1, conv1_IN=True):
        super().__init__()
        assert len(blocks) == len(layers) == len(channels) - 1
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], channels[0], channels[1])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = self._make_layer(blocks[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = self._make_layer(blocks[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU()
        )
        # Kept only so published checkpoints load cleanly; never used for embedding.
        self.classifier = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _make_layer(block_types, in_channels, out_channels):
        layers = [block_types[0](in_channels, out_channels)]
        layers += [b(out_channels, out_channels) for b in block_types[1:]]
        return nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv5(self.conv4(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.global_avgpool(self.featuremaps(x))
        v = v.view(v.size(0), -1)
        return self.fc(v)


def osnet_ain_x1_0(num_classes: int = 1) -> OSNet:
    """Full-width OSNet-AIN (2.2M params, 512-d embedding)."""
    return OSNet(
        blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        num_classes=num_classes,
    )


def osnet_ain_x0_25(num_classes: int = 1) -> OSNet:
    """Quarter-width OSNet-AIN (0.2M params) — ~4x faster on CPU, a little less accurate."""
    return OSNet(
        blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        num_classes=num_classes,
    )
