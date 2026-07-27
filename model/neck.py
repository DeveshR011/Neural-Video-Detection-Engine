"""
BiFPN Neck
==========
Bidirectional feature pyramid with learnable weighted fusion.

Each level is first projected to a common channel width, then refined by
`num_layers` bidirectional passes. Fusion weights are non-negative and
normalized, so each output is a convex combination of its inputs -- this keeps
scales balanced when features have very different magnitudes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, DepthwiseSeparableConv


class WeightedFusion(nn.Module):
    """Normalized weighted sum of N inputs with learnable weights."""

    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_inputs))
        self.eps = eps

    def forward(self, inputs):
        w = F.relu(self.weight)
        w = w / (w.sum() + self.eps)
        out = inputs[0] * w[0]
        for i in range(1, len(inputs)):
            out = out + inputs[i] * w[i]
        return out


class BiFPNLayer(nn.Module):
    """One bidirectional pass: top-down P5->P4->P3, then bottom-up P3->P4->P5."""

    def __init__(self, channels: int):
        super().__init__()
        self.fuse_p4_td = WeightedFusion(2)
        self.fuse_p3_out = WeightedFusion(2)
        self.fuse_p4_out = WeightedFusion(3)
        self.fuse_p5_out = WeightedFusion(2)

        self.conv_p4_td = DepthwiseSeparableConv(channels, channels)
        self.conv_p3_out = DepthwiseSeparableConv(channels, channels)
        self.conv_p4_out = DepthwiseSeparableConv(channels, channels)
        self.conv_p5_out = DepthwiseSeparableConv(channels, channels)

        self.down_p3 = ConvBlock(channels, channels, k=3, s=2)
        self.down_p4 = ConvBlock(channels, channels, k=3, s=2)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="nearest")

    @staticmethod
    def _match(x, ref):
        """Align spatial size after strided downsampling for odd input sizes."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="nearest")
        return x

    def forward(self, p3, p4, p5):
        # Top-down
        p4_td = self.conv_p4_td(self.fuse_p4_td([p4, self._up(p5, p4)]))
        p3_out = self.conv_p3_out(self.fuse_p3_out([p3, self._up(p4_td, p3)]))

        # Bottom-up. P4 receives its original input, the top-down result, and
        # the downsampled P3 -- the extra skip is what distinguishes BiFPN
        # from a plain PANet.
        p3_down = self._match(self.down_p3(p3_out), p4)
        p4_out = self.conv_p4_out(self.fuse_p4_out([p4, p4_td, p3_down]))

        p4_down = self._match(self.down_p4(p4_out), p5)
        p5_out = self.conv_p5_out(self.fuse_p5_out([p5, p4_down]))

        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    """Lateral projection followed by stacked BiFPN layers."""

    def __init__(self, in_channels, channels: int = 128, num_layers: int = 2):
        super().__init__()
        c3, c4, c5 = in_channels
        self.lateral_p3 = ConvBlock(c3, channels, k=1)
        self.lateral_p4 = ConvBlock(c4, channels, k=1)
        self.lateral_p5 = ConvBlock(c5, channels, k=1)
        self.layers = nn.ModuleList(BiFPNLayer(channels) for _ in range(num_layers))
        self.out_channels = (channels, channels, channels)

    def forward(self, p3, p4, p5):
        p3 = self.lateral_p3(p3)
        p4 = self.lateral_p4(p4)
        p5 = self.lateral_p5(p5)
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        return p3, p4, p5
