"""
Pluggable Detection Backbone
============================
Produces a three-level feature pyramid (P3/8, P4/16, P5/32) from an input
image.

Two families are supported:

- ``builtin``: a CSP backbone defined here, trained from scratch. No external
  weights, useful as a baseline and for environments without network access.
- Any ``timm`` model name: loaded with ``features_only=True`` and ImageNet
  pretrained weights. This is the default because a pretrained backbone is
  worth far more than architecture tweaks under a short training budget.

The attention block at P5 is attached by the backbone (not the neck) so that
global context is available before multi-scale fusion begins.
"""

import logging

import torch
import torch.nn as nn

from .blocks import ConvBlock, C2f, SPPF, C2PSA

logger = logging.getLogger("Backbone")


class BuiltinCSPBackbone(nn.Module):
    """CSP backbone with per-stage width/depth scaling.

    Stage strides are 2/4/8/16/32; the last three are returned as P3/P4/P5.
    """

    def __init__(self, width=0.5, depth=0.34, base_channels=64, base_depth=3):
        super().__init__()

        def ch(c):
            return max(16, int(round(c * width / 8) * 8))

        def dp(d):
            return max(1, int(round(d * depth)))

        c1, c2, c3, c4, c5 = (ch(base_channels), ch(base_channels * 2),
                              ch(base_channels * 4), ch(base_channels * 8),
                              ch(base_channels * 16))

        self.stem = ConvBlock(3, c1, k=3, s=2)
        self.stage2 = nn.Sequential(
            ConvBlock(c1, c2, k=3, s=2),
            C2f(c2, c2, n=dp(base_depth), shortcut=True),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(c2, c3, k=3, s=2),
            C2f(c3, c3, n=dp(base_depth * 2), shortcut=True),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(c3, c4, k=3, s=2),
            C2f(c4, c4, n=dp(base_depth * 2), shortcut=True),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(c4, c5, k=3, s=2),
            C2f(c5, c5, n=dp(base_depth), shortcut=True),
            SPPF(c5, c5),
        )
        self.out_channels = (c3, c4, c5)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


class TimmBackbone(nn.Module):
    """Wraps a timm model in ``features_only`` mode, selecting the /8, /16, /32 levels."""

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                f"Backbone '{name}' requires timm. Install it with: pip install timm"
            ) from exc

        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-3, -2, -1),
        )
        info = self.model.feature_info
        self.out_channels = tuple(info.channels())
        strides = tuple(info.reduction())

        if strides != (8, 16, 32):
            logger.warning(
                "Backbone '%s' returns strides %s, expected (8, 16, 32). "
                "Head strides will be set from the actual values.",
                name, strides,
            )
        self.strides = strides
        logger.info("timm backbone '%s' | channels=%s strides=%s pretrained=%s",
                    name, self.out_channels, strides, pretrained)

    def forward(self, x):
        p3, p4, p5 = self.model(x)
        return p3, p4, p5


class Backbone(nn.Module):
    """Feature-pyramid backbone with optional attention at P5.

    Args:
        name: ``"builtin"`` or a timm model name.
        pretrained: load ImageNet weights (timm backbones only).
        attention: attach a C2PSA block to P5.
        attention_blocks: number of stacked PSA blocks.
        width/depth: scaling factors for the builtin backbone.
    """

    def __init__(self, name: str = "builtin", pretrained: bool = True,
                 attention: bool = True, attention_blocks: int = 1,
                 width: float = 0.5, depth: float = 0.34):
        super().__init__()
        self.name = name

        if name == "builtin":
            self.body = BuiltinCSPBackbone(width=width, depth=depth)
            self.strides = (8, 16, 32)
        else:
            self.body = TimmBackbone(name, pretrained=pretrained)
            self.strides = getattr(self.body, "strides", (8, 16, 32))

        self.out_channels = self.body.out_channels

        # Attention only at P5: at stride 32 a 640px input is 20x20 = 400
        # tokens, so global attention costs little. Applying it at P3
        # (6400 tokens) would dominate both memory and runtime.
        c5 = self.out_channels[2]
        self.attention = C2PSA(c5, c5, n=attention_blocks) if attention else nn.Identity()

    def forward(self, x):
        p3, p4, p5 = self.body(x)
        p5 = self.attention(p5)
        return p3, p4, p5


def build_backbone(name="builtin", pretrained=True, attention=True, **kwargs):
    """Construct a backbone, falling back to the builtin CSP model on failure.

    A missing timm install or absent network connection should degrade to a
    trainable model rather than aborting a long run.
    """
    if name == "builtin":
        return Backbone("builtin", attention=attention, **kwargs)

    try:
        return Backbone(name, pretrained=pretrained, attention=attention, **kwargs)
    except Exception as exc:
        logger.error("Failed to build backbone '%s' (%s). Falling back to 'builtin'.",
                     name, exc)
        return Backbone("builtin", attention=attention, **kwargs)
