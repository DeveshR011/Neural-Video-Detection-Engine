"""
Anchor-Free Decoupled Detection Head
====================================
Replaces the anchor-based coupled head.

Per feature-map cell the head predicts:

- ``nc`` class logits (no objectness channel; classification score doubles as
  confidence, matching the task-aligned training objective)
- ``4 * (reg_max + 1)`` box logits forming a discrete distribution over the
  distance from the cell centre to each box side (left, top, right, bottom)

Classification and regression use separate convolutional branches. Coupling
them forces one feature to serve two objectives that want different things,
which measurably costs accuracy.

The DFL distribution matters because box boundaries are often ambiguous
(occlusion, motion blur). A distribution represents that uncertainty and its
expectation gives a sub-cell-accurate box, where direct regression can only
emit a single point estimate.
"""

import math

import torch
import torch.nn as nn

from .blocks import ConvBlock, DFL


def make_anchors(feats, strides, grid_offset=0.5):
    """Build anchor-point centres and their strides for a list of feature maps.

    Returns:
        anchor_points: [A, 2] cell centres in feature-map units
        stride_tensor: [A, 1] stride of the level each point belongs to
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device

    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), float(stride), dtype=dtype, device=device)
        )

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Convert predicted ltrb distances to boxes.

    ``distance``: [..., 4] as (left, top, right, bottom) offsets from the
    anchor point. Distances are non-negative, so the box always contains its
    anchor point -- this is what makes the formulation anchor-free yet stable.
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox, reg_max):
    """Inverse of ``dist2bbox`` for xyxy boxes, clamped to the DFL range.

    ``reg_max`` is the number of bins, so valid indices are 0..reg_max-1. DFL
    brackets each target between bins ``floor(t)`` and ``floor(t)+1``, so the
    target must stay strictly below ``reg_max - 1`` for the upper index to
    remain in range.
    """
    x1y1, x2y2 = bbox.chunk(2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    return torch.cat((lt, rb), -1).clamp_(0, reg_max - 1.01)


class DetectHead(nn.Module):
    """Decoupled anchor-free head with DFL box regression."""

    def __init__(self, num_classes=80, in_channels=(128, 128, 128),
                 reg_max=16, strides=(8, 16, 32), hidden=None):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.num_levels = len(in_channels)
        self.num_outputs = num_classes + 4 * reg_max

        reg_hidden = hidden or max(16, in_channels[0] // 4, reg_max * 4)
        cls_hidden = hidden or max(in_channels[0], num_classes)

        self.reg_branches = nn.ModuleList(
            nn.Sequential(
                ConvBlock(c, reg_hidden, k=3),
                ConvBlock(reg_hidden, reg_hidden, k=3),
                nn.Conv2d(reg_hidden, 4 * reg_max, kernel_size=1),
            )
            for c in in_channels
        )
        self.cls_branches = nn.ModuleList(
            nn.Sequential(
                ConvBlock(c, cls_hidden, k=3),
                ConvBlock(cls_hidden, cls_hidden, k=3),
                nn.Conv2d(cls_hidden, num_classes, kernel_size=1),
            )
            for c in in_channels
        )

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        self._init_biases()

    def _init_biases(self):
        """Prime output biases so early training is stable.

        Classification bias encodes a low prior probability of an object being
        present. Without it the initial loss is dominated by ~8400 confident
        false positives per image and the first epochs are wasted recovering.
        """
        for reg_branch, cls_branch, stride in zip(self.reg_branches,
                                                  self.cls_branches,
                                                  self.strides):
            reg_branch[-1].bias.data.fill_(1.0)
            # Prior of roughly 1 object per 8^2 cells at a 640px input.
            prior = 5.0 / self.num_classes / (640 / stride) ** 2
            cls_branch[-1].bias.data.fill_(math.log(prior / (1 - prior)))

    def forward(self, feats):
        """Return raw per-level outputs concatenated across scales.

        Returns:
            box_dist: [B, 4*reg_max, A] raw DFL logits
            cls_logits: [B, nc, A] raw class logits
            anchor_points: [A, 2] cell centres in feature-map units
            stride_tensor: [A, 1]
        """
        box_outs, cls_outs = [], []
        for feat, reg_branch, cls_branch in zip(feats, self.reg_branches, self.cls_branches):
            box_outs.append(reg_branch(feat))
            cls_outs.append(cls_branch(feat))

        b = feats[0].shape[0]
        box_dist = torch.cat([o.view(b, 4 * self.reg_max, -1) for o in box_outs], dim=2)
        cls_logits = torch.cat([o.view(b, self.num_classes, -1) for o in cls_outs], dim=2)

        anchor_points, stride_tensor = make_anchors(feats, self.strides)
        return box_dist, cls_logits, anchor_points, stride_tensor

    def decode_boxes(self, box_dist, anchor_points, stride_tensor, xywh=False):
        """Integrate the DFL distribution and map to input-image pixel coordinates."""
        distance = self.dfl(box_dist) if self.reg_max > 1 else box_dist
        boxes = dist2bbox(distance.permute(0, 2, 1), anchor_points, xywh=xywh, dim=-1)
        return boxes * stride_tensor
