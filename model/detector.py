"""
SecureYOLO — Anchor-Free Detector
=================================
Composes the pluggable backbone, BiFPN neck, and decoupled DFL head into a
single model, and provides batched inference with vectorized NMS.

Design notes
------------
The security features that previously lived inside ``forward`` (a frozen
random projection on P3/P4/P5, and ``.zero_()`` calls on live activations)
have been removed from the accuracy path. The projection is retained as an
opt-in wrapper because it measurably costs mAP, and zeroing activations
mid-forward provided no security benefit -- the tensors were already about to
be freed, and mutating them in place risks corrupting autograd state. Model
encryption, integrity hashing, and audit logging operate at the checkpoint and
API boundaries instead, where they are effective.

Model size presets follow a width/depth scaling convention:

    n: width 0.25, depth 0.34    s: width 0.50, depth 0.34
    m: width 0.75, depth 0.67    l: width 1.00, depth 1.00
"""

import logging
import math

import torch
import torch.nn as nn
import torchvision

from .backbone import build_backbone
from .head import DetectHead
from .neck import BiFPN

logger = logging.getLogger("SecureYOLO")


SIZE_PRESETS = {
    "n": dict(width=0.25, depth=0.34, fpn_channels=64, bifpn_layers=1),
    "s": dict(width=0.50, depth=0.34, fpn_channels=128, bifpn_layers=2),
    "m": dict(width=0.75, depth=0.67, fpn_channels=192, bifpn_layers=2),
    "l": dict(width=1.00, depth=1.00, fpn_channels=256, bifpn_layers=3),
}


class RandomProjectionHead(nn.Module):
    """Frozen random projection, retained as an opt-in feature-obfuscation layer.

    Disabled by default: it degrades detection accuracy and is only meaningful
    when an untrusted party can observe intermediate feature maps.
    """

    def __init__(self, channels, seed=42):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        gen = torch.Generator().manual_seed(seed)
        weight = torch.empty(channels, channels, 1, 1)
        nn.init.orthogonal_(weight.view(channels, -1), gain=1.0)
        with torch.no_grad():
            noise = torch.randn(weight.shape, generator=gen) * 0.01
            self.proj.weight.copy_(weight + noise)
        self.proj.weight.requires_grad_(False)

    def forward(self, x):
        return self.proj(x)


class SecureYOLO(nn.Module):
    """Anchor-free detector with a hybrid CNN + P5-attention backbone.

    Args:
        num_classes: number of object classes
        backbone: ``"builtin"`` or any timm model name
        pretrained: load ImageNet weights for timm backbones
        size: one of ``n``, ``s``, ``m``, ``l``
        reg_max: DFL bins per box side
        attention: enable the P5 attention block
        secure_features: enable the opt-in random projection (costs accuracy)
    """

    def __init__(self, num_classes: int = 80, backbone: str = "builtin",
                 pretrained: bool = True, size: str = "s", reg_max: int = 16,
                 attention: bool = True, secure_features: bool = False,
                 projection_seed: int = 42, fpn_channels: int = None,
                 bifpn_layers: int = None):
        super().__init__()
        if size not in SIZE_PRESETS:
            raise ValueError(f"size must be one of {list(SIZE_PRESETS)}, got {size!r}")

        preset = SIZE_PRESETS[size]
        fpn_channels = fpn_channels or preset["fpn_channels"]
        bifpn_layers = bifpn_layers or preset["bifpn_layers"]

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.size = size
        self.secure_features = secure_features

        self.backbone = build_backbone(
            backbone, pretrained=pretrained, attention=attention,
            width=preset["width"], depth=preset["depth"],
        )
        self.strides = tuple(self.backbone.strides)

        self.neck = BiFPN(self.backbone.out_channels,
                          channels=fpn_channels, num_layers=bifpn_layers)

        self.head = DetectHead(
            num_classes=num_classes,
            in_channels=self.neck.out_channels,
            reg_max=reg_max,
            strides=self.strides,
        )

        self.projection = (
            RandomProjectionHead(fpn_channels, seed=projection_seed)
            if secure_features else None
        )

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x):
        """Return raw head outputs for loss computation.

        Returns:
            (box_dist [B, 4*reg_max, A], cls_logits [B, nc, A],
             anchor_points [A, 2], stride_tensor [A, 1])
        """
        p3, p4, p5 = self.backbone(x)
        feats = list(self.neck(p3, p4, p5))

        if self.projection is not None:
            feats = [self.projection(f) for f in feats]

        return self.head(feats)

    # ── Inference ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, images, conf_threshold: float = 0.001,
                iou_threshold: float = 0.7, max_detections: int = 300,
                pre_nms_topk: int = 3000, class_agnostic: bool = False):
        """Run inference and return per-image detections.

        The default confidence threshold is deliberately low: mAP rewards
        recall in the tail of the score distribution, so evaluation should not
        discard low-confidence boxes. Raise it for interactive use.

        Args:
            images: [B, 3, H, W] normalized input
            conf_threshold: minimum class score
            iou_threshold: NMS IoU threshold
            max_detections: cap per image
            pre_nms_topk: candidates kept before NMS
            class_agnostic: run NMS across all classes jointly

        Returns:
            list of [N, 6] tensors, rows of (x1, y1, x2, y2, score, class_id)
            in input-image pixel coordinates.
        """
        was_training = self.training
        self.eval()

        box_dist, cls_logits, anchor_points, stride_tensor = self(images)
        boxes = self.head.decode_boxes(box_dist, anchor_points, stride_tensor,
                                       xywh=False)                    # [B, A, 4]
        scores = cls_logits.permute(0, 2, 1).sigmoid()                # [B, A, nc]

        height, width = images.shape[-2:]
        results = []

        for bi in range(boxes.shape[0]):
            box_b = boxes[bi]
            score_b = scores[bi]

            best_score, best_cls = score_b.max(dim=1)
            keep = best_score > conf_threshold
            if not keep.any():
                results.append(box_b.new_zeros((0, 6)))
                continue

            box_k = box_b[keep]
            score_k = best_score[keep]
            cls_k = best_cls[keep]

            if score_k.numel() > pre_nms_topk:
                top = score_k.topk(pre_nms_topk).indices
                box_k, score_k, cls_k = box_k[top], score_k[top], cls_k[top]

            box_k[:, 0::2] = box_k[:, 0::2].clamp(0, width)
            box_k[:, 1::2] = box_k[:, 1::2].clamp(0, height)

            idx = (torch.zeros_like(cls_k) if class_agnostic else cls_k)
            nms_keep = torchvision.ops.batched_nms(box_k, score_k, idx, iou_threshold)
            nms_keep = nms_keep[:max_detections]

            results.append(torch.cat([
                box_k[nms_keep],
                score_k[nms_keep, None],
                cls_k[nms_keep, None].to(box_k.dtype),
            ], dim=1))

        if was_training:
            self.train()
        return results

    @torch.no_grad()
    def predict_padded(self, images, max_detections: int = 300, **kwargs):
        """Inference returning a dense ``[B, max_detections, 5 + nc]`` tensor.

        Layout per row: ``(cx, cy, w, h, score, one-hot class)`` with boxes
        normalized to [0, 1]. Rows beyond the detection count are zero.

        ``predict`` returns a ragged list, which is the correct representation
        for evaluation and for callers that iterate detections. The temporal
        smoother and the benchmark suite instead expect a fixed-size tensor
        they can index and stack, so that shape is produced explicitly here
        rather than by making ``predict`` ambiguous.
        """
        results = self.predict(images, max_detections=max_detections, **kwargs)
        height, width = images.shape[-2:]

        out = images.new_zeros((len(results), max_detections, 5 + self.num_classes))
        for bi, det in enumerate(results):
            n = min(det.shape[0], max_detections)
            if n == 0:
                continue
            box = det[:n, :4]
            out[bi, :n, 0] = (box[:, 0] + box[:, 2]) / 2 / width
            out[bi, :n, 1] = (box[:, 1] + box[:, 3]) / 2 / height
            out[bi, :n, 2] = (box[:, 2] - box[:, 0]) / width
            out[bi, :n, 3] = (box[:, 3] - box[:, 1]) / height
            out[bi, :n, 4] = det[:n, 4]
            out[bi, torch.arange(n, device=out.device),
                5 + det[:n, 5].long()] = 1.0
        return out

    @torch.no_grad()
    def secure_forward(self, frame_tensor, **kwargs):
        """Inference entry point that clears the caller's input buffer afterwards.

        Used by the secure pipeline when a decrypted frame must not persist in
        memory past inference. Operates on the input only -- never on
        activations inside the graph.
        """
        out = self.predict(frame_tensor, **kwargs)
        frame_tensor.zero_()
        return out

    # ── Utilities ─────────────────────────────────────────────────────

    def fuse(self):
        """Fold BatchNorm into the preceding convolution for faster inference."""
        from .blocks import ConvBlock

        fused = 0
        for module in self.modules():
            if isinstance(module, ConvBlock) and hasattr(module, "bn") and \
                    isinstance(module.bn, nn.BatchNorm2d):
                module.conv = _fuse_conv_bn(module.conv, module.bn)
                module.bn = nn.Identity()
                module.forward = module.fuse_forward
                fused += 1
        logger.info("Fused %d conv-bn pairs", fused)
        return self

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def summary(self) -> str:
        total = self.num_parameters()
        trainable = self.num_parameters(trainable_only=True)
        return (
            f"SecureYOLO-{self.size} | backbone={self.backbone.name} "
            f"classes={self.num_classes} reg_max={self.reg_max} "
            f"strides={self.strides} | params={total/1e6:.2f}M "
            f"(trainable {trainable/1e6:.2f}M) "
            f"| secure_features={self.secure_features}"
        )


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a single conv equivalent to conv followed by bn."""
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
        conv.padding, conv.dilation, conv.groups, bias=True,
    ).to(conv.weight.device).requires_grad_(False)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))

    b_conv = (torch.zeros(conv.weight.size(0), device=conv.weight.device)
              if conv.bias is None else conv.bias)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused
