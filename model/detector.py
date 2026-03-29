"""
Secure YOLO-v2 Detection Model — CSP + BiFPN Architecture
===========================================================
Advanced architecture with:
- Depthwise-separable convolutions for efficiency
- CSP (Cross Stage Partial) blocks for richer gradients
- BiFPN (Bidirectional Feature Pyramid) for multi-scale fusion
- Multi-scale detection heads (P3/P4/P5)
- Feature denoising blocks for adversarial robustness
- Random projection head for feature leakage prevention
- In-place activations and explicit tensor zeroing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter + 1e-7
    return inter / union


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        ious = _box_iou_xyxy(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]

    return torch.stack(keep) if keep else boxes.new_zeros((0,), dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════
#  Core Building Blocks
# ══════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Standard Conv + BN + SiLU with in-place activation."""
    def __init__(self, in_c, out_c, k=3, s=1, p=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise 3×3 + pointwise 1×1.
    ~8× fewer FLOPs than standard 3×3 conv at same channel width."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.depthwise = ConvBlock(in_c, in_c, k, s, p, groups=in_c)
        self.pointwise = ConvBlock(in_c, out_c, k=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Bottleneck(nn.Module):
    """Bottleneck residual block with optional shortcut."""
    def __init__(self, in_c, out_c, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(out_c * expansion)
        self.cv1 = ConvBlock(in_c, hidden, k=1)
        self.cv2 = DepthwiseSeparableConv(hidden, out_c, k=3, p=1)
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        if self.add:
            out = out + x  # Residual
        return out


class CSPBlock(nn.Module):
    """Cross Stage Partial block: splits channels, processes one branch
    through stacked bottlenecks, concatenates both branches.
    Produces richer gradients with fewer parameters."""
    def __init__(self, in_c, out_c, n_bottlenecks=2, shortcut=True):
        super().__init__()
        hidden = out_c // 2
        self.cv1 = ConvBlock(in_c, hidden, k=1)          # Branch 1 (processed)
        self.cv2 = ConvBlock(in_c, hidden, k=1)          # Branch 2 (skip)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(hidden, hidden, shortcut) for _ in range(n_bottlenecks)]
        )
        self.cv3 = ConvBlock(hidden * 2, out_c, k=1)     # Fusion

    def forward(self, x):
        branch1 = self.bottlenecks(self.cv1(x))
        branch2 = self.cv2(x)
        return self.cv3(torch.cat([branch1, branch2], dim=1))


class FeatureDenoisingBlock(nn.Module):
    """Lightweight feature denoising via 1×1 conv + BN.
    Acts as a learned smoothing filter that suppresses adversarial
    high-frequency perturbations in the feature space."""
    def __init__(self, channels):
        super().__init__()
        self.denoise = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.denoise(x) + x  # Residual denoising


# ══════════════════════════════════════════════════════════════════════
#  BiFPN — Bidirectional Feature Pyramid Network
# ══════════════════════════════════════════════════════════════════════

class BiFPNLayer(nn.Module):
    """Single BiFPN layer with learnable weighted feature fusion.
    Fuses P3 (80×80), P4 (40×40), P5 (20×20) bidirectionally."""

    def __init__(self, channels):
        super().__init__()
        self.eps = 1e-4

        # Learnable fusion weights (softmax-normalized)
        # Top-down: P5→P4, P4→P3
        self.w_p4_td = nn.Parameter(torch.ones(2))
        self.w_p3_td = nn.Parameter(torch.ones(2))
        # Bottom-up: P3→P4, P4→P5
        self.w_p4_bu = nn.Parameter(torch.ones(3))
        self.w_p5_bu = nn.Parameter(torch.ones(3))

        # Depthwise-separable convs for each fused output
        self.conv_p4_td = DepthwiseSeparableConv(channels, channels)
        self.conv_p3_out = DepthwiseSeparableConv(channels, channels)
        self.conv_p4_out = DepthwiseSeparableConv(channels, channels)
        self.conv_p5_out = DepthwiseSeparableConv(channels, channels)

    def forward(self, p3, p4, p5):
        # ── Top-down pathway ──────────────────────────────────
        # P4_td = Conv(w1*P4 + w2*Resize(P5)) / sum(w)
        w = F.relu(self.w_p4_td)
        w = w / (w.sum() + self.eps)
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4_td = self.conv_p4_td(w[0] * p4 + w[1] * p5_up)

        # P3_out = Conv(w1*P3 + w2*Resize(P4_td)) / sum(w)
        w = F.relu(self.w_p3_td)
        w = w / (w.sum() + self.eps)
        p4_td_up = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_out = self.conv_p3_out(w[0] * p3 + w[1] * p4_td_up)

        # ── Bottom-up pathway ─────────────────────────────────
        # P4_out = Conv(w1*P4 + w2*P4_td + w3*Resize(P3_out)) / sum(w)
        w = F.relu(self.w_p4_bu)
        w = w / (w.sum() + self.eps)
        p3_down = F.interpolate(p3_out, size=p4.shape[2:], mode='nearest')
        p4_out = self.conv_p4_out(w[0] * p4 + w[1] * p4_td + w[2] * p3_down)

        # P5_out = Conv(w1*P5 + w2*P5_bu_skip + w3*Resize(P4_out)) / sum(w)
        w = F.relu(self.w_p5_bu)
        w = w / (w.sum() + self.eps)
        p4_down = F.interpolate(p4_out, size=p5.shape[2:], mode='nearest')
        p5_out = self.conv_p5_out(w[0] * p5 + w[1] * p5 + w[2] * p4_down)

        return p3_out, p4_out, p5_out


# ══════════════════════════════════════════════════════════════════════
#  Random Projection (Security Layer — Frozen)
# ══════════════════════════════════════════════════════════════════════

class RandomProjectionHead(nn.Module):
    """Frozen random projection for feature leakage prevention."""
    def __init__(self, in_c, out_c, seed=42):
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, 1, bias=False)
        gen = torch.Generator()
        gen.manual_seed(seed)
        nn.init.normal_(self.proj.weight, 0.0, 0.01, generator=gen)
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.proj(x)


# ══════════════════════════════════════════════════════════════════════
#  SecureYOLO v2 — Full Model
# ══════════════════════════════════════════════════════════════════════

class SecureYOLO(nn.Module):
    """
    Advanced YOLO-style architecture with security hardening:
    - CSP backbone with depthwise-separable convolutions
    - BiFPN multi-scale feature fusion
    - Multi-scale detection heads (P3 80×80, P4 40×40, P5 20×20)
    - Feature denoising for adversarial robustness
    - Random projection for feature obfuscation
    - In-place ops and intermediate tensor zeroing
    """

    def __init__(self, num_classes: int = 80, fpn_channels: int = 128,
                 projection_seed: int = 42):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.anchors = (
            ((10.0, 13.0), (16.0, 30.0), (33.0, 23.0)),
            ((30.0, 61.0), (62.0, 45.0), (59.0, 119.0)),
            ((116.0, 90.0), (156.0, 198.0), (373.0, 326.0)),
        )
        self.strides = (8, 16, 32)
        self.num_outputs = 3 * (5 + num_classes)  # 3 anchors per scale

        # ── Backbone (CSP stages producing P3, P4, P5) ────────
        self.stem = ConvBlock(3, 32, k=3, s=2, p=1)

        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, k=3, s=2, p=1),
            CSPBlock(64, 64, n_bottlenecks=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, k=3, s=2, p=1),
            CSPBlock(128, 128, n_bottlenecks=2),
        )  # → P3 (80×80)

        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, k=3, s=2, p=1),
            CSPBlock(256, 256, n_bottlenecks=2),
        )  # → P4 (40×40)

        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, k=3, s=2, p=1),
            CSPBlock(512, 512, n_bottlenecks=1),
        )  # → P5 (20×20)

        # ── Lateral convolutions (align channels to fpn_channels) ──
        self.lateral_p3 = ConvBlock(128, fpn_channels, k=1)
        self.lateral_p4 = ConvBlock(256, fpn_channels, k=1)
        self.lateral_p5 = ConvBlock(512, fpn_channels, k=1)

        # ── BiFPN ─────────────────────────────────────────────
        self.bifpn = BiFPNLayer(fpn_channels)

        # ── Feature denoising (adversarial defense) ───────────
        self.denoise_p3 = FeatureDenoisingBlock(fpn_channels)
        self.denoise_p4 = FeatureDenoisingBlock(fpn_channels)
        self.denoise_p5 = FeatureDenoisingBlock(fpn_channels)

        # ── Random projection (security) ──────────────────────
        self.random_projection = RandomProjectionHead(
            fpn_channels, fpn_channels, seed=projection_seed
        )

        # ── Detection heads (per-scale) ───────────────────────
        self.head_p3 = nn.Conv2d(fpn_channels, self.num_outputs, 1)
        self.head_p4 = nn.Conv2d(fpn_channels, self.num_outputs, 1)
        self.head_p5 = nn.Conv2d(fpn_channels, self.num_outputs, 1)

    def _forward_raw_scales(self, x):
        # ── Backbone ──────────────────────────────────────────
        s = self.stem(x)
        s1 = self.stage1(s)
        c3 = self.stage2(s1)    # P3: 80×80
        c4 = self.stage3(c3)    # P4: 40×40
        c5 = self.stage4(c4)    # P5: 20×20

        # Wipe early features
        if not self.training:
            s.zero_(); del s
            s1.zero_(); del s1

        # ── Lateral alignment ─────────────────────────────────
        p3 = self.lateral_p3(c3)
        p4 = self.lateral_p4(c4)
        p5 = self.lateral_p5(c5)

        if not self.training:
            c3.zero_(); del c3
            c4.zero_(); del c4
            c5.zero_(); del c5

        # ── BiFPN fusion ──────────────────────────────────────
        p3, p4, p5 = self.bifpn(p3, p4, p5)

        # ── Feature denoising ─────────────────────────────────
        p3 = self.denoise_p3(p3)
        p4 = self.denoise_p4(p4)
        p5 = self.denoise_p5(p5)

        # ── Random projection (applied per scale) ─────────────
        p3 = self.random_projection(p3)
        p4 = self.random_projection(p4)
        p5 = self.random_projection(p5)

        # ── Detection heads ───────────────────────────────────
        out_p3 = self.head_p3(p3)
        out_p4 = self.head_p4(p4)
        out_p5 = self.head_p5(p5)

        if not self.training:
            p3.zero_(); del p3
            p4.zero_(); del p4
            p5.zero_(); del p5

        b = out_p3.shape[0]
        raw_p3 = out_p3.view(b, 3, 5 + self.num_classes, out_p3.shape[2], out_p3.shape[3]).permute(0, 1, 3, 4, 2).contiguous()
        raw_p4 = out_p4.view(b, 3, 5 + self.num_classes, out_p4.shape[2], out_p4.shape[3]).permute(0, 1, 3, 4, 2).contiguous()
        raw_p5 = out_p5.view(b, 3, 5 + self.num_classes, out_p5.shape[2], out_p5.shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:
            out_p3.zero_(); del out_p3
            out_p4.zero_(); del out_p4
            out_p5.zero_(); del out_p5

        return raw_p3, raw_p4, raw_p5

    def forward(self, x):
        raw_p3, raw_p4, raw_p5 = self._forward_raw_scales(x)

        # Flatten and concatenate all scales:
        # Each output: [B, num_outputs, H, W] → [B, H*W*3, 5+num_classes]
        b = raw_p3.shape[0]
        det_p3 = raw_p3.reshape(b, -1, 5 + self.num_classes)
        det_p4 = raw_p4.reshape(b, -1, 5 + self.num_classes)
        det_p5 = raw_p5.reshape(b, -1, 5 + self.num_classes)

        detections = torch.cat([det_p3, det_p4, det_p5], dim=1)

        if not self.training:
            det_p3.zero_(); del det_p3
            det_p4.zero_(); del det_p4
            det_p5.zero_(); del det_p5

        return detections

    def _decode_scale(self, raw_scale: torch.Tensor, stride: int, anchors, image_hw: tuple[int, int]) -> torch.Tensor:
        b, n_a, h, w, _ = raw_scale.shape
        device = raw_scale.device
        img_h, img_w = image_hw

        yv, xv = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        grid = torch.stack((xv, yv), dim=-1).float().view(1, 1, h, w, 2)

        anchor_tensor = torch.tensor(anchors, device=device, dtype=raw_scale.dtype).view(1, n_a, 1, 1, 2)

        tx_ty = raw_scale[..., 0:2]
        tw_th = raw_scale[..., 2:4]
        obj = raw_scale[..., 4:5]
        cls = raw_scale[..., 5:]

        xy = (torch.sigmoid(tx_ty) + grid) * stride
        wh = torch.exp(tw_th).clamp(max=1e3) * anchor_tensor

        decoded = torch.cat([
            xy[..., 0:1] / max(float(img_w), 1.0),
            xy[..., 1:2] / max(float(img_h), 1.0),
            wh[..., 0:1] / max(float(img_w), 1.0),
            wh[..., 1:2] / max(float(img_h), 1.0),
            torch.sigmoid(obj),
            torch.sigmoid(cls),
        ], dim=-1)

        decoded[..., :4] = decoded[..., :4].clamp(0.0, 1.0)
        return decoded.view(b, -1, 5 + self.num_classes)

    @torch.no_grad()
    def predict(self, frame_tensor: torch.Tensor, conf_threshold: float = 0.35,
                iou_threshold: float = 0.45, max_detections: int = 300,
                pre_nms_topk: int = 1000) -> torch.Tensor:
        was_training = self.training
        self.eval()

        raw_scales = self._forward_raw_scales(frame_tensor)
        image_hw = (frame_tensor.shape[-2], frame_tensor.shape[-1])

        decoded = [
            self._decode_scale(raw_scales[0], self.strides[0], self.anchors[0], image_hw),
            self._decode_scale(raw_scales[1], self.strides[1], self.anchors[1], image_hw),
            self._decode_scale(raw_scales[2], self.strides[2], self.anchors[2], image_hw),
        ]

        pred = torch.cat(decoded, dim=1)
        b = pred.shape[0]
        out = pred.new_zeros((b, max_detections, 5 + self.num_classes))

        for bi in range(b):
            boxes_xywh = pred[bi, :, :4]
            obj = pred[bi, :, 4]
            cls_probs = pred[bi, :, 5:]

            cls_conf, cls_idx = cls_probs.max(dim=1)
            scores = obj * cls_conf
            keep_mask = scores > conf_threshold
            if keep_mask.sum() == 0:
                continue

            boxes_xywh = boxes_xywh[keep_mask]
            scores = scores[keep_mask]
            obj = obj[keep_mask]
            cls_idx = cls_idx[keep_mask]

            # Bound NMS complexity by capping candidate count.
            if pre_nms_topk > 0 and scores.numel() > pre_nms_topk:
                topk = torch.topk(scores, pre_nms_topk, largest=True).indices
                boxes_xywh = boxes_xywh[topk]
                scores = scores[topk]
                obj = obj[topk]
                cls_idx = cls_idx[topk]

            boxes_xyxy = _xywh_to_xyxy(boxes_xywh).clamp(0.0, 1.0)

            final_keep = []
            for c in cls_idx.unique():
                c_mask = cls_idx == c
                c_keep = _nms_xyxy(boxes_xyxy[c_mask], scores[c_mask], iou_threshold)
                if c_keep.numel() > 0:
                    c_indices = torch.where(c_mask)[0][c_keep]
                    final_keep.append(c_indices)

            if not final_keep:
                continue

            final_keep = torch.cat(final_keep)
            final_keep = final_keep[scores[final_keep].argsort(descending=True)]
            final_keep = final_keep[:max_detections]

            n = final_keep.numel()
            out[bi, :n, :4] = boxes_xywh[final_keep]
            out[bi, :n, 4] = obj[final_keep]
            out[bi, :n, 5:] = 0.0
            out[bi, torch.arange(n, device=out.device), 5 + cls_idx[final_keep]] = 1.0

        if was_training:
            self.train()
        return out

    @torch.no_grad()
    def secure_forward(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """Inference-only forward with input cleanup."""
        out = self.predict(frame_tensor)
        frame_tensor.zero_()
        return out
