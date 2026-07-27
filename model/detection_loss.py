"""
Anchor-Free Detection Loss
==========================

    L = lambda_box * CIoU + lambda_dfl * DFL + lambda_cls * BCE

Targets come from the Task-Aligned Assigner, so classification targets are
soft -- scaled by alignment quality -- rather than hard one-hot. The box and
DFL terms are weighted by each positive's target score, concentrating gradient
on confident, well-localized matches.

The entire loss is vectorized over the batch. The implementation this replaces
looped over images in Python and reconstructed feature-map geometry from the
flat prediction count, which was both slow and incorrect.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import TaskAlignedAssigner
from .head import bbox2dist, dist2bbox
from .losses import bbox_iou


class BboxLoss(nn.Module):
    """CIoU regression loss plus Distribution Focal Loss.

    DFL supervises the two bins bracketing the continuous target distance,
    weighted by proximity to each. This teaches the shape of the distribution,
    not merely its mean.
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        # Shapes matter here: `weight` is [N] and `bbox_iou` returns [N]. Using
        # a [N, 1] weight against a [N] iou would broadcast into an [N, N]
        # outer product and silently corrupt the loss.
        weight = target_scores.sum(-1)[fg_mask]                       # [N]

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                       xywh=False, ciou=True)                         # [N]
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.reg_max > 1:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            pred_dist_pos = pred_dist[fg_mask].view(-1, self.reg_max)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb[fg_mask])  # [N, 1]
            loss_dfl = (loss_dfl.squeeze(-1) * weight).sum() / target_scores_sum
        else:
            loss_dfl = pred_dist.sum() * 0.0

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Cross-entropy against the two integer bins bracketing the target.

        The bin range is enforced here, in float32, rather than relying on the
        clamp inside ``bbox2dist``. Under bfloat16 autocast the mantissa is
        only 8 bits, so a value clamped to ``reg_max - 1.01`` (14.99 for 16
        bins) rounds to exactly 15.0 and the upper index ``tr`` overflows to
        16 -- a device-side assert rather than a wrong number, but only under
        AMP and only for boxes near the maximum distance.
        """
        n_bins = pred_dist.shape[1]
        target = target.float().clamp_(0, n_bins - 1 - 1e-3)
        tl = target.long()
        tr = (tl + 1).clamp_(max=n_bins - 1)
        wl = tr.to(target.dtype) - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class DetectionLoss(nn.Module):
    """Detection loss for the anchor-free DFL head.

    Args:
        num_classes: class count
        reg_max: DFL bins per box side
        lambda_box / lambda_cls / lambda_dfl: term weights
        topk / alpha / beta: assigner hyperparameters
    """

    def __init__(self, num_classes: int = 80, reg_max: int = 16,
                 lambda_box: float = 7.5, lambda_cls: float = 0.5,
                 lambda_dfl: float = 1.5, topk: int = 10,
                 alpha: float = 0.5, beta: float = 6.0):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_dfl = lambda_dfl

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(reg_max)
        self.assigner = TaskAlignedAssigner(
            topk=topk, num_classes=num_classes, alpha=alpha, beta=beta
        )
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32),
                             persistent=False)

    def _decode_bboxes(self, anchor_points, pred_dist):
        """Integrate the DFL distribution to ltrb distances, then to xyxy boxes."""
        if self.reg_max > 1:
            b, a, _ = pred_dist.shape
            proj = self.proj.to(pred_dist.dtype)
            pred_dist = pred_dist.view(b, a, 4, self.reg_max).softmax(dim=3).matmul(proj)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    @staticmethod
    def _preprocess_targets(targets, batch_size, scale_tensor, device, dtype):
        """Pad flat (batch_idx, cls, cx, cy, w, h) targets into per-image tensors.

        Returns gt_labels [B, M, 1], gt_bboxes [B, M, 4] xyxy in pixels, and
        mask_gt [B, M, 1] marking real (non-padding) entries.
        """
        if targets.numel() == 0:
            return (torch.zeros(batch_size, 0, 1, device=device, dtype=dtype),
                    torch.zeros(batch_size, 0, 4, device=device, dtype=dtype),
                    torch.zeros(batch_size, 0, 1, device=device, dtype=dtype))

        targets = targets.to(device=device, dtype=dtype)
        batch_idx = targets[:, 0].long()
        counts = torch.bincount(batch_idx, minlength=batch_size)
        max_boxes = int(counts.max().item())

        out = torch.zeros(batch_size, max_boxes, 5, device=device, dtype=dtype)
        for i in range(batch_size):
            matches = batch_idx == i
            n = int(matches.sum())
            if n:
                out[i, :n] = targets[matches, 1:]

        gt_labels = out[..., 0:1]
        gt_xywh = out[..., 1:5] * scale_tensor
        xy, wh = gt_xywh[..., 0:2], gt_xywh[..., 2:4]
        gt_bboxes = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)
        mask_gt = (wh.sum(-1, keepdim=True) > 0).to(dtype)
        return gt_labels, gt_bboxes, mask_gt

    def forward(self, preds, targets, imgsz=None):
        """
        Args:
            preds: (box_dist [B, 4*reg_max, A], cls_logits [B, nc, A],
                    anchor_points [A, 2], stride_tensor [A, 1])
            targets: [N, 6] rows of (batch_idx, class_idx, cx, cy, w, h) with
                box coordinates normalized to [0, 1]
            imgsz: (height, width) of the network input. Inferred from the
                finest feature level when omitted.

        Returns:
            (total_loss, components_dict)
        """
        box_dist, cls_logits, anchor_points, stride_tensor = preds
        device = cls_logits.device
        dtype = cls_logits.dtype
        batch_size = cls_logits.shape[0]

        pred_scores = cls_logits.permute(0, 2, 1).contiguous()   # [B, A, nc]
        pred_dist = box_dist.permute(0, 2, 1).contiguous()       # [B, A, 4*reg_max]

        # Assignment, IoU, and GT all share one coordinate frame: input pixels.
        anchor_points_img = anchor_points * stride_tensor
        pred_bboxes = self._decode_bboxes(anchor_points, pred_dist) * stride_tensor

        if imgsz is None:
            stride_min = stride_tensor.min()
            n_level0 = int((stride_tensor == stride_min).sum().item())
            side = int(round(math.sqrt(n_level0))) * float(stride_min.item())
            imgsz = (side, side)
        h, w = float(imgsz[0]), float(imgsz[1])
        scale_tensor = torch.tensor([w, h, w, h], device=device, dtype=dtype)

        gt_labels, gt_bboxes, mask_gt = self._preprocess_targets(
            targets, batch_size, scale_tensor, device, dtype
        )

        target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach(),
            anchor_points_img,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp(min=1.0)
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum

        if fg_mask.sum():
            # Box terms operate in feature-map units so distances stay inside
            # the DFL range regardless of input resolution.
            loss_box, loss_dfl = self.bbox_loss(
                pred_dist,
                pred_bboxes / stride_tensor,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
        else:
            loss_box = pred_dist.sum() * 0.0
            loss_dfl = pred_dist.sum() * 0.0

        total = (self.lambda_box * loss_box
                 + self.lambda_cls * loss_cls
                 + self.lambda_dfl * loss_dfl)

        return total, {
            "box": float(loss_box.detach()),
            "cls": float(loss_cls.detach()),
            "dfl": float(loss_dfl.detach()),
            "total": float(total.detach()),
            "n_positives": int(fg_mask.sum().detach()),
        }
