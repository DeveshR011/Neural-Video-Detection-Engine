"""
Generic Loss Components
=======================
IoU-family box metrics and classification losses shared across the codebase.

The combined detection objective lives in ``model/detection_loss.py``; this
module holds only reusable pieces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════
#  IoU Family (Box Regression)
# ══════════════════════════════════════════════════════════════════════

def bbox_iou(box1, box2, xywh=True, giou=False, diou=False, ciou=False, eps=1e-7):
    """IoU and its distance-aware variants between broadcastable box tensors.

    Args:
        box1, box2: [..., 4] boxes, xywh if ``xywh`` else xyxy
        giou/diou/ciou: select a variant; plain IoU when all are False

    Returns:
        Tensor of shape [...] with the requested overlap measure.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if not (giou or diou or ciou):
        return iou.squeeze(-1)

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    if ciou or diou:
        c2 = cw.pow(2) + ch.pow(2) + eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
        if ciou:
            v = (4 / math.pi ** 2) * (
                (w2 / h2).atan() - (w1 / h1.clamp(eps)).atan()
            ).pow(2)
            # alpha is a non-gradient trade-off factor; detaching it keeps the
            # aspect-ratio term from destabilizing early training.
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            return (iou - (rho2 / c2 + v * alpha)).squeeze(-1)
        return (iou - rho2 / c2).squeeze(-1)

    c_area = cw * ch + eps
    return (iou - (c_area - union) / c_area).squeeze(-1)


class CIoULoss(nn.Module):
    """Complete IoU Loss with distance, overlap, and aspect ratio penalties.
    Better convergence than vanilla IoU/GIoU for box regression."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] in xywh format
            target_boxes: [N, 4] in xywh format
        """
        # Convert xywh → xyxy
        p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        t_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        t_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        t_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        t_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Intersection
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # Union
        area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
        area_t = (t_x2 - t_x1) * (t_y2 - t_y1)
        union = area_p + area_t - inter + 1e-7

        iou = inter / union

        # Distance penalty (center distance / diagonal of enclosing box)
        cx_p, cy_p = pred_boxes[:, 0], pred_boxes[:, 1]
        cx_t, cy_t = target_boxes[:, 0], target_boxes[:, 1]
        center_dist_sq = (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2

        enclose_x1 = torch.min(p_x1, t_x1)
        enclose_y1 = torch.min(p_y1, t_y1)
        enclose_x2 = torch.max(p_x2, t_x2)
        enclose_y2 = torch.max(p_y2, t_y2)
        diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

        # Aspect ratio penalty
        w_p, h_p = pred_boxes[:, 2], pred_boxes[:, 3]
        w_t, h_t = target_boxes[:, 2], target_boxes[:, 3]
        v = (4 / math.pi ** 2) * (torch.atan(w_t / (h_t + 1e-7)) - torch.atan(w_p / (h_p + 1e-7))) ** 2
        alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - center_dist_sq / diag_sq - alpha * v
        loss = 1 - ciou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ══════════════════════════════════════════════════════════════════════
#  Classification Losses
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    Down-weights well-classified examples, focuses on hard negatives.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] raw logits
            target: [N, C] one-hot or soft targets
        """
        p = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class VarifocalLoss(nn.Module):
    """Varifocal Loss: IoU-aware classification scoring.
    Targets are set to IoU quality for positives, 0 for negatives.
    Asymmetric: only down-weights negative examples."""

    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: [N] raw logits
            target: [N] IoU-quality targets (0 for negatives, IoU for positives)
        """
        p = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Asymmetric weighting: only modulate negatives
        positive_mask = target > 0
        weight = torch.where(
            positive_mask,
            target,  # Positives weighted by IoU quality
            self.alpha * p.pow(self.gamma)  # Negatives down-weighted
        )

        loss = weight * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ══════════════════════════════════════════════════════════════════════
#  Temporal Jitter Penalty
# ══════════════════════════════════════════════════════════════════════

class TemporalJitterLoss(nn.Module):
    """L2 penalty on frame-to-frame bounding box displacement.
    Encourages stable predictions across consecutive frames during training."""

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, current_boxes, previous_boxes):
        """
        Args:
            current_boxes: [N, 4] predicted boxes for current frame
            previous_boxes: [N, 4] predicted boxes for previous frame
        Returns:
            Weighted L2 jitter penalty
        """
        if previous_boxes is None:
            return torch.tensor(0.0, device=current_boxes.device)

        jitter = F.mse_loss(current_boxes, previous_boxes)
        return self.weight * jitter
