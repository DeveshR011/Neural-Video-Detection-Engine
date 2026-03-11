"""
Detection Training Losses
===========================
CIoU, Focal, Varifocal, and Temporal Jitter losses for
high-accuracy object detection training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════
#  IoU Family Losses (Box Regression)
# ══════════════════════════════════════════════════════════════════════

def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    """Compute IoU between two sets of boxes. Returns IoU tensor."""
    if xywh:
        # Convert xywh → xyxy
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter + eps

    iou = inter / union
    return iou


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


# ══════════════════════════════════════════════════════════════════════
#  Combined Detection Loss
# ══════════════════════════════════════════════════════════════════════

class DetectionLoss(nn.Module):
    """Combined loss for detection training:
    L_total = λ_box * CIoU + λ_cls * Focal + λ_obj * BCE + λ_jitter * Temporal

    Handles mismatch between prediction grid (N anchors) and sparse GT
    targets (M objects) via IoU-based assignment: each prediction is
    assigned to its best-matching GT, or treated as negative."""

    def __init__(self, num_classes=80, lambda_box=5.0, lambda_cls=1.0,
                 lambda_obj=1.0, lambda_jitter=0.1, iou_assign_thresh=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_jitter = lambda_jitter
        self.iou_assign_thresh = iou_assign_thresh

        self.box_loss = CIoULoss()
        self.cls_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.jitter_loss = TemporalJitterLoss(weight=lambda_jitter)

    def _assign_targets(self, pred_boxes, gt_boxes, gt_mask):
        """
        Assign each prediction to its best-matching GT object via IoU.

        Args:
            pred_boxes: [N, 4] predicted boxes (xywh)
            gt_boxes: [M, 4] ground truth boxes (xywh)
            gt_mask: [M] objectness mask (1=valid GT, 0=padding)

        Returns:
            assigned_gt_idx: [N] index of assigned GT (-1 = negative)
            positive_mask: [N] bool mask of positive predictions
        """
        n_pred = pred_boxes.shape[0]
        n_gt = gt_boxes.shape[0]

        # Find valid GT objects
        valid = gt_mask > 0.5
        n_valid = valid.sum().item()

        if n_valid == 0:
            return (torch.full((n_pred,), -1, device=pred_boxes.device, dtype=torch.long),
                    torch.zeros(n_pred, device=pred_boxes.device, dtype=torch.bool))

        valid_gt = gt_boxes[valid]  # [K, 4]

        # Compute IoU between all predictions and valid GTs
        ious = bbox_iou(pred_boxes.unsqueeze(1).expand(-1, n_valid, -1),
                       valid_gt.unsqueeze(0).expand(n_pred, -1, -1))  # [N, K]

        # Best GT for each prediction
        best_iou, best_idx = ious.max(dim=1)  # [N], [N]

        # Only assign if IoU exceeds threshold
        positive_mask = best_iou > self.iou_assign_thresh

        # Map back to original GT indices
        valid_indices = torch.where(valid)[0]  # [K]
        assigned_gt_idx = torch.full((n_pred,), -1, device=pred_boxes.device, dtype=torch.long)
        assigned_gt_idx[positive_mask] = valid_indices[best_idx[positive_mask]]

        return assigned_gt_idx, positive_mask

    def forward(self, predictions, targets, prev_boxes=None):
        """
        Args:
            predictions: [B, N_anchors, 5+C] detection predictions
            targets: dict with 'boxes' [B, M, 4], 'classes' [B, M, C], 'obj_mask' [B, M]
        """
        B = predictions.shape[0]
        device = predictions.device

        total_box = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        n_pos = 0

        for b in range(B):
            pred_b = predictions[b]  # [N, 5+C]
            pred_boxes = pred_b[:, :4]
            pred_obj = pred_b[:, 4]
            pred_cls = pred_b[:, 5:]

            gt_boxes = targets['boxes'][b]      # [M, 4]
            gt_classes = targets['classes'][b]   # [M, C]
            gt_mask = targets['obj_mask'][b]     # [M]

            # Assign predictions to GT
            assigned_idx, pos_mask = self._assign_targets(pred_boxes.detach(), gt_boxes, gt_mask)

            # Objectness target: 1 for positive, 0 for negative
            obj_target = pos_mask.float()
            total_obj = total_obj + self.obj_loss(pred_obj, obj_target)

            # Box + class loss only on positives
            if pos_mask.sum() > 0:
                pos_pred_boxes = pred_boxes[pos_mask]
                pos_pred_cls = pred_cls[pos_mask]
                pos_gt_boxes = gt_boxes[assigned_idx[pos_mask]]
                pos_gt_cls = gt_classes[assigned_idx[pos_mask]]

                total_box = total_box + self.box_loss(pos_pred_boxes, pos_gt_boxes)
                total_cls = total_cls + self.cls_loss(pos_pred_cls, pos_gt_cls)
                n_pos += pos_mask.sum().item()

        # Average over batch
        total_obj = total_obj / B
        if n_pos > 0:
            total_box = total_box / B
            total_cls = total_cls / B

        loss_box = self.lambda_box * total_box
        loss_cls = self.lambda_cls * total_cls
        loss_obj = self.lambda_obj * total_obj
        loss_jitter = self.jitter_loss(predictions[..., :4], prev_boxes)

        total = loss_box + loss_cls + loss_obj + loss_jitter

        return {
            'total': total,
            'box': loss_box.item(),
            'cls': loss_cls.item(),
            'obj': loss_obj.item(),
            'jitter': loss_jitter.item(),
            'n_positives': n_pos,
        }
