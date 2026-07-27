"""
Task-Aligned Assigner (TAL)
===========================
Decides which predictions are responsible for which ground-truth objects.

The previous implementation matched predictions to GT by IoU alone, using a
fixed threshold. That has two well-known failure modes:

1. Early in training every predicted box is poor, so IoU-only matching selects
   almost nothing and the model receives no learning signal.
2. It optimizes localization and classification independently, so the network
   can produce a well-localized box with a low class score. NMS then discards
   it, and the metric never rewards the good box.

TAL instead scores each (prediction, GT) pair by an alignment metric

    t = s^alpha * u^beta

where ``s`` is the predicted class score for that GT's class and ``u`` is the
IoU. Selecting the top-k by ``t`` couples the two objectives: a prediction is
positive only if it is both confident and accurate. Because ``s`` is non-zero
from initialization, positives exist from the first step.

Fully vectorized over the batch -- no Python loop over images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou_aligned(box1, box2, eps=1e-7):
    """IoU between broadcastable xyxy box tensors."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp_(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp_(0)
    area1 = (b1_x2 - b1_x1).clamp_(0) * (b1_y2 - b1_y1).clamp_(0)
    area2 = (b2_x2 - b2_x1).clamp_(0) * (b2_y2 - b2_y1).clamp_(0)
    return (inter / (area1 + area2 - inter + eps)).squeeze(-1)


def select_candidates_in_gts(anchor_points, gt_bboxes, eps=1e-9):
    """Mask of anchor points falling inside each GT box.

    Args:
        anchor_points: [A, 2] centres in input-image pixels
        gt_bboxes: [B, M, 4] xyxy
    Returns:
        [B, M, A] boolean mask
    """
    n_anchors = anchor_points.shape[0]
    b, m, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, dim=2)
    deltas = torch.cat(
        (anchor_points[None] - lt, rb - anchor_points[None]), dim=2
    ).view(b, m, n_anchors, 4)
    return deltas.amin(dim=3) > eps


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """Resolve anchors assigned to several GTs by keeping the highest-IoU GT.

    Without this an anchor could receive contradictory regression targets from
    two overlapping objects.
    """
    fg_mask = mask_pos.sum(dim=1)

    if fg_mask.max() > 1:
        multi_gt_mask = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
        max_overlaps_idx = overlaps.argmax(dim=1)

        is_max_overlaps = torch.zeros_like(mask_pos, dtype=mask_pos.dtype)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

        mask_pos = torch.where(multi_gt_mask, is_max_overlaps, mask_pos).float()
        fg_mask = mask_pos.sum(dim=1)

    target_gt_idx = mask_pos.argmax(dim=1)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """Assign GT objects to predictions by a joint classification/IoU metric.

    Args:
        topk: candidate positives kept per GT
        num_classes: class count
        alpha: exponent on the classification score
        beta: exponent on IoU. Default 6.0 weights localization heavily, which
            keeps the metric from collapsing onto over-confident wrong boxes.
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores: [B, A, nc] predicted class scores (sigmoid applied)
            pd_bboxes: [B, A, 4] predicted boxes, xyxy, input-image pixels
            anchor_points: [A, 2] centres, input-image pixels
            gt_labels: [B, M, 1] class indices
            gt_bboxes: [B, M, 4] xyxy, input-image pixels
            mask_gt: [B, M, 1] validity mask for padded GT slots

        Returns:
            target_bboxes: [B, A, 4]
            target_scores: [B, A, nc] soft targets normalized by alignment
            fg_mask: [B, A] bool, positives
            target_gt_idx: [B, A]
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.long),
            )

        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )

        target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Rescale soft targets so the strongest match for each GT reaches its
        # peak IoU. This is what makes the class score predict localization
        # quality, which in turn makes NMS ranking meaningful.
        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(dim=-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt):
        mask_in_gts = select_candidates_in_gts(anchor_points, gt_bboxes)
        align_metric, overlaps = self._get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        mask_topk = self._select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def _get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[1]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na],
                               dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na],
                                  dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1).long()

        # Gather each GT's class score at every anchor, but only where the
        # anchor is a valid candidate -- avoids materializing [B, M, A, nc].
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou_aligned(gt_boxes, pd_boxes).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def _select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """Keep the top-k anchors per GT by alignment metric."""
        num_anchors = metrics.shape[-1]
        topk = min(self.topk, num_anchors)
        topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)

        if topk_mask is None:
            topk_mask = (topk_metrics.max(dim=-1, keepdim=True)[0] > self.eps)
        topk_mask = topk_mask[..., :topk]
        topk_idxs.masked_fill_(~topk_mask, 0)

        count = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
        for k in range(topk):
            count.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        # A duplicate index means the same anchor was selected twice; drop it.
        count.masked_fill_(count > 1, 0)
        return count.to(metrics.dtype)

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(self.bs, dtype=torch.int64,
                                 device=gt_labels.device).unsqueeze(-1)
        flat_idx = target_gt_idx + batch_ind * self.n_max_boxes

        target_labels = gt_labels.long().flatten()[flat_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[flat_idx]

        target_labels.clamp_(0, self.num_classes - 1)
        target_scores = F.one_hot(target_labels, self.num_classes).to(gt_bboxes.dtype)

        fg_scores_mask = fg_mask.unsqueeze(-1).expand(-1, -1, self.num_classes).bool()
        target_scores = torch.where(fg_scores_mask, target_scores,
                                    torch.zeros_like(target_scores))
        return target_bboxes, target_scores
