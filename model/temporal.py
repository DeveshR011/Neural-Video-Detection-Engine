"""
Temporal Smoothing for Video Detection
=========================================
Lightweight temporal stabilization without LSTM overhead:
- EMA logit smoother across frames
- Box momentum tracker for flicker reduction
- All in-memory, no disk writes, clearable on shutdown
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger("TemporalSmoothing")


class EMALogitSmoother:
    """
    Exponential Moving Average smoother for detection logits.
    Stabilizes predictions across consecutive video frames
    with zero learnable parameters.
    
    new_output = α * current + (1 - α) * previous_ema
    """

    def __init__(self, alpha: float = 0.3, device: torch.device = None):
        """
        Args:
            alpha: Weight for current frame (higher = less smoothing).
                   0.3 provides good stability without excessive lag.
        """
        self.alpha = alpha
        self.device = device
        self._ema: Optional[torch.Tensor] = None

    def smooth(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Apply EMA smoothing to detection output.
        
        Args:
            detections: [B, N, 5+C] detection tensor
        Returns:
            Smoothed detection tensor (same shape)
        """
        if self._ema is None:
            self._ema = detections.clone().detach()
            return detections

        # EMA update
        with torch.no_grad():
            self._ema = self.alpha * detections + (1 - self.alpha) * self._ema.to(detections.device)

        return self._ema.clone()

    def reset(self):
        """Clear temporal state (secure shutdown)."""
        if self._ema is not None:
            self._ema.zero_()
            self._ema = None

    @property
    def is_initialized(self) -> bool:
        return self._ema is not None


class BoxMomentumTracker:
    """
    Smooths bounding box coordinates using momentum-based tracking.
    Reduces frame-to-frame jitter in box positions.
    
    Uses per-box matching via IoU to associate boxes across frames,
    then applies momentum smoothing to matched pairs.
    """

    def __init__(self, momentum: float = 0.7, iou_threshold: float = 0.3):
        """
        Args:
            momentum: Weight for previous position (higher = smoother, more lag)
            iou_threshold: Minimum IoU to consider a match
        """
        self.momentum = momentum
        self.iou_threshold = iou_threshold
        self._prev_boxes: Optional[torch.Tensor] = None
        self._prev_confs: Optional[torch.Tensor] = None

    def smooth(self, boxes: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
        """
        Apply momentum smoothing to box coordinates.
        
        Args:
            boxes: [N, 4] in xywh format
            confidences: [N] confidence scores
        Returns:
            Smoothed boxes [N, 4]
        """
        if self._prev_boxes is None or self._prev_boxes.numel() == 0:
            self._prev_boxes = boxes.clone().detach()
            self._prev_confs = confidences.clone().detach()
            return boxes

        smoothed = boxes.clone()

        # Match current boxes to previous via IoU
        if boxes.numel() > 0 and self._prev_boxes.numel() > 0:
            ious = self._compute_iou_matrix(boxes, self._prev_boxes)

            for i in range(len(boxes)):
                if ious.shape[1] > 0:
                    best_iou, best_idx = ious[i].max(0)
                    if best_iou > self.iou_threshold:
                        # Apply momentum smoothing
                        smoothed[i] = (
                            (1 - self.momentum) * boxes[i] +
                            self.momentum * self._prev_boxes[best_idx]
                        )

        self._prev_boxes = smoothed.clone().detach()
        self._prev_confs = confidences.clone().detach()

        return smoothed

    def _compute_iou_matrix(self, boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix between two sets of xywh boxes."""
        # Convert xywh → xyxy
        a_x1 = boxes_a[:, 0] - boxes_a[:, 2] / 2
        a_y1 = boxes_a[:, 1] - boxes_a[:, 3] / 2
        a_x2 = boxes_a[:, 0] + boxes_a[:, 2] / 2
        a_y2 = boxes_a[:, 1] + boxes_a[:, 3] / 2

        b_x1 = boxes_b[:, 0] - boxes_b[:, 2] / 2
        b_y1 = boxes_b[:, 1] - boxes_b[:, 3] / 2
        b_x2 = boxes_b[:, 0] + boxes_b[:, 2] / 2
        b_y2 = boxes_b[:, 1] + boxes_b[:, 3] / 2

        n_a, n_b = len(boxes_a), len(boxes_b)
        ious = torch.zeros(n_a, n_b, device=boxes_a.device)

        for i in range(n_a):
            inter_x1 = torch.max(a_x1[i], b_x1)
            inter_y1 = torch.max(a_y1[i], b_y1)
            inter_x2 = torch.min(a_x2[i], b_x2)
            inter_y2 = torch.min(a_y2[i], b_y2)

            inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            area_a = (a_x2[i] - a_x1[i]) * (a_y2[i] - a_y1[i])
            area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
            union = area_a + area_b - inter + 1e-7

            ious[i] = inter / union

        return ious

    def reset(self):
        """Clear tracking state (secure shutdown)."""
        if self._prev_boxes is not None:
            self._prev_boxes.zero_()
            self._prev_boxes = None
        if self._prev_confs is not None:
            self._prev_confs.zero_()
            self._prev_confs = None


class TemporalSmoother:
    """
    Combined temporal smoothing pipeline integrating EMA and box momentum.
    Drop-in module for the secure pipeline runner.
    """

    def __init__(self, ema_alpha: float = 0.3, box_momentum: float = 0.7,
                 iou_threshold: float = 0.3, num_classes: int = 80):
        self.ema_smoother = EMALogitSmoother(alpha=ema_alpha)
        self.box_tracker = BoxMomentumTracker(momentum=box_momentum, iou_threshold=iou_threshold)
        self.num_classes = num_classes

    def smooth(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Apply full temporal smoothing pipeline.
        
        Args:
            detections: [B, N, 5+C] raw detection tensor
        Returns:
            Temporally smoothed detections
        """
        # 1. EMA on full logits
        smoothed = self.ema_smoother.smooth(detections)

        # 2. Box momentum on confident detections
        for b in range(smoothed.shape[0]):
            confs = torch.sigmoid(smoothed[b, :, 4])
            mask = confs > 0.3
            if mask.sum() > 0:
                boxes = smoothed[b, mask, :4]
                tracked = self.box_tracker.smooth(boxes, confs[mask])
                smoothed[b, mask, :4] = tracked

        return smoothed

    def reset(self):
        """Secure reset — wipe all temporal state."""
        self.ema_smoother.reset()
        self.box_tracker.reset()
        logger.info("Temporal smoother state wiped.")
