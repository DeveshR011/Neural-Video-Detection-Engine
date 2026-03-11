"""
Adversarial & Temporal Attack Resilience
=========================================
Extended defenses: PGD/FGSM attack generators for adversarial training,
compression jitter, temporal consistency verification across consecutive
frames, and stochastic preprocessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
import logging
from typing import Optional
from collections import deque

logger = logging.getLogger("AdversarialRobustness")


# ══════════════════════════════════════════════════════════════════════
#  Adversarial Attack Generators (for training)
# ══════════════════════════════════════════════════════════════════════

class FGSMAttacker:
    """Fast Gradient Sign Method attack generator."""

    def __init__(self, model: nn.Module, epsilon: float = 0.03):
        self.model = model
        self.epsilon = epsilon

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 loss_fn: nn.Module = None) -> torch.Tensor:
        """Generate FGSM adversarial examples."""
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        images_adv = images.clone().detach().requires_grad_(True)
        outputs = self.model(images_adv)

        # Flatten outputs for loss computation
        b = outputs.shape[0]
        loss = loss_fn(outputs.view(b, -1), labels.view(b, -1))
        loss.backward()

        # FGSM perturbation: x_adv = x + epsilon * sign(grad)
        perturbation = self.epsilon * images_adv.grad.sign()
        adv_images = (images + perturbation).clamp(0.0, 1.0)

        return adv_images.detach()


class PGDAttacker:
    """Projected Gradient Descent attack generator (iterative FGSM)."""

    def __init__(self, model: nn.Module, epsilon: float = 0.03,
                 alpha: float = 0.007, num_steps: int = 10):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 loss_fn: nn.Module = None) -> torch.Tensor:
        """Generate PGD adversarial examples with random start."""
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        # Random start within epsilon ball
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(
            -self.epsilon, self.epsilon
        )
        adv_images = adv_images.clamp(0.0, 1.0)

        for _ in range(self.num_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)

            b = outputs.shape[0]
            loss = loss_fn(outputs.view(b, -1), labels.view(b, -1))
            loss.backward()

            with torch.no_grad():
                adv_images = adv_images + self.alpha * adv_images.grad.sign()
                # Project back into epsilon-ball
                delta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
                adv_images = (images + delta).clamp(0.0, 1.0)

        return adv_images.detach()


# ══════════════════════════════════════════════════════════════════════
#  Runtime Preprocessing Defenses
# ══════════════════════════════════════════════════════════════════════

class AdversarialRobustness:
    """Enhanced runtime defenses for adversarial attack vectors."""

    @staticmethod
    def _add_gaussian_noise(tensor: torch.Tensor, std: float = 0.05) -> torch.Tensor:
        """Inject random perturbations to diffuse precision PGD attacks."""
        noise = torch.randn_like(tensor) * std
        return tensor + noise

    @staticmethod
    def _random_resizing(tensor: torch.Tensor) -> torch.Tensor:
        """Randomly resize the image slightly and crop back to destroy adversarial patterns."""
        b, c, h, w = tensor.shape
        scale = np.random.uniform(0.9, 1.1)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

        if scale > 1.0:
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            return resized[:, :, top:top + h, left:left + w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            return F.pad(resized, (pad_w, w - new_w - pad_w, pad_h, h - new_h - pad_h))

    @staticmethod
    def _compression_jitter(tensor: torch.Tensor,
                            quality_range: tuple = (70, 95)) -> torch.Tensor:
        """
        Randomly apply JPEG compression artifacts to break adversarial patterns.
        Operates on BxCxHxW tensors in [0, 1] range.
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning("Pillow not installed; skipping compression jitter.")
            return tensor

        b, c, h, w = tensor.shape
        quality = np.random.randint(quality_range[0], quality_range[1] + 1)
        result = torch.empty_like(tensor)

        for i in range(b):
            # Convert to PIL Image
            img_np = (tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)

            # Compress and decompress via JPEG
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            img_compressed = Image.open(buf)
            img_arr = np.array(img_compressed).astype(np.float32) / 255.0

            result[i] = torch.from_numpy(img_arr).permute(2, 0, 1)

        return result.to(tensor.device)

    @staticmethod
    def preprocess_frame(frame_tensor: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Applies all inference-time defenses before the tensor hits the model."""

        # 1. Random Resizing and Cropping
        processed = AdversarialRobustness._random_resizing(frame_tensor)

        # 2. JPEG Compression Jitter (50% probability to balance speed)
        if np.random.random() < 0.5:
            processed = AdversarialRobustness._compression_jitter(processed)

        # 3. Gaussian Denoising/Perturbation
        processed = AdversarialRobustness._add_gaussian_noise(processed, std=0.01)

        return processed

    @staticmethod
    def apply_confidence_threshold(predictions: torch.Tensor,
                                   threshold: float = 0.3) -> torch.Tensor:
        """Zero out low-confidence boxes which are common in evasion attacks."""
        conf_scores = torch.sigmoid(predictions[..., 4])
        mask = conf_scores > threshold
        predictions[~mask] = 0.0
        return predictions


# ══════════════════════════════════════════════════════════════════════
#  Temporal Consistency Verification
# ══════════════════════════════════════════════════════════════════════

class TemporalConsistencyChecker:
    """
    Monitors detection consistency across consecutive frames.
    Flags frames where predictions deviate abruptly from the temporal
    window, indicating possible adversarial video injection.
    
    Uses IoU-based scoring between consecutive frame detections.
    """

    def __init__(self, window_size: int = 5, iou_threshold: float = 0.15,
                 alert_threshold: float = 0.5):
        """
        Args:
            window_size: Number of previous frames to consider
            iou_threshold: Minimum IoU for a box to count as "consistent"
            alert_threshold: Fraction of boxes that must be consistent
        """
        self.window_size = window_size
        self.iou_threshold = iou_threshold
        self.alert_threshold = alert_threshold
        self._history: deque = deque(maxlen=window_size)
        self._alert_count = 0

    def check(self, detections: torch.Tensor) -> tuple[bool, float]:
        """
        Check a new frame's detections against the temporal window.
        
        Args:
            detections: Tensor of shape [N, 5+] where columns are (x, y, w, h, conf, ...)
            
        Returns:
            (is_consistent, consistency_score)
        """
        if len(self._history) == 0:
            self._history.append(detections.detach().cpu())
            return True, 1.0

        # Compare against the most recent frame
        prev_detections = self._history[-1]
        consistency = self._compute_consistency(prev_detections, detections.detach().cpu())

        self._history.append(detections.detach().cpu())

        is_consistent = consistency >= self.alert_threshold
        if not is_consistent:
            self._alert_count += 1
            logger.warning(
                f"TEMPORAL ANOMALY: consistency={consistency:.2%} "
                f"(threshold={self.alert_threshold:.0%}). "
                f"Total alerts: {self._alert_count}"
            )

        return is_consistent, consistency

    def _compute_consistency(self, prev: torch.Tensor, curr: torch.Tensor) -> float:
        """Compute IoU-based consistency between two frames' detections."""
        if prev.numel() == 0 and curr.numel() == 0:
            return 1.0
        if prev.numel() == 0 or curr.numel() == 0:
            return 0.0

        # Filter to confident detections only
        prev_conf = prev[torch.sigmoid(prev[:, 4]) > 0.3]
        curr_conf = curr[torch.sigmoid(curr[:, 4]) > 0.3]

        if prev_conf.numel() == 0 and curr_conf.numel() == 0:
            return 1.0
        if prev_conf.numel() == 0 or curr_conf.numel() == 0:
            return 0.0

        # Compute pairwise IoU between xywh boxes
        matched = 0
        for box in curr_conf:
            ious = self._batch_iou(box[:4].unsqueeze(0), prev_conf[:, :4])
            if ious.max() > self.iou_threshold:
                matched += 1

        total = max(len(curr_conf), 1)
        return matched / total

    @staticmethod
    def _batch_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU between one box and a batch. Boxes in xywh format."""
        # Convert xywh to xyxy
        b1_x1 = box[:, 0] - box[:, 2] / 2
        b1_y1 = box[:, 1] - box[:, 3] / 2
        b1_x2 = box[:, 0] + box[:, 2] / 2
        b1_y2 = box[:, 1] + box[:, 3] / 2

        b2_x1 = boxes[:, 0] - boxes[:, 2] / 2
        b2_y1 = boxes[:, 1] - boxes[:, 3] / 2
        b2_x2 = boxes[:, 0] + boxes[:, 2] / 2
        b2_y2 = boxes[:, 1] + boxes[:, 3] / 2

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = b1_area + b2_area - inter_area + 1e-6

        return inter_area / union

    @property
    def alert_count(self) -> int:
        return self._alert_count
