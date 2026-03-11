"""
Confidence Calibration & OOD Detection
=========================================
Temperature scaling for posterior calibration, ECE/Brier metrics,
and a lightweight out-of-distribution detection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("Calibration")


# ══════════════════════════════════════════════════════════════════════
#  Temperature Scaling
# ══════════════════════════════════════════════════════════════════════

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for confidence calibration.
    Learns a single scalar temperature T on the validation set.
    
    calibrated_logit = logit / T
    
    T > 1 → softer (less overconfident)
    T < 1 → sharper (more confident)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature."""
        return logits / self.temperature.clamp(min=0.01)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 100):
        """
        Optimize temperature on validation data using NLL loss.
        
        Args:
            logits: [N, C] model output logits
            labels: [N] ground truth class indices
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = nll(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        logger.info(f"Temperature calibrated: T = {self.temperature.item():.4f}")


# ══════════════════════════════════════════════════════════════════════
#  Calibration Metrics
# ══════════════════════════════════════════════════════════════════════

class CalibrationMetrics:
    """Computes Expected Calibration Error (ECE) and Brier Score."""

    @staticmethod
    def expected_calibration_error(
        confidences: torch.Tensor,
        accuracies: torch.Tensor,
        n_bins: int = 15,
    ) -> float:
        """
        ECE: weighted average of |confidence - accuracy| across bins.
        
        Args:
            confidences: [N] predicted confidence scores (max softmax prob)
            accuracies: [N] binary correctness indicators (1=correct, 0=wrong)
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(confidences)

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            n_bin = mask.sum().item()

            if n_bin > 0:
                avg_confidence = confidences[mask].mean().item()
                avg_accuracy = accuracies[mask].float().mean().item()
                ece += (n_bin / total) * abs(avg_confidence - avg_accuracy)

        return ece

    @staticmethod
    def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Brier Score: mean squared error of probabilistic predictions.
        Lower is better. Range: [0, 2] for multi-class.
        
        Args:
            probs: [N, C] predicted class probabilities
            targets: [N, C] one-hot encoded targets
        """
        return F.mse_loss(probs, targets).item()

    @staticmethod
    def compute_from_detections(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int = 80,
    ) -> dict:
        """
        Compute calibration metrics from raw detection outputs.
        
        Args:
            predictions: [N, 5+C] detection predictions
            targets: [N, 5+C] ground truth
        """
        pred_conf = torch.sigmoid(predictions[:, 4])
        pred_cls_probs = torch.sigmoid(predictions[:, 5:])

        # Object confidence calibration
        target_obj = (targets[:, 4] > 0.5).float()
        obj_correct = ((pred_conf > 0.5) == (target_obj > 0.5)).long()

        ece = CalibrationMetrics.expected_calibration_error(pred_conf, obj_correct)

        # Class probability Brier score
        target_cls = targets[:, 5:]
        brier = CalibrationMetrics.brier_score(pred_cls_probs, target_cls)

        return {
            'ece': round(ece, 4),
            'brier': round(brier, 4),
            'mean_confidence': round(pred_conf.mean().item(), 4),
            'confidence_std': round(pred_conf.std().item(), 4),
        }


# ══════════════════════════════════════════════════════════════════════
#  Out-of-Distribution Detection Head
# ══════════════════════════════════════════════════════════════════════

class OODDetector(nn.Module):
    """
    Lightweight binary classifier on detection features for
    out-of-distribution input detection.
    
    Operates on the flattened detection output — uses energy-based
    scoring to flag anomalous inputs that may be adversarial or OOD.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.threshold = 0.5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict OOD score (higher = more likely OOD)."""
        return torch.sigmoid(self.classifier(features))

    def is_ood(self, features: torch.Tensor) -> tuple[bool, float]:
        """
        Binary OOD decision.
        
        Returns:
            (is_ood, ood_score)
        """
        with torch.no_grad():
            score = self.forward(features).mean().item()
        return score > self.threshold, score

    @staticmethod
    def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Energy-based OOD score: E(x) = -T * logsumexp(logits / T).
        Lower energy → in-distribution. Higher energy → OOD.
        """
        return -temperature * torch.logsumexp(logits / temperature, dim=-1)


# ══════════════════════════════════════════════════════════════════════
#  Confidence Calibration Pipeline
# ══════════════════════════════════════════════════════════════════════

class CalibrationPipeline:
    """
    End-to-end calibration pipeline for secure detection:
    1. Temperature scaling
    2. OOD detection
    3. Calibration metrics
    """

    def __init__(self, num_classes: int = 80, feature_dim: int = 255):
        self.temp_scaler = TemperatureScaler()
        self.ood_detector = OODDetector(input_dim=feature_dim)
        self.metrics = CalibrationMetrics()
        self.num_classes = num_classes

    def calibrate_predictions(self, raw_detections: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to detection class logits.
        """
        calibrated = raw_detections.clone()
        # Scale class logits only (indices 5+)
        calibrated[..., 5:] = self.temp_scaler(raw_detections[..., 5:])
        return calibrated

    def check_ood(self, detections: torch.Tensor) -> tuple[bool, float]:
        """Check if the detection output suggests OOD input."""
        cls_logits = detections[..., 5:]
        energy = OODDetector.energy_score(cls_logits)
        mean_energy = energy.mean().item()
        # High energy = OOD
        is_ood = mean_energy > -5.0  # Threshold set empirically
        return is_ood, mean_energy

    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute full calibration report."""
        flat_pred = predictions.reshape(-1, predictions.shape[-1])
        flat_tgt = targets.reshape(-1, targets.shape[-1])

        # Filter to non-empty predictions for meaningful metrics
        mask = flat_pred[:, 4].abs() > 0
        if mask.sum() == 0:
            return {'ece': 0.0, 'brier': 0.0, 'samples': 0}

        metrics = CalibrationMetrics.compute_from_detections(
            flat_pred[mask], flat_tgt[mask], self.num_classes
        )
        metrics['samples'] = mask.sum().item()
        return metrics
