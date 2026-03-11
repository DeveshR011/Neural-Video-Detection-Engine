"""
Adversarial Training Module
==============================
Lightweight PGD adversarial training, input gradient regularization,
and logit smoothing for robust detection under adversarial attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Callable

logger = logging.getLogger("AdversarialTrainer")


class AdversarialTrainer:
    """
    Integrates adversarial training into the detection training loop.
    
    Training strategy:
    - 50% clean batches + 50% PGD-adversarial batches
    - Input gradient regularization to smooth loss landscape
    - Logit smoothing with soft labels
    """

    def __init__(
        self,
        model: nn.Module,
        pgd_epsilon: float = 2 / 255,
        pgd_alpha: float = 0.5 / 255,
        pgd_steps: int = 3,
        grad_reg_weight: float = 0.1,
        label_smoothing: float = 0.1,
        adversarial_ratio: float = 0.5,
    ):
        """
        Args:
            model: Detection model
            pgd_epsilon: Maximum perturbation magnitude (L∞)
            pgd_alpha: Step size per PGD iteration
            pgd_steps: Number of PGD iterations (3 for speed)
            grad_reg_weight: Weight for input gradient penalty
            label_smoothing: Soft label smoothing factor
            adversarial_ratio: Fraction of batches to attack
        """
        self.model = model
        self.pgd_epsilon = pgd_epsilon
        self.pgd_alpha = pgd_alpha
        self.pgd_steps = pgd_steps
        self.grad_reg_weight = grad_reg_weight
        self.label_smoothing = label_smoothing
        self.adversarial_ratio = adversarial_ratio

    def generate_adversarial(
        self,
        images: torch.Tensor,
        loss_fn: Callable,
        targets: dict,
    ) -> torch.Tensor:
        """
        Generate PGD adversarial examples (lightweight 3-step variant).
        
        Args:
            images: Clean input images [B, C, H, W]
            loss_fn: Loss function f(model_output, targets) → scalar
            targets: Target dict for loss computation
        """
        adv = images.clone().detach()

        # Random initialization within ε-ball
        adv = adv + torch.empty_like(adv).uniform_(-self.pgd_epsilon, self.pgd_epsilon)
        adv = adv.clamp(0.0, 1.0)

        for _ in range(self.pgd_steps):
            adv.requires_grad_(True)
            outputs = self.model(adv)
            loss = loss_fn(outputs, targets)

            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                adv = adv + self.pgd_alpha * grad.sign()
                # Project back into ε-ball around original
                delta = torch.clamp(adv - images, -self.pgd_epsilon, self.pgd_epsilon)
                adv = (images + delta).clamp(0.0, 1.0)

        return adv.detach()

    def compute_gradient_regularization(
        self,
        images: torch.Tensor,
        loss_fn: Callable,
        targets: dict,
    ) -> torch.Tensor:
        """
        Input gradient regularization: penalize ‖∂L/∂x‖₂ to smooth
        the loss landscape and improve adversarial robustness.
        """
        images_grad = images.clone().detach().requires_grad_(True)
        outputs = self.model(images_grad)
        loss = loss_fn(outputs, targets)

        input_grad = torch.autograd.grad(
            loss, images_grad, create_graph=True, retain_graph=True
        )[0]

        grad_norm = input_grad.norm(p=2, dim=(1, 2, 3)).mean()
        return self.grad_reg_weight * grad_norm

    def smooth_labels(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Apply label smoothing: hard labels → soft labels.
        e.g., 1.0 → 0.9, 0.0 → 0.1/(num_classes-1)
        """
        smoothed = targets * (1 - self.label_smoothing)
        smoothed = smoothed + self.label_smoothing / num_classes
        return smoothed

    def training_step(
        self,
        images: torch.Tensor,
        targets: dict,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        num_classes: int = 80,
    ) -> dict:
        """
        Execute one adversarial training step.
        
        Returns:
            dict with loss components
        """
        # Decide whether this batch gets adversarial treatment
        use_adversarial = torch.rand(1).item() < self.adversarial_ratio

        # Label smoothing
        if 'classes' in targets:
            targets['classes'] = self.smooth_labels(targets['classes'], num_classes)

        if use_adversarial:
            # Generate adversarial examples using the main loss function
            def _loss_fn(outputs, tgt):
                return loss_fn(outputs, tgt)['total']

            adv_images = self.generate_adversarial(images, _loss_fn, targets)

            # Forward on adversarial
            self.model.train()
            outputs = self.model(adv_images)
            losses = loss_fn(outputs, targets)

            # Add gradient regularization on clean images
            grad_reg = self.compute_gradient_regularization(images, _loss_fn, targets)
            total_loss = losses['total'] + grad_reg
        else:
            # Clean forward pass
            self.model.train()
            outputs = self.model(images)
            losses = loss_fn(outputs, targets)
            total_loss = losses['total']
            grad_reg = torch.tensor(0.0)

        # Backward + step
        optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        optimizer.step()

        return {
            **losses,
            'grad_reg': grad_reg.item(),
            'adversarial': use_adversarial,
            'total_with_reg': total_loss.item(),
        }
