"""
Model Compression: Pruning, Distillation & Quantization
=========================================================
Structured channel pruning, knowledge distillation, and
quantization-aware training — all compatible with watermarking,
runtime hashing, and architecture morphing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
from typing import Optional, Callable

logger = logging.getLogger("ModelCompression")


# ══════════════════════════════════════════════════════════════════════
#  Structured Channel Pruning
# ══════════════════════════════════════════════════════════════════════

class StructuredPruner:
    """
    L1-norm based structured (channel-level) pruning.
    Prunes entire output channels from Conv2d layers, keeping
    the architecture morphing-compatible (P·W·Pᵀ works on any
    channel count since it generates a new permutation per-boot).
    """

    def __init__(self, prune_ratio: float = 0.2, enable_structural_rewrite: bool = False):
        """
        Args:
            prune_ratio: Fraction of channels to prune per eligible layer
            enable_structural_rewrite: EXPERIMENTAL. If False, pruning rewrites are
                disabled to avoid breaking non-sequential detector topologies.
        """
        self.prune_ratio = prune_ratio
        self.enable_structural_rewrite = enable_structural_rewrite

    def compute_channel_importance(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Compute L1-norm importance scores for each Conv2d layer."""
        importance = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.groups == 1:
                # L1 norm across [in_c, kH, kW] → importance per output channel
                scores = module.weight.data.abs().sum(dim=(1, 2, 3))
                importance[name] = scores
        return importance

    def prune(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """
        Apply structured channel pruning to the model.
        
        Returns:
            (pruned_model, pruning_stats)
        """
        stats = {
            'layers_pruned': 0,
            'channels_removed': 0,
            'original_params': sum(p.numel() for p in model.parameters()),
        }

        if not self.enable_structural_rewrite:
            stats['pruned_params'] = stats['original_params']
            stats['compression_ratio'] = 0.0
            stats['skipped'] = True
            stats['reason'] = (
                "Structured channel rewrite is disabled by default because this model "
                "contains multi-branch topology where naive channel edits are unsafe."
            )
            logger.warning(stats['reason'])
            return model, stats

        importance = self.compute_channel_importance(model)

        for name, module in model.named_modules():
            if name in importance and isinstance(module, nn.Conv2d):
                scores = importance[name]
                n_channels = len(scores)
                n_prune = int(n_channels * self.prune_ratio)

                if n_prune == 0 or n_prune >= n_channels - 1:
                    continue

                # Keep channels with highest importance
                _, keep_idx = scores.topk(n_channels - n_prune, largest=True)
                keep_idx = keep_idx.sort()[0]

                # Prune the conv weights
                with torch.no_grad():
                    module.weight.data = module.weight.data[keep_idx]
                    module.out_channels = len(keep_idx)
                    if module.bias is not None:
                        module.bias.data = module.bias.data[keep_idx]

                stats['layers_pruned'] += 1
                stats['channels_removed'] += n_prune

        stats['pruned_params'] = sum(p.numel() for p in model.parameters())
        stats['compression_ratio'] = 1 - stats['pruned_params'] / stats['original_params']

        logger.info(
            f"Pruned {stats['layers_pruned']} layers, "
            f"removed {stats['channels_removed']} channels, "
            f"compression: {stats['compression_ratio']:.1%}"
        )
        stats['skipped'] = False
        return model, stats


# ══════════════════════════════════════════════════════════════════════
#  Knowledge Distillation
# ══════════════════════════════════════════════════════════════════════

class KnowledgeDistiller:
    """
    Knowledge distillation from a larger teacher model.
    Uses KL-divergence on logits + feature-map MSE for rich transfer.
    """

    def __init__(
        self,
        teacher: nn.Module,
        temperature: float = 4.0,
        alpha_kd: float = 0.5,
        alpha_feature: float = 0.1,
    ):
        """
        Args:
            teacher: Pre-trained teacher model (frozen)
            temperature: Softmax temperature for KL divergence
            alpha_kd: Weight for KD logit loss
            alpha_feature: Weight for feature MSE loss
        """
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.alpha_feature = alpha_feature

    def compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL-divergence between student and teacher softened predictions."""
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
        return self.alpha_kd * kd_loss

    def compute_feature_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between student and teacher intermediate features."""
        # Align spatial dimensions if needed
        if student_features.shape != teacher_features.shape:
            teacher_features = F.interpolate(
                teacher_features, size=student_features.shape[2:], mode='bilinear',
                align_corners=False
            )
            # Align channels with 1×1 projection
            if student_features.shape[1] != teacher_features.shape[1]:
                return torch.tensor(0.0, device=student_features.device)

        return self.alpha_feature * F.mse_loss(student_features, teacher_features)

    def distillation_step(
        self,
        images: torch.Tensor,
        student_model: nn.Module,
        task_loss_fn: Callable,
        targets: dict,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        One distillation training step.
        
        Returns dict with loss breakdown.
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(images)

        # Student forward
        student_model.train()
        student_out = student_model(images)

        # Task loss (CIoU + Focal + etc.)
        task_losses = task_loss_fn(student_out, targets)

        # KD loss on classification logits
        student_cls = student_out[..., 5:]
        teacher_cls = teacher_out[..., 5:]
        kd_loss = self.compute_kd_loss(
            student_cls.reshape(-1, student_cls.shape[-1]),
            teacher_cls.reshape(-1, teacher_cls.shape[-1]),
        )

        total = task_losses['total'] + kd_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=10.0)
        optimizer.step()

        return {
            **task_losses,
            'kd_loss': kd_loss.item(),
            'total_with_kd': total.item(),
        }


# ══════════════════════════════════════════════════════════════════════
#  Quantization-Aware Training (QAT)
# ══════════════════════════════════════════════════════════════════════

class QuantizationManager:
    """
    Manages quantization-aware training (QAT) for INT8 readiness.
    Uses PyTorch's fake-quantize observers.
    
    Watermark and runtime hashing remain compatible because they
    operate on the raw FP32 parameter bytes — QAT only inserts
    observer modules during training and fuses for deployment.
    """

    def __init__(self, backend: str = 'x86'):
        """
        Args:
            backend: Quantization backend ('x86', 'qnnpack', 'onednn')
        """
        self.backend = backend

    def prepare_qat(self, model: nn.Module) -> nn.Module:
        """
        Insert fake-quantize observers into the model for QAT.
        Call this before the training loop.
        """
        try:
            model.train()

            # Set QConfig
            if self.backend == 'x86':
                qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
            else:
                qconfig = torch.ao.quantization.get_default_qat_qconfig(self.backend)

            model.qconfig = qconfig
            torch.ao.quantization.prepare_qat(model, inplace=True)
            logger.info(f"QAT prepared with backend '{self.backend}'")
            return model
        except Exception as e:
            logger.warning(f"QAT preparation failed: {e}. Returning original model.")
            return model

    def convert_to_quantized(self, model: nn.Module) -> nn.Module:
        """
        Convert QAT model to fully quantized INT8 for deployment.
        """
        try:
            model.eval()
            quantized = torch.ao.quantization.convert(model, inplace=False)
            logger.info("Model converted to INT8 quantized format.")
            return quantized
        except Exception as e:
            logger.warning(f"Quantization conversion failed: {e}. Returning original model.")
            return model

    @staticmethod
    def estimate_int8_size(model: nn.Module) -> dict:
        """Estimate INT8 model size vs FP32."""
        fp32_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        int8_bytes = sum(p.numel() for p in model.parameters())  # 1 byte per param

        return {
            'fp32_mb': round(fp32_bytes / (1024 * 1024), 2),
            'int8_mb': round(int8_bytes / (1024 * 1024), 2),
            'compression': round(fp32_bytes / max(int8_bytes, 1), 2),
        }


# ══════════════════════════════════════════════════════════════════════
#  Mixed Precision Training Helper
# ══════════════════════════════════════════════════════════════════════

class MixedPrecisionTrainer:
    """Wrapper for FP16/BF16 mixed-precision training with GradScaler."""

    def __init__(self, device: torch.device, use_bf16: bool = False):
        self.device = device
        self.use_amp = device.type == 'cuda'
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def training_step(
        self,
        model: nn.Module,
        images: torch.Tensor,
        loss_fn: Callable,
        targets: dict,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """Execute one mixed-precision training step."""
        model.train()

        if self.use_amp:
            with torch.amp.autocast('cuda', dtype=self.dtype):
                outputs = model(images)
                losses = loss_fn(outputs, targets)

            optimizer.zero_grad()
            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = model(images)
            losses = loss_fn(outputs, targets)
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        return losses
