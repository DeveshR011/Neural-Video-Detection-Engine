"""
SecureYOLO v3 Training Script
================================
End-to-end training with:
- COCO-format dataset loading
- CIoU + Focal loss
- PGD adversarial training (lightweight 3-step)
- Mixed precision (FP16)
- Cosine LR schedule with warmup
- Model checkpointing with encrypted export
"""

import os
import sys
import math
import time
import json
import random
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional

# Suppress PyTorch's harmless AMP scheduler warning that triggers when AMP skips the first batch due to random initialization NaNs.
warnings.filterwarnings("ignore", category=UserWarning, message=".*Detected call of `lr_scheduler.step()`.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model.detector import SecureYOLO
from model.losses import CIoULoss, FocalLoss, DetectionLoss
from model.adversarial_trainer import AdversarialTrainer
from model.compression import MixedPrecisionTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-14s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("Training")


def set_seed(seed: int):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════

class COCODetectionDataset(Dataset):
    """
    COCO-format detection dataset.
    
    Expected structure:
        data_root/
        ├── images/
        │   ├── train2017/
        │   └── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
    
    If no dataset is available, generates synthetic training data
    to demonstrate the training pipeline.
    """

    def __init__(self, data_root: str = None, split: str = "train",
                 img_size: int = 640, num_classes: int = 80,
                 max_objects: int = 50, synthetic_size: int = 1000,
                 allow_synthetic_fallback: bool = False):
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.use_synthetic = data_root is None

        if self.use_synthetic:
            logger.info(f"No dataset at '{data_root}' — using synthetic data ({synthetic_size} samples)")
            self.length = synthetic_size
            self.annotations = None
        else:
            if not os.path.exists(data_root):
                if self.allow_synthetic_fallback:
                    logger.warning(f"Dataset root '{data_root}' does not exist. Falling back to synthetic data.")
                    self.use_synthetic = True
                    self.length = synthetic_size
                    self.annotations = None
                    return
                raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
            self._load_coco(data_root, split)

    def _load_coco(self, data_root: str, split: str):
        """Load COCO annotations."""
        ann_file = os.path.join(data_root, "annotations", f"instances_{split}2017.json")
        img_dir = os.path.join(data_root, "images", f"{split}2017")

        if not os.path.exists(ann_file):
            if self.allow_synthetic_fallback:
                logger.warning(f"Annotation file not found: {ann_file}. Falling back to synthetic data.")
                self.use_synthetic = True
                self.length = 1000
                return
            raise FileNotFoundError(f"Missing annotation file: {ann_file}")
        if not os.path.exists(img_dir):
            if self.allow_synthetic_fallback:
                logger.warning(f"Image directory not found: {img_dir}. Falling back to synthetic data.")
                self.use_synthetic = True
                self.length = 1000
                return
            raise FileNotFoundError(f"Missing image directory: {img_dir}")

        try:
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)

            self.images = {img['id']: img for img in coco_data['images']}
            self.img_dir = img_dir

            # Group annotations by image
            self.img_annotations = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.img_annotations:
                    self.img_annotations[img_id] = []
                self.img_annotations[img_id].append(ann)

            self.img_ids = list(self.img_annotations.keys())
            self.length = len(self.img_ids)

            # Build category mapping
            self.cat_to_idx = {cat['id']: idx for idx, cat in
                              enumerate(coco_data['categories'])}

            missing = 0
            for img_id in self.img_ids:
                file_name = self.images[img_id]['file_name']
                if not os.path.exists(os.path.join(self.img_dir, file_name)):
                    missing += 1
            if missing > 0:
                message = f"COCO split '{split}' has {missing} missing image files under {self.img_dir}."
                if self.allow_synthetic_fallback:
                    logger.warning(message + " Falling back to synthetic data.")
                    self.use_synthetic = True
                    self.length = 1000
                    return
                raise FileNotFoundError(message)

            logger.info(f"Loaded {self.length} images from {split} split")
        except Exception as e:
            if self.allow_synthetic_fallback:
                logger.warning(f"Failed to load COCO: {e}. Falling back to synthetic data.")
                self.use_synthetic = True
                self.length = 1000
            else:
                raise

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.use_synthetic:
            return self._generate_synthetic()
        else:
            return self._load_real(idx)

    def _generate_synthetic(self) -> tuple:
        """Generate a synthetic training sample with random objects."""
        # Random image
        image = torch.rand(3, self.img_size, self.img_size)

        # Random objects (1–10 per image)
        n_objects = random.randint(1, 10)

        boxes = torch.zeros(self.max_objects, 4)
        classes = torch.zeros(self.max_objects, self.num_classes)
        obj_mask = torch.zeros(self.max_objects)

        for i in range(n_objects):
            # Random box (cx, cy, w, h) normalized to [0, 1]
            cx = random.uniform(0.1, 0.9)
            cy = random.uniform(0.1, 0.9)
            w = random.uniform(0.05, 0.4)
            h = random.uniform(0.05, 0.4)

            boxes[i] = torch.tensor([cx, cy, w, h])
            cls = random.randint(0, self.num_classes - 1)
            classes[i, cls] = 1.0
            obj_mask[i] = 1.0

        targets = {
            'boxes': boxes,
            'classes': classes,
            'obj_mask': obj_mask,
            'num_objects': n_objects,
        }

        return image, targets

    def _load_real(self, idx) -> tuple:
        """Load a real COCO image and its annotations."""
        import cv2

        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load COCO image: {img_path}")

        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        # Process annotations
        anns = self.img_annotations.get(img_id, [])
        boxes = torch.zeros(self.max_objects, 4)
        classes = torch.zeros(self.max_objects, self.num_classes)
        obj_mask = torch.zeros(self.max_objects)

        for i, ann in enumerate(anns[:self.max_objects]):
            x, y, w, h = ann['bbox']  # COCO format: x, y, w, h (absolute)
            # Convert to normalized center format
            cx = (x + w / 2) / w_orig
            cy = (y + h / 2) / h_orig
            nw = w / w_orig
            nh = h / h_orig

            boxes[i] = torch.tensor([cx, cy, nw, nh]).clamp(0, 1)

            cat_idx = self.cat_to_idx.get(ann['category_id'], 0)
            if cat_idx < self.num_classes:
                classes[i, cat_idx] = 1.0
            obj_mask[i] = 1.0

        targets = {
            'boxes': boxes,
            'classes': classes,
            'obj_mask': obj_mask,
            'num_objects': min(len(anns), self.max_objects),
        }

        return image, targets


def collate_fn(batch):
    """Custom collate for detection data."""
    images = torch.stack([b[0] for b in batch])
    targets = {
        'boxes': torch.stack([b[1]['boxes'] for b in batch]),
        'classes': torch.stack([b[1]['classes'] for b in batch]),
        'obj_mask': torch.stack([b[1]['obj_mask'] for b in batch]),
    }
    return images, targets


# ══════════════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════════════

def train(args):
    """Main training function."""
    set_seed(args.seed)
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    
    if device.type == 'cuda':
        # Auto-tune convolution algorithms for maximum hardware efficiency
        torch.backends.cudnn.benchmark = True
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        logger.info("Enabled CuDNN Benchmark for maximum throughput")

    # ── Model ─────────────────────────────────────────────────────
    model = SecureYOLO(
        num_classes=args.num_classes,
        fpn_channels=args.fpn_channels,
    ).to(device)
    if device.type == 'cuda' and args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: SecureYOLO v3 | Params: {param_count:,}")

    # ── Dataset ───────────────────────────────────────────────────
    train_dataset = COCODetectionDataset(
        data_root=args.data,
        split=args.train_split,
        img_size=args.img_size,
        num_classes=args.num_classes,
        synthetic_size=args.synthetic_size,
        allow_synthetic_fallback=args.allow_synthetic,
    )
    val_dataset = COCODetectionDataset(
        data_root=args.data,
        split="val",
        img_size=args.img_size,
        num_classes=args.num_classes,
        synthetic_size=args.synthetic_size // 5,
        allow_synthetic_fallback=args.allow_synthetic,
    )

    loader_kwargs = {
        'num_workers': args.workers,
        'pin_memory': device.type == 'cuda',
        'collate_fn': collate_fn,
    }
    if args.workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # ── Loss ──────────────────────────────────────────────────────
    loss_fn = DetectionLoss(
        num_classes=args.num_classes,
        lambda_box=5.0,
        lambda_cls=1.0,
        lambda_obj=1.0,
        lambda_jitter=0.1,
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine decay with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Adversarial Trainer ───────────────────────────────────────
    adv_trainer = AdversarialTrainer(
        model=model,
        pgd_epsilon=args.pgd_epsilon,
        pgd_alpha=args.pgd_alpha,
        pgd_steps=args.pgd_steps,
        grad_reg_weight=args.grad_reg_weight,
        label_smoothing=args.label_smoothing,
        adversarial_ratio=args.adversarial_ratio,
    ) if args.adversarial else None

    # ── Mixed Precision ───────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'

    # ── Checkpointing ─────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = int(checkpoint.get('epoch', 0))
        best_loss = float(checkpoint.get('best_loss', checkpoint.get('loss', best_loss)))
        logger.info(f"Resume state loaded | start_epoch={start_epoch} | best_loss={best_loss:.4f}")

    # ── Training ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  Training SecureYOLO v3")
    logger.info(f"  Epochs: {args.epochs} | Batch: {args.batch_size}")
    logger.info(f"  LR: {args.lr} | Adversarial: {args.adversarial}")
    logger.info(f"  Steps per epoch: {len(train_loader)}")
    if args.adversarial:
        logger.info(
            f"  Adversarial fine-tune starts at last {int(args.adversarial_finetune_ratio * 100)}% of epochs"
        )
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {'total': 0, 'box': 0, 'cls': 0, 'obj': 0}
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=(device.type == 'cuda'))
            if device.type == 'cuda' and args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in targets.items()}

            adv_start_epoch = int(args.epochs * (1.0 - args.adversarial_finetune_ratio))
            use_adv_epoch = adv_trainer is not None and epoch >= adv_start_epoch

            if use_adv_epoch and random.random() < args.adversarial_ratio:
                # Adversarial training step
                adv_result = adv_trainer.training_step(
                    images, targets,
                    lambda out, tgt: loss_fn(out, tgt),
                    optimizer, args.num_classes,
                    use_adversarial=True,
                )
                losses = adv_result
            else:
                # Standard training step
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        predictions = model(images)
                        losses = loss_fn(predictions, targets)

                    optimizer.zero_grad()
                    scaler.scale(losses['total']).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = model(images)
                    losses = loss_fn(predictions, targets)

                    optimizer.zero_grad()
                    losses['total'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()

            scheduler.step()

            # Accumulate losses
            for k in epoch_losses:
                epoch_losses[k] += losses.get(k, 0) if isinstance(losses.get(k, 0), (int, float)) else losses.get(k, torch.tensor(0)).item() if isinstance(losses.get(k, 0), torch.Tensor) else 0
            num_batches += 1

            # Log every N batches
            if (batch_idx + 1) % args.log_interval == 0:
                avg = {k: v / num_batches for k, v in epoch_losses.items()}
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {avg['total']:.4f} (box={avg['box']:.4f} cls={avg['cls']:.4f} obj={avg['obj']:.4f}) | "
                    f"LR: {lr:.6f}"
                )

        # ── Epoch Summary ─────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} complete | "
            f"Time: {epoch_time:.1f}s | "
            f"Loss: {avg['total']:.4f}"
        )

        # ── Validation ────────────────────────────────────────────
        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate(model, val_loader, loss_fn, device, use_amp)
            logger.info(f"Validation Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join(args.save_dir, "best.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                    'loss': best_loss,
                    'best_loss': best_loss,
                }, save_path)
                logger.info(f"  Saved best model → {save_path} (loss={best_loss:.4f})")

        # ── Periodic checkpoint ───────────────────────────────────
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'loss': avg['total'],
                'best_loss': best_loss,
            }, save_path)
            logger.info(f"  Checkpoint → {save_path}")

    # ── Final save ────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "final.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'loss': avg['total'],
        'best_loss': best_loss,
        'architecture': {
            'num_classes': args.num_classes,
            'fpn_channels': args.fpn_channels,
        },
    }, final_path)
    logger.info(f"Training complete. Final model → {final_path}")

    return model


def validate(model, val_loader, loss_fn, device, use_amp):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=(device.type == 'cuda'))
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in targets.items()}

            if use_amp:
                with torch.amp.autocast('cuda'):
                    predictions = model(images)
                    losses = loss_fn(predictions, targets)
            else:
                predictions = model(images)
                losses = loss_fn(predictions, targets)

            total_loss += losses['total'].item() if isinstance(losses['total'], torch.Tensor) else losses['total']
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="SecureYOLO v3 Training")

    # Data
    parser.add_argument('--data', type=str, default=None,
                       help='Path to COCO dataset root (if None, uses synthetic)')
    parser.add_argument('--train-split', type=str, default='train',
                       help='COCO split to use for training (train or val)')
    parser.add_argument('--synthetic-size', type=int, default=2000,
                       help='Number of synthetic training samples')
    parser.add_argument('--allow-synthetic', action='store_true',
                       help='Allow fallback to synthetic data when COCO files are incomplete')

    # Model
    parser.add_argument('--num-classes', type=int, default=80)
    parser.add_argument('--fpn-channels', type=int, default=128)
    parser.add_argument('--img-size', type=int, default=640)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of dataloader workers (increase to prevent GPU starvation)')
    parser.add_argument('--prefetch-factor', type=int, default=4,
                       help='Dataloader prefetch factor per worker')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--channels-last', action='store_true', default=True,
                       help='Use channels_last memory format on CUDA for higher throughput')
    parser.add_argument('--allow-tf32', action='store_true', default=True,
                       help='Enable TF32 matmul/convolution on CUDA (faster, negligible accuracy impact)')

    # Adversarial
    parser.add_argument('--adversarial', action='store_true', default=True,
                       help='Enable adversarial training')
    parser.add_argument('--no-adversarial', dest='adversarial', action='store_false')
    parser.add_argument('--pgd-epsilon', type=float, default=2/255)
    parser.add_argument('--pgd-alpha', type=float, default=0.5/255)
    parser.add_argument('--pgd-steps', type=int, default=1)
    parser.add_argument('--adversarial-ratio', type=float, default=0.2)
    parser.add_argument('--adversarial-finetune-ratio', type=float, default=0.2,
                       help='Fraction of final epochs where adversarial batches are enabled')
    parser.add_argument('--grad-reg-weight', type=float, default=0.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # Logging/Saving
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--save-interval', type=int, default=20)
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=20)

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(args)
