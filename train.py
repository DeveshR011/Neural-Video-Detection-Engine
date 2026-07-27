"""
SecureYOLO Training
===================
Training entry point for the anchor-free detector.

    py -3.14 train.py --dataset voc  --data-root ./VOCdevkit --epochs 60
    py -3.14 train.py --dataset coco --data-root ./coco       --epochs 40

Implements the pieces that matter for accuracy under a short compute budget:
EMA weight averaging, bf16 autocast, gradient accumulation to reach an
effective batch size larger than 6 GB of VRAM allows, cosine decay with
warmup, and mosaic disabled for the final epochs.

Every run writes to ``runs/<name>/`` with checkpoints, a metrics CSV, and the
resolved configuration, so a result can be traced back to what produced it.
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.build import build_dataloaders
from evaluation.coco_eval import evaluate, format_metrics
from model.detection_loss import DetectionLoss
from model.detector import SecureYOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-10s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Train")


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class ModelEMA:
    """Exponential moving average of model weights.

    The averaged weights are consistently better than the final weights,
    typically by 1-2 mAP, because they suppress the noise of late-training SGD
    steps. The decay ramps in from a low value so the average is not anchored
    to the random initialization.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(self._bare(model)).eval()
        self.updates = updates
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _bare(model):
        return model.module if hasattr(model, "module") else model

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay_fn(self.updates)
        msd = self._bare(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()
            else:
                # Integer buffers (e.g. num_batches_tracked) are copied, not averaged.
                v.copy_(msd[k])


def build_optimizer(model, name="auto", lr=0.001, momentum=0.937, weight_decay=0.0005):
    """Build an optimizer with decay applied only to convolution/linear weights.

    Applying weight decay to BatchNorm parameters and biases measurably hurts
    detection accuracy, so they are placed in separate parameter groups.
    """
    decay, no_decay = [], []
    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param_name == "bias" or isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                no_decay.append(param)
            else:
                decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if name == "auto":
        name = "adamw"
    name = name.lower()

    if name == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=(momentum, 0.999))
    elif name == "sgd":
        optimizer = torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer {name!r}")

    logger.info("Optimizer %s | lr %g | decay params %d | no-decay params %d",
                name, lr, len(decay), len(no_decay))
    return optimizer


def cosine_schedule(epoch, epochs, lr_final_ratio=0.01, warmup_epochs=3.0):
    """Warmup followed by cosine decay, returned as a multiplier on base lr."""
    if epoch < warmup_epochs:
        return (epoch + 1) / (warmup_epochs + 1e-8)
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)
    return lr_final_ratio + (1 - lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, ema, optimizer, epoch, best_map, args, class_names):
    payload = {
        "epoch": epoch,
        "best_map": best_map,
        "model": model.state_dict(),
        "ema": ema.ema.state_dict() if ema else None,
        "ema_updates": ema.updates if ema else 0,
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "class_names": class_names,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch,
                    args, ema=None, amp_dtype=None, accumulate=1):
    model.train()
    n_batches = len(loader)
    totals = {"box": 0.0, "cls": 0.0, "dfl": 0.0, "total": 0.0}
    seen = 0
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    for bi, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                preds = model(images)
                loss, components = loss_fn(preds, targets,
                                           imgsz=images.shape[-2:])
        else:
            preds = model(images)
            loss, components = loss_fn(preds, targets, imgsz=images.shape[-2:])

        # Scale so gradient magnitude is independent of the accumulation count.
        scaled = loss / accumulate

        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (bi + 1) % accumulate == 0 or (bi + 1) == n_batches:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        for k in totals:
            totals[k] += components[k]
        seen += 1

        if args.log_interval and (bi + 1) % args.log_interval == 0:
            elapsed = time.perf_counter() - start
            ips = ((bi + 1) * images.shape[0]) / elapsed
            mem = (torch.cuda.max_memory_allocated() / 1e9
                   if device.type == "cuda" else 0.0)
            logger.info(
                "  e%-3d %5d/%-5d loss %7.3f (box %.3f cls %.3f dfl %.3f) "
                "pos %4d | %5.1f img/s | %.2f GB",
                epoch, bi + 1, n_batches, totals["total"] / seen,
                totals["box"] / seen, totals["cls"] / seen, totals["dfl"] / seen,
                components["n_positives"], ips, mem,
            )

    elapsed = time.perf_counter() - start
    return {k: v / max(seen, 1) for k, v in totals.items()}, elapsed


def main():
    parser = argparse.ArgumentParser(description="Train SecureYOLO")

    # Data
    parser.add_argument("--dataset", default="voc", choices=["voc", "coco", "coco-minitrain"])
    parser.add_argument("--data-root", default="./VOCdevkit")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)

    # Model
    parser.add_argument("--backbone", default="builtin",
                        help="'builtin' or a timm model name, e.g. cspdarknet53, convnext_tiny")
    parser.add_argument("--size", default="s", choices=["n", "s", "m", "l"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-attention", action="store_true",
                        help="disable the P5 attention block")
    parser.add_argument("--secure-features", action="store_true",
                        help="enable the random projection layer (reduces mAP)")
    parser.add_argument("--reg-max", type=int, default=16)

    # Optimization
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=8,
                        help="gradient accumulation steps; effective batch = batch * accumulate")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-final-ratio", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="auto", choices=["auto", "adamw", "sgd"])
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    parser.add_argument("--no-ema", action="store_true")

    # Augmentation
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="disable mosaic for the final N epochs")

    # Run control
    parser.add_argument("--name", default="exp")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--resume", default="", help="path to a checkpoint")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--time-budget", type=float, default=0.0,
                        help="stop cleanly after N hours (0 = unlimited)")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    amp_dtype = None
    scaler = None
    if args.amp != "off" and device.type == "cuda":
        if args.amp == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")

    logger.info("Device %s | AMP %s | run dir %s", device,
                amp_dtype if amp_dtype else "off", run_dir)

    # ── Data ──────────────────────────────────────────────────────────
    bundle = build_dataloaders(
        dataset=args.dataset, data_root=args.data_root, img_size=args.imgsz,
        batch_size=args.batch, workers=args.workers, augment=True,
        mosaic_prob=args.mosaic, mixup_prob=args.mixup, seed=args.seed,
    )
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    num_classes = bundle["num_classes"]
    class_names = bundle["class_names"]

    # ── Model ─────────────────────────────────────────────────────────
    model = SecureYOLO(
        num_classes=num_classes, backbone=args.backbone,
        pretrained=not args.no_pretrained, size=args.size,
        reg_max=args.reg_max, attention=not args.no_attention,
        secure_features=args.secure_features,
    ).to(device)
    logger.info(model.summary())

    loss_fn = DetectionLoss(num_classes=num_classes, reg_max=args.reg_max).to(device)
    optimizer = build_optimizer(model, args.optimizer, lr=args.lr,
                                weight_decay=args.weight_decay)
    ema = None if args.no_ema else ModelEMA(model)

    start_epoch, best_map = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ema and ckpt.get("ema"):
            ema.ema.load_state_dict(ckpt["ema"])
            ema.updates = ckpt.get("ema_updates", 0)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", 0.0)
        logger.info("Resumed from %s at epoch %d (best mAP %.4f)",
                    args.resume, start_epoch, best_map)

    effective_batch = args.batch * args.accumulate
    logger.info("Batch %d x accumulate %d = effective batch %d | %d train batches/epoch",
                args.batch, args.accumulate, effective_batch, len(train_loader))

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "num_classes": num_classes,
                   "class_names": class_names}, f, indent=2)

    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "lr", "loss", "box", "cls", "dfl",
                 "mAP50-95", "mAP50", "mAP75", "epoch_seconds"])

    # ── Training ──────────────────────────────────────────────────────
    mosaic_closed = False
    train_start = time.perf_counter()
    stop_reason = "completed"

    for epoch in range(start_epoch, args.epochs):
        if (not mosaic_closed and args.close_mosaic > 0
                and epoch >= args.epochs - args.close_mosaic):
            logger.info("Closing mosaic/mixup for the final %d epochs", args.close_mosaic)
            bundle["train_dataset"].close_mosaic()
            mosaic_closed = True

        lr_scale = cosine_schedule(epoch, args.epochs, args.lr_final_ratio,
                                   args.warmup_epochs)
        current_lr = args.lr * lr_scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        logger.info("Epoch %d/%d | lr %.2e", epoch + 1, args.epochs, current_lr)
        losses, elapsed = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch + 1,
            args, ema=ema, amp_dtype=amp_dtype, accumulate=args.accumulate,
        )
        logger.info("Epoch %d done in %.1fs | loss %.4f (box %.3f cls %.3f dfl %.3f)",
                    epoch + 1, elapsed, losses["total"], losses["box"],
                    losses["cls"], losses["dfl"])

        metrics = None
        is_last = (epoch + 1) == args.epochs
        if args.eval_interval and ((epoch + 1) % args.eval_interval == 0 or is_last):
            eval_model = ema.ema if ema else model
            metrics = evaluate(
                eval_model, val_loader, bundle["gt_json"], bundle["id_map"],
                device, class_names=class_names, amp_dtype=amp_dtype,
            )
            logger.info("\n%s", format_metrics(metrics, f"Epoch {epoch + 1} validation"))

            if metrics["mAP50-95"] > best_map:
                best_map = metrics["mAP50-95"]
                save_checkpoint(run_dir / "best.pt", model, ema, optimizer,
                                epoch, best_map, args, class_names)
                logger.info("New best mAP@[.5:.95] %.4f -> best.pt", best_map)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{current_lr:.6f}", f"{losses['total']:.4f}",
                f"{losses['box']:.4f}", f"{losses['cls']:.4f}", f"{losses['dfl']:.4f}",
                f"{metrics['mAP50-95']:.4f}" if metrics else "",
                f"{metrics['mAP50']:.4f}" if metrics else "",
                f"{metrics['mAP75']:.4f}" if metrics else "",
                f"{elapsed:.1f}",
            ])

        save_checkpoint(run_dir / "last.pt", model, ema, optimizer, epoch,
                        best_map, args, class_names)
        if args.save_interval and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(run_dir / f"epoch_{epoch + 1}.pt", model, ema,
                            optimizer, epoch, best_map, args, class_names)

        if args.time_budget > 0:
            hours = (time.perf_counter() - train_start) / 3600
            if hours >= args.time_budget:
                logger.warning("Time budget of %.2f h reached at epoch %d; stopping.",
                               args.time_budget, epoch + 1)
                stop_reason = "time_budget"
                break

    total_hours = (time.perf_counter() - train_start) / 3600
    logger.info("Training %s after %.2f h | best mAP@[.5:.95] %.4f",
                stop_reason, total_hours, best_map)

    # ── Final evaluation with per-class breakdown ──────────────────────
    best_path = run_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        eval_model = SecureYOLO(
            num_classes=num_classes, backbone=args.backbone, pretrained=False,
            size=args.size, reg_max=args.reg_max,
            attention=not args.no_attention, secure_features=args.secure_features,
        ).to(device)
        eval_model.load_state_dict(ckpt["ema"] if ckpt.get("ema") else ckpt["model"])

        metrics = evaluate(
            eval_model, val_loader, bundle["gt_json"], bundle["id_map"], device,
            class_names=class_names, amp_dtype=amp_dtype, per_class=True,
            save_json=str(run_dir / "predictions.json"),
        )
        logger.info("\n%s", format_metrics(metrics, "FINAL (best checkpoint, EMA)"))

        metrics["training_hours"] = total_hours
        metrics["stop_reason"] = stop_reason
        with open(run_dir / "final_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote %s", run_dir / "final_metrics.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
