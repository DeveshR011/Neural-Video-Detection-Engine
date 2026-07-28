"""
SecureYOLO Evaluation
=====================
Score a trained checkpoint against a validation split.

    py -3.14 val.py --weights runs/voc120/best.pt --dataset voc --data-root ./VOCdevkit
    py -3.14 val.py --weights runs/voc120/last.pt --per-class

Model configuration is read from the checkpoint, so the architecture flags
used at training time do not need to be repeated. By default the EMA weights
are scored, since those are what training selects on.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from data.build import build_dataloaders
from evaluation.coco_eval import evaluate, format_metrics
from evaluation.tta import build_tta
from model.detector import SecureYOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-10s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Val")


def load_model(weights, device, use_ema=True, num_classes=None):
    """Rebuild the model from the configuration stored in the checkpoint."""
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    class_names = ckpt.get("class_names", [])
    num_classes = num_classes or len(class_names) or saved.get("num_classes", 80)

    model = SecureYOLO(
        num_classes=num_classes,
        backbone=saved.get("backbone", "builtin"),
        pretrained=False,
        size=saved.get("size", "s"),
        reg_max=saved.get("reg_max", 16),
        attention=not saved.get("no_attention", False),
        secure_features=saved.get("secure_features", False),
    ).to(device)

    state = ckpt.get("ema") if (use_ema and ckpt.get("ema")) else ckpt["model"]
    source = "EMA" if (use_ema and ckpt.get("ema")) else "raw"
    model.load_state_dict(state)
    model.eval()

    logger.info("Loaded %s (%s weights) from epoch %s | best mAP recorded %.4f",
                weights, source, ckpt.get("epoch", "?"), ckpt.get("best_map", 0.0))
    return model, ckpt, class_names


def main():
    parser = argparse.ArgumentParser(description="Evaluate a SecureYOLO checkpoint")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="voc", choices=["voc", "coco", "coco-minitrain"])
    parser.add_argument("--data-root", default="./VOCdevkit")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="defaults to the training resolution in the checkpoint")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="low by default: mAP rewards recall in the score tail")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--per-class", action="store_true")
    parser.add_argument("--tta", default="off", choices=["off", "flip", "scale", "full"],
                        help="test-time augmentation preset (costs extra forward passes)")
    parser.add_argument("--tta-iou", type=float, default=0.65,
                        help="IoU threshold used when merging TTA views")
    parser.add_argument("--raw-weights", action="store_true",
                        help="score the raw weights instead of the EMA average")
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-json", default="")
    parser.add_argument("--out", default="", help="write metrics to this JSON path")
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )

    amp_dtype = None
    if args.amp != "off" and device.type == "cuda":
        amp_dtype = (torch.bfloat16
                     if args.amp == "bf16" and torch.cuda.is_bf16_supported()
                     else torch.float16)

    ckpt_peek = torch.load(args.weights, map_location="cpu", weights_only=False)
    imgsz = args.imgsz or ckpt_peek.get("args", {}).get("imgsz", 640)
    del ckpt_peek

    bundle = build_dataloaders(
        dataset=args.dataset, data_root=args.data_root, img_size=imgsz,
        batch_size=args.batch, workers=args.workers, augment=False,
        val_batch_size=args.batch,
    )

    model, ckpt, _ = load_model(args.weights, device,
                                use_ema=not args.raw_weights,
                                num_classes=bundle["num_classes"])

    logger.info("Evaluating at %dpx | conf %.4f | NMS IoU %.2f | max_det %d | TTA %s",
                imgsz, args.conf, args.iou, args.max_det, args.tta)

    eval_target = build_tta(model, mode=args.tta, merge_iou=args.tta_iou)

    metrics = evaluate(
        eval_target, bundle["val_loader"], bundle["gt_json"], bundle["id_map"], device,
        class_names=bundle["class_names"], conf_threshold=args.conf,
        iou_threshold=args.iou, max_detections=args.max_det,
        amp_dtype=amp_dtype, per_class=args.per_class,
        save_json=args.save_json or None,
    )

    title = f"{args.dataset} | {Path(args.weights).name} | epoch {ckpt.get('epoch', '?')}"
    logger.info("\n%s", format_metrics(metrics, title))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote %s", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
