"""
COCO mAP Evaluation
===================
Runs inference over a validation split and scores it with ``pycocotools``.

Correctness hinges on the coordinate round trip: predictions are produced in
letterboxed space and must be mapped back through the recorded ``(ratio, pad)``
to original image pixels before scoring. A mistake here shifts every box and
depresses mAP with no visible error, so the inverse uses the same
``scale_boxes`` implementation as the data pipeline.

Reported metrics follow the standard COCO protocol:
mAP@[.5:.95], mAP@.5, mAP@.75, and the small/medium/large breakdown.
"""

import contextlib
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from data.transforms import scale_boxes

logger = logging.getLogger("Eval")


@torch.no_grad()
def run_inference(model, val_loader, id_map, device, conf_threshold=0.001,
                  iou_threshold=0.7, max_detections=300, amp_dtype=None,
                  progress_every=50):
    """Collect detections in COCO result format.

    Returns (results, seconds_of_inference).
    """
    model.eval()
    results = []
    inference_time = 0.0
    n_batches = len(val_loader)

    for bi, (images, metas) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)

        start = time.perf_counter()
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model.predict(images, conf_threshold=conf_threshold,
                                      iou_threshold=iou_threshold,
                                      max_detections=max_detections)
        else:
            preds = model.predict(images, conf_threshold=conf_threshold,
                                  iou_threshold=iou_threshold,
                                  max_detections=max_detections)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time += time.perf_counter() - start

        for det, meta in zip(preds, metas):
            if det.numel() == 0:
                continue
            det = det.float().cpu().numpy()
            boxes = scale_boxes(det[:, :4], meta["ratio"], meta["pad"],
                                meta["orig_shape"])
            image_id = id_map[meta["id"]]

            for (x1, y1, x2, y2), score, cls in zip(boxes, det[:, 4], det[:, 5]):
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                results.append({
                    "image_id": int(image_id),
                    "category_id": int(cls) + 1,
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(w), 2), round(float(h), 2)],
                    "score": round(float(score), 5),
                })

        if progress_every and (bi + 1) % progress_every == 0:
            logger.info("  inference %d/%d batches", bi + 1, n_batches)

    return results, inference_time


def score_results(gt_json, results, class_names=None, per_class=False):
    """Score COCO-format detections against a ground-truth JSON.

    Returns a metrics dict. An empty detection list yields all-zero metrics
    rather than raising, so an untrained model still produces a valid report.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    metrics = {
        "mAP50-95": 0.0, "mAP50": 0.0, "mAP75": 0.0,
        "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
        "AR100": 0.0, "num_detections": len(results),
    }

    if not results:
        logger.warning("No detections produced; reporting zero mAP.")
        return metrics

    # pycocotools writes to stdout unconditionally; capture it.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_json)
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = evaluator.stats
    metrics.update({
        "mAP50-95": float(stats[0]), "mAP50": float(stats[1]),
        "mAP75": float(stats[2]), "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]), "mAP_large": float(stats[5]),
        "AR100": float(stats[8]),
    })

    if per_class and class_names:
        precisions = evaluator.eval["precision"]  # [T, R, K, A, M]
        per_class_ap = {}
        for idx, name in enumerate(class_names):
            if idx >= precisions.shape[2]:
                break
            p = precisions[:, :, idx, 0, -1]
            p = p[p > -1]
            per_class_ap[name] = float(np.mean(p)) if p.size else float("nan")
        metrics["per_class_AP"] = per_class_ap

    return metrics


def evaluate(model, val_loader, gt_json, id_map, device, class_names=None,
             conf_threshold=0.001, iou_threshold=0.7, max_detections=300,
             amp_dtype=None, per_class=False, save_json=None):
    """Full evaluation: inference, optional dump, and COCO scoring."""
    results, inference_time = run_inference(
        model, val_loader, id_map, device,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        max_detections=max_detections, amp_dtype=amp_dtype,
    )

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(results, f)
        logger.info("Saved detections: %s", save_json)

    metrics = score_results(gt_json, results, class_names=class_names,
                            per_class=per_class)

    n_images = len(val_loader.dataset)
    metrics["inference_ms_per_image"] = (inference_time / max(n_images, 1)) * 1000
    metrics["fps"] = n_images / inference_time if inference_time > 0 else 0.0
    return metrics


def format_metrics(metrics, title="Validation"):
    """Render a metrics dict as an aligned text block."""
    lines = [
        f"  {title}",
        f"  {'-' * 46}",
        f"  mAP@[.5:.95] : {metrics['mAP50-95']:.4f}",
        f"  mAP@.5       : {metrics['mAP50']:.4f}",
        f"  mAP@.75      : {metrics['mAP75']:.4f}",
        f"  small/med/lg : {metrics['mAP_small']:.4f} / "
        f"{metrics['mAP_medium']:.4f} / {metrics['mAP_large']:.4f}",
        f"  AR@100       : {metrics['AR100']:.4f}",
        f"  detections   : {metrics['num_detections']}",
    ]
    if "inference_ms_per_image" in metrics:
        lines.append(f"  speed        : {metrics['inference_ms_per_image']:.1f} ms/img "
                     f"({metrics['fps']:.1f} FPS)")

    if "per_class_AP" in metrics:
        lines.append(f"  {'-' * 46}")
        ranked = sorted(metrics["per_class_AP"].items(), key=lambda kv: -kv[1])
        for name, ap in ranked:
            lines.append(f"    {name:<20s} {ap:.4f}")

    return "\n".join(lines)
