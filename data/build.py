"""
Dataset and DataLoader Construction
===================================
Resolves a dataset name to train/val splits, builds loaders, and materializes
a COCO-format ground-truth JSON for evaluation.

VOC is converted to COCO JSON once at preparation time so that both benchmarks
score through a single metric implementation. Two separate mAP implementations
would be two opportunities to report a number that is not comparable.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (VOC_CLASSES, DetectionDataset, ValDetectionDataset,
                      load_coco_records, load_voc_records)

logger = logging.getLogger("DataBuild")


DATASET_SPECS = {
    "coco": dict(num_classes=80, kind="coco"),
    "coco-minitrain": dict(num_classes=80, kind="coco"),
    "voc": dict(num_classes=20, kind="voc"),
}


def records_to_coco_json(records, class_names, out_path):
    """Write records as a COCO ground-truth JSON for pycocotools.

    Image ids must be integers for pycocotools, so string ids (VOC) are mapped
    to a stable integer index. The same mapping is applied to detections.
    """
    images, annotations, id_map = [], [], {}
    ann_id = 1

    for i, rec in enumerate(records):
        int_id = i + 1
        id_map[rec["id"]] = int_id
        images.append({
            "id": int_id,
            "file_name": os.path.basename(rec["image_path"]),
            "width": rec["width"], "height": rec["height"],
        })
        for box, cls in zip(rec["boxes"], rec["classes"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            annotations.append({
                "id": ann_id, "image_id": int_id, "category_id": int(cls) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": "generated ground truth"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    logger.info("Wrote COCO ground truth: %s (%d images, %d annotations)",
                out_path, len(images), len(annotations))
    return out_path, id_map


def build_records(dataset: str, data_root: str, split: str):
    """Return (records, class_names) for a dataset name and split."""
    dataset = dataset.lower()
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {dataset!r}. "
                         f"Choose from {list(DATASET_SPECS)}")

    kind = DATASET_SPECS[dataset]["kind"]

    if kind == "coco":
        subset = "train2017" if split == "train" else "val2017"
        ann_name = f"instances_{subset}.json"
        if dataset == "coco-minitrain" and split == "train":
            mini = os.path.join(data_root, "annotations", "instances_minitrain2017.json")
            if os.path.exists(mini):
                ann_name = "instances_minitrain2017.json"
            else:
                logger.warning(
                    "coco-minitrain requested but %s not found; falling back to "
                    "full instances_train2017.json", mini)

        ann_file = os.path.join(data_root, "annotations", ann_name)
        img_dir = os.path.join(data_root, "images", subset)
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Missing annotations: {ann_file}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing image directory: {img_dir}")

        records, class_names, _ = load_coco_records(ann_file, img_dir)
        if not records:
            raise RuntimeError(
                f"No usable images found for {dataset}/{split}. "
                f"Annotations exist at {ann_file} but no listed image was found "
                f"in {img_dir}.")
        return records, class_names

    image_sets = ([("2007", "trainval"), ("2012", "trainval")] if split == "train"
                  else [("2007", "test")])
    records, class_names = load_voc_records(data_root, image_sets)
    if not records:
        raise RuntimeError(f"No VOC records found under {data_root} for {image_sets}")
    return records, class_names


def build_dataloaders(dataset="voc", data_root="./VOCdevkit", img_size=640,
                      batch_size=16, workers=4, augment=True, hyp=None,
                      mosaic_prob=1.0, mixup_prob=0.1, cache_gt_json=None,
                      val_batch_size=None, seed=0):
    """Build train and val loaders plus evaluation metadata.

    Returns a dict with ``train_loader``, ``val_loader``, ``train_dataset``,
    ``val_dataset``, ``class_names``, ``num_classes``, ``gt_json``, ``id_map``.
    """
    train_records, class_names = build_records(dataset, data_root, "train")
    val_records, _ = build_records(dataset, data_root, "val")

    logger.info("%s | train images %d | val images %d | classes %d",
                dataset, len(train_records), len(val_records), len(class_names))

    train_dataset = DetectionDataset(
        train_records, img_size=img_size, augment=augment,
        mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, hyp=hyp,
        class_names=class_names,
    )
    val_dataset = ValDetectionDataset(val_records, img_size=img_size,
                                      class_names=class_names)

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin = torch.cuda.is_available()

    # Validation workers are not kept alive between epochs. Holding a second
    # persistent pool doubles the resident worker processes for the whole run,
    # and on Windows each one reserves shared memory against the paging file.
    # Exhausting that limit surfaces as "Couldn't open shared file mapping ...
    # error code 1455" mid-epoch, after hours of training. The val loader is
    # used once every eval_interval epochs, so respawning it is cheap.
    val_workers = min(workers, 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=pin, drop_last=True,
        collate_fn=DetectionDataset.collate_fn, generator=generator,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size or batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=pin, drop_last=False,
        collate_fn=ValDetectionDataset.collate_fn,
        persistent_workers=False,
    )

    gt_json = cache_gt_json or os.path.join("runs", "cache", f"{dataset}_val_gt.json")
    gt_json, id_map = records_to_coco_json(val_records, class_names, gt_json)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "class_names": class_names,
        "num_classes": len(class_names),
        "gt_json": gt_json,
        "id_map": id_map,
    }
