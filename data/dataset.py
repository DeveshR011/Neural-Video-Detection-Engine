"""
Detection Dataset
=================
A single dataset class serving both COCO-JSON and VOC-XML sources.

Annotations are parsed once into a common in-memory record list, so the
augmentation and batching path has exactly one implementation. Each record is
``{"image_path", "boxes" (xyxy px), "classes", "width", "height", "id"}``.

``__getitem__`` returns ``(image_tensor, labels)`` where labels is
``[n, 5]`` of ``(class_idx, cx, cy, w, h)`` with box coordinates normalized to
the letterboxed image. ``collate_fn`` prepends the batch index, producing the
``[N, 6]`` layout the loss consumes.
"""

import json
import logging
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import (augment_hsv, letterbox, mixup, mosaic4,
                         random_perspective, xyxy2xywhn)

logger = logging.getLogger("Dataset")

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def load_coco_records(ann_file, img_dir, class_map=None):
    """Parse a COCO instances JSON into records.

    Category ids in COCO are non-contiguous (1..90 for 80 classes), so they
    are remapped to a contiguous 0-based index. The inverse map is returned
    for writing evaluation results back in original category ids.
    """
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda c: c["id"])
    if class_map is None:
        class_map = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]

    images = {img["id"]: img for img in data["images"]}
    per_image = {img_id: [] for img_id in images}

    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        per_image[ann["image_id"]].append((x, y, x + w, y + h, class_map[ann["category_id"]]))

    records, missing = [], 0
    for img_id, img in images.items():
        path = os.path.join(img_dir, img["file_name"])
        if not os.path.exists(path):
            missing += 1
            continue
        entries = per_image[img_id]
        boxes = np.array([e[:4] for e in entries], dtype=np.float32).reshape(-1, 4)
        classes = np.array([e[4] for e in entries], dtype=np.int64)
        records.append({
            "image_path": path, "boxes": boxes, "classes": classes,
            "width": img["width"], "height": img["height"], "id": img_id,
        })

    if missing:
        logger.warning("%d/%d images listed in annotations are missing from %s",
                       missing, len(images), img_dir)
    return records, class_names, class_map


def load_voc_records(voc_root, image_sets, class_names=VOC_CLASSES, keep_difficult=False):
    """Parse VOC XML annotations.

    Args:
        voc_root: directory containing ``VOC2007`` / ``VOC2012``
        image_sets: list of ``(year, split)`` e.g. ``[("2007","trainval")]``
    """
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    records = []

    for year, split in image_sets:
        base = Path(voc_root) / f"VOC{year}"
        split_file = base / "ImageSets" / "Main" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"VOC split file not found: {split_file}")

        ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        for img_id in ids:
            xml_path = base / "Annotations" / f"{img_id}.xml"
            img_path = base / "JPEGImages" / f"{img_id}.jpg"
            if not xml_path.exists() or not img_path.exists():
                continue

            root = ET.parse(xml_path).getroot()
            size = root.find("size")
            width, height = int(size.find("width").text), int(size.find("height").text)

            entries = []
            for obj in root.findall("object"):
                difficult = int(obj.findtext("difficult", "0"))
                if difficult and not keep_difficult:
                    continue
                name = obj.findtext("name")
                if name not in name_to_idx:
                    continue
                bb = obj.find("bndbox")
                # VOC coordinates are 1-based inclusive.
                x1 = float(bb.findtext("xmin")) - 1
                y1 = float(bb.findtext("ymin")) - 1
                x2 = float(bb.findtext("xmax")) - 1
                y2 = float(bb.findtext("ymax")) - 1
                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    continue
                entries.append((x1, y1, x2, y2, name_to_idx[name]))

            boxes = np.array([e[:4] for e in entries], dtype=np.float32).reshape(-1, 4)
            classes = np.array([e[4] for e in entries], dtype=np.int64)
            records.append({
                "image_path": str(img_path), "boxes": boxes, "classes": classes,
                "width": width, "height": height, "id": f"{year}_{img_id}",
            })

    return records, list(class_names)


class DetectionDataset(Dataset):
    """Detection dataset with mosaic, mixup, affine, HSV, and flip augmentation.

    Args:
        records: list of annotation records
        img_size: square training resolution
        augment: enable training augmentations
        mosaic_prob / mixup_prob / flip_prob: augmentation probabilities
        hyp: dict overriding affine ranges
    """

    def __init__(self, records, img_size=640, augment=False, mosaic_prob=1.0,
                 mixup_prob=0.1, flip_prob=0.5, hyp=None, class_names=None):
        self.records = records
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob if augment else 0.0
        self.mixup_prob = mixup_prob if augment else 0.0
        self.flip_prob = flip_prob if augment else 0.0
        self.class_names = class_names or []

        default_hyp = dict(degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
                           perspective=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)
        self.hyp = {**default_hyp, **(hyp or {})}

        if not records:
            raise ValueError("DetectionDataset received an empty record list")

    def __len__(self):
        return len(self.records)

    def close_mosaic(self):
        """Disable mosaic and mixup for the final epochs.

        Mosaic creates image statistics that never occur at test time. Turning
        it off late lets the model adapt to real images and reliably adds mAP.
        """
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0

    def load_image(self, index):
        """Load an image and its boxes, resized so the long side is img_size."""
        rec = self.records[index]
        img = cv2.imread(rec["image_path"])
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {rec['image_path']}")

        h, w = img.shape[:2]
        r = self.img_size / max(h, w)
        if r != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            img = cv2.resize(img, (max(1, int(w * r)), max(1, int(h * r))),
                             interpolation=interp)

        boxes = rec["boxes"].copy() * r
        return img, boxes, rec["classes"].copy()

    def _load_mosaic(self, index):
        indices = [index] + random.choices(range(len(self.records)), k=3)
        random.shuffle(indices)
        loaded = [self.load_image(i) for i in indices]

        canvas, boxes, classes, border = mosaic4(
            [x[0] for x in loaded], [x[1] for x in loaded],
            [x[2] for x in loaded], self.img_size,
        )
        return random_perspective(
            canvas, boxes, classes,
            degrees=self.hyp["degrees"], translate=self.hyp["translate"],
            scale=self.hyp["scale"], shear=self.hyp["shear"],
            perspective=self.hyp["perspective"], border=border,
        )

    def __getitem__(self, index):
        use_mosaic = random.random() < self.mosaic_prob

        if use_mosaic:
            img, boxes, classes = self._load_mosaic(index)
            if random.random() < self.mixup_prob:
                img2, boxes2, classes2 = self._load_mosaic(
                    random.randint(0, len(self.records) - 1))
                img, boxes, classes = mixup(img, boxes, classes, img2, boxes2, classes2)
        else:
            img, boxes, classes = self.load_image(index)
            img, ratio, pad = letterbox(img, self.img_size, scaleup=self.augment)
            if len(boxes):
                boxes = boxes * ratio[0]
                boxes[:, [0, 2]] += pad[0]
                boxes[:, [1, 3]] += pad[1]
            if self.augment:
                img, boxes, classes = random_perspective(
                    img, boxes, classes,
                    degrees=self.hyp["degrees"], translate=self.hyp["translate"],
                    scale=self.hyp["scale"], shear=self.hyp["shear"],
                    perspective=self.hyp["perspective"],
                )

        if self.augment:
            img = augment_hsv(img, self.hyp["hsv_h"], self.hyp["hsv_s"], self.hyp["hsv_v"])
            if random.random() < self.flip_prob:
                img = np.fliplr(img)
                if len(boxes):
                    w = img.shape[1]
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        h, w = img.shape[:2]
        boxes_n = xyxy2xywhn(boxes.astype(np.float32), w, h)

        keep = (boxes_n[:, 2] > 0.002) & (boxes_n[:, 3] > 0.002) if len(boxes_n) \
            else np.zeros(0, dtype=bool)
        boxes_n, classes = boxes_n[keep], classes[keep]

        labels = np.zeros((len(boxes_n), 5), dtype=np.float32)
        if len(boxes_n):
            labels[:, 0] = classes
            labels[:, 1:] = boxes_n

        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW
        return torch.from_numpy(img).float() / 255.0, torch.from_numpy(labels)

    @staticmethod
    def collate_fn(batch):
        """Stack images and build flat [N, 6] targets of (batch_idx, cls, cx, cy, w, h)."""
        images, labels = zip(*batch)
        targets = []
        for i, lab in enumerate(labels):
            if lab.shape[0]:
                idx = torch.full((lab.shape[0], 1), float(i))
                targets.append(torch.cat([idx, lab], dim=1))
        targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
        return torch.stack(images, 0), targets


class ValDetectionDataset(Dataset):
    """Deterministic letterboxed dataset for evaluation.

    Returns the letterbox transform parameters alongside each image so
    predictions can be mapped back to original image coordinates.
    """

    def __init__(self, records, img_size=640, class_names=None):
        self.records = records
        self.img_size = img_size
        self.class_names = class_names or []

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        img = cv2.imread(rec["image_path"])
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {rec['image_path']}")
        orig_h, orig_w = img.shape[:2]

        img, ratio, pad = letterbox(img, self.img_size, scaleup=False)
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(img).float() / 255.0

        meta = {
            "id": rec["id"], "ratio": ratio, "pad": pad,
            "orig_shape": (orig_h, orig_w),
        }
        return tensor, meta

    @staticmethod
    def collate_fn(batch):
        images, metas = zip(*batch)
        return torch.stack(images, 0), list(metas)
