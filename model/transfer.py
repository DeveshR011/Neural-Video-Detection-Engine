"""
Cross-Dataset Weight Transfer
=============================
Transfers a detector trained on one label set to another.

Everything except the classification head is label-agnostic: the backbone,
neck, and box-regression branch describe *where* objects are, not *what* they
are, so those tensors copy directly.

The classification head is the exception. Its final convolution has one output
channel per class, so a checkpoint trained on 80 classes cannot be loaded into
a 20-class model. The usual workaround is to discard that layer and randomly
re-initialize it, which throws away everything the pretraining learned about
classes that appear in *both* label sets.

Every PASCAL VOC class is present in COCO, so for a COCO -> VOC transfer the
head can instead be *sliced*: for each destination class, copy the weights of
the matching source class. This preserves the learned classifier for all 20
classes rather than relearning them.
"""

import logging

import torch

logger = logging.getLogger("Transfer")


# VOC class name -> COCO class name. Every VOC class has a COCO counterpart;
# several differ only in naming convention.
VOC_TO_COCO = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "sofa": "couch",
    "train": "train",
    "tvmonitor": "tv",
}


def _normalize(name):
    return name.strip().lower().replace("_", " ").replace("-", " ")


def build_class_index_map(src_names, dst_names, alias=None):
    """Map each destination class to its index in the source label set.

    Args:
        src_names: class names of the pretrained checkpoint
        dst_names: class names of the model being initialized
        alias: optional destination-name -> source-name overrides

    Returns:
        (index_map, matched, unmatched) where index_map[i] is the source index
        for destination class i, or -1 when there is no counterpart.
    """
    alias = alias or {}
    src_lookup = {_normalize(n): i for i, n in enumerate(src_names)}

    index_map, matched, unmatched = [], [], []
    for name in dst_names:
        key = _normalize(alias.get(name, name))
        idx = src_lookup.get(key, src_lookup.get(_normalize(name), -1))
        index_map.append(idx)
        (matched if idx >= 0 else unmatched).append(name)

    return index_map, matched, unmatched


def transfer_state_dict(src_state, dst_model, src_names, dst_names, alias=None):
    """Build a state dict for ``dst_model`` from a differently-classed checkpoint.

    Tensors whose shape already matches are copied directly. Classification
    tensors whose first dimension equals the source class count are sliced
    per-class using the name mapping; destination classes with no counterpart
    keep their initialized values.

    Returns:
        (new_state, report) where report describes what was transferred.
    """
    if alias is None and set(dst_names) <= set(VOC_TO_COCO):
        alias = VOC_TO_COCO

    index_map, matched, unmatched = build_class_index_map(src_names, dst_names, alias)
    n_src, n_dst = len(src_names), len(dst_names)

    dst_state = dst_model.state_dict()
    new_state = {}
    copied = sliced = skipped = 0
    names_differ = list(src_names) != list(dst_names)

    for key, dst_tensor in dst_state.items():
        if key not in src_state:
            new_state[key] = dst_tensor
            skipped += 1
            continue

        src_tensor = src_state[key]

        # Whether a tensor is class-indexed is decided by which module it
        # belongs to, not by its shape. Two label sets of equal size but
        # different content produce identically shaped classifier tensors, and
        # copying those directly would silently mis-map every class.
        # ``num_batches_tracked`` buffers also live under cls_branches and are
        # 0-dimensional, so the rank check must come before any indexing.
        is_class_indexed = (
            "cls_branches" in key
            and src_tensor.dim() > 0
            and dst_tensor.dim() > 0
            and src_tensor.shape[0] == n_src
            and dst_tensor.shape[0] == n_dst
            and src_tensor.shape[1:] == dst_tensor.shape[1:]
        )

        if is_class_indexed and (names_differ or n_src != n_dst):
            out = dst_tensor.clone()
            for dst_idx, src_idx in enumerate(index_map):
                if src_idx >= 0:
                    out[dst_idx] = src_tensor[src_idx]
            new_state[key] = out
            sliced += 1
            continue

        if src_tensor.shape == dst_tensor.shape:
            new_state[key] = src_tensor.clone()
            copied += 1
            continue

        new_state[key] = dst_tensor
        skipped += 1

    report = {
        "copied": copied,
        "sliced": sliced,
        "skipped": skipped,
        "matched_classes": matched,
        "unmatched_classes": unmatched,
    }
    return new_state, report


def log_transfer_report(report, src_names, dst_names):
    logger.info("Class transfer: %d/%d destination classes matched a source class",
                len(report["matched_classes"]), len(dst_names))
    logger.info("Tensors: %d copied, %d class-sliced, %d left initialized",
                report["copied"], report["sliced"], report["skipped"])
    if report["unmatched_classes"]:
        logger.warning("No source counterpart for: %s",
                       ", ".join(report["unmatched_classes"]))
