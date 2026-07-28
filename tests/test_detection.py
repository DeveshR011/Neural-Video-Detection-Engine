"""
Detection Pipeline Tests
========================
Run with:  py -3.14 -m pytest tests/ -v

Several tests encode specific defects found during the rewrite and exist to
stop them recurring:

- ``test_dfl_bin_range_under_bf16``: bfloat16 rounding pushed a DFL target
  clamped to 14.99 up to 15.0, overflowing the bin count.
- ``test_bbox_iou_shape_matches_weight``: an [N] IoU multiplied by an [N, 1]
  weight broadcast into an [N, N] outer product and silently corrupted the
  box loss.
- ``test_overfit_single_batch``: the end-to-end guarantee that the loss is
  trainable at all. The previous loss would have failed this.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.transforms import letterbox, scale_boxes, xywhn2xyxy, xyxy2xywhn
from model.assigner import TaskAlignedAssigner, select_candidates_in_gts
from model.blocks import DFL, C2PSA
from model.detection_loss import BboxLoss, DetectionLoss
from model.detector import SecureYOLO
from model.head import DetectHead, bbox2dist, dist2bbox, make_anchors
from model.losses import bbox_iou

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Building blocks ───────────────────────────────────────────────────

def test_dfl_integral_matches_expectation():
    """DFL must compute the expectation of the softmax distribution."""
    reg_max = 16
    dfl = DFL(reg_max)
    logits = torch.randn(2, 4 * reg_max, 10)

    out = dfl(logits)
    assert out.shape == (2, 4, 10)

    probs = logits.view(2, 4, reg_max, 10).softmax(dim=2)
    expected = (probs * torch.arange(reg_max).view(1, 1, reg_max, 1)).sum(dim=2)
    assert torch.allclose(out, expected, atol=1e-5)


def test_dfl_on_one_hot_returns_bin_index():
    """A distribution concentrated on bin k must integrate to exactly k."""
    reg_max = 16
    dfl = DFL(reg_max)
    logits = torch.full((1, 4 * reg_max, 1), -1e4)
    for side in range(4):
        logits[0, side * reg_max + 7, 0] = 1e4
    assert torch.allclose(dfl(logits), torch.full((1, 4, 1), 7.0), atol=1e-3)


def test_c2psa_preserves_shape():
    block = C2PSA(64, 64, n=1)
    x = torch.randn(2, 64, 20, 20)
    assert block(x).shape == x.shape


# ── Box geometry ──────────────────────────────────────────────────────

def test_dist2bbox_bbox2dist_roundtrip():
    """Encoding a box to ltrb distances and back must be lossless."""
    anchors = torch.tensor([[10.0, 10.0], [20.0, 30.0]])
    boxes = torch.tensor([[6.0, 7.0, 14.0, 13.0], [15.0, 25.0, 26.0, 36.0]])

    dist = bbox2dist(anchors, boxes, reg_max=64)
    recovered = dist2bbox(dist, anchors, xywh=False)
    assert torch.allclose(recovered, boxes, atol=1e-5)


def test_bbox2dist_respects_bin_range():
    """Targets must stay strictly below reg_max - 1 so the upper index is valid."""
    reg_max = 16
    anchors = torch.zeros(4, 2)
    huge = torch.tensor([[-1000.0, -1000.0, 1000.0, 1000.0]]).repeat(4, 1)
    dist = bbox2dist(anchors, huge, reg_max)
    assert dist.max() < reg_max - 1, f"max {dist.max()} would overflow {reg_max} bins"


def test_dfl_bin_range_under_bf16():
    """Regression: bfloat16 rounding must not push the upper bin out of range.

    bfloat16 has an 8-bit mantissa, so spacing near 15 is 0.0625 and a target
    clamped to 14.99 rounds to exactly 15.0. With 16 bins the upper bracket
    index becomes 16 and cross_entropy raises a device-side assert.
    """
    reg_max = 16
    pred = torch.randn(8, reg_max, dtype=torch.float32)
    target = torch.full((8, 1), reg_max - 1.01, dtype=torch.bfloat16)

    assert float(target[0]) == 15.0, "precondition: bf16 rounds 14.99 to 15.0"
    loss = BboxLoss._df_loss(pred, target)
    assert torch.isfinite(loss).all()


def test_bbox_iou_shape_matches_weight():
    """Regression: iou and per-positive weight must have identical shape.

    A shape mismatch broadcasts into an [N, N] outer product, inflating the
    loss by a factor of N without any error.
    """
    n = 7
    b1 = torch.rand(n, 4)
    b2 = torch.rand(n, 4)
    b1[:, 2:] += b1[:, :2] + 0.1
    b2[:, 2:] += b2[:, :2] + 0.1

    iou = bbox_iou(b1, b2, xywh=False, ciou=True)
    assert iou.shape == (n,), f"expected [{n}], got {tuple(iou.shape)}"

    weight = torch.rand(n)
    assert ((1 - iou) * weight).shape == (n,)


def test_ciou_bounded():
    """CIoU lies in [-1.5, 1], so the 1 - CIoU loss cannot exceed 2.5."""
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[500.0, 500.0, 510.0, 510.0]])
    loss = 1 - bbox_iou(b1, b2, xywh=False, ciou=True)
    assert 0 <= float(loss) <= 2.5, f"1-CIoU out of range: {float(loss)}"


def test_identical_boxes_give_iou_one():
    b = torch.tensor([[3.0, 4.0, 9.0, 12.0]])
    assert pytest.approx(1.0, abs=1e-5) == float(bbox_iou(b, b, xywh=False))
    assert pytest.approx(1.0, abs=1e-5) == float(bbox_iou(b, b, xywh=False, ciou=True))


# ── Anchors ───────────────────────────────────────────────────────────

def test_make_anchors_counts_and_centres():
    feats = [torch.zeros(1, 8, 80, 80), torch.zeros(1, 8, 40, 40), torch.zeros(1, 8, 20, 20)]
    points, strides = make_anchors(feats, (8, 16, 32))

    assert points.shape[0] == 80 * 80 + 40 * 40 + 20 * 20 == 8400
    assert strides.shape == (8400, 1)
    # First cell centre sits at (0.5, 0.5) in feature units.
    assert torch.allclose(points[0], torch.tensor([0.5, 0.5]))
    assert float(strides[0]) == 8.0 and float(strides[-1]) == 32.0


# ── Assigner ──────────────────────────────────────────────────────────

def test_select_candidates_inside_gt():
    anchors = torch.tensor([[5.0, 5.0], [50.0, 50.0], [15.0, 15.0]])
    gt = torch.tensor([[[0.0, 0.0, 20.0, 20.0]]])
    mask = select_candidates_in_gts(anchors, gt)
    assert mask.shape == (1, 1, 3)
    assert mask[0, 0].tolist() == [True, False, True]


def test_assigner_selects_positives_near_gt():
    """The assigner must produce positives and place them inside the GT box."""
    torch.manual_seed(0)
    n_anchors, nc = 400, 4
    grid = torch.stack(torch.meshgrid(
        torch.arange(20) * 16.0 + 8, torch.arange(20) * 16.0 + 8, indexing="ij"
    ), dim=-1).view(-1, 2)

    gt_box = torch.tensor([[[100.0, 100.0, 200.0, 200.0]]])
    gt_labels = torch.tensor([[[2.0]]])
    mask_gt = torch.ones(1, 1, 1)

    pred_boxes = torch.cat([grid - 40, grid + 40], dim=1).unsqueeze(0)
    pred_scores = torch.full((1, n_anchors, nc), 0.5)

    assigner = TaskAlignedAssigner(topk=10, num_classes=nc)
    t_boxes, t_scores, fg, t_idx = assigner(
        pred_scores, pred_boxes, grid, gt_labels, gt_box, mask_gt)

    assert fg.sum() > 0, "assigner produced no positives"
    assert fg.sum() <= 10
    centres = grid[fg[0]]
    assert (centres[:, 0] >= 100).all() and (centres[:, 0] <= 200).all()
    # Assigned targets must be the GT box, and the class must be 2.
    assert torch.allclose(t_boxes[0][fg[0]][0], gt_box[0, 0])
    assert t_scores[0][fg[0]][:, 2].max() > 0


def test_assigner_handles_no_gt():
    assigner = TaskAlignedAssigner(topk=10, num_classes=4)
    _, _, fg, _ = assigner(
        torch.rand(2, 100, 4), torch.rand(2, 100, 4), torch.rand(100, 2),
        torch.zeros(2, 0, 1), torch.zeros(2, 0, 4), torch.zeros(2, 0, 1))
    assert fg.sum() == 0


# ── Model ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size", ["n", "s"])
def test_model_forward_shapes(size):
    model = SecureYOLO(num_classes=20, size=size, backbone="builtin")
    box_dist, cls_logits, points, strides = model(torch.randn(2, 3, 320, 320))
    n_anchors = 40 * 40 + 20 * 20 + 10 * 10

    assert box_dist.shape == (2, 4 * model.reg_max, n_anchors)
    assert cls_logits.shape == (2, 20, n_anchors)
    assert points.shape == (n_anchors, 2)
    assert strides.shape == (n_anchors, 1)


def test_model_handles_non_square_input():
    model = SecureYOLO(num_classes=20, size="n")
    _, cls_logits, points, _ = model(torch.randn(1, 3, 480, 640))
    expected = 60 * 80 + 30 * 40 + 15 * 20
    assert points.shape[0] == expected
    assert cls_logits.shape[2] == expected


def test_predict_returns_valid_detections():
    model = SecureYOLO(num_classes=20, size="n").eval()
    results = model.predict(torch.randn(2, 3, 320, 320), conf_threshold=0.0)

    assert len(results) == 2
    for det in results:
        assert det.shape[1] == 6
        if det.shape[0]:
            assert (det[:, 0] <= det[:, 2]).all(), "x1 must not exceed x2"
            assert (det[:, 1] <= det[:, 3]).all(), "y1 must not exceed y2"
            assert (det[:, 4] >= 0).all() and (det[:, 4] <= 1).all()
            assert (det[:, 5] >= 0).all() and (det[:, 5] < 20).all()


def test_secure_features_flag_changes_parameters():
    plain = SecureYOLO(num_classes=20, size="n", secure_features=False)
    secure = SecureYOLO(num_classes=20, size="n", secure_features=True)
    assert plain.projection is None
    assert secure.projection is not None
    # The projection must be frozen, not trained.
    assert not any(p.requires_grad for p in secure.projection.parameters())


# ── Loss ──────────────────────────────────────────────────────────────

def test_loss_components_finite_and_positive():
    torch.manual_seed(0)
    model = SecureYOLO(num_classes=20, size="n")
    preds = model(torch.randn(2, 3, 320, 320))
    targets = torch.tensor([
        [0, 3, 0.5, 0.5, 0.2, 0.3],
        [1, 7, 0.3, 0.4, 0.1, 0.2],
    ], dtype=torch.float32)

    loss, comp = DetectionLoss(num_classes=20)(preds, targets, imgsz=(320, 320))
    assert torch.isfinite(loss)
    assert comp["n_positives"] > 0
    assert 0 <= comp["box"] <= 2.5, f"CIoU term out of range: {comp['box']}"
    for key in ("box", "cls", "dfl"):
        assert comp[key] >= 0


def test_loss_handles_empty_targets():
    model = SecureYOLO(num_classes=20, size="n")
    preds = model(torch.randn(2, 3, 320, 320))
    loss, comp = DetectionLoss(num_classes=20)(preds, torch.zeros((0, 6)),
                                               imgsz=(320, 320))
    assert torch.isfinite(loss)
    assert comp["n_positives"] == 0
    assert comp["box"] == 0.0


def test_loss_backward_produces_gradients():
    model = SecureYOLO(num_classes=20, size="n")
    preds = model(torch.randn(2, 3, 320, 320))
    targets = torch.tensor([[0, 3, 0.5, 0.5, 0.2, 0.3]], dtype=torch.float32)
    loss, _ = DetectionLoss(num_classes=20)(preds, targets, imgsz=(320, 320))
    loss.backward()

    with_grad = [p for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(with_grad) > 10, "too few parameters received gradient"


def test_loss_decreases_for_better_boxes():
    """A prediction closer to the target must yield a lower box loss."""
    torch.manual_seed(0)
    anchors = torch.tensor([[10.0, 10.0]])
    target = torch.tensor([[6.0, 6.0, 14.0, 14.0]])
    close = torch.tensor([[6.2, 6.2, 13.8, 13.8]])
    far = torch.tensor([[2.0, 2.0, 18.0, 18.0]])

    loss_close = 1 - bbox_iou(close, target, xywh=False, ciou=True)
    loss_far = 1 - bbox_iou(far, target, xywh=False, ciou=True)
    assert float(loss_close) < float(loss_far)


# ── Transforms ────────────────────────────────────────────────────────

def test_letterbox_roundtrip_recovers_boxes():
    """The inverse transform must recover original coordinates exactly.

    An error here shifts every prediction and depresses mAP with no visible
    failure, so it is checked directly.
    """
    img = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 20.0, 110.0, 220.0],
                      [300.0, 100.0, 700.0, 470.0]], dtype=np.float32)

    lb, ratio, pad = letterbox(img, 640, scaleup=False)
    assert lb.shape[:2] == (640, 640)

    fwd = boxes * ratio[0]
    fwd[:, [0, 2]] += pad[0]
    fwd[:, [1, 3]] += pad[1]

    back = scale_boxes(fwd.copy(), ratio, pad, (480, 720))
    assert np.abs(back - boxes).max() < 1e-2


def test_letterbox_preserves_aspect_ratio():
    img = np.random.randint(0, 255, (300, 900, 3), dtype=np.uint8)
    lb, ratio, pad = letterbox(img, 640, scaleup=False)
    assert lb.shape[:2] == (640, 640)
    assert pytest.approx(640 / 900, rel=1e-3) == ratio[0]


def test_xyxy_xywhn_roundtrip():
    boxes = np.array([[10.0, 20.0, 110.0, 220.0]], dtype=np.float32)
    n = xyxy2xywhn(boxes.copy(), 720, 480)
    assert (n >= 0).all() and (n <= 1).all()
    assert np.abs(xywhn2xyxy(n.copy(), 720, 480) - boxes).max() < 1e-3


# ── End to end ────────────────────────────────────────────────────────

def test_overfit_single_batch():
    """The definitive check that the loss is trainable.

    A correct detector must drive the loss down and recover the target boxes
    from a fixed batch. The previous implementation, which reconstructed grid
    geometry from the flat prediction count, could not pass this.
    """
    torch.manual_seed(0)
    model = SecureYOLO(num_classes=4, size="n").to(DEVICE)
    loss_fn = DetectionLoss(num_classes=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    images = torch.randn(2, 3, 320, 320, device=DEVICE)
    targets = torch.tensor([
        [0, 0, 0.30, 0.30, 0.20, 0.20],
        [0, 1, 0.70, 0.70, 0.25, 0.25],
        [1, 2, 0.50, 0.50, 0.40, 0.30],
    ], device=DEVICE)

    model.train()
    first = None
    for _ in range(120):
        loss, comp = loss_fn(model(images), targets, imgsz=(320, 320))
        first = first if first is not None else comp["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert comp["total"] < first * 0.3, \
        f"loss only fell {first:.2f} -> {comp['total']:.2f}"
    assert comp["box"] < 0.1, f"box loss did not converge: {comp['box']:.4f}"

    model.eval()
    results = model.predict(images, conf_threshold=0.25, iou_threshold=0.5)
    assert results[0].shape[0] >= 2, "failed to recover both boxes in image 0"

    # First target: centre (0.30, 0.30), size (0.20, 0.20) at 320px.
    expected = torch.tensor([64.0, 64.0, 128.0, 128.0], device=DEVICE)
    best = min(results[0], key=lambda d: (d[:4] - expected).abs().max())
    assert (best[:4] - expected).abs().max() < 10, \
        f"box off by more than 10px: {best[:4].tolist()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_amp_bf16_training_step():
    """A full bf16 autocast step must stay finite -- guards the DFL range bug."""
    model = SecureYOLO(num_classes=20, size="n").to(DEVICE)
    loss_fn = DetectionLoss(num_classes=20).to(DEVICE)
    images = torch.randn(2, 3, 320, 320, device=DEVICE)
    # A box spanning nearly the whole image maximizes ltrb distances.
    targets = torch.tensor([[0, 1, 0.5, 0.5, 0.98, 0.98]], device=DEVICE)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, comp = loss_fn(model(images), targets, imgsz=(320, 320))
    loss.backward()

    assert torch.isfinite(loss), "bf16 step produced a non-finite loss"
    assert comp["n_positives"] > 0


# ── Cross-dataset transfer ────────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def test_every_voc_class_maps_to_coco():
    """All 20 VOC classes exist in COCO, so none should fall back to random init."""
    from data.dataset import VOC_CLASSES
    from model.transfer import VOC_TO_COCO, build_class_index_map

    _, matched, unmatched = build_class_index_map(
        COCO_CLASSES, list(VOC_CLASSES), VOC_TO_COCO)
    assert not unmatched, f"unmapped VOC classes: {unmatched}"
    assert len(matched) == 20


def test_voc_coco_alias_targets_are_real_coco_classes():
    """Guards against typos in the alias table, which would silently drop a class."""
    from model.transfer import VOC_TO_COCO
    for voc_name, coco_name in VOC_TO_COCO.items():
        assert coco_name in COCO_CLASSES, \
            f"alias {voc_name} -> {coco_name!r} is not a COCO class"


@pytest.mark.parametrize("size", ["n", "s"])
def test_head_width_independent_of_class_count(size):
    """Head geometry must not depend on num_classes.

    If it did, an 80-class and a 20-class model would have structurally
    different heads and no weights could transfer between them.
    """
    a = SecureYOLO(num_classes=80, size=size)
    b = SecureYOLO(num_classes=20, size=size)
    for branch_a, branch_b in zip(a.head.cls_branches, b.head.cls_branches):
        # All layers except the final per-class conv must match exactly.
        assert branch_a[0].conv.weight.shape == branch_b[0].conv.weight.shape
        assert branch_a[1].conv.weight.shape == branch_b[1].conv.weight.shape


@pytest.mark.parametrize("size", ["n", "s"])
def test_transfer_copies_all_tensors_and_slices_classes(size):
    """A COCO -> VOC transfer must leave nothing randomly initialized."""
    from data.dataset import VOC_CLASSES
    from model.transfer import build_class_index_map, transfer_state_dict, VOC_TO_COCO

    src = SecureYOLO(num_classes=80, size=size)
    dst = SecureYOLO(num_classes=20, size=size)
    new_state, report = transfer_state_dict(
        src.state_dict(), dst, COCO_CLASSES, list(VOC_CLASSES))
    dst.load_state_dict(new_state)

    assert report["skipped"] == 0, "some tensors were left at their init values"
    # weight + bias of the final conv at each of the three levels
    assert report["sliced"] == 6

    idx, _, _ = build_class_index_map(COCO_CLASSES, list(VOC_CLASSES), VOC_TO_COCO)
    key = "head.cls_branches.0.2.weight"
    for dst_i, src_i in enumerate(idx):
        assert torch.allclose(dst.state_dict()[key][dst_i], src.state_dict()[key][src_i])

    assert torch.allclose(dst.backbone.body.stem.conv.weight,
                          src.backbone.body.stem.conv.weight)


def test_transfer_leaves_unmatched_classes_initialized():
    """A destination class with no source counterpart keeps its own weights."""
    from model.transfer import transfer_state_dict

    src = SecureYOLO(num_classes=3, size="n")
    dst = SecureYOLO(num_classes=3, size="n")
    original = dst.state_dict()["head.cls_branches.0.2.weight"].clone()

    new_state, report = transfer_state_dict(
        src.state_dict(), dst, ["cat", "dog", "bird"], ["cat", "dog", "wombat"])

    assert report["unmatched_classes"] == ["wombat"]
    assert torch.allclose(new_state["head.cls_branches.0.2.weight"][2], original[2])
