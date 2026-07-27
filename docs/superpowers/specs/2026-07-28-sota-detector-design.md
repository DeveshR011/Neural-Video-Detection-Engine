# SOTA Detector Rewrite — Design Spec

Date: 2026-07-28
Status: Approved

## Problem

The repository contains a hand-rolled anchor-based YOLOv3-era detector with no
evaluation harness. Three defects block any benchmark result:

1. **Untrainable loss.** `DetectionLoss._decode_flat_boxes` infers feature-map
   geometry from the flat prediction count (`base = round((cells*16/21)**0.5)`)
   and silently degrades to `sigmoid(raw)` when the arithmetic fails. Target
   assignment is global IoU matching over those guessed boxes, inside a Python
   loop over the batch.
2. **No mAP path.** `benchmarks/benchmark.py` measures latency on synthetic
   tensors only. There is no COCO evaluation, so model quality is unmeasured.
3. **Security layer inside the accuracy path.** A frozen random projection is
   applied to P3/P4/P5 features and `.zero_()` is called on live activations
   during eval. Both cost accuracy for no security benefit at that location.

Secondary gaps: no EMA, no mosaic/mixup, no pretrained weights, no letterbox,
no train2017 images on disk, no tests.

## Constraints

- Single RTX 4050 laptop GPU, 6 GB VRAM.
- Total training budget: a few hours to one day, with checkpoint/resume.
- Numbers must be defensible against published baselines.

## Architecture

Anchor-free, YOLOv8/v11-class detector with a hybrid CNN + attention backbone.

```
input 640x640
  -> Backbone (timm, ImageNet-pretrained, features_only, out_indices=P3/P4/P5)
  -> PSA attention block on P5 only (20x20 = 400 tokens)
  -> BiFPN neck (retained from existing code, repeated n times)
  -> Decoupled head per scale:
       cls branch -> [B, nc, H, W]
       reg branch -> [B, 4*(reg_max+1), H, W]   (DFL distribution)
  -> Anchor-free decode: ltrb distances from grid cell center, x stride
```

**Rationale for the hybrid.** A plain ViT backbone loses under these
constraints: attention at 640² (1600 tokens) exceeds 6 GB at usable batch
sizes, ViTs need MAE pretraining plus long schedules to beat a pretrained CNN,
and every detector on the real-time speed/accuracy Pareto front (YOLOv8/v10/v11,
RT-DETR) uses a convolutional backbone. Attention is applied only at P5, where
the token count is 400 and global context matters most — the same choice made
by YOLOv11 (C2PSA) and RT-DETR (AIFI on P5). The backbone is selectable by
flag so hierarchical alternates (ConvNeXt-T, EfficientViT) can be A/B tested
once the pipeline is validated.

### Key components

| Component | Choice | Reason |
|---|---|---|
| Box regression | DFL, `reg_max=16` | Learns a distribution over offsets; ~+2 mAP over direct regression |
| Assignment | Task-Aligned Assigner, `topk=10`, `alpha=0.5`, `beta=6.0` | Dynamic, aligns cls confidence with localization quality |
| Box loss | CIoU + DFL | Standard for the class |
| Cls loss | BCE over TAL-normalized soft targets | Matches v8; avoids objectness branch entirely |
| Weight averaging | EMA, decay 0.9999 with warmup ramp | Consistently worth 1–2 mAP |
| Precision | AMP, bf16 preferred | Fits 6 GB, ~1.8x throughput |
| NMS | Batched, vectorized `torchvision.ops.batched_nms` | Replaces per-class Python loop |

No objectness channel. Output per anchor point is `nc + 4*(reg_max+1)`.

## Data pipeline

- **Letterbox** resize preserving aspect ratio, with padding recorded for
  inverse-mapping predictions back to original image coordinates at eval time.
- **Mosaic** (4-image), **mixup**, HSV jitter, horizontal flip, random
  scale/translate. Mosaic disabled for the final `close_mosaic` epochs.
- **Datasets**: unified `DetectionDataset` with COCO-JSON and VOC-XML readers,
  producing a common `(image, boxes_xyxy, class_ids)` record.
- **Collate** to a flat `(batch_idx, cls, xywh_normalized)` target tensor —
  the layout the vectorized loss consumes.

## Evaluation harness

`eval/coco_eval.py` — runs inference over a val split, maps boxes back through
the letterbox transform to original coordinates, writes COCO-format detection
JSON, and scores with `pycocotools`:

- mAP@[.5:.95], mAP@.5, mAP@.75
- mAP small / medium / large
- Per-class AP table

VOC is scored through the same path by converting VOC annotations to COCO JSON
once at dataset-prep time, so there is exactly one metric implementation.

## Security layer refactor

Every existing security feature is retained, but moved out of the forward path:

- Random projection and mid-forward `.zero_()`: **removed** from the detector.
  The projection is preserved as an opt-in `--secure-features` wrapper applied
  after the neck, off by default, documented as costing mAP.
- Model encryption, hash-chain audit log, integrity registry, device-bound
  keys: unchanged, applied at checkpoint load/save and at inference API
  boundaries.
- Adversarial training remains available via flag, off for benchmark runs.

The README must state which numbers were produced with security features on
versus off.

## Benchmark plan

**PASCAL VOC 07+12 only.** Train on VOC2007 trainval + VOC2012 trainval
(16,551 images), evaluate on VOC2007 test (4,952 images) — the protocol used
by the published baselines. COCO was dropped: the 19 GB download plus ~10 h of
training did not fit the wall-clock budget, and a single well-executed
benchmark is worth more than two rushed ones.

Measured throughput: 31.4 img/s at 640px, batch 16, bf16 on the RTX 4050
(4.4 GB of 6.4 GB used) — 8.8 minutes per epoch.

Run is split across two sessions using `--resume`, which restores model, EMA,
optimizer state, epoch counter, and best-mAP. The cosine schedule is computed
from the total epoch count, so splitting does not distort it. Resume refuses
to start if any schedule- or architecture-critical flag differs from the
original run.

**Comparison set.** Our protocol is an ImageNet-pretrained backbone with the
detector trained from random init, so the honest comparison is against:

| Model | VOC07 test mAP@0.5 |
|---|---|
| Faster R-CNN (VGG16) | 73.2 |
| SSD512 | 79.8 |
| YOLOv3 | ~80.3 |
| R-FCN (ResNet-101) | 80.5 |
| RefineDet512 | 81.8 |

**Reporting honesty.**

- Results in the 88–92 range reported for YOLOv5/v8 on VOC start from a
  *COCO-pretrained detector* and fine-tune. That transfers detection knowledge
  from 118k images and 80 classes. Our number must never be compared against
  those without stating the difference in protocol.
- We report COCO-style AP50 via pycocotools on VOC data. The official VOC
  metric uses different interpolation; the two typically agree within ~1
  point but are not identical. The README must say so.
- The headline claim states dataset, epochs, hardware, and wall-clock.

## Testing

- Shape//dtype contract tests for backbone, neck, head at multiple input sizes.
- DFL integral correctness against a closed-form expectation.
- TAL assigner: synthetic case with known-correct assignment.
- Loss: overfit-a-single-batch test — loss must drop below a threshold in N
  steps. This is the regression test that would have caught the original bug.
- Letterbox round-trip: box coordinates survive forward and inverse transform.
- End-to-end smoke: 2-step train + eval on a tiny fixture, asserting mAP > 0.

## Out of scope

- Distributed / multi-GPU training.
- ONNX/TensorRT export.
- Real SGX/SEV enclave implementations (remain documented stubs).
