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

**Phase 1 — PASCAL VOC 07+12** (16.5k train, 20 classes). Validates the
rewritten loss and assigner with a fast feedback loop. Target: 80–85 mAP@0.5,
comparable to YOLOv3 (80.3), SSD512 (79.8), Faster R-CNN (73.2). Expected
wall-clock: 4–8 hours.

**Phase 2 — COCO minitrain (25k subset)**, evaluated on full val2017. Target:
28–33 mAP@[.5:.95]. Expected wall-clock: 10–20 hours.

**Reporting honesty.** Published YOLOv8 numbers use 118k images for ~500
epochs. Our COCO result must be reported as "X mAP trained on COCO minitrain
(25k images) in N GPU-hours on a single RTX 4050", never compared bare against
full-COCO results.

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
