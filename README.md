# Neural Video Detection Engine

Anchor-free object detector with a hybrid CNN + attention backbone, trained
and evaluated end-to-end on a single 6 GB laptop GPU. Includes an on-device
security layer (model encryption, integrity hashing, audit logging) that sits
outside the inference path and therefore costs no accuracy.

**82.6 mAP@0.5 on PASCAL VOC2007 test** — trained in 10 GPU-hours on an RTX
4050 Laptop (6 GB), ImageNet-pretrained backbone, detector trained from
random init.

Engineering notes on three silent-correctness bugs found and fixed during the
rewrite are in [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md).

---

## Results

VOC2007 test (4,952 images), trained on VOC2007 + VOC2012 trainval (16,551
images). Metric is COCO-style AP via `pycocotools`.

| Configuration | mAP@[.5:.95] | mAP@.5 | mAP@.75 | FPS |
|---|---:|---:|---:|---:|
| Base (epoch 60, EMA) | 0.6092 | 0.8102 | 0.6668 | 87.1 |
| \+ flip TTA | 0.6213 | 0.8232 | 0.6799 | 44.7 |
| **\+ multi-scale TTA** | **0.6343** | **0.8264** | **0.6996** | 14.1 |

Object-size breakdown at the best setting: small **0.1396**, medium 0.3964,
large **0.7241**. Small-object AP is the binding constraint on the headline
number.

### How this compares

Published VOC2007 test results using the same protocol — ImageNet-pretrained
backbone, detector trained from random init:

| Model | mAP@0.5 |
|---|---:|
| Faster R-CNN (VGG16) | 73.2 |
| SSD300 | 74.3 |
| Faster R-CNN (ResNet-101) | 76.4 |
| YOLOv2 544 | 78.6 |
| SSD512 | 79.8 |
| YOLOv3 | ~80.3 |
| R-FCN (ResNet-101) | 80.5 |
| DSSD513 | 81.5 |
| RefineDet512 | 81.8 |
| **This work** | **82.6** |
| RefineDet512+ (multi-scale test) | 83.8 |

**Two caveats, stated plainly:**

1. Results in the **88–92** range reported for YOLOv5/v8 on VOC start from a
   **COCO-pretrained detector** and fine-tune on VOC. That transfers detection
   knowledge from 118k images across 80 classes. This model starts from an
   ImageNet-classification backbone with the detector head randomly
   initialized. The numbers are not comparable, and the table above
   deliberately lists only same-protocol results.
2. This reports **COCO-style AP50** via `pycocotools` on VOC data. The official
   VOC metric uses different interpolation. The two typically agree within
   ~1 point but are not identical.

Raw metrics and the full training curve are in [`results/`](results/).

---

## Architecture

```
input 640x640
  │
  ├─ Backbone ── timm model, ImageNet-pretrained (default resnet34)
  │              returns P3/8, P4/16, P5/32
  │                        │
  │                        └─ C2PSA attention on P5 only
  │
  ├─ Neck ────── BiFPN, learnable weighted fusion, 2 bidirectional passes
  │
  └─ Head ────── decoupled, anchor-free
                   cls branch → [B, nc, A]
                   reg branch → [B, 4*(reg_max), A]   DFL distribution
                   decode: ltrb distances from cell centre × stride
```

### Why a CNN backbone with attention only at P5

A plain ViT backbone is the wrong choice under these constraints:

- **Memory.** Attention at 640² is 1600 tokens; activations exceed 6 GB at
  usable batch sizes. Dropping to 384px costs more accuracy than the backbone
  gains.
- **Data appetite.** ViTs lack locality and translation-equivariance priors.
  ViTDet's results come from MAE self-supervised pretraining plus 100 epochs
  on full COCO. On 16.5k images for 10 hours, a ViT lands below a pretrained
  CNN.
- **Precedent.** Every detector on the real-time speed/accuracy Pareto front —
  YOLOv8/v10/v11, RT-DETR — uses a convolutional backbone. RT-DETR applies its
  transformer only in the encoder, not the backbone.

Attention *is* worth it at low resolution late in the network. At P5 a 640px
input is 20×20 = 400 tokens, so global context is cheap exactly where it helps
most. This mirrors YOLOv11's C2PSA and RT-DETR's AIFI-on-P5.

### Key design choices

| Component | Choice | Rationale |
|---|---|---|
| Box regression | DFL, `reg_max=16` | Predicts a distribution over offsets; its expectation gives sub-cell accuracy and represents boundary ambiguity |
| Assignment | Task-Aligned Assigner, `topk=10`, `α=0.5`, `β=6.0` | Scores pairs by `s^α · u^β`, coupling classification confidence to localization quality. Positives exist from step 1 |
| Box loss | CIoU + DFL | Standard for this detector class |
| Cls loss | BCE over alignment-normalized soft targets | No objectness branch; class score doubles as confidence |
| Weight averaging | EMA, decay 0.9999 with ramp | Consistently worth 1–2 mAP |
| Precision | bf16 autocast | No loss scaling needed on sm_89; ~1.8× throughput |
| NMS | `torchvision.ops.batched_nms` | Replaces a per-class Python loop |

Backbones are swappable via `--backbone` (any `timm` name). Measured on the
RTX 4050 at 640px, batch 16:

| Backbone | Params | Throughput | Peak VRAM | Pretrained |
|---|---:|---:|---:|:--:|
| `builtin` (CSP) | 8.2M | 56.8 img/s | 1.87 GB | ✗ |
| **`resnet34`** | 24.4M | 32.1 img/s | 2.35 GB | ✓ |
| `efficientnet_b0` | 6.0M | 29.6 img/s | 3.88 GB | ✓ |
| `cspresnet50` | 26.7M | 26.1 img/s | 4.32 GB | ✓ |
| `convnext_tiny` | 32.2M | 14.2 img/s | 5.43 GB | ✓ |

---

## Getting started

Requires Python 3.12+ and a CUDA-capable GPU (CPU works but is impractically
slow for training).

```bash
pip install -r requirements.txt
```

Download PASCAL VOC (~2.9 GB, resumable):

```bash
python scripts/download_voc.py --root ./VOCdevkit
```

Train:

```bash
python train.py --dataset voc --data-root ./VOCdevkit --backbone resnet34 --size s --imgsz 640 --batch 16 --accumulate 4 --epochs 120 --eval-interval 20 --name voc120
```

Evaluate a checkpoint, with test-time augmentation and a per-class breakdown:

```bash
python val.py --weights runs/voc120/best.pt --dataset voc --data-root ./VOCdevkit --tta full --per-class
```

Run the tests:

```bash
python -m pytest tests/ -v
```

### Useful flags

| Flag | Purpose |
|---|---|
| `--accumulate N` | Gradient accumulation; effective batch = `batch × N`. Reaches large effective batches on small VRAM |
| `--time-budget H` | Stop cleanly after H hours, writing a resumable checkpoint |
| `--resume PATH` | Continue an interrupted run. Refuses to start if schedule- or architecture-critical flags differ |
| `--init-weights PATH` | Start a **new** schedule from existing weights (fine-tuning) |
| `--close-mosaic N` | Disable mosaic for the final N epochs |
| `--tta {off,flip,scale,full}` | Test-time augmentation at evaluation |
| `--secure-features` | Enable the random projection layer (**reduces mAP**) |

Training splits cleanly across sessions: `--time-budget` stops at a wall-clock
limit and `--resume` continues with the cosine schedule intact, since the
schedule is computed from the total epoch count rather than the session.

---

## Security layer

This project began as a security-oriented detector. The features below are
real and tested, but they operate at **checkpoint and API boundaries**, not
inside the forward pass.

### Implemented

- **Model encryption** — AES-256-GCM with device-bound AAD (`crypto/`)
- **Integrity registry** — per-parameter hashes, verified at load
  (`security/runtime_integrity.py`)
- **Hash-chain audit log** — HMAC-SHA256 chained entries (`security/secure_log.py`)
- **Watermarking** — parameter-space watermark with verification; measured
  90.5% bit accuracy at strength 0.001 (`model/watermark.py`)
- **Side-channel mitigations** — timing noise injection, CUDA schedule
  randomization (`security/side_channel.py`)

### Experimental

- Architecture morphing — disabled by default, no output-invariance guarantee
- Adversarial training and preprocessing — available by flag, off for benchmarks
- Structured pruning — feature-gated off
- Random feature projection — opt-in via `--secure-features`, **costs mAP**

### Stubs — not production implementations

- **SGX / SEV enclave backends are interface-level placeholders.** They define
  the API surface and do not provide hardware-backed isolation.
- Security benchmarks requiring a certified hardware trust chain are not
  implemented.

### Effect on accuracy

The original design applied a frozen random projection to P3/P4/P5 features
and called `.zero_()` on live activations during inference. Both were removed
from the forward pass: the projection degrades accuracy and only matters under
a threat model where an adversary observes intermediate activations, and
zeroing activations mutated tensors still referenced by autograd for no
benefit. See [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md) §4.

**Every benchmark number in this README was produced with
`secure_features=False`.** The security layer costs zero mAP.

---

## Project structure

```
model/
  detector.py        SecureYOLO: backbone + neck + head, predict, NMS, fusion
  backbone.py        pluggable timm / builtin CSP backbone, P5 attention
  neck.py            BiFPN with learnable weighted fusion
  head.py            decoupled anchor-free head, DFL, anchor generation
  assigner.py        Task-Aligned Assigner
  detection_loss.py  CIoU + DFL + BCE, fully vectorized
  losses.py          IoU-family metrics, focal / varifocal losses
data/
  dataset.py         COCO-JSON and VOC-XML readers, augmentation pipeline
  transforms.py      letterbox, mosaic, mixup, affine, HSV
  build.py           dataloader construction, COCO ground-truth export
evaluation/
  coco_eval.py       pycocotools mAP harness
  tta.py             test-time augmentation
security/, crypto/   encryption, integrity, audit logging, enclave stubs
tests/               27 tests, including regressions for each fixed bug
train.py             training entry point
val.py               evaluation entry point
scripts/             dataset downloaders
```

---

## Known limitations

- **Small-object AP is 0.14 against 0.72 for large objects.** This is the
  dominant term holding the headline number down. It is bounded by P3 stride
  and 640px input, not by training length.
- **Accuracy plateaued at epoch 40.** Epochs 40→78 moved mAP@0.5 by −0.04
  while training loss fell from 3.35 to 2.89 — fitting without generalizing.
  The limit is initialization and capacity, not schedule.
- **VOC only.** COCO support is implemented and tested but no COCO training
  run was performed; the 19 GB download plus ~10 h of training did not fit the
  compute budget.
- Enclave backends are stubs (see above).

### What would raise the number

| Change | Expected gain | Cost |
|---|---|---|
| COCO-pretrained detector initialization | +6–9 mAP | ~10 h COCO pretraining |
| Higher input resolution (896px) | +1.5–2.5, mostly small objects | full retrain, smaller batch |
| Larger backbone | +1–2 | ~14 h at 14 img/s |

---

## License

Research prototype. See repository for details.
