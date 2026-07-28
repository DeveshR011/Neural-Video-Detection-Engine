# COCO Pretrain → VOC Fine-tune

Runbook for the two-stage schedule that raises VOC accuracy past what
ImageNet-only initialization can reach.

## Why this helps

The current 82.6 mAP@0.5 comes from an ImageNet-**classification** backbone
with the detection head trained from random init on 16,551 VOC images. The
model has to learn what an object *is* from VOC alone.

Pretraining on COCO first means the detector already localizes and classifies
before it ever sees VOC. Every VOC class exists in COCO, so nothing has to be
learned from scratch during fine-tuning.

**Class-sliced head transfer.** The usual approach discards the classification
head when the class count changes and re-initializes it randomly. That throws
away exactly the knowledge worth keeping. `model/transfer.py` instead maps each
VOC class to its COCO counterpart and copies the corresponding weight rows:

```
aeroplane   -> airplane      motorbike   -> motorcycle
diningtable -> dining table  pottedplant -> potted plant
sofa        -> couch         tvmonitor   -> tv
```

All 20 map, so **zero tensors are left randomly initialized** — verified by
`test_transfer_copies_all_tensors_and_slices_classes`.

## Expected outcome

| | mAP@0.5 |
|---|---:|
| Current (ImageNet backbone + TTA) | 82.6 |
| **After this pipeline** | **85–88** |
| Full-COCO pretrained detectors (YOLOv5/v8 fine-tuned) | 88–92 |

**85–88 is the honest range; 88 is the optimistic end.** The gap to 88–92
remains because those models pretrain on all 118k COCO images for ~500 epochs
across multiple GPUs — several hundred GPU-hours. This pipeline uses 25k images
for ~30 epochs in 7 hours, so it captures part of that benefit, not all of it.

Reaching the top of the range reliably would need a full-COCO pretrain, which
is a cloud-GPU job (~20–30 A100-hours), not a laptop one.

## Running it

One command, ~11 hours, safe to leave overnight:

```bash
bash scripts/run_pretrain_finetune.sh
```

Tunable through the environment:

```bash
SUBSET=35000 PRETRAIN_EPOCHS=35 PRETRAIN_BUDGET=8 bash scripts/run_pretrain_finetune.sh
```

### Time budget

| Stage | Work | Time |
|---|---|---|
| 0 | COCO subset download (~4 GB, 25k images) | ~40 min |
| 1 | COCO pretrain, 30 epochs @ ~14 min | ~7 h |
| 2 | VOC fine-tune, 25 epochs @ ~8.6 min | ~3.6 h |
| 3 | Final multi-scale TTA evaluation | ~10 min |

The subset builder downloads only the images it selects — about 4 GB rather
than the full 19 GB train2017 archive — and skips files already present, so an
interrupted download resumes.

### If it stops early

Both stages write `last.pt` every epoch and enforce a wall-clock budget. To
continue a stage, repeat its command with `--resume`:

```bash
py -3.14 train.py --dataset coco-minitrain --data-root ./coco --backbone resnet34 --size s --imgsz 640 --batch 16 --accumulate 4 --epochs 30 --name coco_pretrain --resume runs/coco_pretrain/last.pt
```

Every other flag must match the original invocation — resume compares
schedule- and architecture-critical flags against the checkpoint and refuses to
continue on a mismatch, since a changed schedule would silently alter the
learning-rate trajectory.

## Choices worth knowing

**Subset size (25k).** Selected class-balanced rather than uniformly at random.
A random COCO subset badly under-samples rare classes, and mAP averages over
all 80 equally, so balancing matters more than raw count. Selection is seeded,
so a reported number ties to a specific image set.

**Fine-tune LR is 4e-4, not 1e-3.** The model already detects these objects;
the schedule only needs to adapt it to VOC's annotation conventions. Too high
an LR destroys the pretrained representation — a 1.5e-4 fine-tune with warmup
that ramped to 3e-4 was measured *1.3 points below* its starting checkpoint at
epoch 3, because the warmup knocked the model out of its converged minimum.

**Stage 1 evaluates on COCO val2017.** Expect roughly 25–30 mAP@[.5:.95] there.
That number is not comparable to published COCO results, which train on all
118k images — it exists only to confirm the pretrain is working before stage 2
consumes it.

## Verifying before committing a night to it

A short dry run confirms both stages start and the transfer engages:

```bash
py -3.14 scripts/download_coco_subset.py --root ./coco --images 500
py -3.14 train.py --dataset coco-minitrain --data-root ./coco --epochs 1 --batch 8 --eval-interval 1 --name smoke_coco
py -3.14 train.py --dataset voc --data-root ./VOCdevkit --epochs 1 --batch 8 --eval-interval 1 --name smoke_voc --init-weights runs/smoke_coco/best.pt
```

Stage 2's log should report:

```
Class transfer: 20/20 destination classes matched a source class
Tensors: N copied, 6 class-sliced, 0 left initialized
```

`0 left initialized` is the line that matters. Anything else means the transfer
silently fell back to random weights for part of the model.
