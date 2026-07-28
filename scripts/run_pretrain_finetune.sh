#!/usr/bin/env bash
# Two-stage COCO pretrain -> VOC fine-tune.
#
#   bash scripts/run_pretrain_finetune.sh
#
# Stage 1 trains on a class-balanced 25k subset of COCO train2017 (80 classes).
# Stage 2 fine-tunes on VOC, transferring every tensor including the
# classification head, which is sliced per class because all 20 VOC classes
# exist in COCO.
#
# Each stage has a wall-clock budget and writes resumable checkpoints, so an
# interrupted run continues with --resume rather than restarting.
#
# Total: roughly 11 hours. Safe to leave running overnight.

set -u  # undefined variables are errors; failures are handled per-stage

PY=${PY:-"py -3.14"}
COCO_ROOT=${COCO_ROOT:-./coco}
VOC_ROOT=${VOC_ROOT:-./VOCdevkit}
SUBSET=${SUBSET:-25000}

BACKBONE=${BACKBONE:-resnet34}
SIZE=${SIZE:-s}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-4}
WORKERS=${WORKERS:-4}

PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-30}
PRETRAIN_BUDGET=${PRETRAIN_BUDGET:-7}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-25}
FINETUNE_BUDGET=${FINETUNE_BUDGET:-4}

mkdir -p runs logs
STAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "logs/pipeline_${STAMP}.log"; }

# ── Stage 0: data ─────────────────────────────────────────────────────
if [ ! -f "${COCO_ROOT}/annotations/instances_minitrain2017.json" ]; then
  log "Building COCO subset (${SUBSET} images). Downloads only what it needs."
  $PY scripts/download_coco_subset.py --root "${COCO_ROOT}" --images "${SUBSET}" \
      2>&1 | tee -a "logs/subset_${STAMP}.log"
  if [ ! -f "${COCO_ROOT}/annotations/instances_minitrain2017.json" ]; then
    log "ERROR: subset build failed; stopping before training."
    exit 1
  fi
else
  log "COCO subset already present, skipping download."
fi

# ── Stage 1: COCO pretrain ────────────────────────────────────────────
log "Stage 1/2: COCO pretrain, ${PRETRAIN_EPOCHS} epochs, ${PRETRAIN_BUDGET}h budget"
$PY train.py \
  --dataset coco-minitrain --data-root "${COCO_ROOT}" \
  --backbone "${BACKBONE}" --size "${SIZE}" --imgsz "${IMGSZ}" \
  --batch "${BATCH}" --accumulate "${ACCUM}" --workers "${WORKERS}" \
  --epochs "${PRETRAIN_EPOCHS}" --lr 1e-3 --warmup-epochs 3 \
  --mosaic 1.0 --mixup 0.15 --close-mosaic 5 \
  --eval-interval 10 --save-interval 10 \
  --time-budget "${PRETRAIN_BUDGET}" --name coco_pretrain \
  2>&1 | tee -a "logs/stage1_${STAMP}.log"

if [ ! -f runs/coco_pretrain/best.pt ]; then
  log "ERROR: stage 1 produced no checkpoint; stopping."
  exit 1
fi
log "Stage 1 complete."

# ── Stage 2: VOC fine-tune ────────────────────────────────────────────
# Lower LR than training from scratch: the model already detects these
# objects, so the schedule only needs to adapt it to VOC's annotation style.
log "Stage 2/2: VOC fine-tune from COCO weights, ${FINETUNE_EPOCHS} epochs"
$PY train.py \
  --dataset voc --data-root "${VOC_ROOT}" \
  --backbone "${BACKBONE}" --size "${SIZE}" --imgsz "${IMGSZ}" \
  --batch "${BATCH}" --accumulate "${ACCUM}" --workers "${WORKERS}" \
  --epochs "${FINETUNE_EPOCHS}" --lr 4e-4 --warmup-epochs 1 \
  --mosaic 1.0 --mixup 0.1 --close-mosaic 8 \
  --eval-interval 5 --save-interval 10 \
  --time-budget "${FINETUNE_BUDGET}" --name voc_from_coco \
  --init-weights runs/coco_pretrain/best.pt \
  2>&1 | tee -a "logs/stage2_${STAMP}.log"

if [ ! -f runs/voc_from_coco/best.pt ]; then
  log "ERROR: stage 2 produced no checkpoint."
  exit 1
fi

# ── Final: scored with test-time augmentation ─────────────────────────
log "Final evaluation with multi-scale TTA"
$PY val.py --weights runs/voc_from_coco/best.pt \
  --dataset voc --data-root "${VOC_ROOT}" \
  --tta full --per-class --out results/voc_from_coco_tta.json \
  2>&1 | tee -a "logs/final_${STAMP}.log"

log "Pipeline complete."
log "Baseline to beat: 0.8264 mAP@.5 / 0.6343 mAP@[.5:.95] (results/voc_tta_full.json)"
