#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT=runs/smoke
mkdir -p "$ROOT"
TIMESTEPS=${TIMESTEPS:-100000}
ENVS=${ENVS:-4}
PREV=""

for STAGE in 0 1 2 3; do
  OUT="$ROOT/stage${STAGE}"
  RESUME=""
  if [ -n "$PREV" ]; then
    RESUME="--resume $PREV"
  fi
  echo "=== stage ${STAGE} -> ${OUT} ==="
  python train_sac.py \
    --stage "$STAGE" \
    --timesteps "$TIMESTEPS" \
    --n-envs "$ENVS" \
    --buffer-size 200000 \
    --learning-starts 5000 \
    --eval-freq 10000 \
    --checkpoint-freq 25000 \
    --eval-episodes 5 \
    --device cpu \
    $RESUME \
    --out "$OUT" \
    2>&1 | tee "$ROOT/stage${STAGE}.log"

  BEST="$OUT/best/best_model.zip"
  if [ ! -f "$BEST" ]; then
    BEST="$OUT/sac_stage${STAGE}.zip"
  fi
  python evaluate.py \
    --model "$BEST" \
    --stage "$STAGE" \
    --episodes 5 \
    --out "$ROOT/stage${STAGE}_eval.json" \
    2>&1 | tee "$ROOT/stage${STAGE}_eval.log"

  if ! python record_video.py \
    --model "$BEST" \
    --stage "$STAGE" \
    --out "$ROOT/stage${STAGE}.mp4" \
    --gif "$ROOT/stage${STAGE}.gif" \
    --seconds 8 \
    --width 640 \
    --height 360 \
    --action-scale 0.5; then
    echo "video rendering failed (non-fatal)"
  fi

  PREV="$BEST"
done

python plot_training.py --runs "$ROOT" --out "$ROOT/training_curve.png" || true
