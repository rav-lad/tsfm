#!/usr/bin/env bash
# Model V2 (future patch tokens, horizons up to 1024, contexts up to 2048) on the local GPU, fp32.
#
#     bash approaches/approach1_on_the_fly/run_v2.sh [name] [data config] [episodes]   # from tsfm/; rerun to resume
#
# Default: 4M episodes (the v1_long budget) for a first comparison with
# v1_long_best_sym (GIFT rel. MASE 0.963), then GIFT-Eval with 2048-point contexts.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv-rocm/Scripts/python.exe
NAME=${1:-v2}
DATA=${2:-configs/train_v2.yaml}
EPISODES=${3:-4000000}
DIR=runs/approach1/$NAME
OUT=reports/gift_eval/approach1
mkdir -p "$DIR"

"$PY" -u scripts/train.py --model v2 --data-config "$DATA" --run-dir "$DIR" --total-episodes "$EPISODES" \
    --batch-size 64 --num-workers 12 --no-amp --resume 2>&1 | tee -a "$DIR/train.log"
if [ -f "$DIR/eval.json" ]; then
    "$PY" -u scripts/eval_gift.py --model "$DIR/best.pt" --out-dir "$OUT" --max-context 2048 2>&1 | tee "$DIR/gift_eval.log"
fi
