#!/usr/bin/env bash
# Phase-1 ablations (2026-11-13): what fixes level anchoring? Each run = v1_long recipe
# (train_v1.yaml, V1, 4M episodes, batch 64, lr 1e-3) on the local GPU in fp32
# (fp32 is faster than bf16 on the RX 6800 XT: ~2 060 vs ~1 490 ep/s), ~35 min, then GIFT-Eval.
#
#     bash approaches/approach1_on_the_fly/run_ablations_p1.sh      # from tsfm/; rerun to resume
#
# Baseline: runs/approach1/v1_long (same recipe, CPU fp32), re-scored with symmetric averaging.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv-rocm/Scripts/python.exe
OUT=reports/gift_eval/approach1

run() {  # name, data config, extra train flags...
    local name=$1 data=$2; shift 2
    local dir=runs/approach1/$name
    mkdir -p "$dir"
    "$PY" -u scripts/train.py --model v1 --data-config "$data" --run-dir "$dir" --total-episodes 4000000 \
        --batch-size 64 --num-workers 12 --no-amp --resume "$@" 2>&1 | tee -a "$dir/train.log"
    if [ -f "$dir/eval.json" ] && [ ! -f "$OUT/${name}_best_sym.md" ]; then
        "$PY" -u scripts/eval_gift.py --model "$dir/best.pt" --out-dir "$OUT" 2>&1 | tee "$dir/gift_eval.log"
    fi
}

[ -f "$OUT/v1_long_best_sym.md" ] || "$PY" -u scripts/eval_gift.py --model runs/approach1/v1_long/best.pt --out-dir "$OUT"

run abl_harmonic configs/train_v1.yaml --loss-weighting harmonic
run abl_naive_scale configs/train_v1.yaml --loss-weighting naive_scale
run abl_aug configs/train_v1_aug.yaml
