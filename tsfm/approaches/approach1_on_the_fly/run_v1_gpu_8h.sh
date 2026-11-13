#!/usr/bin/env bash
# V1, ~8 h on the local RX 6800 XT (ROCm, Windows), on-the-fly generation, then GIFT-Eval.
#
#     bash approaches/approach1_on_the_fly/run_v1_gpu_8h.sh          # from tsfm/; rerun to resume
#
# Needs .venv-rocm (scripts/setup_rocm_windows.ps1). Same recipe as runs/approach1/v1_long
# (train_v1.yaml, batch 64, lr 1e-3, cosine) with 7.5x the episodes. Measured before launch:
# ~1 100-1 150 episodes/s in bf16 with 12 workers, data wait 0 % -> 30M episodes ~ 7.5 h.
# bf16, not fp16: fp16 on this GPU diverged at batch 256 (non-finite weights after 25-125 steps).
# --stop-after-hours is a guard: if it fires, the checkpoint is kept and rerunning resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv-rocm/Scripts/python.exe
RUN=${RUN:-runs/approach1/v1_gpu_8h}
EPISODES=${EPISODES:-30000000}
mkdir -p "$RUN"

"$PY" -u scripts/train.py --model v1 --data-config configs/train_v1.yaml --run-dir "$RUN" \
    --total-episodes "$EPISODES" --batch-size 64 --num-workers 12 --amp-dtype bf16 \
    --stop-after-hours 8.5 --resume 2>&1 | tee -a "$RUN/train.log"

if [ -f "$RUN/eval.json" ]; then  # written only when the schedule is complete
    "$PY" -u scripts/eval_gift.py --model "$RUN/best.pt" --out-dir reports/gift_eval/approach1 2>&1 \
        | tee "$RUN/gift_eval.log"
fi
