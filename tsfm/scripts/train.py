"""Train a model (V0 point, V1 quantiles, V2 future tokens) on the on-the-fly synthetic stream.

    uv run python scripts/train.py                                   # full run (train_v0.yaml)
    uv run python scripts/train.py --model v1 --run-dir runs/v1      # quantile model (pinball loss)
    uv run python scripts/train.py --model v2 --data-config configs/train_v2.yaml --run-dir runs/v2   # horizons to 1024
    uv run python scripts/train.py --total-episodes 20000 --run-dir runs/smoke
    uv run python scripts/train.py --run-dir runs/v0 --resume        # continue from last.pt
    uv run python scripts/train.py --run-dir runs/v1 --stop-after-hours 11 [--resume]  # time-limited session

Every TrainConfig field is a flag (``--batch-size``, ``--lr``, ``--val-suites val_id val_ood_noise`` ...).
After training, all val_* suites are scored and written to <run_dir>/eval.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import typing
from pathlib import Path

from tsfm.training.trainer import Trainer, TrainConfig


def _add_flags(ap: argparse.ArgumentParser) -> None:
    hints = typing.get_type_hints(TrainConfig)
    for f in dataclasses.fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        hint = hints[f.name]
        if hint is bool:
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif typing.get_origin(hint) is list:
            ap.add_argument(flag, nargs="*", default=default)
        else:
            base = next((t for t in typing.get_args(hint) if t is not type(None)), hint)
            ap.add_argument(flag, type=base, default=default)


def _keep_awake() -> None:
    """Windows: no sleep while this process runs (long unattended runs); the display may still turn off."""
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main() -> None:
    _keep_awake()
    ap = argparse.ArgumentParser()
    _add_flags(ap)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-final-eval", action="store_true")
    args = vars(ap.parse_args())
    resume = args.pop("resume")
    final_eval = not args.pop("no_final_eval")

    trainer = Trainer(TrainConfig(**args), resume=resume)
    trainer.train()
    if trainer.step < trainer.total_steps:
        print(f"session stopped at step {trainer.step}/{trainer.total_steps}; continue with --resume")
    elif final_eval:
        root = Path(trainer.cfg.val_root)
        suites = sorted(p.name for p in root.iterdir() if p.name.startswith("val_") and (p / "manifest.json").exists())
        print("final evaluation (full suites):")
        results = trainer.evaluate(suites, max_batches=None)
        (trainer.run_dir / "eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"-> {trainer.run_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
