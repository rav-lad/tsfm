"""Minimal training loop for models V0 / V1 (roadmap phase 3, step B).

- Loss: MAE for a point model (V0), pinball over the quantile levels for a
  quantile model (V1), both on the normalized scale. ``loss_weighting`` (quantile
  models) multiplies per-position weights: ``harmonic`` (1/h, Timer-M1) and/or
  ``naive_scale`` (1 / one-step variability of the history, MASE-like).

- Data: infinite on-the-fly stream; ``step`` = batch index, so resuming is
  ``start_batch = step`` and reproduces the exact remaining stream.
- AdamW (no weight decay on norms / biases), linear warmup + cosine decay,
  gradient clipping, autocast on CUDA (bf16, or fp16 + loss scaling on GPUs
  without bf16 such as Kaggle's T4 / P100).
- Logs (``metrics.jsonl``): loss, lr, grad norm, episodes/s and the fraction of
  wall time spent waiting for the DataLoader (the number that says whether
  on-the-fly generation is free).
- A step with a non-finite gradient is skipped (counted in ``skipped_steps``);
  ``MAX_CONSECUTIVE_SKIPS`` in a row abort the run with ``crash.pt``.
- Periodic evaluation on frozen suites, ``last.pt`` / ``best.pt`` checkpoints.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from tsfm.data import load_config
from tsfm.data.config import config_hash
from tsfm.models import MODEL_PRESETS, PatchTransformerV0, PatchV0Config
from tsfm.models.patch_v0 import forecast_origin

from .evaluate import evaluate_suite
from .loader import make_train_loader, split_batch, to_torch
from .losses import harmonic_horizon_weights, masked_mae, masked_pinball, naive_scale_weights

LOSS_WEIGHTINGS = ("harmonic", "naive_scale")
MAX_CONSECUTIVE_SKIPS = 20  # non-finite steps in a row before giving up


@dataclass
class TrainConfig:
    model: str = "v0"  # architecture preset, see tsfm.models.MODEL_PRESETS
    data_config: str = "configs/train_v0.yaml"
    run_dir: str = "runs/v0"
    total_episodes: int | None = None  # default: curriculum.total_episodes of the data config
    batch_size: int = 64
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 8
    device: str = "auto"
    amp: bool = True  # autocast, CUDA only
    amp_dtype: str = "auto"  # auto | bf16 | fp16 (auto: bf16 if the GPU supports it)
    seed: int = 0
    log_every: int = 50
    eval_every: int = 1000
    eval_max_batches: int | None = 20  # periodic eval on the first 20 x 64 episodes
    stop_after_steps: int | None = None  # end this session early (time-limited GPUs); resume later
    stop_after_hours: float | None = None  # same, by wall time (checkpoint at the step it is reached)
    val_root: str = "data/holdouts/holdout_v1"
    val_suites: list[str] = field(default_factory=lambda: ["val_id"])
    best_metric_suite: str = "val_id"
    loss_weighting: list[str] = field(default_factory=list)  # subset of LOSS_WEIGHTINGS


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> PatchTransformerV0:
    """Rebuild the model of a ``last.pt`` / ``best.pt`` checkpoint (eval mode)."""
    ckpt = torch.load(checkpoint, map_location=device)
    mcfg = dict(ckpt["model_config"])
    mcfg["quantiles"] = tuple(mcfg.get("quantiles", ()))
    model = PatchTransformerV0(PatchV0Config(**mcfg)).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


class Trainer:
    def __init__(self, cfg: TrainConfig, model_cfg: PatchV0Config | None = None, resume: bool = False) -> None:
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(cfg.device)
        self.data_config = load_config(cfg.data_config)
        total = cfg.total_episodes or self.data_config["curriculum"]["total_episodes"]
        self.total_steps = int(total) // cfg.batch_size

        torch.manual_seed(cfg.seed)
        if model_cfg is None and cfg.model not in MODEL_PRESETS:
            raise ValueError(f"unknown model preset {cfg.model!r}: {sorted(MODEL_PRESETS)}")
        self.model_cfg = model_cfg or MODEL_PRESETS[cfg.model]
        unknown = set(cfg.loss_weighting) - set(LOSS_WEIGHTINGS)
        if unknown:
            raise ValueError(f"unknown loss_weighting {sorted(unknown)}: {LOSS_WEIGHTINGS}")
        self.model = PatchTransformerV0(self.model_cfg).to(self.device)
        self.opt = make_optimizer(self.model, cfg.lr, cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_factor(s, cfg.warmup_steps, self.total_steps, cfg.min_lr_ratio)
        )
        self.amp_dtype = resolve_amp_dtype(cfg.amp_dtype) if cfg.amp and self.device.type == "cuda" else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype is torch.float16)
        self.step = 0
        self.skipped_steps = 0
        self.best = math.inf
        self.eval_cache: dict[str, Any] = {}
        if resume and (self.run_dir / "last.pt").exists():
            self._load(self.run_dir / "last.pt")
        elif not resume:
            (self.run_dir / "metrics.jsonl").unlink(missing_ok=True)
        self._write_run_info()

    # ------------------------------------------------------------------ io
    def _write_run_info(self) -> None:
        info = {
            "train_config": asdict(self.cfg),
            "model_config": asdict(self.model_cfg),
            "data_config": self.data_config,
            "data_config_sha256": config_hash(self.data_config),
            "n_parameters": self.model.num_parameters(),
            "total_steps": self.total_steps,
            "device": str(self.device),
            "amp_dtype": str(self.amp_dtype),
            "torch": torch.__version__,
        }
        (self.run_dir / "run.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")

    def _log(self, record: dict[str, Any]) -> None:
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save(self, name: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step,
                "best": self.best,
                "model_config": asdict(self.model_cfg),
                "train_config": asdict(self.cfg),
            },
            self.run_dir / name,
        )

    def _load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.sched.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step, self.best = ckpt["step"], ckpt["best"]
        print(f"resumed from {path} at step {self.step}")

    # ------------------------------------------------------------------ loop
    def loss(self, out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor],
             inputs: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        y, mask = labels["target_future"], labels["loss_mask"]
        if self.model.quantile_levels:
            w = None
            if "harmonic" in self.cfg.loss_weighting:
                w = harmonic_horizon_weights(y.shape[-1], y.device)
            if "naive_scale" in self.cfg.loss_weighting:
                origin = forecast_origin(inputs["padding_mask"], inputs["lengths"])
                ws = naive_scale_weights(inputs["input_values"], inputs["input_mask"], inputs["padding_mask"], origin)
                w = ws if w is None else w * ws
            return masked_pinball(out["quantiles"].float(), y, mask, self.model.quantile_levels, w)
        if self.cfg.loss_weighting:
            raise ValueError("loss_weighting needs a quantile model")
        return masked_mae(out["prediction"].float(), y, mask)

    def evaluate(self, suites: list[str], max_batches: int | None) -> dict[str, dict[str, Any]]:
        out = {}
        for name in suites:
            res = evaluate_suite(self.model, Path(self.cfg.val_root) / name, self.device,
                                 self.cfg.batch_size, max_batches, self.eval_cache)
            out[name] = res
            self._log({"step": self.step, "eval": name, **{k: v for k, v in res.items() if k != "by_family"}})
            print(f"  eval {name:<16} MASE model {res['mase_model']:.3f} | seasonal naive "
                  f"{res['mase_seasonal_naive']:.3f} | best baseline {res['mase_best']:.3f} | "
                  f"win rate {res['win_rate_vs_best']:.0%}"
                  + (f" | SQL {res['sql_model']:.3f} | coverage {res['coverage']:.0%}"
                     f" (nominal {res['coverage_nominal']:.0%})" if "sql_model" in res else ""))
        return out

    def _periodic_eval(self) -> None:
        if not self.cfg.val_suites:
            return
        res = self.evaluate(self.cfg.val_suites, self.cfg.eval_max_batches)
        score = res.get(self.cfg.best_metric_suite, {}).get("mase_model")
        if score is not None and score < self.best:
            self.best = score
            self.save("best.pt")

    def train(self) -> None:
        cfg = self.cfg
        if self.step >= self.total_steps:
            print("nothing to do: training already finished")
            return
        loader = make_train_loader(cfg.data_config, cfg.batch_size, cfg.num_workers,
                                   start_batch=self.step, num_batches=self.total_steps - self.step)
        stop_at = self.total_steps
        if cfg.stop_after_steps is not None:
            stop_at = min(stop_at, self.step + cfg.stop_after_steps)
        deadline = time.time() + cfg.stop_after_hours * 3600 if cfg.stop_after_hours is not None else math.inf
        print(f"device {self.device} | {self.model.num_parameters():,} params | "
              f"steps {self.step} -> {self.total_steps} | batch {cfg.batch_size} | workers {cfg.num_workers} | "
              f"amp {self.amp_dtype}")

        self.model.train()
        it = iter(loader)
        win_t0, win_wait, win_steps, win_skips = time.perf_counter(), 0.0, 0, 0
        consecutive_skips = 0
        loss_sum = torch.zeros((), device=self.device)
        grad_sum = torch.zeros((), device=self.device)
        while self.step < stop_at:
            t_wait = time.perf_counter()
            batch = next(it)
            win_wait += time.perf_counter() - t_wait

            inputs, labels = split_batch(to_torch(batch, self.device))
            with torch.autocast(self.device.type, dtype=self.amp_dtype or torch.bfloat16,
                                enabled=self.amp_dtype is not None):
                out = self.model(**inputs)
            loss = self.loss(out, labels, inputs)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            finite = self.scaler.is_enabled() or bool(torch.isfinite(grad_norm))
            if finite:
                self.scaler.step(self.opt)  # skipped on inf/nan gradients when fp16 loss scaling is on
                self.scaler.update()
                consecutive_skips = 0
            else:  # rare non-finite step (seen once on ROCm with V2): drop it, keep the weights
                self.skipped_steps += 1
                consecutive_skips += 1
                win_skips += 1
                if consecutive_skips > MAX_CONSECUTIVE_SKIPS:
                    self.save("crash.pt")
                    raise FloatingPointError(f"{consecutive_skips} non-finite steps in a row at step {self.step}")
            self.sched.step()
            self.step += 1
            win_steps += 1
            if time.time() >= deadline:
                stop_at = self.step  # the log / eval / checkpoint branches below fire on this step
            if finite:
                loss_sum += loss.detach()
                grad_sum += grad_norm.detach()

            if self.step % cfg.log_every == 0 or self.step == stop_at:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - win_t0
                kept = max(win_steps - win_skips, 1)
                mean_loss = (loss_sum / kept).item()
                if not math.isfinite(mean_loss):
                    self.save("crash.pt")
                    raise FloatingPointError(f"non-finite loss at step {self.step} (state saved to crash.pt)")
                rec = {
                    "step": self.step,
                    "loss": mean_loss,
                    "grad_norm": (grad_sum / kept).item(),
                    "skipped_steps": win_skips,
                    "lr": self.sched.get_last_lr()[0],
                    "episodes_per_s": win_steps * cfg.batch_size / dt,
                    "data_wait_frac": win_wait / dt,
                }
                self._log(rec)
                print(f"step {self.step:>6}/{self.total_steps} | loss {rec['loss']:.4f} | "
                      f"grad {rec['grad_norm']:.2f} | lr {rec['lr']:.2e} | "
                      f"{rec['episodes_per_s']:.0f} ep/s | data wait {rec['data_wait_frac']:.0%}"
                      + (f" | skipped {win_skips} non-finite" if win_skips else ""))
                win_t0, win_wait, win_steps, win_skips = time.perf_counter(), 0.0, 0, 0
                loss_sum.zero_()
                grad_sum.zero_()

            if self.step % cfg.eval_every == 0 or self.step == stop_at:
                t_eval = time.perf_counter()
                self._periodic_eval()
                self.save("last.pt")
                win_t0 += time.perf_counter() - t_eval  # eval time is not training time
