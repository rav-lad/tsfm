"""Training loop: schedule, a tiny run, exact resume, frozen-suite evaluation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tsfm.data import SyntheticEpisodeSource, load_config
from tsfm.data.frozen import write_suite
from tsfm.models import V1_QUANTILES, PatchTransformerV0, PatchV0Config
from tsfm.training.evaluate import evaluate_suite
from tsfm.training.trainer import Trainer, TrainConfig, load_model, lr_factor

CONFIG = Path(__file__).parents[2] / "configs" / "train_v0.yaml"
TINY = PatchV0Config(n_layers=1, d_model=64, n_heads=2, ffn_hidden=128, embed_hidden=64, head_hidden=128)


def _cfg(run_dir: Path, **kw) -> TrainConfig:
    base = dict(data_config=str(CONFIG), run_dir=str(run_dir), total_episodes=6 * 8, batch_size=8,
                num_workers=0, warmup_steps=2, log_every=1, eval_every=1000, val_suites=[])
    return TrainConfig(**{**base, **kw})


def test_lr_schedule():
    assert lr_factor(0, 10, 100, 0.1) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100, 0.1) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100, 0.1) == pytest.approx(0.1)
    assert lr_factor(55, 10, 100, 0.1) == pytest.approx(0.55)


def test_tiny_run_logs_and_checkpoints(tmp_path):
    t = Trainer(_cfg(tmp_path), TINY)
    t.train()
    assert t.step == 6
    lines = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == list(range(1, 7))
    assert all(0.0 <= r["data_wait_frac"] <= 1.0 for r in lines)
    assert (tmp_path / "last.pt").exists() and (tmp_path / "run.json").exists()


def test_resume_is_exact(tmp_path):
    straight = Trainer(_cfg(tmp_path / "a"), TINY)
    straight.train()

    first = Trainer(_cfg(tmp_path / "b", stop_after_steps=3), TINY)
    first.train()
    assert first.step == 3
    second = Trainer(_cfg(tmp_path / "b"), TINY, resume=True)
    assert second.step == 3
    second.train()

    for (k, a), b in zip(straight.model.state_dict().items(), second.model.state_dict().values()):
        assert torch.equal(a, b), k


def test_evaluate_suite_reports_baselines(tmp_path):
    src = SyntheticEpisodeSource(load_config(CONFIG), split="val_id")
    write_suite(tmp_path / "suite", [src.episode(i) for i in range(20)], {"name": "test"})
    cache: dict = {}
    res = evaluate_suite(PatchTransformerV0(TINY), tmp_path / "suite", batch_size=8, cache=cache)
    assert res["n_series"] == 20
    for k in ("mase_model", "mase_naive", "mase_seasonal_naive", "mase_best", "win_rate_vs_best"):
        assert k in res
    assert res["mase_best"] <= res["mase_seasonal_naive"]
    again = evaluate_suite(PatchTransformerV0(TINY), tmp_path / "suite", batch_size=8, cache=cache)
    assert again["mase_best"] == res["mase_best"]  # baselines served from the cache


def test_v1_trains_with_pinball_and_reloads(tmp_path):
    t = Trainer(_cfg(tmp_path, model="v1"), replace(TINY, quantiles=V1_QUANTILES))
    t.train()
    model = load_model(tmp_path / "last.pt")
    assert model.quantile_levels == V1_QUANTILES
    for a, b in zip(t.model.state_dict().values(), model.state_dict().values()):
        assert torch.equal(a.cpu(), b)


def test_unknown_model_preset(tmp_path):
    with pytest.raises(ValueError, match="preset"):
        Trainer(_cfg(tmp_path, model="v9"))


def test_evaluate_suite_quantile_metrics(tmp_path):
    src = SyntheticEpisodeSource(load_config(CONFIG), split="val_id")
    write_suite(tmp_path / "suite", [src.episode(i) for i in range(20)], {"name": "test"})
    res = evaluate_suite(PatchTransformerV0(replace(TINY, quantiles=V1_QUANTILES)), tmp_path / "suite", batch_size=8)
    assert res["sql_model"] > 0
    assert 0.0 <= res["coverage"] <= 1.0
    assert res["coverage_nominal"] == pytest.approx(0.8)
