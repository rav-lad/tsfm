"""Sprint D4 tests: holdout exclusions, frozen suites, reproducibility (§11, §16 D4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tsfm.data.config import deep_merge, load_config
from tsfm.data.dataset import SyntheticEpisodeSource, curriculum_probability
from tsfm.data.evaluation import baseline_forecasts, pinball_loss, weighted_quantile_loss
from tsfm.data.frozen import FrozenEpisodeSet
from tsfm.data.holdouts import build_suite, suite_configs
from tsfm.data.rng import split_code

ROOT = Path(__file__).parents[2]
HOLDOUT = ROOT / "configs" / "holdout_v1.yaml"
TRAIN = load_config(ROOT / "configs" / "data_v1.yaml")


@pytest.fixture(scope="module")
def suites():
    return suite_configs(HOLDOUT)


def _periods(meta) -> list[float]:
    lat = meta["latent_parameters"]
    if meta["family"] == "trend_seasonal":
        return [s["period"] for s in lat["seasonalities"]]
    if meta["family"] == "complex_periodic":
        if "components" in lat:
            return [c["period"] for c in lat["components"]]
        return [lat.get("period", lat.get("initial_period"))]
    return []


def test_training_never_sees_held_out_kinds_or_period_band():
    src = SyntheticEpisodeSource(deep_merge(TRAIN, {"curriculum": {"milestones": None}}), "train")
    lo, hi = TRAIN["prior"]["period_ratio_band"]
    for i in range(1500):
        ep = src.episode(i)
        assert ep.meta["kind"] not in ("chirp", "bursts")
        for P in _periods(ep.meta):
            assert not lo <= P / ep.context_length <= hi, (ep.meta["kind"], P, ep.context_length)
        assert ep.meta.get("latent_parameters", {}).get("noise_regime") != "extreme_noise"
        for e in ep.meta.get("graph", {"edges": []})["edges"]:
            assert e["op"] != "cointegration"


def test_ood_suites_contain_only_held_out_structure(suites):
    _, s = suites
    kinds = SyntheticEpisodeSource(s["val_ood_kinds"]["config"], s["val_ood_kinds"]["split"])
    assert {kinds.episode(i).meta["kind"] for i in range(100)} == {"chirp", "bursts"}

    per = SyntheticEpisodeSource(s["val_ood_period"]["config"], s["val_ood_period"]["split"])
    lo, hi = TRAIN["prior"]["period_ratio_band"]
    ratios = [P / ep.context_length for ep in (per.episode(i) for i in range(100)) for P in _periods(ep.meta)]
    assert ratios and all(lo - 1e-9 <= r <= hi + 1e-9 for r in ratios)

    noise = SyntheticEpisodeSource(s["val_ood_noise"]["config"], s["val_ood_noise"]["split"])
    assert {noise.episode(i).meta["latent_parameters"]["noise_regime"] for i in range(50)} == {"extreme_noise"}

    hor = SyntheticEpisodeSource(s["val_ood_horizon"]["config"], s["val_ood_horizon"]["split"])
    assert {hor.episode(i).prediction_length for i in range(20)} == {384}

    rel = SyntheticEpisodeSource(s["val_ood_relation"]["config"], s["val_ood_relation"]["split"])
    ops = {e["op"] for i in range(50) for e in rel.episode(i).meta["graph"]["edges"]}
    assert "cointegration" in ops


def test_suites_have_disjoint_seed_namespaces(suites):
    _, s = suites
    codes = [split_code(v["split"]) for v in s.values()] + [split_code("train")]
    assert len(set(codes)) == len(codes)
    seeds = {}
    for name, v in s.items():
        src = SyntheticEpisodeSource(v["config"], v["split"])
        seeds[name] = {src.episode(i).meta["record_seed"] for i in range(30)}
    train = SyntheticEpisodeSource(deep_merge(TRAIN, {"curriculum": {"milestones": None}}), "train")
    seeds["train"] = {train.episode(i).meta["record_seed"] for i in range(30)}
    names = list(seeds)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            assert not seeds[names[a]] & seeds[names[b]], (names[a], names[b])


def test_frozen_roundtrip_and_reproducibility(tmp_path, suites):
    spec, s = suites
    m1 = build_suite("val_mv_id", s["val_mv_id"], spec, tmp_path / "a", n=40)
    m2 = build_suite("val_mv_id", s["val_mv_id"], spec, tmp_path / "b", n=40)
    assert m1["content_sha256"] == m2["content_sha256"]  # rebuild is bit-identical
    ds = FrozenEpisodeSet(tmp_path / "a" / "val_mv_id")
    src = SyntheticEpisodeSource(s["val_mv_id"]["config"], s["val_mv_id"]["split"])
    assert len(ds) == 40
    for i in (0, 17, 39):
        a, b = ds[i], src.episode(i)
        assert np.array_equal(a.values, b.values)
        assert np.array_equal(a.observed_mask, b.observed_mask)
        assert np.array_equal(a.loss_mask, b.loss_mask)
        assert np.array_equal(a.roles, b.roles)
        assert a.meta["record_seed"] == b.meta["record_seed"]
    assert m1["config_sha256"] and m1["seed"] == TRAIN["seed"]


def test_frozen_detects_tampering(tmp_path, suites):
    spec, s = suites
    build_suite("val_id", s["val_id"], spec, tmp_path, n=10)
    meta = tmp_path / "val_id" / "meta.jsonl"
    meta.write_text(meta.read_text(encoding="utf-8").replace('"split"', '"spl1t"', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        FrozenEpisodeSet(tmp_path / "val_id")


def test_curriculum_interpolation():
    ms = TRAIN["curriculum"]["milestones"]
    assert curriculum_probability(ms, 0.0) == pytest.approx(0.10)
    assert curriculum_probability(ms, 0.25) == pytest.approx(0.15)
    assert curriculum_probability(ms, 1.0) == pytest.approx(0.40)
    src = SyntheticEpisodeSource(TRAIN, "train")
    total = TRAIN["curriculum"]["total_episodes"]
    early = np.mean([src.episode(i).n_channels > 1 for i in range(400)])
    late = np.mean([src.episode(total - 1 - i).n_channels > 1 for i in range(400)])
    assert 0.04 < early < 0.18 and 0.30 < late < 0.50


def test_metrics_sanity():
    q = np.array([0.1, 0.5, 0.9])
    y = np.zeros(10)
    perfect = np.zeros((10, 3))
    assert np.allclose(pinball_loss(y, perfect, q), 0)
    assert pinball_loss(y, perfect + 1, q)[0] == pytest.approx(0.9)
    assert weighted_quantile_loss(np.ones(5), np.ones((5, 3)), q) == 0
    hist = np.sin(2 * np.pi * np.arange(240) / 24)
    sn = baseline_forecasts(hist, 48)["seasonal_naive"]
    assert np.allclose(sn, np.sin(2 * np.pi * np.arange(240, 288) / 24), atol=1e-9)
