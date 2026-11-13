"""G5 (regime SDE) and G7 (composite) specific tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsfm.data.config import deep_merge, load_config
from tsfm.data.dataset import SyntheticEpisodeSource
from tsfm.data.episode import Support
from tsfm.data.generators import CompositeGenerator, RegimeSDEGenerator
from tsfm.data.generators.regime_sde import _ou_path, markov_regimes


def test_markov_switch_rate_matches_p_stay():
    rng = np.random.default_rng(0)
    for p in (0.9, 0.98, 0.995):
        r = markov_regimes(rng, 200_000, 3, p)
        rate = (np.diff(r) != 0).mean()
        assert abs(rate - (1 - p)) < 0.15 * (1 - p) + 2e-4
        assert set(np.unique(r)) == {0, 1, 2}


def test_exact_ou_stationary_variance():
    rng = np.random.default_rng(1)
    n, theta, sigma = 200_000, 0.05, 0.7
    y = _ou_path(0.0, np.zeros(n), np.full(n, theta), np.full(n, sigma), rng.standard_normal(n))
    assert abs(y[1000:].var() / (sigma**2 / (2 * theta)) - 1) < 0.1


def test_every_sde_kind_is_stable_and_positive_when_declared():
    for kind in ("regime_ou", "regime_drift", "stochastic_vol", "jump_diffusion", "regime_gbm"):
        gen = RegimeSDEGenerator(kinds={kind: 1.0})
        for s in range(40):
            rec = gen.sample(np.random.default_rng(s), 2000, 1024)
            assert np.all(np.isfinite(rec.values)) and np.abs(rec.values).max() < 1e8, kind
            if rec.metadata["support"] == Support.NONNEGATIVE:
                assert rec.values.min() >= 0


def test_regime_metadata_reflects_configured_persistence():
    gen = RegimeSDEGenerator(kinds={"regime_ou": 1.0}, regime_stay_probability_range=(0.995, 0.999))
    sw = [gen.sample(np.random.default_rng(s), 2000, 512).metadata["latent_parameters"]["n_switches"] for s in range(50)]
    assert np.mean(sw) < 2000 * 0.006


def test_composite_records_components_and_dominant_family():
    gen = CompositeGenerator()
    combos = set()
    for s in range(150):
        m = gen.sample(np.random.default_rng(s), 600, 512).metadata
        assert m["dominant_family"] == m["components"][0]
        assert "composite" not in m["components"]
        combos.add(m["latent_parameters"]["combination"])
    assert combos == {"additive", "modulated", "heteroscedastic", "trend_noise", "time_warp"}


def test_holdout_exclusions_propagate_into_composites():
    cfg = deep_merge(
        load_config(Path(__file__).parents[2] / "configs" / "data_v1.yaml"),
        {"curriculum": {"milestones": None}},
    )
    src = SyntheticEpisodeSource(cfg, "train")
    seen = 0
    for i in range(400):
        ep = src.episode(i, family="composite")
        for c in ep.meta["latent_parameters"]["component_details"]:
            seen += 1
            assert c["kind"] not in ("chirp", "bursts")
    assert seen > 800
