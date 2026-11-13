"""Sprint D2 generator unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from tsfm.data.episode import Support
from tsfm.data.generators import REGISTRY
from tsfm.data.generators.autoregressive import ar_max_root_inverse, pacf_to_ar
from tsfm.data.generators.base import sample_period
from tsfm.data.validation import check_record

FAMILIES = sorted(REGISTRY)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("length,ctx", [(160, 128), (1500, 1024)])
def test_shape_finite_support_metadata(family, length, ctx):
    gen = REGISTRY[family]()
    for seed in range(60):
        rec = gen.sample(np.random.default_rng(seed), length, ctx)
        assert rec.values.shape == (length, 1)
        check_record(rec)  # finite, no overflow, support, metadata keys
        assert rec.metadata["family"] == family
        assert rec.metadata["predictability_hint"] in {"easy", "medium", "hard", "unpredictable"}


@pytest.mark.parametrize("family", FAMILIES)
def test_reproducible(family):
    gen = REGISTRY[family]()
    a = gen.sample(np.random.default_rng(42), 600, 512)
    b = gen.sample(np.random.default_rng(42), 600, 512)
    c = gen.sample(np.random.default_rng(43), 600, 512)
    assert np.array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)


@pytest.mark.parametrize("family", FAMILIES)
def test_kind_diversity(family):
    gen = REGISTRY[family]()
    kinds = {gen.sample(np.random.default_rng(s), 400, 256).metadata["kind"] for s in range(200)}
    assert len(kinds) >= 4


def test_pacf_parametrisation_is_always_stable():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        phi = pacf_to_ar(rng.uniform(-0.99, 0.99, rng.integers(1, 8)))
        assert ar_max_root_inverse(phi) < 1.0


def test_ar_metadata_matches_stationarity():
    gen = REGISTRY["autoregressive"]()
    for s in range(300):
        lat = gen.sample(np.random.default_rng(s), 300, 256).metadata["latent_parameters"]
        if lat["stationary"]:
            assert lat["max_inverse_root"] < 1.0


def test_counts_are_nonnegative_integers():
    gen = REGISTRY["events_counts"]()
    n_int = 0
    for s in range(300):
        rec = gen.sample(np.random.default_rng(s), 500, 256)
        if rec.metadata["support"] == Support.INTEGER:
            n_int += 1
            assert rec.values.min() >= 0 and np.all(rec.values == np.round(rec.values))
    assert n_int > 50


def test_zero_inflation_is_applied():
    gen = REGISTRY["events_counts"](kinds={"intermittent_demand": 1.0})
    for s in range(50):
        rec = gen.sample(np.random.default_rng(s), 2000, 1024)
        p0 = rec.metadata["latent_parameters"]["zero_inflation"]
        assert (rec.values == 0).mean() >= p0 - 0.05


def test_dominant_frequency_of_clean_sine():
    # Noise-free G1-style check: the FFT peak matches the configured period.
    gen = REGISTRY["complex_periodic"](waveforms={"sine": 1.0})
    hits = 0
    for s in range(40):
        rec = gen.sample(np.random.default_rng(s), 2048, 1024)
        lat = rec.metadata["latent_parameters"]
        if lat["noise_regime"] != "low_noise" or "fm_depth" in lat or lat["amplitude_modulation"] != "none":
            continue
        x = rec.values[:, 0] - rec.values[:, 0].mean()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        f_peak = np.argmax(spec[1:]) + 1
        assert abs(len(x) / f_peak - lat["period"]) / lat["period"] < 0.1
        hits += 1
    assert hits >= 3


def test_periods_mostly_identifiable():
    rng = np.random.default_rng(0)
    ctx = 256
    periods = [sample_period(rng, ctx) for _ in range(5000)]
    unamb = [P for P, info in periods if not info["ambiguous"]]
    assert all(P <= ctx / 1.5 + 1e-9 for P in unamb)
    frac_amb = 1 - len(unamb) / len(periods)
    assert 0.02 < frac_amb < 0.10
