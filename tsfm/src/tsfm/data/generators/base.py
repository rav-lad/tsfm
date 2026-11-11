"""Generator interface and shared sampling helpers.

A generator maps ``(rng, length, context_length)`` to a :class:`SyntheticRecord`.
``context_length`` is a hint so that dimensionless quantities (period / L, number
of visible cycles) can be sampled relative to what the model will actually see
(reference §6.2). Generators must never look at anything else.

Every record's metadata contains at least::

    family, generator_version, kind, support, latent_parameters, components,
    predictability_hint
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..episode import SyntheticRecord, Support

DISCRETE_PERIODS = (7, 12, 24, 48, 52, 96, 168, 365)
# extreme_noise is never in the default training mix: it exists for the H2 holdout.
SNR_REGIMES = {"low_noise": 0.05, "mid_noise": 0.3, "high_noise": 1.0, "extreme_noise": 2.5}


class Generator(ABC):
    """Base class.

    Shared *prior* parameters (used by holdouts, reference §11.2 H2):
    - ``period_ratio_band``: ``[lo, hi]`` band of P / L
    - ``period_band_mode``: ``"exclude"`` (train) or ``"only"`` (OOD suite)
    - ``noise_regimes``: distribution over SNR regimes overriding the defaults
    """

    family: str = "base"
    version: str = "0.1.0"

    def __init__(self, **params: Any) -> None:
        self.params = params
        band = params.get("period_ratio_band")
        self.period_kw: dict[str, Any] = (
            {"band": tuple(band), "band_mode": params.get("period_band_mode", "exclude")} if band else {}
        )
        self.noise_regimes: dict[str, float] | None = params.get("noise_regimes")

    def noise(self, rng: np.random.Generator, signal: np.ndarray, default: dict[str, float] | None = None):
        """add_noise_snr with the holdout override taking precedence."""
        return add_noise_snr(rng, signal, regimes=self.noise_regimes or default)

    def period(self, rng: np.random.Generator, context_length: int, **kw: Any):
        return sample_period(rng, context_length, **{**kw, **self.period_kw})

    @abstractmethod
    def _sample(self, rng: np.random.Generator, length: int, context_length: int) -> tuple[np.ndarray, dict]:
        """Return ``(values[T] or [T, C], metadata)``."""

    def sample(self, rng: np.random.Generator, length: int, context_length: int | None = None) -> SyntheticRecord:
        if length < 2:
            raise ValueError("length must be >= 2")
        context_length = int(context_length or length)
        values, meta = self._sample(rng, int(length), context_length)
        values = np.asarray(values, dtype=np.float64)
        base_meta = {
            "family": self.family,
            "generator_version": self.version,
            "support": Support.REAL,
            "components": [],
            "latent_parameters": {},
        }
        base_meta.update(meta)
        return SyntheticRecord(values=values, metadata=base_meta)


# --------------------------------------------------------------------------- helpers


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def signed_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.choice((-1.0, 1.0)) * log_uniform(rng, low, high))


def choose(rng: np.random.Generator, options: dict[str, float]) -> str:
    keys = list(options)
    p = np.array([options[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def sample_period(
    rng: np.random.Generator,
    context_length: int,
    min_period: float = 4.0,
    min_cycles: float = 1.5,
    p_discrete: float = 0.5,
    p_ambiguous: float = 0.05,
    ratio_range: tuple[float, float] = (0.02, 0.6),
    band: tuple[float, float] | None = None,
    band_mode: str = "exclude",
) -> tuple[float, dict]:
    """Sample a seasonal period in steps.

    Half from "calendar-like" integer periods, half as a ratio P/L. Periods are
    restricted to show at least ``min_cycles`` cycles in the context, except for a
    small fraction ``p_ambiguous`` of intentionally under-identified cases.

    ``band`` (on P/L) supports holdouts: ``exclude`` rejects periods inside it
    (training), ``only`` samples log-uniformly inside it (OOD suite).
    """
    if band is not None:
        lo, hi = band
        if band_mode == "only":
            P = log_uniform(rng, lo, hi) * context_length
            return max(P, min_period), {"period_source": "holdout_band", "ambiguous": False}
        if band_mode != "exclude":
            raise ValueError(f"band_mode must be 'exclude' or 'only', got {band_mode!r}")
        for _ in range(50):
            P, info = sample_period(rng, context_length, min_period, min_cycles, p_discrete, p_ambiguous, ratio_range)
            if not lo <= P / context_length <= hi:
                return P, info
        # pathological config: push the period just below the band
        return max(min_period, 0.95 * lo * context_length), {"period_source": "band_fallback", "ambiguous": False}

    ambiguous = rng.random() < p_ambiguous
    max_period = context_length * (2.0 if ambiguous else 1.0 / min_cycles)
    if rng.random() < p_discrete:
        ok = [p for p in DISCRETE_PERIODS if min_period <= p <= max_period]
        if ok:
            P = float(rng.choice(ok))
            return P, {"period_source": "discrete", "ambiguous": ambiguous}
    lo, hi = ratio_range
    hi = min(hi, max_period / context_length) if not ambiguous else 2.0
    lo = max(lo, min_period / context_length)
    if lo >= hi:
        lo = hi * 0.5
    P = log_uniform(rng, lo, hi) * context_length
    return max(P, min_period), {"period_source": "ratio", "ambiguous": ambiguous}


def sample_noise(rng: np.random.Generator, n: int, kind: str | None = None) -> tuple[np.ndarray, dict]:
    """Unit-scale noise: gaussian, student-t (heavy tails) or AR(1)-coloured."""
    kind = kind or choose(rng, {"gaussian": 0.6, "student_t": 0.2, "ar1": 0.2})
    info: dict[str, Any] = {"noise_kind": kind}
    if kind == "gaussian":
        e = rng.standard_normal(n)
    elif kind == "student_t":
        df = float(rng.uniform(2.5, 8.0))
        e = rng.standard_t(df, n) / np.sqrt(df / (df - 2.0))
        info["df"] = df
    elif kind == "ar1":
        from scipy.signal import lfilter

        phi = float(rng.uniform(0.2, 0.9))
        e = lfilter([1.0], [1.0, -phi], rng.standard_normal(n)) * np.sqrt(1 - phi**2)
        info["phi"] = phi
    else:  # pragma: no cover
        raise ValueError(kind)
    return e, info


DEFAULT_NOISE_REGIMES = {"low_noise": 0.35, "mid_noise": 0.4, "high_noise": 0.25}


def add_noise_snr(
    rng: np.random.Generator,
    signal: np.ndarray,
    regime: str | None = None,
    regimes: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Add noise whose std is a fraction of the signal std (difficulty regime)."""
    regime = regime or choose(rng, regimes or DEFAULT_NOISE_REGIMES)
    ratio = SNR_REGIMES[regime] * float(rng.uniform(0.5, 1.5))
    sig_std = float(np.std(signal))
    if sig_std < 1e-8:
        sig_std = 1.0
    e, info = sample_noise(rng, len(signal))
    info.update(noise_regime=regime, noise_to_signal=ratio)
    return signal + ratio * sig_std * e, info


HINT_BY_NOISE = {"low_noise": "easy", "mid_noise": "medium", "high_noise": "hard", "extreme_noise": "hard"}
