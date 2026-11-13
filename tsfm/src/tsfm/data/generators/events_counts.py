"""G6 — events, spikes, intermittency and counts.

Kinds:
- intermittent_demand : zero-inflated Poisson / negative-binomial (integer >= 0)
- count_process       : Poisson/NB with a latent log-intensity (AR + season)
- spikes              : isolated heavy-tailed spikes on a base process
- bursts              : clustered events (burst windows of elevated activity)
- level_shifts        : piecewise-constant levels / plateaus with noise
- promotions          : positive baseline with multiplicative uplift windows
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from ..episode import Support
from .base import Generator, add_noise_snr, choose, log_uniform, sample_period

KINDS = {
    "intermittent_demand": 0.20,
    "count_process": 0.20,
    "spikes": 0.15,
    "bursts": 0.15,
    "level_shifts": 0.15,
    "promotions": 0.15,
}


def _counts(rng: np.random.Generator, mu: np.ndarray, dist: str) -> tuple[np.ndarray, dict]:
    mu = np.clip(mu, 0.0, 1e6)
    if dist == "poisson":
        return rng.poisson(mu).astype(float), {"dist": "poisson"}
    r = log_uniform(rng, 0.5, 20.0)  # NB dispersion; var = mu + mu^2 / r
    lam = rng.gamma(r, mu / r)
    return rng.poisson(lam).astype(float), {"dist": "negative_binomial", "dispersion_r": r}


def _log_intensity(rng, length, context_length, base_rate, period_fn=sample_period):
    t = np.arange(length)
    z = np.full(length, np.log(base_rate))
    info: dict = {"base_rate": base_rate}
    if rng.random() < 0.5:
        P, pinfo = period_fn(rng, context_length)
        a = float(rng.uniform(0.2, 1.2))
        z += a * np.sin(2 * np.pi * t / P + rng.uniform(0, 2 * np.pi))
        info.update(season_period=P, season_log_amplitude=a, **pinfo)
    if rng.random() < 0.5:
        phi = float(rng.uniform(0.5, 0.98))
        s = float(rng.uniform(0.05, 0.4))
        z += lfilter([1.0], [1.0, -phi], s * np.sqrt(1 - phi**2) * rng.standard_normal(length))
        info.update(latent_ar_phi=phi, latent_ar_sigma=s)
    if rng.random() < 0.3:
        slope = float(rng.normal(0, 0.5))
        z += slope * t / context_length
        info["log_trend"] = slope
    return z, info


class EventsCountsGenerator(Generator):
    family = "events_counts"
    version = "0.1.0"

    def _sample(self, rng, length, context_length):
        kind = choose(rng, self.params.get("kinds", KINDS))
        lat: dict = {}
        support = Support.REAL
        hint = "medium"

        if kind == "intermittent_demand":
            rate = log_uniform(rng, 0.5, 20.0)
            z, info = _log_intensity(rng, length, context_length, rate, self.period)
            p_zero = float(rng.uniform(0.3, 0.95))
            counts, dinfo = _counts(rng, np.exp(z), choose(rng, {"poisson": 0.5, "nb": 0.5}))
            demand = rng.random(length) >= p_zero
            y = counts * demand
            support, hint = Support.INTEGER, "hard"
            lat.update(zero_inflation=p_zero, **info, **dinfo)

        elif kind == "count_process":
            rate = log_uniform(rng, 0.3, 200.0)
            z, info = _log_intensity(rng, length, context_length, rate, self.period)
            y, dinfo = _counts(rng, np.exp(z), choose(rng, {"poisson": 0.5, "nb": 0.5}))
            support = Support.INTEGER
            hint = "medium" if rate > 5 else "hard"
            lat.update(**info, **dinfo)

        elif kind in ("spikes", "bursts"):
            base_kind = choose(rng, {"flat": 0.3, "ar1": 0.4, "seasonal": 0.3})
            t = np.arange(length)
            if base_kind == "flat":
                base = float(rng.normal(0, 2)) + 0.3 * rng.standard_normal(length)
            elif base_kind == "ar1":
                phi = float(rng.uniform(0.3, 0.95))
                base = lfilter([1.0], [1.0, -phi], rng.standard_normal(length) * np.sqrt(1 - phi**2))
            else:
                P, _ = self.period(rng, context_length)
                base = np.sin(2 * np.pi * t / P + rng.uniform(0, 2 * np.pi))
            base, ninfo = add_noise_snr(rng, base + 1e-3 * rng.standard_normal(length), regime="low_noise")
            scale = float(np.std(base)) or 1.0
            ev = np.zeros(length)
            if kind == "spikes":
                rate = log_uniform(rng, 0.002, 0.05)
                where = rng.random(length) < rate
                sizes = scale * log_uniform(rng, 2.0, 10.0) * (1 + rng.pareto(3.0, length))
                sign = 1.0 if rng.random() < 0.7 else float(rng.choice([-1.0, 1.0]))
                ev[where] = sign * sizes[where]
                decay = float(rng.uniform(0.0, 0.8))
                if decay > 0:  # spikes that relax over a few steps
                    ev = lfilter([1.0], [1.0, -decay], ev)
                lat.update(spike_rate=rate, spike_decay=decay, n_events=int(where.sum()))
            else:
                p_start = log_uniform(rng, 0.003, 0.03)
                mean_len = log_uniform(rng, 3.0, 30.0)
                inside = np.zeros(length, dtype=bool)
                starts = np.flatnonzero(rng.random(length) < p_start)
                for s, L_b in zip(starts, 1 + rng.geometric(1.0 / mean_len, len(starts))):
                    inside[s : s + int(L_b)] = True
                intensity = scale * log_uniform(rng, 1.0, 6.0)
                ev[inside] = intensity * rng.exponential(1.0, int(inside.sum()))
                lat.update(burst_start_prob=p_start, burst_mean_length=mean_len, burst_fraction=float(inside.mean()))
            y = base + ev
            hint = "hard"
            lat.update(base_kind=base_kind, **ninfo)

        elif kind == "level_shifts":
            rate = log_uniform(rng, 0.002, 0.03)
            shifts = rng.random(length) < rate
            sizes = rng.normal(0, log_uniform(rng, 0.5, 4.0), length)
            level = np.cumsum(np.where(shifts, sizes, 0.0)) + float(rng.normal(0, 2))
            y, ninfo = self.noise(rng, level + 1e-3 * rng.standard_normal(length))
            if np.std(level) < 1e-6:  # no shift drawn: noise relative to unit scale
                y = level + 0.3 * rng.standard_normal(length)
            lat.update(shift_rate=rate, n_shifts=int(shifts.sum()), **ninfo)
            hint = "hard"

        else:  # promotions: positive baseline x (1 + uplift) during promo windows
            t = np.arange(length)
            base_level = log_uniform(rng, 1.0, 100.0)
            P, pinfo = self.period(rng, context_length)
            season = 1 + float(rng.uniform(0, 0.4)) * np.sin(2 * np.pi * t / P + rng.uniform(0, 2 * np.pi))
            promo = np.zeros(length, dtype=bool)
            regular = rng.random() < 0.5
            if regular:  # planned, periodic promotions
                every = int(log_uniform(rng, 20, max(21, context_length // 2)))
                dur = int(rng.integers(2, max(3, every // 4)))
                offset = int(rng.integers(0, every))
                promo[((t - offset) % every) < dur] = True
                lat.update(promo_every=every, promo_duration=dur)
            else:
                p_start = log_uniform(rng, 0.005, 0.03)
                starts = np.flatnonzero(rng.random(length) < p_start)
                for s in starts:
                    promo[s : s + int(rng.integers(2, 15))] = True
                lat["promo_start_prob"] = p_start
            uplift = log_uniform(rng, 0.2, 3.0)
            mu = base_level * season * (1 + uplift * promo)
            y, dinfo = _counts(rng, mu, choose(rng, {"poisson": 0.5, "nb": 0.5}))
            support = Support.INTEGER
            hint = "medium" if regular else "hard"
            lat.update(base_level=base_level, uplift=uplift, regular=regular, promo_fraction=float(promo.mean()), **pinfo, **dinfo)
            # The promo calendar is a natural KNOWN_FUTURE covariate for D5;
            # emitted as metadata-only for now (univariate stage).
            lat["has_known_future_driver"] = True

        meta = {
            "kind": kind,
            "support": support,
            "components": [kind],
            "predictability_hint": hint,
            "latent_parameters": lat,
        }
        return y, meta
