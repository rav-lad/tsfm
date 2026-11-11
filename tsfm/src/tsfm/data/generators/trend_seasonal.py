"""G1 — trends + classical seasonalities (ForecastPFN-style prior, reference §4 G1).

    y_t = trend(t) (+|x) sum_k A_k * shape_k(2 pi t / P_k + phi_k) * damping(t) + noise

Time is expressed in units of the context length (u = t / L) so trend slopes
have comparable visual effect whatever L is.
"""

from __future__ import annotations

import numpy as np

from .base import HINT_BY_NOISE, Generator, choose, log_uniform, signed_log_uniform


def _trend(rng: np.random.Generator, u: np.ndarray) -> tuple[np.ndarray, dict]:
    kind = choose(
        rng,
        {"none": 0.2, "linear": 0.3, "quadratic": 0.1, "piecewise_linear": 0.2, "logistic": 0.1, "bounded_exp": 0.1},
    )
    p: dict = {"trend_kind": kind}
    if kind == "none":
        tr = np.zeros_like(u)
    elif kind == "linear":
        b1 = signed_log_uniform(rng, 0.05, 3.0)
        tr = b1 * u
        p["slope"] = b1
    elif kind == "quadratic":
        b1 = float(rng.normal(0, 1))
        b2 = signed_log_uniform(rng, 0.05, 1.5)
        tr = b1 * u + b2 * u**2
        p.update(slope=b1, curvature=b2)
    elif kind == "piecewise_linear":
        n_cp = int(rng.integers(1, 5))
        cps = np.sort(rng.uniform(u[0], u[-1], n_cp))
        slopes = rng.normal(0, 1.5, n_cp + 1)
        tr = slopes[0] * u
        for cp, (s_prev, s_next) in zip(cps, zip(slopes[:-1], slopes[1:])):
            tr = tr + (s_next - s_prev) * np.maximum(u - cp, 0.0)
        p.update(changepoints=cps.tolist(), slopes=slopes.tolist())
    elif kind == "logistic":
        cap = signed_log_uniform(rng, 0.5, 4.0)
        k = log_uniform(rng, 2.0, 20.0)
        u0 = float(rng.uniform(u[0], u[-1]))
        tr = cap / (1.0 + np.exp(-k * (u - u0)))
        p.update(cap=cap, rate=k, midpoint=u0)
    else:  # bounded_exp: exponential growth/decay that saturates (never explodes)
        cap = signed_log_uniform(rng, 0.5, 4.0)
        k = log_uniform(rng, 0.3, 5.0)
        tr = cap * (1.0 - np.exp(-k * (u - u[0])))
        p.update(cap=cap, rate=k)
    return tr, p


def _season_shape(rng: np.random.Generator, phase: np.ndarray) -> tuple[np.ndarray, dict]:
    """Sinusoid or short Fourier series (non-sinusoidal but smooth season)."""
    n_harm = int(rng.choice([1, 1, 2, 3, 4]))
    if n_harm == 1:
        return np.sin(phase), {"harmonics": 1}
    amps = np.r_[1.0, rng.uniform(0, 0.6, n_harm - 1) / np.arange(2, n_harm + 1)]
    phases = np.r_[0.0, rng.uniform(0, 2 * np.pi, n_harm - 1)]
    s = sum(a * np.sin((h + 1) * phase + ph) for h, (a, ph) in enumerate(zip(amps, phases)))
    s = s / np.max(np.abs(s) + 1e-12)
    return s, {"harmonics": n_harm, "harmonic_amplitudes": amps.tolist()}


class TrendSeasonalGenerator(Generator):
    family = "trend_seasonal"
    version = "0.1.0"

    def _sample(self, rng, length, context_length):
        t = np.arange(length, dtype=float)
        u = t / context_length
        trend, tp = _trend(rng, u)
        K = int(rng.choice(self.params.get("harmonics", [0, 1, 2, 3]), p=self.params.get("harmonics_p", [0.1, 0.45, 0.3, 0.15])))
        mode = choose(rng, {"additive": 0.75, "multiplicative": 0.25})
        damping = choose(rng, {"none": 0.85, "damped": 0.1, "growing": 0.05})
        lam = log_uniform(rng, 0.1, 1.5) if damping != "none" else 0.0
        if damping == "damped":
            env = np.exp(-lam * u)
        elif damping == "growing":
            env = np.exp(0.5 * lam * u)
        else:
            env = 1.0

        season = np.zeros(length)
        seasons = []
        for _ in range(K):
            P, pinfo = self.period(rng, context_length)
            phi = float(rng.uniform(0, 2 * np.pi))
            A = log_uniform(rng, 0.1, 3.0)
            shape, sinfo = _season_shape(rng, 2 * np.pi * t / P + phi)
            season += A * shape
            seasons.append({"period": P, "amplitude": A, "phase": phi, **pinfo, **sinfo})
        season = season * env

        level = float(rng.normal(0, 2))
        if mode == "multiplicative" and K > 0:
            # level x (1 + relative seasonality); keep the base positive
            base = np.abs(level) + 1.0 + trend - trend.min()
            rel = 0.8 * season / (np.max(np.abs(season)) + 1e-12) * rng.uniform(0.1, 1.0)
            signal = base * (1.0 + rel)
        else:
            signal = level + trend + season

        y, ninfo = self.noise(rng, signal)
        if K == 0 and tp["trend_kind"] == "none":
            hint = "unpredictable"
        else:
            hint = HINT_BY_NOISE[ninfo["noise_regime"]]
        meta = {
            "kind": f"{tp['trend_kind']}+K{K}",
            "components": ["trend", *(["seasonality"] * K), "noise"],
            "predictability_hint": hint,
            "latent_parameters": {
                "level": level,
                **tp,
                "n_seasonalities": K,
                "seasonalities": seasons,
                "mode": mode,
                "damping": damping,
                "damping_rate": lam,
                **ninfo,
            },
        }
        return y, meta
