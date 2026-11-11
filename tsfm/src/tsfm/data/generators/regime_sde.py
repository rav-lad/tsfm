"""G5 — SDEs with regime switching and volatility (reference §4 G5, TempoPFN-inspired).

    dy_t = theta_{r_t} (mu_{r_t} - y_t) dt + sigma_{r_t} dW_t,   r_t Markov chain

Kinds:
- regime_ou       : mean reversion with regime-specific (mu, theta, sigma)
- regime_drift    : drift/volatility switches (trend changes, no reversion)
- stochastic_vol  : Heston-like variance process -> volatility clustering
- jump_diffusion  : OU or Brownian base + compound-Poisson jumps
- regime_gbm      : log-space regime drift/vol -> positive series

Numerics (documented, as required by §4 G5):
- time unit = one observation step;
- OU segments use the *exact* discretization of the OU transition (no
  Euler bias, stable for any theta);
- stochastic_vol uses Euler–Maruyama with ``substeps`` per step and full
  truncation of the variance (v -> max(v, 0));
- regime changes happen on the observation grid.
The recursions run as plain-Python loops over floats (no numba dependency),
about 1–3 ms per 1,500-step series.
"""

from __future__ import annotations

import numpy as np

from ..episode import Support
from .base import Generator, choose, log_uniform

KINDS = {"regime_ou": 0.30, "regime_drift": 0.20, "stochastic_vol": 0.20, "jump_diffusion": 0.15, "regime_gbm": 0.15}


def markov_regimes(rng: np.random.Generator, n: int, K: int, p_stay: float) -> np.ndarray:
    """Regime path with geometric durations; a switch always changes regime."""
    switches = rng.random(n) > p_stay
    switches[0] = False
    seg = np.cumsum(switches)
    labels = np.empty(seg[-1] + 1, dtype=np.int64)
    labels[0] = int(rng.integers(0, K))
    for s in range(1, len(labels)):
        nxt = int(rng.integers(0, K - 1))
        labels[s] = nxt if nxt < labels[s - 1] else nxt + 1
    return labels[seg]


def _ou_path(y0: float, mu: np.ndarray, theta: np.ndarray, sigma: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Exact OU transition with piecewise-constant parameters (dt = 1)."""
    a = np.exp(-theta)
    sd = sigma * np.sqrt((1 - a**2) / (2 * theta))
    b = mu * (1 - a) + sd * eps
    a_l, b_l = a.tolist(), b.tolist()
    out = [0.0] * len(a_l)
    y = y0
    for t in range(len(a_l)):
        y = a_l[t] * y + b_l[t]
        out[t] = y
    return np.array(out)


class RegimeSDEGenerator(Generator):
    family = "regime_sde"
    version = "0.1.0"

    def _sample(self, rng, length, context_length):
        kind = choose(rng, self.params.get("kinds", KINDS))
        lo_stay, hi_stay = self.params.get("regime_stay_probability_range", (0.90, 0.999))
        n = length
        K = int(rng.choice([2, 2, 3]))
        p_stay = float(rng.uniform(lo_stay, hi_stay))
        r = markov_regimes(rng, n, K, p_stay)
        eps = rng.standard_normal(n)
        lat: dict = {"n_regimes": K, "p_stay": p_stay, "n_switches": int((np.diff(r) != 0).sum()),
                     "regime_fractions": np.bincount(r, minlength=K).astype(float).__truediv__(n).tolist(), "dt": 1.0}
        support = Support.REAL
        hint = "hard"

        if kind == "regime_ou":
            mu = rng.normal(0, 2, K)
            # half-life between 1 step and ~L steps
            theta = np.log(2) / np.array([log_uniform(rng, 1.0, max(2.0, float(context_length))) for _ in range(K)])
            sigma = np.array([log_uniform(rng, 0.05, 1.0) for _ in range(K)])
            y = _ou_path(float(mu[r[0]]), mu[r], theta[r], sigma[r], eps)
            lat.update(mu=mu.tolist(), theta=theta.tolist(), sigma=sigma.tolist())
            hint = "medium" if p_stay > 0.98 else "hard"

        elif kind == "regime_drift":
            drift = rng.normal(0, 0.05, K)
            sigma = np.array([log_uniform(rng, 0.02, 0.5) for _ in range(K)])
            y = np.cumsum(drift[r] + sigma[r] * eps)
            lat.update(drift=drift.tolist(), sigma=sigma.tolist())

        elif kind == "stochastic_vol":
            sub = int(self.params.get("substeps", 2))
            dt = 1.0 / sub
            kappa = log_uniform(rng, 0.005, 0.2)  # variance mean reversion (per step)
            vbar = log_uniform(rng, 1e-3, 0.25)
            xi = float(rng.uniform(0.1, 1.0)) * np.sqrt(2 * kappa * vbar)  # mostly Feller-compatible
            rho = float(rng.uniform(-0.8, 0.2))  # leverage effect
            mu = float(rng.normal(0, 0.01))
            z1 = rng.standard_normal(n * sub).tolist()
            z2 = rng.standard_normal(n * sub).tolist()
            sq = np.sqrt(1 - rho**2)
            v, x, out = vbar, 0.0, [0.0] * n
            sdt = np.sqrt(dt)
            for t in range(n * sub):
                vp = v if v > 0.0 else 0.0
                sv = vp**0.5
                x += mu * dt + sv * sdt * z1[t]
                v += kappa * (vbar - vp) * dt + xi * sv * sdt * (rho * z1[t] + sq * z2[t])
                if (t + 1) % sub == 0:
                    out[(t + 1) // sub - 1] = x
            y = np.array(out)
            lat = {"kappa": kappa, "vbar": vbar, "xi": xi, "rho": rho, "drift": mu, "substeps": sub, "dt": dt,
                   "scheme": "euler_maruyama_full_truncation"}

        elif kind == "jump_diffusion":
            base = choose(rng, {"ou": 0.5, "brownian": 0.5})
            sigma = log_uniform(rng, 0.05, 0.5)
            if base == "ou":
                theta = np.log(2) / log_uniform(rng, 2.0, max(3.0, float(context_length)))
                y = _ou_path(0.0, np.zeros(n), np.full(n, theta), np.full(n, sigma), eps)
            else:
                theta = 0.0
                y = np.cumsum(sigma * eps)
            rate = log_uniform(rng, 0.002, 0.03)
            jumps = (rng.random(n) < rate) * rng.normal(0, log_uniform(rng, 1.0, 8.0) * sigma, n)
            if base == "ou":  # jumps decay with the OU reversion
                from scipy.signal import lfilter

                y = y + lfilter([1.0], [1.0, -np.exp(-theta)], jumps)
            else:
                y = y + np.cumsum(jumps)
            lat = {"base": base, "theta": float(theta), "sigma": sigma, "jump_rate": rate, "n_jumps": int((jumps != 0).sum()), "dt": 1.0}

        else:  # regime_gbm: positive, multiplicative dynamics
            drift = rng.normal(0, 0.002, K)
            vol = np.array([log_uniform(rng, 0.003, 0.04) for _ in range(K)])
            logy = np.log(log_uniform(rng, 1.0, 1000.0)) + np.cumsum(drift[r] - 0.5 * vol[r] ** 2 + vol[r] * eps)
            y = np.exp(np.clip(logy, -30, 30))
            support = Support.NONNEGATIVE
            lat.update(drift=drift.tolist(), vol=vol.tolist())

        y = y + float(rng.normal(0, 2)) if support == Support.REAL else y
        meta = {
            "kind": kind,
            "support": support,
            "components": [kind],
            "predictability_hint": hint,
            "latent_parameters": {"kind": kind, **lat},
        }
        return y, meta
