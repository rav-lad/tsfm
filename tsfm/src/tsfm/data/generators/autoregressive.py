"""G4 — AR / ARMA / random walk / mean reversion / white noise (reference §4 G4).

Stable AR(p) coefficients are drawn through partial autocorrelations in (-1, 1)
(Levinson–Durbin), which guarantees all roots lie outside the unit circle.
Non-stationary processes (random walks) are kept on purpose and labelled.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .base import Generator, choose, log_uniform, sample_noise

KINDS = {
    "white_noise": 0.10,
    "ar1": 0.25,
    "ar_p": 0.20,
    "arma": 0.15,
    "random_walk": 0.15,
    "near_unit_root": 0.15,
}


def pacf_to_ar(pacf: np.ndarray) -> np.ndarray:
    """Partial autocorrelations -> AR coefficients phi_1..phi_p (stable if |pacf|<1)."""
    phi = np.zeros(0)
    for k, a in enumerate(pacf):
        phi = np.r_[phi - a * phi[::-1], a] if k else np.array([a])
    return phi


def ar_max_root_inverse(phi: np.ndarray) -> float:
    """Max modulus of the inverse characteristic roots (<1 means stationary)."""
    if len(phi) == 0:
        return 0.0
    return float(np.max(np.abs(np.roots(np.r_[1.0, -phi]))))


class AutoregressiveGenerator(Generator):
    family = "autoregressive"
    version = "0.1.0"

    def _sample(self, rng, length, context_length):
        kind = choose(rng, self.params.get("kinds", KINDS))
        phi_range = self.params.get("ar1_phi_range", (-0.95, 0.95))
        burn = 200
        n = length + burn
        e, ninfo = sample_noise(rng, n, kind=choose(rng, {"gaussian": 0.75, "student_t": 0.25}))
        sigma = log_uniform(rng, 0.1, 2.0)
        e = sigma * e
        phi = np.zeros(0)
        theta = np.zeros(0)
        stationary = True
        drift = 0.0

        if kind == "white_noise":
            x = e
        elif kind == "ar1":
            phi = np.array([rng.uniform(*phi_range)])
        elif kind == "ar_p":
            p = int(rng.integers(2, 6))
            phi = pacf_to_ar(rng.uniform(-0.9, 0.9, p))
        elif kind == "arma":
            p, q = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            phi = pacf_to_ar(rng.uniform(-0.9, 0.9, p))
            theta = pacf_to_ar(rng.uniform(-0.8, 0.8, q))  # invertible MA
        elif kind == "near_unit_root":
            sign = 1.0 if rng.random() < 0.85 else -1.0  # mostly positive persistence
            phi = np.array([sign * (1 - log_uniform(rng, 0.002, 0.05))])
        elif kind == "random_walk":
            stationary = False
            drift = float(rng.normal(0, 0.05)) * sigma if rng.random() < 0.5 else 0.0

        if kind == "random_walk":
            x = np.cumsum(e[burn:] + drift)
        elif kind != "white_noise":
            x = lfilter(np.r_[1.0, theta], np.r_[1.0, -phi], e)[burn:]
        else:
            x = x[burn:]

        mean = float(rng.normal(0, 3))
        y = mean + x
        max_root = ar_max_root_inverse(phi)
        hint = {
            "white_noise": "unpredictable",
            "random_walk": "hard",
            "near_unit_root": "hard",
        }.get(kind, "medium" if max_root > 0.5 else "hard")
        meta = {
            "kind": kind,
            "components": [kind],
            "predictability_hint": hint,
            "latent_parameters": {
                "phi": phi.tolist(),
                "theta": theta.tolist(),
                "sigma": sigma,
                "mean": mean,
                "drift": drift,
                "stationary": stationary,
                "max_inverse_root": max_root,
                **ninfo,
            },
        }
        return y, meta
