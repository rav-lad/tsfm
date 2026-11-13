"""G3 — Gaussian processes with composed kernels (KernelSynth-style).

A random kernel expression is built from a bank (RBF, Matérn, Periodic,
RationalQuadratic, Linear, White) by combining 1..max_kernels kernels with
random ``+`` / ``*`` (as in Chronos' KernelSynth).

Sampling, with *the same covariance* in both paths:

- ``length <= max_exact_length``: exact sampling, Cholesky of the covariance
  matrix (O(T^3), jitter recorded in metadata).
- longer series: Random Fourier Features (Rahimi & Recht 2007). The expression
  is expanded into a sum of product terms. For each term:
    * stationary factors -> one spectral frequency per factor, summed
      (the spectral measure of a product is the convolution of the measures);
    * Linear factors     -> deterministic multiplier sqrt(v) * (x - c), exact;
    * a White factor     -> the term is white noise with the matching variance.
  Independent terms are summed. Only the stationary part is approximate
  (Monte Carlo error O(1/sqrt(D))).

Time is in units of the context length: ``x = t / L``, so length-scales are
relative to what the model sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import toeplitz
from scipy.special import ive

from .base import Generator, choose, log_uniform, sample_period

STATIONARY = ("rbf", "matern", "periodic", "rational_quadratic")
BANK = {"rbf": 0.20, "matern": 0.20, "periodic": 0.25, "rational_quadratic": 0.10, "linear": 0.15, "white": 0.10}


@dataclass
class Kernel:
    kind: str
    p: dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- exact
    def matrix(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        if self.kind == "white":
            return p["variance"] * np.eye(len(x))
        if self.kind == "linear":
            d = x - p["offset"]
            return p["variance"] * np.outer(d, d)
        dx = np.diff(x)
        if len(x) > 1 and np.allclose(dx, dx[0], rtol=0, atol=1e-12):
            # regular grid: stationary covariance is Toeplitz -> T evaluations, not T^2
            return p["variance"] * toeplitz(self.correlation(np.arange(len(x)) * dx[0]))
        r = np.abs(x[:, None] - x[None, :])
        return p["variance"] * self.correlation(r)

    def correlation(self, r: np.ndarray) -> np.ndarray:
        """Stationary correlation k(r) / k(0)."""
        p = self.p
        if self.kind == "rbf":
            return np.exp(-0.5 * (r / p["lengthscale"]) ** 2)
        if self.kind == "matern":
            s = r / p["lengthscale"]
            nu = p["nu"]
            if nu == 0.5:
                return np.exp(-s)
            if nu == 1.5:
                a = np.sqrt(3) * s
                return (1 + a) * np.exp(-a)
            a = np.sqrt(5) * s
            return (1 + a + a**2 / 3) * np.exp(-a)
        if self.kind == "periodic":
            return np.exp(-2 * np.sin(np.pi * r / p["period"]) ** 2 / p["lengthscale"] ** 2)
        if self.kind == "rational_quadratic":
            return (1 + r**2 / (2 * p["alpha"] * p["lengthscale"] ** 2)) ** (-p["alpha"])
        raise ValueError(self.kind)

    # ---------------------------------------------------------------- spectral
    def frequencies(self, rng: np.random.Generator, D: int) -> np.ndarray:
        """D angular frequencies from the normalized spectral measure of k(r)/k(0)."""
        p = self.p
        if self.kind == "rbf":
            return rng.standard_normal(D) / p["lengthscale"]
        if self.kind == "matern":
            return rng.standard_t(2 * p["nu"], D) / p["lengthscale"]
        if self.kind == "rational_quadratic":
            a, ls = p["alpha"], p["lengthscale"]
            tau = rng.gamma(a, 1.0 / (a * ls**2), D)  # precision mixture of RBFs
            return rng.standard_normal(D) * np.sqrt(tau)
        if self.kind == "periodic":
            # exp(-2 sin^2(pi r/P)/l^2) = sum_k ive(|k|, 1/l^2) cos(2 pi k r / P)
            z = 1.0 / p["lengthscale"] ** 2
            kmax = int(np.ceil(z + 10 * np.sqrt(z + 1))) + 2
            k = np.arange(-kmax, kmax + 1)
            w = ive(np.abs(k), z)
            harmonics = rng.choice(k, size=D, p=w / w.sum())
            return 2 * np.pi * harmonics / p["period"]
        raise ValueError(self.kind)

    def __str__(self) -> str:
        args = ",".join(f"{k}={v:.3g}" for k, v in self.p.items())
        return f"{self.kind}({args})"


def sample_kernel(rng: np.random.Generator, kind: str, context_length: int, period_fn=sample_period) -> Kernel:
    var = log_uniform(rng, 0.1, 2.0)
    if kind == "rbf":
        return Kernel(kind, {"variance": var, "lengthscale": log_uniform(rng, 0.01, 2.0)})
    if kind == "matern":
        return Kernel(kind, {"variance": var, "lengthscale": log_uniform(rng, 0.01, 2.0), "nu": float(rng.choice([0.5, 1.5, 2.5]))})
    if kind == "periodic":
        P, _ = period_fn(rng, context_length)
        return Kernel(kind, {"variance": var, "period": P / context_length, "lengthscale": log_uniform(rng, 0.3, 3.0)})
    if kind == "rational_quadratic":
        return Kernel(kind, {"variance": var, "lengthscale": log_uniform(rng, 0.01, 2.0), "alpha": log_uniform(rng, 0.1, 10.0)})
    if kind == "linear":
        return Kernel(kind, {"variance": log_uniform(rng, 0.1, 2.0), "offset": float(rng.uniform(-0.5, 1.5))})
    if kind == "white":
        return Kernel(kind, {"variance": log_uniform(rng, 1e-4, 0.25)})
    raise ValueError(kind)


Terms = list[list[Kernel]]  # sum over terms of product over factors


def random_expression(rng: np.random.Generator, context_length: int, max_kernels: int, bank: dict[str, float], period_fn=sample_period) -> tuple[Terms, str]:
    """KernelSynth: draw J kernels, fold them with random + / *. Returns expanded terms."""
    J = int(rng.integers(1, max_kernels + 1))
    kernels = [sample_kernel(rng, choose(rng, bank), context_length, period_fn) for _ in range(J)]
    terms: Terms = [[kernels[0]]]
    expr = str(kernels[0].kind)
    for k in kernels[1:]:
        if rng.random() < 0.5:
            terms = terms + [[k]]
            expr = f"({expr} + {k.kind})"
        else:
            terms = [t + [k] for t in terms]
            expr = f"({expr} * {k.kind})"
    return terms, expr


def covariance(terms: Terms, x: np.ndarray) -> np.ndarray:
    K = np.zeros((len(x), len(x)))
    for term in terms:
        Kt = np.ones((len(x), len(x)))
        for f in term:
            Kt = Kt * f.matrix(x)
        K += Kt
    return K


def sample_exact(rng: np.random.Generator, terms: Terms, x: np.ndarray) -> tuple[np.ndarray, dict]:
    K = covariance(terms, x)
    base = 1e-6 * max(float(np.mean(np.diag(K))), 1e-12)
    for i in range(6):
        jitter = base * 10**i
        try:
            Lc = np.linalg.cholesky(K + jitter * np.eye(len(x)))
            return Lc @ rng.standard_normal(len(x)), {"jitter": jitter, "psd_fallback": False}
        except np.linalg.LinAlgError:
            continue
    # numerically indefinite even with jitter: clip eigenvalues (logged)
    w, V = np.linalg.eigh(K)
    return V @ (np.sqrt(np.clip(w, 0, None)) * rng.standard_normal(len(x))), {"jitter": None, "psd_fallback": True}


def _fourier_basis(x: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """exp(i * outer(x, omega)), [T, D].

    On a regular grid x_t = x_0 + t*dx this is a cumulative product of the unit
    rotation exp(i*omega*dx), much cheaper than T*D cos/sin evaluations
    (round-off grows ~ T * 1e-16). Falls back to the direct formula otherwise.
    """
    dx = np.diff(x)
    if len(x) > 1 and np.allclose(dx, dx[0], rtol=0, atol=1e-12):
        E = np.empty((len(x), len(omega)), dtype=np.complex128)
        E[0] = np.exp(1j * omega * x[0])
        E[1:] = np.exp(1j * omega * dx[0])
        return np.cumprod(E, axis=0)
    return np.exp(1j * np.outer(x, omega))


def sample_rff(rng: np.random.Generator, terms: Terms, x: np.ndarray, n_features: int) -> np.ndarray:
    y = np.zeros(len(x))
    for term in terms:
        var = float(np.prod([f.p["variance"] for f in term if f.kind != "linear"]))
        mult = np.ones(len(x))
        for f in term:
            if f.kind == "linear":
                mult = mult * np.sqrt(f.p["variance"]) * (x - f.p["offset"])
        if any(f.kind == "white" for f in term):
            # white x stationary = diagonal with k_stat(0) = product of variances
            y += mult * np.sqrt(var) * rng.standard_normal(len(x))
            continue
        stat = [f for f in term if f.kind in STATIONARY]
        if not stat:  # product of linear factors only (variance already in mult)
            y += mult * rng.standard_normal()
            continue
        omega = sum(f.frequencies(rng, n_features) for f in stat)
        a, b = rng.standard_normal(n_features), rng.standard_normal(n_features)
        # a.cos(wx) + b.sin(wx) = Re[(a - ib) e^{iwx}]
        g = (_fourier_basis(x, omega) @ (a - 1j * b)).real * np.sqrt(var / n_features)
        y += mult * g
    return y


class GaussianProcessGenerator(Generator):
    family = "gaussian_process"
    version = "0.1.0"

    def _sample(self, rng, length, context_length):
        max_exact = int(self.params.get("max_exact_length", 512))
        n_features = int(self.params.get("rff_features", 384))
        max_kernels = int(self.params.get("max_kernels", 4))
        bank = self.params.get("kernel_bank", BANK)
        if isinstance(bank, list):
            bank = {k: BANK[k] for k in bank}

        terms, expr = random_expression(rng, context_length, max_kernels, bank, self.period)
        x = np.arange(length, dtype=float) / context_length
        info: dict[str, Any]
        if length <= max_exact:
            f, info = sample_exact(rng, terms, x)
            info["method"] = "exact"
        else:
            f = sample_rff(rng, terms, x, n_features)
            info = {"method": "rff", "rff_features": n_features}
        level = float(rng.normal(0, 1))
        kinds = sorted({k.kind for t in terms for k in t})
        only_white = all(any(k.kind == "white" for k in t) for t in terms)
        meta = {
            "kind": expr if len(expr) < 60 else f"{len(terms)}terms:{'+'.join(kinds)}",
            "components": kinds,
            "predictability_hint": "unpredictable" if only_white else "medium",
            "latent_parameters": {
                "expression": expr,
                "terms": [[str(k) for k in t] for t in terms],
                "level": level,
                **info,
            },
        }
        return level + f, meta
