"""G3 tests: the RFF path must reproduce the exact covariance."""

from __future__ import annotations

import numpy as np
import pytest

from tsfm.data.generators.gaussian_process import (
    GaussianProcessGenerator,
    Kernel,
    covariance,
    random_expression,
    sample_exact,
    sample_rff,
)

X = np.arange(60, dtype=float) / 128

CASES = {
    "rbf": [[Kernel("rbf", {"variance": 1.3, "lengthscale": 0.1})]],
    "matern12": [[Kernel("matern", {"variance": 1.0, "lengthscale": 0.05, "nu": 0.5})]],
    "matern52": [[Kernel("matern", {"variance": 0.7, "lengthscale": 0.2, "nu": 2.5})]],
    "periodic": [[Kernel("periodic", {"variance": 1.0, "period": 0.15, "lengthscale": 0.7})]],
    "rq": [[Kernel("rational_quadratic", {"variance": 1.0, "lengthscale": 0.1, "alpha": 0.5})]],
    "rbf*periodic": [[Kernel("rbf", {"variance": 1.0, "lengthscale": 0.3}),
                      Kernel("periodic", {"variance": 0.8, "period": 0.1, "lengthscale": 1.0})]],
    "linear*rbf+white": [
        [Kernel("linear", {"variance": 1.5, "offset": 0.2}), Kernel("rbf", {"variance": 1.0, "lengthscale": 0.2})],
        [Kernel("white", {"variance": 0.1})],
    ],
    "white*periodic": [[Kernel("white", {"variance": 0.2}),
                        Kernel("periodic", {"variance": 2.0, "period": 0.1, "lengthscale": 1.0})]],
}


@pytest.mark.parametrize("name", list(CASES))
def test_rff_covariance_matches_exact(name):
    terms = CASES[name]
    K = covariance(terms, X)
    rng = np.random.default_rng(0)
    S = np.stack([sample_rff(rng, terms, X, n_features=256) for _ in range(4000)])
    K_hat = S.T @ S / len(S)
    err = np.abs(K_hat - K).max() / np.abs(np.diag(K)).max()
    assert err < 0.12, f"{name}: relative max error {err:.3f}"


@pytest.mark.parametrize("name", ["rbf", "periodic", "linear*rbf+white"])
def test_exact_sampler_covariance(name):
    terms = CASES[name]
    K = covariance(terms, X)
    rng = np.random.default_rng(1)
    S = np.stack([sample_exact(rng, terms, X)[0] for _ in range(4000)])
    err = np.abs(S.T @ S / len(S) - K).max() / np.abs(np.diag(K)).max()
    assert err < 0.12


def test_random_covariances_are_psd():
    rng = np.random.default_rng(2)
    x = np.arange(200, dtype=float) / 128
    for _ in range(200):
        terms, _ = random_expression(rng, 128, 4, {"rbf": 1, "matern": 1, "periodic": 1,
                                                   "rational_quadratic": 1, "linear": 1, "white": 1})
        K = covariance(terms, x)
        assert np.allclose(K, K.T)
        w = np.linalg.eigvalsh(K)
        assert w.min() > -1e-6 * max(w.max(), 1e-12)


def test_method_switch_and_fallback_rate():
    gen = GaussianProcessGenerator(max_exact_length=512)
    short = [gen.sample(np.random.default_rng(s), 400, 256).metadata["latent_parameters"] for s in range(100)]
    long = [gen.sample(np.random.default_rng(s), 1300, 1024).metadata["latent_parameters"] for s in range(30)]
    assert all(m["method"] == "exact" for m in short)
    assert all(m["method"] == "rff" for m in long)
    assert np.mean([m["psd_fallback"] for m in short]) < 0.02
