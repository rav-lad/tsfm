"""G7 — non-linear compositions of other families.

    y_t = T(t) + A(t) S(t) + R(t) + eps_t, then optional transforms

Components are drawn from the *configured* base generators (so holdout
exclusions propagate), standardized, then combined with one of:

- additive        : sum_i w_i c_i
- modulated       : c1 * (1 + a * tanh(c2))            amplitude modulation A(t)S(t)
- heteroscedastic : c1 + softplus(c2) * noise         variance driven by a component
- trend_noise     : c1 + c2 + |c1| * small noise      noise level depends on the level
- time_warp       : c1 evaluated on a monotone warped clock (+ c2)

followed by 0–2 transforms: saturation (tanh), soft-clip, softplus (positive
support), signed power, local clipping (sensor saturation on a window).

Metadata keeps the *dominant* family (first component) and the secondary ones,
so ablations can still attribute results.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..episode import Support
from .base import Generator, choose, log_uniform

COMBINATIONS = {"additive": 0.3, "modulated": 0.25, "heteroscedastic": 0.15, "trend_noise": 0.1, "time_warp": 0.2}
TRANSFORMS = {"saturation": 0.25, "soft_clip": 0.2, "softplus": 0.2, "signed_power": 0.2, "local_clip": 0.15}


def _std(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    return (x - float(np.mean(x))) / (s if s > 1e-9 else 1.0)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


class CompositeGenerator(Generator):
    family = "composite"
    version = "0.1.0"

    def __init__(self, components: dict[str, Generator] | None = None, component_mix: dict[str, float] | None = None, **params: Any) -> None:
        super().__init__(**params)
        if components is None:  # standalone use: default base generators
            from . import REGISTRY

            components = {f: cls() for f, cls in REGISTRY.items() if f != self.family}
        self.components = {f: g for f, g in components.items() if f != self.family}
        mix = component_mix or {f: 1.0 for f in self.components}
        mix = {f: w for f, w in mix.items() if f in self.components}
        total = sum(mix.values())
        self.component_mix = {f: w / total for f, w in mix.items()}

    def _component(self, rng, n, ctx) -> tuple[np.ndarray, dict]:
        fam = choose(rng, self.component_mix)
        rec = self.components[fam].sample(rng, n, ctx)
        return _std(rec.values[:, 0]), {"family": fam, "kind": rec.metadata["kind"]}

    def _sample(self, rng, length, context_length):
        n = length
        comb = choose(rng, self.params.get("combinations", COMBINATIONS))
        c1, i1 = self._component(rng, n, context_length)
        c2, i2 = self._component(rng, n, context_length)
        comps = [i1, i2]
        p: dict[str, Any] = {"combination": comb}

        if comb == "additive":
            ws = [1.0, float(rng.choice([-1, 1])) * log_uniform(rng, 0.2, 1.5)]
            y = c1 + ws[1] * c2
            if rng.random() < 0.4:
                c3, i3 = self._component(rng, n, context_length)
                w3 = float(rng.choice([-1, 1])) * log_uniform(rng, 0.2, 1.0)
                y, ws = y + w3 * c3, ws + [w3]
                comps.append(i3)
            p["weights"] = ws
        elif comb == "modulated":
            a = float(rng.uniform(0.2, 0.9))
            y = c1 * (1 + a * np.tanh(c2))
            p["depth"] = a
        elif comb == "heteroscedastic":
            g = log_uniform(rng, 0.1, 1.0)
            y = c1 + g * _softplus(c2) * rng.standard_normal(n)
            p["gain"] = g
        elif comb == "trend_noise":
            g = log_uniform(rng, 0.05, 0.5)
            y = c1 + 0.5 * c2 + g * np.abs(c1) * rng.standard_normal(n)
            p["gain"] = g
        else:  # time_warp: monotone clock u(t) = t + amp * sin(...) with du/dt > 0
            amp = float(rng.uniform(0.1, 0.8))
            P = log_uniform(rng, 0.5, 4.0) * context_length
            t = np.arange(n, dtype=float)
            u = t + amp * P / (2 * np.pi) * np.sin(2 * np.pi * t / P + rng.uniform(0, 2 * np.pi))
            u = (u - u.min()) / (u.max() - u.min()) * (n - 1)  # keep within the grid
            y = np.interp(u, t, c1) + 0.3 * c2
            p.update(warp_amplitude=amp, warp_period=P)

        support = Support.REAL
        transforms = []
        for _ in range(int(rng.choice([0, 1, 1, 2]))):
            tr = choose(rng, TRANSFORMS)
            y = _std(y)
            if tr == "saturation":
                g = log_uniform(rng, 0.3, 2.0)
                y = np.tanh(g * y) / g
                transforms.append({"op": tr, "gain": g})
            elif tr == "soft_clip":
                c = float(rng.uniform(0.8, 2.5))
                y = c * np.tanh(y / c)
                transforms.append({"op": tr, "level": c})
            elif tr == "softplus":
                s = log_uniform(rng, 0.5, 3.0)
                y = _softplus(s * y) / s
                support = Support.NONNEGATIVE
                transforms.append({"op": tr, "sharpness": s})
            elif tr == "signed_power":
                e = log_uniform(rng, 0.5, 2.0)
                y = np.sign(y) * np.abs(y) ** e
                transforms.append({"op": tr, "exponent": e})
            else:  # local clipping on one window (sensor saturation), not global
                w = int(n * rng.uniform(0.05, 0.3))
                s = int(rng.integers(0, n - w))
                q = float(rng.uniform(0.6, 0.95))
                hi = float(np.quantile(y[s : s + w], q))
                y = y.copy()
                y[s : s + w] = np.minimum(y[s : s + w], hi)
                transforms.append({"op": tr, "start": s, "width": w, "quantile": q})
            if support == Support.NONNEGATIVE and tr != "softplus":
                support = Support.REAL  # later transforms may break positivity

        # Components already carry their own noise: extra noise only sometimes,
        # otherwise compositions drift toward pure noise.
        ninfo, level = {"noise_regime": "none"}, 0.0
        if support == Support.REAL:
            if rng.random() < 0.3 or self.noise_regimes:
                y, ninfo = self.noise(rng, y, default={"low_noise": 0.6, "mid_noise": 0.4})
            level = float(rng.normal(0, 2))
            y = y + level
        meta = {
            "kind": comb + ("+" + "+".join(t["op"] for t in transforms) if transforms else ""),
            "support": support,
            "components": [c["family"] for c in comps],
            "dominant_family": comps[0]["family"],
            "predictability_hint": "hard" if comb in ("heteroscedastic", "trend_noise") else "medium",
            "latent_parameters": {**p, "component_details": comps, "transforms": transforms, "level": level, **ninfo},
        }
        return y, meta
