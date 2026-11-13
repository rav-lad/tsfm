"""G2 — complex periodicity and asymmetry (inspired by TempoPFN).

Waveforms are evaluated from a phase function, which allows frequency modulation
(slowly drifting period) and chirps. Sharp waveforms (square, sawtooth) are
rendered on an oversampled grid and box-averaged to reduce aliasing.
"""

from __future__ import annotations

import numpy as np

from .base import HINT_BY_NOISE, Generator, choose, log_uniform

OVERSAMPLE = 4

WAVEFORMS = {
    "sine": 0.10,
    "square": 0.15,
    "sawtooth": 0.15,
    "triangle": 0.10,
    "asymmetric_cycle": 0.15,
    "pulse_train": 0.15,
    "chirp": 0.10,
    "multi_period": 0.10,
}


def _wave(kind: str, cyc: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """``cyc`` is the phase in cycles (not radians). Returns values in ~[-1, 1]."""
    frac = np.mod(cyc, 1.0)
    info: dict = {}
    if kind == "sine":
        return np.sin(2 * np.pi * cyc), info
    if kind == "square":
        duty = float(rng.uniform(0.15, 0.85))
        info["duty_cycle"] = duty
        return np.where(frac < duty, 1.0, -1.0), info
    if kind == "sawtooth":
        direction = float(rng.choice([-1.0, 1.0]))
        info["direction"] = direction
        return direction * (2 * frac - 1), info
    if kind == "triangle":
        return 1 - 4 * np.abs(frac - 0.5), info
    if kind == "asymmetric_cycle":
        # fast rise, slow decay (or the reverse): skew parameter = rise fraction
        rise = float(rng.uniform(0.05, 0.4))
        if rng.random() < 0.5:
            rise = 1 - rise
        info["rise_fraction"] = rise
        up = frac / rise
        down = (1 - frac) / (1 - rise)
        tri = np.where(frac < rise, up, down)
        curv = float(rng.uniform(1.0, 3.0))
        info["curvature"] = curv
        return 2 * tri**curv - 1, info
    if kind == "pulse_train":
        width = float(rng.uniform(0.02, 0.15))
        center = float(rng.uniform(0.2, 0.8))
        info.update(width=width, center=center)
        d = (frac - center + 0.5) % 1.0 - 0.5
        return np.exp(-0.5 * (d / width) ** 2), info
    raise ValueError(kind)  # pragma: no cover


class ComplexPeriodicGenerator(Generator):
    family = "complex_periodic"
    version = "0.1.0"

    def _sample(self, rng, length, context_length):
        kind = choose(rng, self.params.get("waveforms", WAVEFORMS))
        n_fine = length * OVERSAMPLE
        t = np.arange(n_fine, dtype=float) / OVERSAMPLE  # time in steps
        lat: dict = {"waveform": kind}

        def phase_for(P: float, fm: bool) -> tuple[np.ndarray, dict]:
            """Cumulative phase (cycles) with optional slow frequency modulation."""
            f0 = 1.0 / P
            info: dict = {"period": P}
            if fm:
                depth = float(rng.uniform(0.05, 0.25))
                P_mod = log_uniform(rng, 2.0, 8.0) * context_length
                f = f0 * (1 + depth * np.sin(2 * np.pi * t / P_mod + rng.uniform(0, 2 * np.pi)))
                info.update(fm_depth=depth, fm_period=P_mod)
            else:
                f = np.full_like(t, f0)
            cyc = np.cumsum(f) / OVERSAMPLE + rng.uniform(0, 1)
            return cyc, info

        if kind == "chirp":
            P0, pinfo = self.period(rng, context_length, min_period=6.0)
            ratio = log_uniform(rng, 0.4, 2.5)  # final/initial frequency
            T_total = t[-1] + 1
            f = (1.0 / P0) * ratio ** (t / T_total)
            cyc = np.cumsum(f) / OVERSAMPLE
            base = np.sin(2 * np.pi * cyc)
            lat.update(initial_period=P0, frequency_ratio=ratio, **pinfo)
        elif kind == "multi_period":
            n = int(rng.integers(2, 4))
            base = np.zeros(n_fine)
            comps = []
            for _ in range(n):
                sub = choose(rng, {k: v for k, v in WAVEFORMS.items() if k not in ("chirp", "multi_period")})
                P, pinfo = self.period(rng, context_length)
                cyc, finfo = phase_for(P, fm=False)
                w, winfo = _wave(sub, cyc, rng)
                a = log_uniform(rng, 0.2, 1.0)
                base += a * w
                comps.append({"waveform": sub, "amplitude": a, **pinfo, **finfo, **winfo})
            lat["components"] = comps
        else:
            P, pinfo = self.period(rng, context_length)
            fm = rng.random() < 0.25
            cyc, finfo = phase_for(P, fm)
            base, winfo = _wave(kind, cyc, rng)
            lat.update(**pinfo, **finfo, **winfo)

        # anti-aliasing: average the oversampled signal down to the step grid
        base = base.reshape(length, OVERSAMPLE).mean(axis=1)

        # slow amplitude modulation
        am = choose(rng, {"none": 0.7, "sinusoidal": 0.2, "drift": 0.1})
        u = np.arange(length) / context_length
        if am == "sinusoidal":
            depth = float(rng.uniform(0.1, 0.6))
            env = 1 + depth * np.sin(2 * np.pi * u / log_uniform(rng, 1.0, 6.0) + rng.uniform(0, 2 * np.pi))
        elif am == "drift":
            env = np.exp(float(rng.normal(0, 0.5)) * u)
        else:
            env = 1.0
        A = log_uniform(rng, 0.2, 3.0)
        level = float(rng.normal(0, 2))
        trend = float(rng.normal(0, 0.5)) * u if rng.random() < 0.3 else 0.0
        signal = level + trend + A * env * base
        y, ninfo = self.noise(rng, signal, default={"low_noise": 0.5, "mid_noise": 0.35, "high_noise": 0.15})
        lat.update(amplitude=A, level=level, amplitude_modulation=am, **ninfo)
        meta = {
            "kind": kind,
            "components": [kind, "noise"],
            "predictability_hint": HINT_BY_NOISE[ninfo["noise_regime"]],
            "latent_parameters": lat,
        }
        return y, meta
