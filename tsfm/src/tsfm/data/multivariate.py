"""Multivariate relational generator.

Two levels, as in Timer-M1 / t0:
1. *temporal primitives*: independent univariate sources from the G* families;
2. *relational primitives*: operators that build targets from covariates.

Construction of one record with ``C`` channels:

- ``n_targets`` TARGET channels, the rest are covariates;
- each covariate is KNOWN_FUTURE (calendar / planned event / scheduled promo —
  information genuinely known at the cutoff) or PAST_ONLY (any source);
- a fraction of covariates are *decoys* (R8): no edge to any target;
- every target gets a covariate *strength*: ``null`` (no parent), ``weak`` or
  ``strong``, so each batch mixes useful, weak and useless relations;
- target = own univariate baseline + scaled sum of relational effects + noise.

Relational operators (codes R1–R8):

    R1 signed_mix       a * z_t
    R2 lag              a * z_{t - tau}
    R3 nonlinear        tanh / square / relu / threshold of z_{t - tau}
    R4 shared_driver    covariate = b z + eta, target uses latent z (z unobserved)
    R5 regime_response  a_{r_t} z_{t - tau}, r_t a 2-state Markov chain
    R6 event_effect     known event convolved with a carry-over kernel
    R7 cointegration    z random walk, covariate = z + u, target = beta z + v
    R8 decoy            covariate independent of every target

Targets may also depend on *earlier* targets with a strictly positive lag, so
the graph stays acyclic in computation order. Lags are applied on an
extended series and cropped, so the record has no undefined start.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import lfilter

from .episode import Role, Support, SyntheticRecord
from .generators.base import Generator, choose, log_uniform, sample_period, signed_log_uniform

PAST_ONLY_OPS = {"signed_mix": 0.2, "lag": 0.3, "nonlinear": 0.2, "regime_response": 0.1, "shared_driver": 0.1, "cointegration": 0.1}
KNOWN_FUTURE_OPS = {"event_effect": 0.5, "signed_mix": 0.3, "nonlinear": 0.2}
KNOWN_FUTURE_KINDS = {"calendar": 0.4, "planned_event": 0.35, "scheduled_promotion": 0.25}
STRENGTH = {"null": 0.1, "weak": 0.35, "strong": 0.55}
# ratio var(effects) / var(baseline) per strength
STRENGTH_RATIO = {"weak": (0.1, 0.5), "strong": (1.0, 10.0)}


def _standardize(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    return (x - float(np.mean(x))) / (s if s > 1e-9 else 1.0)


def _shift(z: np.ndarray, tau: int) -> np.ndarray:
    """z_{t - tau} with the first tau values repeated (only inside the cropped buffer)."""
    if tau == 0:
        return z
    return np.r_[np.full(tau, z[0]), z[:-tau]]


def known_future_signal(rng: np.random.Generator, n: int, context_length: int, kind: str) -> tuple[np.ndarray, dict]:
    """Information genuinely known in advance: calendars and plans."""
    t = np.arange(n)
    if kind == "calendar":
        # categorical calendar effect (e.g. day-of-week) encoded numerically,
        # or a smooth annual-like cycle
        if rng.random() < 0.6:
            P = int(rng.choice([7, 12, 24, 52]))
            pattern = rng.normal(0, 1, P)
            return pattern[(t + int(rng.integers(0, P))) % P], {"calendar": "categorical", "period": P}
        P, _ = sample_period(rng, context_length)
        return np.sin(2 * np.pi * t / P + rng.uniform(0, 2 * np.pi)), {"calendar": "smooth", "period": float(P)}
    if kind == "planned_event":
        p = log_uniform(rng, 0.005, 0.05)
        x = np.zeros(n)
        for s in np.flatnonzero(rng.random(n) < p):
            x[s : s + int(rng.integers(1, 8))] = 1.0
        return x, {"event_rate": p}
    if kind == "scheduled_promotion":
        every = int(log_uniform(rng, 14, max(15, context_length // 2)))
        dur = int(rng.integers(1, max(2, every // 3)))
        off = int(rng.integers(0, every))
        return (((t - off) % every) < dur).astype(float), {"every": every, "duration": dur}
    raise ValueError(kind)


class MultivariateGenerator(Generator):
    family = "multivariate"
    version = "0.1.0"

    def __init__(self, univariate: dict[str, Generator], univariate_mix: dict[str, float], **params: Any) -> None:
        super().__init__(**params)
        self.univariate = univariate
        self.univariate_mix = univariate_mix
        self.min_channels = int(params.get("min_channels", 2))
        self.max_channels = int(params.get("max_channels", 4))
        self.max_targets = int(params.get("max_targets", 2))
        self.p_known_future = float(params.get("known_future_probability", 0.35))
        self.decoy_fraction_range = tuple(params.get("decoy_fraction_range", (0.0, 0.5)))
        self.null_task_probability = float(params.get("minimum_decoy_fraction_of_tasks", 0.15))
        self.max_lag = int(params.get("max_lag", 32))
        self.past_only_ops = params.get("past_only_ops") or PAST_ONLY_OPS
        self.known_future_ops = params.get("known_future_ops") or KNOWN_FUTURE_OPS
        self.known_future_kinds = params.get("known_future_kinds") or KNOWN_FUTURE_KINDS
        self.strength = params.get("strength") or STRENGTH
        self.target_parent_probability = float(params.get("target_parent_probability", 0.2))

    # --------------------------------------------------------------- pieces
    def _source(self, rng: np.random.Generator, n: int, ctx: int) -> tuple[np.ndarray, dict]:
        fam = choose(rng, self.univariate_mix)
        rec = self.univariate[fam].sample(rng, n, ctx)
        return _standardize(rec.values[:, 0]), {"family": fam, "kind": rec.metadata["kind"]}

    def _stationary_noise(self, rng: np.random.Generator, n: int) -> np.ndarray:
        phi = float(rng.uniform(0.0, 0.9))
        return lfilter([1.0], [1.0, -phi], rng.standard_normal(n)) * np.sqrt(1 - phi**2)

    def _effect(self, rng, op: str, z: np.ndarray, ctx: int) -> tuple[np.ndarray, dict]:
        max_tau = max(0, min(self.max_lag, ctx // 4))
        p: dict[str, Any] = {"op": op}
        if op == "signed_mix":
            a = signed_log_uniform(rng, 0.3, 2.0)
            p["coef"] = a
            return a * z, p
        if op == "lag":
            tau = int(rng.integers(1, max_tau + 1)) if max_tau >= 1 else 0
            a = signed_log_uniform(rng, 0.3, 2.0)
            p.update(coef=a, lag=tau)
            return a * _shift(z, tau), p
        if op == "nonlinear":
            tau = int(rng.integers(0, max_tau + 1))
            f = choose(rng, {"tanh": 0.4, "square": 0.2, "relu": 0.2, "threshold": 0.2})
            zl = _shift(z, tau)
            a = log_uniform(rng, 0.5, 3.0)
            if f == "tanh":
                e = np.tanh(a * zl)
            elif f == "square":
                e = zl**2
            elif f == "relu":
                e = np.maximum(zl - float(rng.normal(0, 0.5)), 0.0)
            else:
                e = (zl > float(rng.normal(0, 0.5))).astype(float)
            sign = float(rng.choice([-1.0, 1.0]))
            p.update(function=f, lag=tau, gain=a, sign=sign)
            return sign * e, p
        if op == "regime_response":
            tau = int(rng.integers(0, max_tau + 1))
            p_stay = float(rng.uniform(0.9, 0.995))
            switches = rng.random(len(z)) > p_stay
            regime = np.cumsum(switches) % 2
            coefs = np.array([signed_log_uniform(rng, 0.2, 2.0), signed_log_uniform(rng, 0.2, 2.0)])
            p.update(lag=tau, p_stay=p_stay, coefs=coefs.tolist(), n_switches=int(switches.sum()))
            return coefs[regime] * _shift(z, tau), p
        if op == "event_effect":
            decay = float(rng.uniform(0.0, 0.9)) if rng.random() < 0.6 else 0.0
            a = signed_log_uniform(rng, 0.5, 3.0)
            e = lfilter([1.0], [1.0, -decay], z) if decay > 0 else z
            p.update(coef=a, carryover_decay=decay)
            return a * e, p
        raise ValueError(op)

    # --------------------------------------------------------------- main
    def _sample(self, rng, length, context_length):
        ctx = context_length
        B = self.max_lag + 1
        n = length + B
        C = int(rng.integers(self.min_channels, self.max_channels + 1))
        n_t = int(rng.integers(1, min(self.max_targets, C - 1) + 1))
        n_cov = C - n_t

        # ---------------- covariates
        null_task = rng.random() < self.null_task_probability
        decoy_frac = 1.0 if null_task else float(rng.uniform(*self.decoy_fraction_range))
        decoys = rng.random(n_cov) < decoy_frac
        if not null_task and decoys.all():  # outside null tasks, keep >= 1 useful covariate
            decoys[int(rng.integers(0, n_cov))] = False
        covs: list[dict[str, Any]] = []
        for j in range(n_cov):
            is_kf = rng.random() < self.p_known_future
            info: dict[str, Any] = {"role": int(Role.KNOWN_FUTURE if is_kf else Role.PAST_ONLY), "decoy": bool(decoys[j])}
            if is_kf:
                kind = choose(rng, self.known_future_kinds)
                x, kinfo = known_future_signal(rng, n, ctx, kind)
                info.update(construction="known_future", kind=kind, **kinfo)
            else:
                construction = "independent"
                if not info["decoy"]:
                    # the operator is decided here because R4/R7 shape the covariate itself
                    planned = choose(rng, self.past_only_ops)
                    info["planned_op"] = planned
                    if planned in ("shared_driver", "cointegration"):
                        construction = planned
                if construction == "shared_driver":
                    zl, sinfo = self._source(rng, n, ctx)
                    b = signed_log_uniform(rng, 0.5, 2.0)
                    x = b * zl + log_uniform(rng, 0.1, 1.0) * rng.standard_normal(n)
                    info.update(latent=zl, loading=b, **{f"latent_{k}": v for k, v in sinfo.items()})
                elif construction == "cointegration":
                    zl = np.cumsum(rng.standard_normal(n)) * 0.1
                    x = zl + 0.1 * self._stationary_noise(rng, n)
                    info.update(latent=zl)
                else:
                    x, sinfo = self._source(rng, n, ctx)
                    info.update(**{f"source_{k}": v for k, v in sinfo.items()})
                info["construction"] = construction
            info["values"] = x
            covs.append(info)

        # ---------------- targets
        targets: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        useful = [j for j, c in enumerate(covs) if not c["decoy"]]
        for i in range(n_t):
            strength = "null" if (null_task or not useful) else choose(rng, self.strength)
            parents: list[tuple[str, int]] = []
            if strength != "null":
                k = int(rng.integers(1, min(3, len(useful)) + 1))
                parents = [("cov", int(j)) for j in rng.choice(useful, size=k, replace=False)]
                if i > 0 and rng.random() < self.target_parent_probability:
                    parents.append(("target", int(rng.integers(0, i))))
            effects = np.zeros(n)
            coint_beta = None
            for kind, j in parents:
                if kind == "target":
                    tau = int(rng.integers(1, max(2, min(self.max_lag, ctx // 4)) + 1))
                    a = signed_log_uniform(rng, 0.3, 1.5)
                    e = a * _shift(targets[j]["values"], tau)
                    p = {"op": "lag", "coef": a, "lag": tau}
                else:
                    c = covs[j]
                    if c["construction"] == "shared_driver":
                        e, p = c["latent"], {"op": "shared_driver", "loading": c["loading"]}
                    elif c["construction"] == "cointegration":
                        coint_beta = signed_log_uniform(rng, 0.5, 2.0)
                        e, p = coint_beta * c["latent"], {"op": "cointegration", "beta": coint_beta}
                    elif c["role"] == Role.KNOWN_FUTURE:
                        e, p = self._effect(rng, choose(rng, self.known_future_ops), c["values"], ctx)
                    else:
                        e, p = self._effect(rng, c["planned_op"], c["values"], ctx)
                effects += e
                edges.append({"parent_kind": kind, "parent": j, "child_target": i, **p})

            if coint_beta is not None:  # baseline must be stationary for cointegration
                base, binfo = self._stationary_noise(rng, n), {"family": "stationary_ar1", "kind": "coint_residual"}
            else:
                base, binfo = self._source(rng, n, ctx)
            ratio = 0.0
            if strength != "null" and np.std(effects) > 1e-9:
                ratio = log_uniform(rng, *STRENGTH_RATIO[strength])
                if coint_beta is not None:
                    y = effects + np.sqrt(1.0 / ratio) * 0.1 * base  # keep the common trend dominant
                else:
                    y = base + np.sqrt(ratio) * _standardize(effects)
            else:
                strength = "null" if strength != "null" and np.std(effects) <= 1e-9 else strength
                y = base
            y = y + 0.05 * rng.standard_normal(n)
            level, scale = float(rng.normal(0, 3)), log_uniform(rng, 0.1, 10.0)
            targets.append({"values": level + scale * y, "strength": strength, "effect_ratio": ratio,
                            "baseline": binfo, "parents": [list(p) for p in parents]})

        # ---------------- assemble, crop the lag buffer, shuffle channels
        channels = [t["values"] for t in targets] + [c["values"] for c in covs]
        roles = [int(Role.TARGET)] * n_t + [c["role"] for c in covs]
        perm = rng.permutation(C)  # perm[new] = old
        inv = np.argsort(perm)  # inv[old] = new
        values = np.stack([channels[o] for o in perm], axis=1)[B:]
        roles = [roles[o] for o in perm]

        def new_index(kind: str, j: int) -> int:
            return int(inv[j if kind == "target" else n_t + j])

        graph_edges = [
            {**{k: v for k, v in e.items() if k not in ("parent", "child_target", "parent_kind")},
             "parent": new_index(e["parent_kind"], e["parent"]), "child": new_index("target", e["child_target"])}
            for e in edges
        ]
        nodes = []
        for old in range(C):
            if old < n_t:
                t = targets[old]
                nodes.append({"role": "TARGET", "strength": t["strength"], "effect_ratio": t["effect_ratio"], "baseline": t["baseline"]})
            else:
                c = covs[old - n_t]
                nodes.append({
                    "role": Role(c["role"]).name, "decoy": c["decoy"], "construction": c["construction"],
                    **{k: v for k, v in c.items() if k not in ("values", "latent", "role", "decoy", "construction")},
                })
        nodes = [nodes[o] | {"index": new} for new, o in enumerate(perm)]
        strengths = sorted({t["strength"] for t in targets})
        meta = {
            "kind": f"C{C}:T{n_t}:KF{sum(r == Role.KNOWN_FUTURE for r in roles)}:" + "/".join(strengths),
            "support": Support.REAL,
            "roles": roles,
            "components": sorted({e["op"] for e in graph_edges}) or ["independent"],
            "predictability_hint": "medium",
            "latent_parameters": {"null_task": bool(null_task), "decoy_fraction": decoy_frac},
            "graph": {"nodes": nodes, "edges": graph_edges},
        }
        return values, meta
