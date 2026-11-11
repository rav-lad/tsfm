"""Contract checks for records and episodes (reference §12.1, §12.4).

Rejection policy (§9.4): only *implementation* anomalies are errors (NaN/Inf,
overflow, shape/mask violations, support violations). Hard or noisy series are
legitimate and never rejected here.
"""

from __future__ import annotations

import numpy as np

from .episode import Episode, Role, Support, SyntheticRecord

MAX_ABS = 1e12


class ContractError(ValueError):
    pass


def check_record(record: SyntheticRecord) -> None:
    v = record.values
    if v.ndim != 2:
        raise ContractError(f"record must be [T, C], got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ContractError("non-finite values in record")
    if np.max(np.abs(v)) > MAX_ABS:
        raise ContractError("overflow: |value| > 1e12")
    support = record.metadata.get("support", Support.REAL)
    if support in (Support.NONNEGATIVE, Support.INTEGER) and np.min(v) < 0:
        raise ContractError(f"{support} record has negative values")
    if support == Support.INTEGER and not np.all(v == np.round(v)):
        raise ContractError("integer record has non-integer values")
    for key in ("family", "generator_version", "support", "latent_parameters"):
        if key not in record.metadata:
            raise ContractError(f"missing metadata key {key!r}")


def check_episode(ep: Episode) -> None:
    C, LH = ep.values.shape
    L, H = ep.context_length, ep.prediction_length
    if LH != L + H:
        raise ContractError("values width != L + H")
    if ep.values.dtype != np.float32:
        raise ContractError("values must be float32")
    if ep.observed_mask.shape != (C, L + H) or ep.observed_mask.dtype != bool:
        raise ContractError("observed_mask must be bool [C, L+H]")
    if ep.loss_mask.shape != (C, H) or ep.loss_mask.dtype != bool:
        raise ContractError("loss_mask must be bool [C, H]")
    if ep.roles.shape != (C,):
        raise ContractError("roles must be [C]")
    if not np.all(np.isin(ep.roles, [int(r) for r in Role])):
        raise ContractError("unknown role")
    if not np.any(ep.roles == Role.TARGET):
        raise ContractError("episode without target")
    hidden_future = ep.roles != Role.KNOWN_FUTURE
    if np.any(ep.observed_mask[hidden_future, L:]):
        raise ContractError("LEAK: future of a target/past-only channel is marked observed")
    if np.any(ep.loss_mask[ep.roles != Role.TARGET]):
        raise ContractError("loss on a non-target channel")
    if not np.all(np.isfinite(ep.values[ep.observed_mask])):
        raise ContractError("non-finite observed value")
    if ep.observed_mask[:, :L].sum(axis=1).min() < 1:
        raise ContractError("channel with empty visible history")
