"""Config loading, overlays and hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from numbers import Number
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_distribution(d: Any) -> bool:
    """A mapping whose values are all numbers (a categorical distribution)."""
    return isinstance(d, dict) and len(d) > 0 and all(isinstance(v, Number) and not isinstance(v, bool) for v in d.values())


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Recursive merge of ``override`` into a copy of ``base``.

    When both sides are distributions (number-valued mappings such as
    ``generator_mix`` or ``waveforms``), the override *replaces* the base
    wholesale, so an overlay like ``{chirp: 1.0}`` really means "chirp only".
    Structured sections (e.g. ``multivariate``) are always merged.
    """
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        b = out.get(k)
        if isinstance(v, dict) and isinstance(b, dict) and not (_is_distribution(v) and _is_distribution(b)):
            out[k] = deep_merge(b, v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
