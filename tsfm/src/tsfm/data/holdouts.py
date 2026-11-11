"""Build the synthetic evaluation suites declared in a holdout config (sprint D4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import config_hash, deep_merge, load_config
from .dataset import SyntheticEpisodeSource
from .frozen import git_commit, write_suite
from .generators import REGISTRY
from .multivariate import MultivariateGenerator


def suite_configs(holdout_path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return (holdout spec, {suite: {"split", "n", "config"}})."""
    holdout_path = Path(holdout_path)
    spec = load_config(holdout_path)
    base = load_config(holdout_path.parent / spec["base_config"])
    base = deep_merge(base, spec.get("common_overrides"))
    suites = {}
    for name, s in spec["suites"].items():
        suites[name] = {"split": s["split"], "n": int(s["n"]), "config": deep_merge(base, s.get("overrides"))}
    return spec, suites


def build_suite(name: str, suite: dict[str, Any], spec: dict[str, Any], out_root: str | Path, n: int | None = None) -> dict[str, Any]:
    cfg = suite["config"]
    src = SyntheticEpisodeSource(cfg, split=suite["split"])
    n = int(n or suite["n"])
    versions = {f: g.version for f, g in REGISTRY.items()} | {"multivariate": MultivariateGenerator.version}
    extra = {
        "suite": name,
        "holdout_version": spec.get("version"),
        "split": suite["split"],
        "seed": src.seed,
        "config_sha256": config_hash(cfg),
        "config": cfg,
        "effective_mix": src.mix,
        "dropped_families": src.dropped_families,
        "generator_versions": versions,
        "git": git_commit(Path(__file__).parent),
    }
    return write_suite(Path(out_root) / name, (src.episode(i) for i in range(n)), extra)
