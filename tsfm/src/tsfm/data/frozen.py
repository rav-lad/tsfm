"""Frozen, immutable episode suites (reference §10.5, §11.1, sprint D4).

Layout of one suite directory::

    <suite>/episodes.npz   ragged arrays (values, masks, roles, offsets, L, H, C)
    <suite>/meta.jsonl     one JSON object per episode (full metadata)
    <suite>/manifest.json  provenance + sha256 of both files

``FrozenEpisodeSet`` refuses to load a suite whose files do not match the
manifest hashes, so a validation set cannot drift silently.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import scipy

from .episode import Episode

FORMAT_VERSION = 1


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(arrays: dict[str, np.ndarray], meta_lines: list[str]) -> str:
    """Deterministic hash of the suite *content* (independent of zip timestamps)."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    for line in meta_lines:
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def git_commit(cwd: Path | None = None) -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def write_suite(out_dir: str | Path, episodes: Iterable[Episode], manifest_extra: dict[str, Any]) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vals, obs, loss, roles, offs, loff = [], [], [], [], [0], [0]
    L, H, C, origin = [], [], [], []
    meta_lines = []
    for ep in episodes:
        vals.append(ep.values.ravel())
        obs.append(ep.observed_mask.ravel())
        loss.append(ep.loss_mask.ravel())
        roles.append(ep.roles)
        offs.append(offs[-1] + ep.values.size)
        loff.append(loff[-1] + ep.loss_mask.size)
        L.append(ep.context_length)
        H.append(ep.prediction_length)
        C.append(ep.n_channels)
        origin.append(ep.origin)
        meta_lines.append(json.dumps(ep.meta, default=_json_default, sort_keys=True))
    npz = out / "episodes.npz"
    arrays = dict(
        values=np.concatenate(vals).astype(np.float32),
        observed_mask=np.concatenate(obs),
        loss_mask=np.concatenate(loss),
        roles=np.concatenate(roles).astype(np.int8),
        offsets=np.array(offs, np.int64),
        loss_offsets=np.array(loff, np.int64),
        context_length=np.array(L, np.int32),
        prediction_length=np.array(H, np.int32),
        n_channels=np.array(C, np.int32),
        origin=np.array(origin, np.int64),
    )
    np.savez(npz, **arrays)
    meta_path = out / "meta.jsonl"
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    manifest = {
        "format_version": FORMAT_VERSION,
        "n_episodes": len(L),
        # file hashes detect corruption; the zip container embeds timestamps, so
        # reproducibility across builds is checked with content_sha256 instead
        "files": {"episodes.npz": sha256_file(npz), "meta.jsonl": sha256_file(meta_path)},
        "content_sha256": content_hash(arrays, meta_lines),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        **manifest_extra,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default, sort_keys=True), encoding="utf-8")
    return manifest


class FrozenEpisodeSet:
    def __init__(self, suite_dir: str | Path, verify: bool = True) -> None:
        self.dir = Path(suite_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        if verify:
            for name, digest in self.manifest["files"].items():
                actual = sha256_file(self.dir / name)
                if actual != digest:
                    raise ValueError(f"{self.dir / name}: sha256 mismatch (manifest {digest[:12]}, file {actual[:12]})")
        z = np.load(self.dir / "episodes.npz")
        self._a = {k: z[k] for k in z.files}
        self._meta = (self.dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()
        self._roff = np.r_[0, np.cumsum(self._a["n_channels"])]

    def __len__(self) -> int:
        return len(self._a["context_length"])

    def __getitem__(self, i: int) -> Episode:
        a = self._a
        if not 0 <= i < len(self):
            raise IndexError(i)
        C, L, H = int(a["n_channels"][i]), int(a["context_length"][i]), int(a["prediction_length"][i])
        s, e = a["offsets"][i], a["offsets"][i + 1]
        ls, le = a["loss_offsets"][i], a["loss_offsets"][i + 1]
        return Episode(
            values=a["values"][s:e].reshape(C, L + H).copy(),
            observed_mask=a["observed_mask"][s:e].reshape(C, L + H).copy(),
            loss_mask=a["loss_mask"][ls:le].reshape(C, H).copy(),
            roles=a["roles"][self._roff[i] : self._roff[i + 1]].copy(),
            context_length=L,
            prediction_length=H,
            origin=int(a["origin"][i]),
            meta=json.loads(self._meta[i]),
        )

    def __iter__(self) -> Iterator[Episode]:
        for i in range(len(self)):
            yield self[i]
