"""Sprint D5 acceptance tests.

"Done when: useful relation measurable, decoy useless, no future leak, variable C."
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from tsfm.data import RobustScaler, Role, collate_episodes
from tsfm.data.config import deep_merge, load_config
from tsfm.data.dataset import SyntheticEpisodeSource
from tsfm.data.evaluation import relation_table
from tsfm.data.multivariate import _shift
from tsfm.data.validation import check_episode

CFG = deep_merge(
    load_config(Path(__file__).parents[2] / "configs" / "data_v1.yaml"),
    {"curriculum": {"milestones": None}, "multivariate": {"probability": 1.0}},
)


@pytest.fixture(scope="module")
def episodes():
    src = SyntheticEpisodeSource(CFG, "debug")
    return [src.episode(i) for i in range(300)]


def test_variable_channels_and_roles(episodes):
    Cs = {e.n_channels for e in episodes}
    assert Cs == {2, 3, 4}
    for e in episodes:
        check_episode(e)
        assert (e.roles == Role.TARGET).sum() >= 1 and (e.roles != Role.TARGET).sum() >= 1
        assert len(e.meta["graph"]["nodes"]) == e.n_channels


def test_known_future_channels_are_calendar_or_plans_only(episodes):
    for e in episodes:
        for n in e.meta["graph"]["nodes"]:
            if n["role"] == "KNOWN_FUTURE":
                assert n["kind"] in {"calendar", "planned_event", "scheduled_promotion"}
                assert n["construction"] == "known_future"


def test_graph_edges_point_to_targets_and_respect_acyclicity(episodes):
    for e in episodes:
        for edge in e.meta["graph"]["edges"]:
            assert e.roles[edge["child"]] == Role.TARGET
            if e.roles[edge["parent"]] == Role.TARGET:  # target -> target needs a positive lag
                assert edge["op"] == "lag" and edge["lag"] >= 1
        decoys = [n["index"] for n in e.meta["graph"]["nodes"] if n.get("decoy")]
        assert not any(edge["parent"] in decoys for edge in e.meta["graph"]["edges"])


def test_strength_mix_contains_useful_weak_and_null(episodes):
    s = [n["strength"] for e in episodes for n in e.meta["graph"]["nodes"] if n["role"] == "TARGET"]
    frac = {k: s.count(k) / len(s) for k in ("null", "weak", "strong")}
    assert 0.15 < frac["null"] < 0.35 and frac["weak"] > 0.15 and frac["strong"] > 0.3


def test_no_future_leak_multivariate(episodes):
    rng = np.random.default_rng(0)
    for e in episodes[:150]:
        L = e.context_length
        hidden = [c for c in range(e.n_channels) if e.roles[c] != Role.KNOWN_FUTURE]
        e2 = copy.deepcopy(e)
        for c in hidden:  # targets and past-only covariates
            e2.values[c, L:] = rng.normal(0, 1e4, e.prediction_length)
        a, b = RobustScaler().transform(e), RobustScaler().transform(e2)
        assert np.array_equal(a["input_values"], b["input_values"])
        assert np.array_equal(a["scaling_stats"], b["scaling_stats"])
        kf = np.flatnonzero(e.roles == Role.KNOWN_FUTURE)
        assert e.observed_mask[kf, L:].all() if len(kf) else True


def test_lag_operator_offset():
    z = np.zeros(100)
    z[40] = 1.0
    assert np.argmax(_shift(z, 7)) == 47


def test_useful_relations_measurable_and_decoys_useless(episodes):
    rows = [r for e in episodes for r in relation_table(e)]
    kf_strong = [r["lift"] for r in rows if r["role"] == "KNOWN_FUTURE" and r["edge"] and r["strength"] == "strong"]
    kf_decoy = [r["lift"] for r in rows if r["role"] == "KNOWN_FUTURE" and r["op"] == "decoy"]
    po_strong = [r["history_dependence"] for r in rows if r["role"] == "PAST_ONLY" and r["edge"] and r["strength"] == "strong"]
    po_decoy = [r["history_dependence"] for r in rows if r["role"] == "PAST_ONLY" and r["op"] == "decoy"]
    assert len(kf_strong) > 30 and len(kf_decoy) > 20 and len(po_strong) > 50 and len(po_decoy) > 30
    assert np.median(kf_strong) > 0.10  # covariate lift on useful known-future drivers
    assert abs(np.median(kf_decoy)) < 0.03  # decoys bring nothing
    assert np.median(po_strong) > 2.5 * np.median(po_decoy)


def test_known_future_permutation_destroys_lift(episodes):
    """Shuffling a useful known-future driver in time must remove its lift."""
    from tsfm.data.evaluation import covariate_lift

    rng = np.random.default_rng(1)
    base, perm = [], []
    for e in episodes:
        for r in relation_table(e):
            if r["role"] == "KNOWN_FUTURE" and r["edge"] and r["strength"] == "strong":
                base.append(r["lift"])
                e2 = copy.deepcopy(e)
                e2.values[r["covariate"]] = rng.permutation(e2.values[r["covariate"]])
                perm.append(covariate_lift(e2, r["target"], [r["covariate"]]))
    assert np.median(perm) < 0.25 * np.median(base)


def test_collate_mixed_channel_counts(episodes):
    src = SyntheticEpisodeSource(deep_merge(CFG, {"multivariate": {"probability": 0.5}}), "debug")
    eps = [src.episode(i) for i in range(32)]
    b = collate_episodes(eps, pad_to_multiple=32)
    Cmax = max(e.n_channels for e in eps)
    assert b["input_values"].shape[1] == Cmax
    for i, e in enumerate(eps):
        assert (b["channel_roles"][i, e.n_channels :] == -1).all()
        assert not b["padding_mask"][i, e.n_channels :].any()
        assert not b["loss_mask"][i, : e.n_channels][e.roles != Role.TARGET].any()
