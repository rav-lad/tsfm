"""Sprint D1 acceptance tests (reference §11.3, §16 D1).

"Done when: future-invariance test + replay test + observed-zero vs missing pass."
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from tsfm.data import (
    Episode,
    Role,
    RobustScaler,
    SyntheticEpisodeSource,
    collate_episodes,
    load_config,
)
from tsfm.data.validation import ContractError, check_episode

CONFIG = Path(__file__).parents[2] / "configs" / "data_v0.yaml"


@pytest.fixture(scope="module")
def source():
    return SyntheticEpisodeSource(load_config(CONFIG), split="train")


@pytest.fixture(scope="module")
def episodes(source):
    return [source.episode(i) for i in range(300)]


def _perturb_future(ep: Episode, channels, rng) -> Episode:
    ep2 = copy.deepcopy(ep)
    L = ep.context_length
    for c in channels:
        ep2.values[c, L:] = rng.normal(1e3, 1e3, ep.prediction_length).astype(np.float32)
    return ep2


# --------------------------------------------------------------- 1, 2, 4: future invariance
def test_changing_target_future_does_not_change_model_input(episodes):
    rng = np.random.default_rng(0)
    scaler = RobustScaler()
    for ep in episodes:
        ep2 = _perturb_future(ep, ep.target_channels, rng)
        assert np.array_equal(ep.model_input(), ep2.model_input())
        a, b = scaler.transform(ep), scaler.transform(ep2)
        assert np.array_equal(a["input_values"], b["input_values"])
        assert np.array_equal(a["input_mask"], b["input_mask"])
        assert np.array_equal(a["scaling_stats"], b["scaling_stats"])  # test 4
        # labels did change, so the test is not vacuous
        assert not np.array_equal(a["target_future"], b["target_future"])


def test_changing_missing_history_does_not_change_input(episodes):
    rng = np.random.default_rng(1)
    checked = 0
    for ep in episodes:
        hidden = ~ep.observed_mask[:, : ep.context_length]
        if not hidden.any():
            continue
        ep2 = copy.deepcopy(ep)
        ep2.values[:, : ep.context_length][hidden] = rng.normal(0, 1e6, hidden.sum())
        assert np.array_equal(RobustScaler().transform(ep)["input_values"], RobustScaler().transform(ep2)["input_values"])
        assert np.array_equal(RobustScaler().stats(ep)[1], RobustScaler().stats(ep2)[1])
        checked += 1
    assert checked > 20


def _two_channel_episode(L=64, H=16):
    rng = np.random.default_rng(3)
    values = rng.normal(size=(3, L + H)).astype(np.float32)
    roles = np.array([Role.TARGET, Role.PAST_ONLY, Role.KNOWN_FUTURE], np.int8)
    obs = np.zeros((3, L + H), bool)
    obs[:, :L] = True
    obs[2, L:] = True
    loss = np.zeros((3, H), bool)
    loss[0] = True
    return Episode(values, obs, loss, roles, L, H, origin=L, meta={})


def test_past_only_future_is_hidden_and_known_future_is_visible():
    ep = _two_channel_episode()
    check_episode(ep)
    rng = np.random.default_rng(4)
    ep_po = _perturb_future(ep, [1], rng)  # test 2: past-only future
    assert np.array_equal(RobustScaler().transform(ep)["input_values"], RobustScaler().transform(ep_po)["input_values"])
    ep_kf = _perturb_future(ep, [2], rng)  # test 3: known-future may change input
    assert not np.array_equal(RobustScaler().transform(ep)["input_values"], RobustScaler().transform(ep_kf)["input_values"])
    t = RobustScaler().transform(ep)
    assert not t["target_future"][1:].any()  # non-target labels zeroed


def test_contract_detects_leak():
    ep = _two_channel_episode()
    ep.observed_mask[0, ep.context_length] = True
    with pytest.raises(ContractError, match="LEAK"):
        check_episode(ep)


# --------------------------------------------------------------- 5: replay / determinism
def test_replay_is_bitwise_identical(source):
    cfg = load_config(CONFIG)
    other = SyntheticEpisodeSource(cfg, split="train")
    for i in [0, 7, 123, 10_000, 99_999]:
        a, b = source.episode(i), other.episode(i)
        assert np.array_equal(a.values, b.values)
        assert np.array_equal(a.observed_mask, b.observed_mask)
        assert a.meta["record_seed"] == b.meta["record_seed"]


def test_order_independence(source):
    fwd = [source.episode(i).values for i in range(20)]
    bwd = [source.episode(i).values for i in reversed(range(20))][::-1]
    assert all(np.array_equal(x, y) for x, y in zip(fwd, bwd))


def test_splits_use_disjoint_streams():
    cfg = load_config(CONFIG)
    tr = SyntheticEpisodeSource(cfg, "train")
    va = SyntheticEpisodeSource(cfg, "val_id")
    seeds_tr = {tr.episode(i).meta["record_seed"] for i in range(200)}
    seeds_va = {va.episode(i).meta["record_seed"] for i in range(200)}
    assert not seeds_tr & seeds_va


# --------------------------------------------------------------- observed zero vs missing
def test_observed_zero_is_distinguishable_from_missing():
    L, H = 64, 16
    values = np.ones((1, L + H), np.float32)
    values[0, 10] = 0.0  # a true observed zero
    obs = np.zeros((1, L + H), bool)
    obs[0, :L] = True
    obs[0, 20] = False  # a missing value
    ep = Episode(values, obs, np.ones((1, H), bool), np.array([Role.TARGET], np.int8), L, H, L, {})
    t = RobustScaler(method="mean_std").transform(ep)
    assert t["input_mask"][0, 10] and not t["input_mask"][0, 20]
    assert t["input_values"][0, 20] == 0.0


def test_intermittent_counts_keep_true_zeros_observed(source):
    eps = [source.episode(i, family="events_counts") for i in range(300)]
    zeros = [e for e in eps if e.meta["kind"] == "intermittent_demand"]
    assert zeros
    for e in zeros:
        h = e.values[0, : e.context_length]
        m = e.observed_mask[0, : e.context_length]
        assert np.any((h == 0) & m)  # zeros are observed values, not missing


# --------------------------------------------------------------- normalization
def test_constant_series_uses_fallback_scale():
    L, H = 64, 8
    ep = Episode(
        np.full((1, L + H), 5.0, np.float32),
        np.r_[np.ones(L, bool), np.zeros(H, bool)][None],
        np.ones((1, H), bool),
        np.array([Role.TARGET], np.int8),
        L, H, L, {},
    )
    loc, scale, fb = RobustScaler().stats(ep)
    assert loc[0] == 5.0 and scale[0] == 1.0 and fb[0]


# --------------------------------------------------------------- collate
def test_collate_alignment_and_masks(episodes):
    batch = collate_episodes(episodes[:32], pad_to_multiple=32)
    B, C, W = batch["input_values"].shape
    H_max = batch["target_future"].shape[-1]
    L_max = W - H_max
    assert L_max % 32 == 0 and H_max % 32 == 0
    for b, ep in enumerate(episodes[:32]):
        L, H = ep.context_length, ep.prediction_length
        assert batch["padding_mask"][b, 0, L_max - L : L_max + H].all()
        assert not batch["padding_mask"][b, 0, : L_max - L].any()
        assert not batch["input_mask"][b, 0, L_max:].any()  # no target future visible
        assert not batch["input_values"][b, 0, L_max:].any()
        assert batch["loss_mask"][b, 0].sum() == H
    # padded input positions are never observed
    assert not (batch["input_mask"] & ~batch["padding_mask"]).any()
