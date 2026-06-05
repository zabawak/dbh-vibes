"""Tests for the hardened team clustering core (pure numpy/sklearn — no torch, no video).

These exercise the logic that made team ID unstable (docs/team-clustering.md) on *synthetic*
embeddings, so they need no labelled data and no model:

- per-track aggregation (one vote per player, not per crop),
- determinism / run-to-run stability on identical input,
- clean two-team separation,
- the over-segment + size-merge fix for the degenerate "goalies become a team" split,
- colour-anchored stable T0/T1 labels,
- the label-free quality + confidence signals.
"""

from __future__ import annotations

import numpy as np

from dbh_vibes.team_siglip import (
    ClusterInfo,
    aggregate_track_embeddings,
    cluster_team_embeddings,
    order_labels_by_color,
    team_confidence,
    torso_color_hsv,
)

DIM = 32


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _blob(center: np.ndarray, n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """n embeddings clustered around a unit center with small gaussian noise."""
    pts = center[None, :] + noise * rng.standard_normal((n, len(center)))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _two_team_centers() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros(DIM); a[0] = 1.0
    b = np.zeros(DIM); b[1] = 1.0
    return a, b


def _partition_matches(labels: np.ndarray, group_a: range, group_b: range) -> bool:
    """True if every member of group_a shares a label and group_b shares the *other* label."""
    la = set(labels[list(group_a)])
    lb = set(labels[list(group_b)])
    return len(la) == 1 and len(lb) == 1 and la != lb


# ---- per-track aggregation -------------------------------------------------------------------

def test_aggregate_pools_per_track_and_normalises():
    # Track 1 has 3 crops, track 2 has 1 — aggregation must give one row each, unit norm.
    crops = np.array([[3.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0], [1.0, 1.0, 0]])
    owners = [1, 1, 1, 2]
    out = aggregate_track_embeddings(crops, owners, track_ids=[1, 2])
    assert out.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_aggregate_is_robust_to_crop_count():
    # The same track contributing 1 vs 50 identical-direction crops yields the same embedding —
    # this is the property that made per-crop clustering crop-count sensitive.
    rng = np.random.default_rng(0)
    center = _unit(rng.standard_normal(DIM))
    few = _blob(center, 1, 0.0, rng)
    many = _blob(center, 50, 0.0, rng)
    a = aggregate_track_embeddings(few, [7] * len(few), [7])
    b = aggregate_track_embeddings(many, [7] * len(many), [7])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_aggregate_empty():
    assert aggregate_track_embeddings(np.empty((0, DIM)), [], []).shape[0] == 0


# ---- clean two-team separation ---------------------------------------------------------------

def test_two_clean_teams_split_correctly():
    rng = np.random.default_rng(1)
    a, b = _two_team_centers()
    emb = np.vstack([_blob(a, 8, 0.05, rng), _blob(b, 8, 0.05, rng)])
    labels, info = cluster_team_embeddings(emb)
    assert _partition_matches(labels, range(0, 8), range(8, 16))
    assert info.n_micro == 2              # two clean teams: no spurious over-segmentation
    assert info.silhouette > 0.5          # well separated
    assert info.team_sizes == (8, 8)


def test_clustering_is_deterministic():
    rng = np.random.default_rng(2)
    a, b = _two_team_centers()
    emb = np.vstack([_blob(a, 7, 0.08, rng), _blob(b, 9, 0.08, rng)])
    l1, _ = cluster_team_embeddings(emb)
    l2, _ = cluster_team_embeddings(emb)
    np.testing.assert_array_equal(l1, l2)   # identical input -> identical teams, every run


# ---- the degenerate-split fix: goalies must not become a team --------------------------------

def test_goalie_outliers_do_not_form_a_team():
    # Two skater teams of 10, plus 2 goalies whose gear is visually distinct (a third, far blob).
    # A naive KMeans(k=2) lumps the 20 skaters into one cluster and the 2 goalies into the other —
    # the 28-vs-6 style failure. Over-segment + size-merge must instead split the skaters and fold
    # the goalies into a team.
    rng = np.random.default_rng(3)
    a, b = _two_team_centers()
    goalie = np.zeros(DIM); goalie[2] = 1.0
    emb = np.vstack([
        _blob(a, 10, 0.05, rng),
        _blob(b, 10, 0.05, rng),
        _blob(goalie, 2, 0.05, rng),
    ])
    labels, info = cluster_team_embeddings(emb)

    assert info.n_micro == 3                       # the goalie group was peeled off, not merged in
    assert _partition_matches(labels, range(0, 10), range(10, 20))  # skaters split cleanly
    assert min(info.team_sizes) >= 10              # neither "team" is just the goalies
    assert sum(info.team_sizes) == 22


# ---- scale-decorrelation: crop near/far must not drive the split ------------------------------

def test_scale_does_not_hijack_split_when_sizes_given():
    # Build embeddings where a strong "scale" axis dominates (mimicking near-vs-far fisheye crops),
    # plus a weak kit axis. Crucially, size is independent of team (both teams span all sizes).
    rng = np.random.default_rng(5)
    n = 10
    kit_dir = np.zeros(DIM); kit_dir[0] = 1.0
    scale_dir = np.zeros(DIM); scale_dir[5] = 1.0
    sizes = np.concatenate([np.geomspace(500, 50000, n), np.geomspace(500, 50000, n)])
    kit_sign = np.array([1.0] * n + [-1.0] * n)            # first n = team A, next n = team B
    lsz = (np.log(sizes) - np.log(sizes).mean()) / np.log(sizes).std()
    emb = (0.30 * kit_sign[:, None] * kit_dir[None, :]
           + 2.0 * lsz[:, None] * scale_dir[None, :]
           + 0.03 * rng.standard_normal((2 * n, DIM)))

    # Without sizes the scale axis dominates and the split tracks size, not kit.
    base_labels, _ = cluster_team_embeddings(emb)
    base_size_corr = abs(np.corrcoef(base_labels, lsz)[0, 1])
    assert base_size_corr > 0.6                            # baseline is hijacked by scale

    # With per-track sizes, the scale-correlated PCs are dropped and the split follows the kit.
    labels, _ = cluster_team_embeddings(emb, sizes=sizes)
    assert _partition_matches(labels, range(0, n), range(n, 2 * n))   # teams recovered
    assert abs(np.corrcoef(labels, lsz)[0, 1]) < 0.4                  # no longer size-driven


def test_sizes_dont_break_a_clean_color_split():
    # When kit (not scale) is the real structure, passing sizes must not harm a good split.
    rng = np.random.default_rng(6)
    a, b = _two_team_centers()
    emb = np.vstack([_blob(a, 8, 0.05, rng), _blob(b, 8, 0.05, rng)])
    sizes = rng.uniform(1000, 40000, size=16)   # random sizes, uncorrelated with team
    labels, _ = cluster_team_embeddings(emb, sizes=sizes)
    assert _partition_matches(labels, range(0, 8), range(8, 16))


# ---- colour-anchored stable labels -----------------------------------------------------------

def test_labels_anchored_to_saturation():
    # Team with raw label 1 wears the saturated (pinnie) kit -> it must become stable T0.
    labels = np.array([0, 0, 1, 1])
    bright_pinnie = [120.0, 220.0, 200.0]   # [hue, high sat, val]
    dull = [60.0, 30.0, 90.0]               # low saturation
    colors = np.array([dull, dull, bright_pinnie, bright_pinnie])
    remap = order_labels_by_color(labels, colors)
    assert remap[1] == 0    # saturated kit -> T0
    assert remap[0] == 1


def test_label_anchor_is_orientation_independent():
    # Same two colour groups, raw KMeans ids swapped: the saturated team is still T0.
    colors_grp_sat = [120.0, 220.0, 200.0]
    colors_grp_dull = [60.0, 30.0, 90.0]
    r1 = order_labels_by_color(np.array([0, 0, 1, 1]),
                               np.array([colors_grp_sat, colors_grp_sat,
                                         colors_grp_dull, colors_grp_dull]))
    r2 = order_labels_by_color(np.array([1, 1, 0, 0]),
                               np.array([colors_grp_sat, colors_grp_sat,
                                         colors_grp_dull, colors_grp_dull]))
    # In r1 the saturated team is raw-0; in r2 it is raw-1. Both must map the saturated team to T0.
    assert r1[0] == 0 and r2[1] == 0


# ---- label-free quality + confidence ---------------------------------------------------------

def test_confidence_high_for_well_separated_tracks():
    rng = np.random.default_rng(4)
    a, b = _two_team_centers()
    emb = np.vstack([_blob(a, 6, 0.03, rng), _blob(b, 6, 0.03, rng)])
    labels, info = cluster_team_embeddings(emb)
    conf = team_confidence(info, labels)
    assert conf.shape == (12,)
    assert conf.min() >= 0.0 and conf.max() <= 1.0
    assert conf.mean() > 0.7    # clean split -> confident


def test_confidence_defaults_when_single_team():
    info = ClusterInfo(0.0, (1, 0), 1)
    conf = team_confidence(info, np.zeros(1, dtype=int))
    assert conf.tolist() == [0.5]


# ---- torso colour helper ---------------------------------------------------------------------

def test_torso_color_pure_red_crop():
    crop = np.zeros((40, 20, 3), dtype=np.uint8)
    crop[:, :, 2] = 255   # BGR: pure red
    hue, sat, val = torso_color_hsv(crop)
    assert sat > 200 and val > 200      # vivid
    assert hue < 5 or hue > 175         # red sits at the hue-circle origin


def test_torso_color_gray_is_desaturated():
    crop = np.full((40, 20, 3), 128, dtype=np.uint8)
    _, sat, _ = torso_color_hsv(crop)
    assert sat < 10
