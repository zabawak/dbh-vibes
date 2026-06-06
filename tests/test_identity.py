"""Tests for the Phase 3 identity re-ID core (pure numpy/sklearn — no torch, no video).

These exercise the logic that stitches fragmented tracks into per-player identities on *synthetic*
embeddings + frame spans, so they need no model and no labelled data:

- the temporal cannot-link matrix (a person can't be two tracks at once),
- stitching same-appearance, non-overlapping fragments into one identity,
- the constraint vetoing a merge of two look-alike but simultaneously-on-surface players,
- determinism / run-to-run stability,
- data-driven identity count via the distance threshold, and pinning it via n_identities,
- the label-free confidence signal.
"""

from __future__ import annotations

import numpy as np

from dbh_vibes.identity import (
    assign_identities,
    cluster_identities,
    constrained_agglomerative,
    identity_confidence,
    temporal_overlap_matrix,
)

DIM = 32


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _person(center: np.ndarray, n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """n track-embeddings clustered tightly around one person's appearance center."""
    pts = center[None, :] + noise * rng.standard_normal((n, len(center)))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


# ---- temporal cannot-link matrix -------------------------------------------------------------

def test_overlap_matrix_flags_concurrent_tracks():
    # 0 and 1 overlap (0-10, 5-15); 2 is disjoint (20-30).
    spans = [(0, 10), (5, 15), (20, 30)]
    m = temporal_overlap_matrix(spans)
    assert m[0, 1] and m[1, 0]
    assert not m[0, 2] and not m[1, 2]
    assert not m.diagonal().any()       # a track never conflicts with itself


def test_overlap_matrix_min_gap_blocks_near_adjacent():
    spans = [(0, 10), (12, 20)]          # 2-frame gap
    assert not temporal_overlap_matrix(spans, min_gap=0)[0, 1]
    assert temporal_overlap_matrix(spans, min_gap=5)[0, 1]   # gap < 5 → still blocked


# ---- stitching fragments of the same person --------------------------------------------------

def test_fragments_of_one_person_stitch_together():
    # One person seen as three disjoint-in-time fragments (same appearance) must fold into ONE
    # identity; a clearly different person stays separate.
    rng = np.random.default_rng(0)
    p1 = _unit(rng.standard_normal(DIM))
    p2 = _unit(rng.standard_normal(DIM))
    emb = np.vstack([_person(p1, 3, 0.02, rng), _person(p2, 2, 0.02, rng)])
    spans = [(0, 10), (20, 30), (40, 50),   # person 1: three disjoint shifts
             (0, 10), (20, 30)]             # person 2: two disjoint shifts (concurrent w/ p1)
    labels, info = cluster_identities(emb, spans, distance_threshold=0.5)
    assert info.n_identities == 2
    assert len(set(labels[:3])) == 1                 # the three p1 fragments share an identity
    assert len(set(labels[3:])) == 1                 # the two p2 fragments share an identity
    assert labels[0] != labels[3]                    # and the two people are distinct


def test_cannot_link_keeps_lookalikes_apart():
    # Two players with *identical* appearance who are on the surface at the same time must NOT be
    # merged — appearance alone would fuse them; the temporal constraint forbids it.
    rng = np.random.default_rng(1)
    center = _unit(rng.standard_normal(DIM))
    emb = np.vstack([_person(center, 1, 0.0, rng), _person(center, 1, 0.0, rng)])
    spans = [(0, 100), (0, 100)]          # fully concurrent
    labels, info = cluster_identities(emb, spans, distance_threshold=0.9)
    assert info.n_identities == 2         # constraint prevented the look-alike merge
    assert labels[0] != labels[1]
    assert info.n_blocked_merges >= 1     # the vetoed merge was counted


def test_constrained_agglomerative_floor_from_concurrency():
    # Three mutually-concurrent tracks can never collapse below three identities, no matter how
    # similar they look — the max concurrent count is a hard floor on the roster estimate.
    pts = _unit(np.ones(DIM))[None, :].repeat(3, axis=0)   # identical direction
    cannot = np.array([[False, True, True], [True, False, True], [True, True, False]])
    labels, blocked = constrained_agglomerative(pts, cannot, distance_threshold=2.0)
    assert len(set(labels.tolist())) == 3


# ---- determinism -----------------------------------------------------------------------------

def test_identity_clustering_is_deterministic():
    rng = np.random.default_rng(2)
    centers = [_unit(rng.standard_normal(DIM)) for _ in range(4)]
    emb = np.vstack([_person(c, 2, 0.03, rng) for c in centers])
    spans = [(i * 100, i * 100 + 10) for i in range(8)]     # all disjoint
    l1, _ = cluster_identities(emb, spans, distance_threshold=0.5)
    l2, _ = cluster_identities(emb, spans, distance_threshold=0.5)
    np.testing.assert_array_equal(l1, l2)


# ---- count control: threshold (data-driven) vs pinned roster ---------------------------------

def test_distance_threshold_drives_identity_count():
    # Four well-separated people, two disjoint fragments each → a loose threshold finds 4 identities.
    rng = np.random.default_rng(3)
    centers = [_unit(np.eye(DIM)[i]) for i in range(4)]
    emb = np.vstack([_person(c, 2, 0.02, rng) for c in centers])
    spans = [(i * 100, i * 100 + 10) for i in range(8)]
    labels, info = cluster_identities(emb, spans, distance_threshold=0.5)
    assert info.n_identities == 4
    assert sorted(info.sizes) == [2, 2, 2, 2]


def test_n_identities_pins_roster_size():
    # Same data, but force three identities: two of the four people get merged into one bucket.
    rng = np.random.default_rng(4)
    centers = [_unit(np.eye(DIM)[i]) for i in range(4)]
    emb = np.vstack([_person(c, 2, 0.02, rng) for c in centers])
    spans = [(i * 100, i * 100 + 10) for i in range(8)]
    labels, info = cluster_identities(emb, spans, n_identities=3)
    assert info.n_identities == 3


# ---- confidence ------------------------------------------------------------------------------

def test_confidence_high_for_well_separated_identities():
    rng = np.random.default_rng(5)
    centers = [_unit(np.eye(DIM)[i]) for i in range(3)]
    emb = np.vstack([_person(c, 2, 0.01, rng) for c in centers])
    spans = [(i * 100, i * 100 + 10) for i in range(6)]
    labels, info = cluster_identities(emb, spans, distance_threshold=0.5)
    assert info.conf is not None
    assert info.conf.min() >= 0.0 and info.conf.max() <= 1.0
    assert info.conf.mean() > 0.7


def test_confidence_singleton_is_neutral():
    pts = np.array([[1.0, 0.0, 0.0]])
    conf = identity_confidence(pts, np.array([0]), pts)
    assert conf.tolist() == [0.5]


# ---- orchestration shim ----------------------------------------------------------------------

def test_assign_identities_maps_tracks_to_ids():
    rng = np.random.default_rng(6)
    p1 = _unit(rng.standard_normal(DIM))
    p2 = _unit(rng.standard_normal(DIM))
    emb = np.vstack([_person(p1, 2, 0.02, rng), _person(p2, 1, 0.02, rng)])
    present_ids = [10, 11, 12]
    spans = {10: (0, 10), 11: (20, 30), 12: (0, 10)}
    a = assign_identities(emb, present_ids, spans, distance_threshold=0.5)
    assert set(a.track_identity) == {10, 11, 12}
    assert a.track_identity[10] == a.track_identity[11]      # both are person 1, disjoint in time
    assert a.track_identity[10] != a.track_identity[12]      # person 2 is distinct
    assert a.n_identities == 2
    assert all(0.0 <= c <= 1.0 for c in a.track_conf.values())


def test_assign_identities_empty():
    a = assign_identities(np.empty((0, DIM)), [], {})
    assert a.track_identity == {}
    assert a.n_identities == 0
