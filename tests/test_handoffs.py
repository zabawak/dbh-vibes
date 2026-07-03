"""Tests for spatiotemporal handoff linking — the identity-recall lever.

A tracker "handoff" is a near-certain same-person link: track j starts moments after — and next
to — where track i ended. These tests cover the pure detector (`detect_handoffs`), the
cannot-link-safe union (`merge_handoff_groups`), and the seeded clustering path
(`cluster_identities(handoffs=...)`). No video, no model.
"""

from __future__ import annotations

import numpy as np

from dbh_vibes.identity import (
    cluster_identities,
    constrained_agglomerative,
    detect_handoffs,
    merge_handoff_groups,
    temporal_overlap_matrix,
)

FPS = 30.0
W = 1280.0


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class TestDetectHandoffs:
    def test_close_in_time_and_space_links(self):
        # Track 0 ends at frame 100 @ (400, 300); track 1 starts at frame 115 @ (410, 305).
        spans = [(0, 100), (115, 200)]
        exits = [(400.0, 300.0), (900.0, 500.0)]
        entries = [(0.0, 0.0), (410.0, 305.0)]
        assert detect_handoffs(spans, exits, entries, fps=FPS, frame_width=W) == [(0, 1)]

    def test_too_far_apart_in_space_does_not_link(self):
        spans = [(0, 100), (115, 200)]
        exits = [(400.0, 300.0), (0.0, 0.0)]
        entries = [(0.0, 0.0), (600.0, 300.0)]   # 200px > 5% of 1280
        assert detect_handoffs(spans, exits, entries, fps=FPS, frame_width=W) == []

    def test_too_long_a_gap_does_not_link(self):
        spans = [(0, 100), (100 + int(3 * FPS), 400)]   # 3s gap > 2s default
        exits = [(400.0, 300.0), (0.0, 0.0)]
        entries = [(0.0, 0.0), (402.0, 301.0)]
        assert detect_handoffs(spans, exits, entries, fps=FPS, frame_width=W) == []

    def test_overlapping_tracks_never_link(self):
        # gap <= 0 (concurrent tracks) is not a handoff, whatever the positions say.
        spans = [(0, 100), (90, 200)]
        exits = [(400.0, 300.0), (0.0, 0.0)]
        entries = [(0.0, 0.0), (400.0, 300.0)]
        assert detect_handoffs(spans, exits, entries, fps=FPS, frame_width=W) == []

    def test_missing_positions_never_link(self):
        spans = [(0, 100), (115, 200)]
        assert detect_handoffs(spans, [None, None], [None, None], fps=FPS, frame_width=W) == []

    def test_contested_exit_links_nothing(self):
        # Two tracks (1, 2) both start near track 0's exit within the window — crossing paths.
        # The nearest candidate is as likely wrong as right, so neither may link.
        spans = [(0, 100), (110, 200), (112, 220)]
        exits = [(400.0, 300.0), (0.0, 0.0), (0.0, 0.0)]
        entries = [(0.0, 0.0), (405.0, 302.0), (410.0, 296.0)]
        assert detect_handoffs(spans, exits, entries, fps=FPS, frame_width=W) == []

    def test_contested_entry_links_nothing(self):
        # Two tracks (0, 1) both end near track 2's entry within the window.
        spans = [(0, 100), (0, 105), (115, 220)]
        exits = [(400.0, 300.0), (404.0, 298.0), (0.0, 0.0)]
        entries = [(900.0, 900.0), (800.0, 800.0), (402.0, 301.0)]
        assert detect_handoffs(spans, exits, entries, fps=FPS, frame_width=W) == []

    def test_chain_of_handoffs(self):
        # 0 -> 1 -> 2: two links forming one three-fragment person.
        spans = [(0, 100), (110, 200), (215, 300)]
        exits = [(400.0, 300.0), (500.0, 350.0), (0.0, 0.0)]
        entries = [(0.0, 0.0), (405.0, 302.0), (498.0, 352.0)]
        assert detect_handoffs(spans, exits, entries, fps=FPS, frame_width=W) == [(0, 1), (1, 2)]


class TestMergeHandoffGroups:
    def test_chain_unions_into_one_group(self):
        cannot = np.zeros((3, 3), dtype=bool)
        groups = merge_handoff_groups(3, [(0, 1), (1, 2)], cannot)
        assert sorted(map(sorted, groups)) == [[0, 1, 2]]

    def test_cannot_link_violating_union_is_skipped(self):
        # 0-1 is a handoff, but 1 conflicts with 2 and a 0-2 handoff also fired (contradiction):
        # the union that would put 1 and 2 together must be refused.
        cannot = np.zeros((3, 3), dtype=bool)
        cannot[1, 2] = cannot[2, 1] = True
        groups = merge_handoff_groups(3, [(0, 1), (0, 2)], cannot)
        assert sorted(map(sorted, groups)) == [[0, 1], [2]]

    def test_no_handoffs_all_singletons(self):
        cannot = np.zeros((2, 2), dtype=bool)
        assert merge_handoff_groups(2, [], cannot) == [[0], [1]]


class TestSeededClustering:
    def test_initial_groups_survive_into_labels(self):
        # Three points, appearance says all far apart; a seeded group {0,1} must stay together.
        pts = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
        cannot = np.zeros((3, 3), dtype=bool)
        labels, _ = constrained_agglomerative(
            pts, cannot, distance_threshold=0.05, initial_groups=[[0, 1], [2]]
        )
        assert labels[0] == labels[1]
        assert labels[0] != labels[2]

    def test_handoff_beats_appearance_in_cluster_identities(self):
        # Fragments 0 and 1 look totally different (orthogonal embeddings) but hand off cleanly:
        # without the handoff they stay separate at a strict threshold; with it they are one person.
        emb = np.vstack([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0]), _unit([0, 0, 1, 0])])
        spans = [(0, 100), (110, 200), (0, 200)]
        no_link, _ = cluster_identities(emb, spans, distance_threshold=0.05)
        assert no_link[0] != no_link[1]
        linked, info = cluster_identities(
            emb, spans, distance_threshold=0.05, handoffs=[(0, 1)]
        )
        assert linked[0] == linked[1]
        assert linked[0] != linked[2]
        assert info.n_handoffs == 1

    def test_handoff_violating_temporal_constraint_is_dropped(self):
        # A (bogus) handoff between time-overlapping tracks must not merge them.
        emb = np.vstack([_unit([1, 0, 0]), _unit([1, 0.01, 0])])
        spans = [(0, 100), (50, 150)]
        conflict = temporal_overlap_matrix(spans)
        assert conflict[0, 1]
        labels, info = cluster_identities(
            emb, spans, distance_threshold=0.5, handoffs=[(0, 1)]
        )
        assert labels[0] != labels[1]
        assert info.n_handoffs == 0

    def test_roster_pinning_composes_with_handoffs(self):
        rng = np.random.default_rng(0)
        emb = np.vstack([_unit(rng.normal(size=8)) for _ in range(6)])
        spans = [(0, 50), (60, 100), (0, 100), (120, 160), (0, 160), (170, 200)]
        labels, _ = cluster_identities(emb, spans, n_identities=3, handoffs=[(0, 1)])
        assert labels[0] == labels[1]
        assert len(set(labels.tolist())) == 3
