"""Tests for full-game mode's pure cores: the live-time axis and cross-segment identity merge."""

from __future__ import annotations

import numpy as np
import pytest

from dbh_vibes.game import GameIdentityMerge, LiveTimeline, merge_segment_identities
from dbh_vibes.segments import PlaySegment


def _seg(index, start, end):
    return PlaySegment(index=index, start_frame=start, end_frame=end)


class TestLiveTimeline:
    def test_first_segment_maps_identically(self):
        tl = LiveTimeline.from_segments([_seg(0, 0, 99), _seg(1, 200, 299)])
        assert tl.to_live(0) == 0
        assert tl.to_live(99) == 99

    def test_dead_time_is_compressed_out(self):
        # Frames 100..199 are dead; segment 1 starts right after segment 0 on the live axis.
        tl = LiveTimeline.from_segments([_seg(0, 0, 99), _seg(1, 200, 299)])
        assert tl.to_live(200) == 100
        assert tl.to_live(299) == 199

    def test_round_trip(self):
        tl = LiveTimeline.from_segments([_seg(0, 50, 149), _seg(1, 400, 449)])
        for gf in (50, 149, 400, 425, 449):
            assert tl.to_game(tl.to_live(gf)) == gf

    def test_frame_outside_segments_raises(self):
        tl = LiveTimeline.from_segments([_seg(0, 0, 99)])
        with pytest.raises(ValueError):
            tl.to_live(150)

    def test_unordered_segments_are_sorted(self):
        tl = LiveTimeline.from_segments([_seg(1, 200, 299), _seg(0, 0, 99)])
        assert tl.to_live(200) == 100

    def test_stoppage_gap_shrinks_to_zero_on_live_axis(self):
        # A player present at the end of seg 0 and the start of seg 1: game gap is 101 frames of
        # dead time, live gap is 1 frame -> shift detection on the live axis will bridge it.
        tl = LiveTimeline.from_segments([_seg(0, 0, 99), _seg(1, 200, 299)])
        assert tl.to_live(200) - tl.to_live(99) == 1


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class TestMergeSegmentIdentities:
    def test_same_person_across_segments_merges(self):
        # Segment 0 and 1 each saw two people; person A and person B look the same across segments.
        a, b = _unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])
        merge = merge_segment_identities(
            [
                (0, np.array([0, 1]), np.vstack([a, b])),
                (1, np.array([0, 1]), np.vstack([a + 0.01, b + 0.01])),
            ],
            distance_threshold=0.35,
        )
        assert merge.n_game_players == 2
        assert merge.game_id[(0, 0)] == merge.game_id[(1, 0)]
        assert merge.game_id[(0, 1)] == merge.game_id[(1, 1)]
        assert merge.game_id[(0, 0)] != merge.game_id[(0, 1)]

    def test_same_segment_identities_never_merge(self):
        # Two identical-looking identities *within* one segment must stay separate (the
        # within-segment clustering already ruled them different people).
        a = _unit([1, 0, 0, 0])
        merge = merge_segment_identities(
            [(0, np.array([0, 1]), np.vstack([a, a]))], distance_threshold=1.5
        )
        assert merge.n_game_players == 2

    def test_roster_pins_game_count(self):
        rng = np.random.default_rng(0)
        segs = []
        for s in range(3):
            vecs = np.vstack([_unit(rng.normal(size=8)) for _ in range(4)])
            segs.append((s, np.arange(4), vecs))
        merge = merge_segment_identities(segs, roster=5)
        assert merge.n_game_players >= 4          # same-segment cannot-link floors at 4
        assert merge.n_game_players <= 12

    def test_empty_input(self):
        merge = merge_segment_identities([])
        assert merge == GameIdentityMerge({}, 0, 0, 0)

    def test_game_ids_are_contiguous_and_cover_all(self):
        rng = np.random.default_rng(1)
        segs = [(s, np.arange(3), np.vstack([_unit(rng.normal(size=6)) for _ in range(3)]))
                for s in range(2)]
        merge = merge_segment_identities(segs, distance_threshold=0.2)
        assert set(merge.game_id) == {(s, i) for s in range(2) for i in range(3)}
        assert set(merge.game_id.values()) == set(range(merge.n_game_players))
