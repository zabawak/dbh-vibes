"""Tests for the auto-clip / dead-time segmentation (pure stdlib, no heavy deps)."""

from __future__ import annotations

import csv

from dbh_vibes.segments import (
    PlaySegment,
    frame_segment_index,
    segment_play,
    total_live_seconds,
    write_segments_csv,
)

FPS = 10.0  # 10 fps => 1 frame = 0.1s, keeps the second-knobs easy to reason about


def _bounds(segs: list[PlaySegment]) -> list[tuple[int, int]]:
    return [(s.start_frame, s.end_frame) for s in segs]


def test_empty_input_yields_no_segments():
    assert segment_play([], FPS) == []


def test_all_idle_yields_no_segments():
    assert segment_play([False] * 50, FPS) == []


def test_single_live_run_is_one_segment():
    flags = [False] * 5 + [True] * 30 + [False] * 5
    segs = segment_play(flags, FPS, min_segment_seconds=1.0, bridge_gap_seconds=0.0)
    assert _bounds(segs) == [(5, 34)]
    assert segs[0].n_frames == 30


def test_inclusive_bounds_and_second_conversion():
    flags = [True] * 20  # frames 0..19 at 10 fps => 2.0s
    seg = segment_play(flags, FPS, min_segment_seconds=1.0, bridge_gap_seconds=0.0)[0]
    assert seg.start_frame == 0 and seg.end_frame == 19
    assert seg.start_seconds(FPS) == 0.0
    assert seg.end_seconds(FPS) == 2.0       # end of frame 19 is t=20/10
    assert seg.duration_seconds(FPS) == 2.0


def test_short_runs_are_dropped():
    # A 3-frame (0.3s) blip plus a real 25-frame run; min is 1.0s => only the real one survives.
    flags = [True] * 3 + [False] * 10 + [True] * 25
    segs = segment_play(flags, FPS, min_segment_seconds=1.0, bridge_gap_seconds=0.0)
    assert _bounds(segs) == [(13, 37)]


def test_short_gap_is_bridged():
    # Two long live runs split by a 5-frame (0.5s) idle gap; bridge=1.0s merges them.
    flags = [True] * 20 + [False] * 5 + [True] * 20
    segs = segment_play(flags, FPS, min_segment_seconds=1.0, bridge_gap_seconds=1.0)
    assert _bounds(segs) == [(0, 44)]


def test_long_gap_is_not_bridged():
    # A 15-frame (1.5s) gap exceeds the 1.0s bridge => stays two segments.
    flags = [True] * 20 + [False] * 15 + [True] * 20
    segs = segment_play(flags, FPS, min_segment_seconds=1.0, bridge_gap_seconds=1.0)
    assert _bounds(segs) == [(0, 19), (35, 54)]


def test_leading_and_trailing_idle_is_never_bridged():
    # Idle runs at the very start/end aren't flanked on both sides, so they're left as idle.
    flags = [False] * 8 + [True] * 20 + [False] * 8
    segs = segment_play(flags, FPS, min_segment_seconds=1.0, bridge_gap_seconds=2.0)
    assert _bounds(segs) == [(8, 27)]


def test_segments_are_reindexed_contiguously():
    flags = [True] * 20 + [False] * 15 + [True] * 20 + [False] * 15 + [True] * 20
    segs = segment_play(flags, FPS, min_segment_seconds=1.0, bridge_gap_seconds=0.5)
    assert [s.index for s in segs] == [0, 1, 2]


def test_total_live_seconds():
    flags = [True] * 20 + [False] * 15 + [True] * 30
    segs = segment_play(flags, FPS, min_segment_seconds=1.0, bridge_gap_seconds=0.5)
    # 20 frames + 30 frames = 50 frames at 10 fps = 5.0s
    assert total_live_seconds(segs, FPS) == 5.0


def test_frame_segment_index_maps_only_live_frames():
    flags = [False] * 5 + [True] * 10 + [False] * 5
    segs = segment_play(flags, FPS, min_segment_seconds=0.5, bridge_gap_seconds=0.0)
    mapping = frame_segment_index(segs, frame_count=20)
    assert mapping[:5] == [None] * 5
    assert mapping[5:15] == [0] * 10
    assert mapping[15:] == [None] * 5


def test_write_segments_csv(tmp_path):
    flags = [True] * 20 + [False] * 15 + [True] * 30
    segs = segment_play(flags, FPS, min_segment_seconds=1.0, bridge_gap_seconds=0.5)
    out = tmp_path / "segments.csv"
    write_segments_csv(out, segs, FPS)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert rows[0]["segment"] == "0"
    assert rows[0]["start_frame"] == "0"
    assert rows[0]["end_frame"] == "19"
    assert rows[0]["duration_s"] == "2.0"
    assert rows[1]["start_frame"] == "35"


def test_zero_fps_is_safe():
    # Defensive: a malformed video reporting 0 fps shouldn't crash segmentation.
    flags = [True] * 5
    segs = segment_play(flags, fps=0.0)
    assert _bounds(segs) == [(0, 4)]
    assert segs[0].duration_seconds(0.0) == 0.0
