"""Tests for the auto-clip pre-pass segmentation (pure logic — no video/model needed)."""

from __future__ import annotations

import json

from dbh_vibes.autoclip import segments_from_sampled_active
from dbh_vibes.segments import (
    PlaySegment,
    pad_segments,
    segment_record,
    write_segments_json,
)

FPS = 30.0
STRIDE = 10  # each sampled frame covers 10 original frames (1/3 s at 30 fps)


def _bounds(segs):
    return [(s.start_frame, s.end_frame) for s in segs]


# ---- segments_from_sampled_active (stride expansion + segment_play + padding) ----

def test_single_live_block_padded_and_clamped():
    # 3 idle, 30 live, 3 idle sampled frames -> one segment; padding clamps to [0, last].
    active = [False] * 3 + [True] * 30 + [False] * 3
    total = len(active) * STRIDE  # 360 frames
    segs = segments_from_sampled_active(active, fps=FPS, stride=STRIDE, total_frames=total)
    assert len(segs) == 1
    assert segs[0].index == 0
    assert segs[0].start_frame == 0            # 1s pad past the live block start -> clamped to 0
    assert segs[0].end_frame == total - 1      # and clamped to the last frame


def test_two_blocks_large_gap_stay_separate():
    active = [True] * 15 + [False] * 30 + [True] * 15  # 10s idle gap >> merge_gap (2s)
    total = len(active) * STRIDE
    segs = segments_from_sampled_active(active, fps=FPS, stride=STRIDE, total_frames=total)
    assert len(segs) == 2
    assert [s.index for s in segs] == [0, 1]
    assert segs[0].end_frame < segs[1].start_frame


def test_small_gap_merges():
    active = [True] * 15 + [False] * 3 + [True] * 15  # 1s idle gap <= merge_gap (2s)
    total = len(active) * STRIDE
    segs = segments_from_sampled_active(active, fps=FPS, stride=STRIDE, total_frames=total)
    assert len(segs) == 1


def test_short_blip_dropped():
    active = [True] * 2 + [False] * 30  # ~0.67s of "play" -> below min_segment (3s)
    total = len(active) * STRIDE
    segs = segments_from_sampled_active(active, fps=FPS, stride=STRIDE, total_frames=total)
    assert segs == []


def test_min_segment_measured_before_padding():
    # 2s detected run; even with 1s padding each side (which would make it 4s) the 3s minimum
    # refers to *detected* play, so it must still be dropped.
    active = [True] * 6 + [False] * 30  # 60 frames = 2.0s of play
    total = len(active) * STRIDE
    segs = segments_from_sampled_active(
        active, fps=FPS, stride=STRIDE, total_frames=total,
        min_segment_seconds=3.0, pad_seconds=1.0,
    )
    assert segs == []


def test_no_activity_yields_no_segments():
    assert segments_from_sampled_active([False] * 50, fps=FPS, stride=STRIDE, total_frames=500) == []


def test_segments_clamped_within_video():
    active = [True] * 40
    total = len(active) * STRIDE
    segs = segments_from_sampled_active(active, fps=FPS, stride=STRIDE, total_frames=total)
    assert len(segs) == 1
    assert segs[0].start_frame >= 0
    assert segs[0].end_frame <= total - 1


# ---- pad_segments (used by the pre-pass; pure) ----

def test_pad_extends_and_clamps():
    segs = [PlaySegment(0, 100, 200)]
    out = pad_segments(segs, fps=10.0, pad_seconds=2.0, frame_count=1000)  # pad = 20 frames
    assert _bounds(out) == [(80, 220)]


def test_pad_clamps_at_video_bounds():
    segs = [PlaySegment(0, 5, 995)]
    out = pad_segments(segs, fps=10.0, pad_seconds=2.0, frame_count=1000)
    assert _bounds(out) == [(0, 999)]  # clamped to [0, frame_count-1]


def test_pad_merges_overlaps_it_creates():
    # Two segments 30 frames apart; 2s pad (20 frames) on each closes the gap -> merge.
    segs = [PlaySegment(0, 100, 200), PlaySegment(1, 230, 300)]
    out = pad_segments(segs, fps=10.0, pad_seconds=2.0, frame_count=1000)
    assert _bounds(out) == [(80, 320)]
    assert out[0].index == 0


def test_pad_keeps_distinct_segments_separate():
    segs = [PlaySegment(0, 100, 200), PlaySegment(1, 400, 500)]
    out = pad_segments(segs, fps=10.0, pad_seconds=1.0, frame_count=1000)  # pad = 10
    assert _bounds(out) == [(90, 210), (390, 510)]
    assert [s.index for s in out] == [0, 1]


def test_pad_empty():
    assert pad_segments([], fps=30.0, pad_seconds=1.0, frame_count=100) == []


# ---- shared schema: segment_record + JSON manifest ----

def test_segment_record_schema():
    rec = segment_record(PlaySegment(0, 0, 29), fps=10.0)
    assert rec == {
        "segment": 0, "start_frame": 0, "end_frame": 29, "n_frames": 30,
        "start_time_s": 0.0, "end_time_s": 3.0, "duration_s": 3.0,
    }


def test_write_segments_json_manifest(tmp_path):
    segs = [PlaySegment(0, 0, 299), PlaySegment(1, 600, 899)]
    out = tmp_path / "segments.json"
    written = write_segments_json(
        out, segs, fps=30.0,
        extra={"source": "game.mp4", "total_seconds": 60.0, "savings_frac": 0.6667},
    )
    on_disk = json.loads(out.read_text())
    assert on_disk == written
    assert on_disk["n_segments"] == 2
    assert on_disk["source"] == "game.mp4"
    assert on_disk["savings_frac"] == 0.6667
    assert on_disk["live_seconds"] == 20.0  # (300 + 300) / 30
    assert len(on_disk["segments"]) == 2
    assert on_disk["segments"][0]["start_frame"] == 0
    assert on_disk["segments"][1]["end_frame"] == 899
