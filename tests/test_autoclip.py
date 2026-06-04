"""Unit tests for the pure segmentation logic of auto-clip (no video / model needed)."""

from dbh_vibes.autoclip import segments_from_activity

FPS = 30.0
STRIDE = 10  # each sampled frame covers 10 original frames (1/3 s at 30 fps)


def test_single_live_block():
    # 3 idle, 30 live, 3 idle sampled frames -> one segment, padded and clamped to [0, total].
    active = [False] * 3 + [True] * 30 + [False] * 3
    total_frames = len(active) * STRIDE  # 360 frames = 12 s
    segs = segments_from_activity(active, fps=FPS, stride=STRIDE, total_frames=total_frames)
    assert len(segs) == 1
    s = segs[0]
    assert s.start_frame == 0  # padding clamped at the start
    assert s.end_frame == total_frames  # and clamped at the end
    assert s.index == 0


def test_two_blocks_large_gap_stay_separate():
    active = [True] * 15 + [False] * 30 + [True] * 15  # 10 s idle gap >> merge_gap (2 s)
    total_frames = len(active) * STRIDE
    segs = segments_from_activity(active, fps=FPS, stride=STRIDE, total_frames=total_frames)
    assert len(segs) == 2
    assert segs[0].index == 0 and segs[1].index == 1
    assert segs[0].end_frame < segs[1].start_frame


def test_small_gap_merges():
    active = [True] * 15 + [False] * 3 + [True] * 15  # 1 s idle gap <= merge_gap (2 s)
    total_frames = len(active) * STRIDE
    segs = segments_from_activity(active, fps=FPS, stride=STRIDE, total_frames=total_frames)
    assert len(segs) == 1


def test_short_blip_dropped():
    active = [True] * 2 + [False] * 30  # ~0.67 s of "play" -> below min_segment (3 s)
    total_frames = len(active) * STRIDE
    segs = segments_from_activity(active, fps=FPS, stride=STRIDE, total_frames=total_frames)
    assert segs == []


def test_min_segment_measured_before_padding():
    # A 2 s detected run with 1 s padding each side would be 4 s once padded, but min_segment
    # refers to detected play, so a min_segment of 3 s must still drop it.
    active = [True] * 6 + [False] * 30  # 60 frames = 2.0 s of play
    total_frames = len(active) * STRIDE
    segs = segments_from_activity(
        active, fps=FPS, stride=STRIDE, total_frames=total_frames,
        min_segment_seconds=3.0, pad_seconds=1.0,
    )
    assert segs == []


def test_no_activity_yields_no_segments():
    active = [False] * 50
    segs = segments_from_activity(active, fps=FPS, stride=STRIDE, total_frames=500)
    assert segs == []


def test_segments_are_clamped_within_video():
    active = [True] * 40
    total_frames = len(active) * STRIDE
    segs = segments_from_activity(active, fps=FPS, stride=STRIDE, total_frames=total_frames)
    assert len(segs) == 1
    assert segs[0].start_frame >= 0
    assert segs[0].end_frame <= total_frames
