"""Auto-clip / dead-time segmentation (Phase 2 quick win).

A full game recording is mostly dead time — breaks, line changes, between-game lulls — with
bursts of live play in between. `activity.py` already gives us a per-frame "is the game live?"
signal; this module turns that boolean series into a handful of contiguous **live-play
segments** with frame/second bounds.

Two payoffs, both called out in docs/feature-ideas.md:
  - **Auto-clip / dead-time skip.** Export (and downstream, process) only the live segments —
    a big compute saving on a 38-minute game that is mostly idle.
  - **Shift segmentation foundation.** Shifts happen *within* live play; a clean list of
    live segments is the scaffolding the Phase 3 shift detector hangs off.

Pure stdlib (no numpy/cv2) so it stays trivially testable and import-cheap.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlaySegment:
    """A contiguous run of live-play frames. Both bounds are inclusive."""

    index: int
    start_frame: int
    end_frame: int

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def start_seconds(self, fps: float) -> float:
        return self.start_frame / fps if fps else 0.0

    def end_seconds(self, fps: float) -> float:
        # End of the last live frame, i.e. start of the frame after it.
        return (self.end_frame + 1) / fps if fps else 0.0

    def duration_seconds(self, fps: float) -> float:
        return self.n_frames / fps if fps else 0.0


def segment_play(
    per_frame_active: list[bool],
    fps: float,
    min_segment_seconds: float = 2.0,
    bridge_gap_seconds: float = 1.0,
) -> list[PlaySegment]:
    """Collapse a per-frame live/idle signal into live-play segments.

    Args:
        per_frame_active: one bool per frame (from ``ActivitySummary.per_frame_active``).
        fps: frames per second, used to convert the second-based knobs to frames.
        min_segment_seconds: drop live runs shorter than this (debounces stray live frames).
        bridge_gap_seconds: merge two live runs separated by an idle gap no longer than this
            (a brief whole-court occlusion or count dip shouldn't chop one play in two).

    Returns:
        Live-play segments in order, re-indexed 0..n-1 over the kept segments.
    """
    flags = [bool(x) for x in per_frame_active]
    if not flags:
        return []

    bridge_frames = max(0, round(bridge_gap_seconds * fps)) if fps else 0
    min_frames = max(1, round(min_segment_seconds * fps)) if fps else 1

    flags = _bridge_gaps(flags, bridge_frames)

    segments: list[PlaySegment] = []
    n = len(flags)
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        if (j - i + 1) >= min_frames:
            segments.append(PlaySegment(index=len(segments), start_frame=i, end_frame=j))
        i = j + 1
    return segments


def _bridge_gaps(flags: list[bool], bridge_frames: int) -> list[bool]:
    """Fill short idle gaps that sit between two live runs, returning a new list."""
    if bridge_frames <= 0:
        return list(flags)
    out = list(flags)
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not out[j + 1]:
            j += 1
        flanked = i - 1 >= 0 and out[i - 1] and j + 1 < n and out[j + 1]
        if flanked and (j - i + 1) <= bridge_frames:
            for k in range(i, j + 1):
                out[k] = True
        i = j + 1
    return out


def total_live_seconds(segments: list[PlaySegment], fps: float) -> float:
    """Total live-play time across all segments, in seconds."""
    return sum(s.duration_seconds(fps) for s in segments)


def frame_segment_index(segments: list[PlaySegment], frame_count: int) -> list[int | None]:
    """Map each frame index 0..frame_count-1 to its segment index, or None if idle.

    Handy for routing frames to per-segment clip writers in a single decode pass.
    """
    mapping: list[int | None] = [None] * frame_count
    for seg in segments:
        lo = max(0, seg.start_frame)
        hi = min(frame_count - 1, seg.end_frame)
        for f in range(lo, hi + 1):
            mapping[f] = seg.index
    return mapping


def write_segments_csv(path: str | Path, segments: list[PlaySegment], fps: float) -> None:
    """Write one row per live-play segment: frame bounds plus second bounds and duration."""
    fields = ["segment", "start_frame", "end_frame", "n_frames",
              "start_time_s", "end_time_s", "duration_s"]
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in segments:
            w.writerow({
                "segment": s.index,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "n_frames": s.n_frames,
                "start_time_s": round(s.start_seconds(fps), 2),
                "end_time_s": round(s.end_seconds(fps), 2),
                "duration_s": round(s.duration_seconds(fps), 2),
            })
