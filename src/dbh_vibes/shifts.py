"""Shift detection — turn per-player track fragments into true on-surface shifts.

The headline per-player stat the project set out to produce is **time on surface broken into
shifts** ("time on ice"). Phase 3 re-ID (`identity.py`) already stitches the fragmented track ids
back into per-player **identities**, but the per-player roll-up counted ``n_shifts`` as simply the
*number of track fragments* — and that over-counts. ByteTrack drops and re-acquires a player who is
briefly occluded, mid-court, even though that player never left the surface; each re-acquisition is
a new track id, so one continuous shift fragments into several.

A **true shift** is a contiguous on-surface stretch. This module segments each identity's timeline
into shifts by merging its track fragments whose temporal gap is short (an occlusion / tracker
re-acquire — still the same shift) and splitting on a long gap (the player went to the bench and
came back — a new shift).

Why a *temporal* gap is the bench signal here: the Phase 2 surface filter (`surface.py`) already
drops off-surface detections, so a player sitting on the bench simply has **no on-surface track**.
Their identity's on-surface timeline therefore goes dark for the duration of the bench trip — a long
temporal gap — and lights back up when they step on for the next shift. So gap-based segmentation
*is* bench detection, riding on the surface filter rather than needing a hand-drawn bench polygon.
(A dedicated entry/exit zone, deferred, would sharpen the on/off instant; the gap is the robust,
zero-calibration version available today.)

A useful consequence of the re-ID temporal cannot-link constraint: all fragments of one identity are
guaranteed *disjoint* in time, so a player's fragment spans form a clean, non-overlapping timeline —
merging them is well-defined.

Pure stdlib (no numpy/cv2), like ``segments.py`` and ``boxscore.py``, so the core is trivially
unit-testable and import-cheap.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shift:
    """One contiguous on-surface stretch for a player. Both frame bounds are inclusive.

    ``n_fragments`` is how many track fragments were stitched into this shift (>1 means the tracker
    lost and re-acquired the player mid-shift; the temporal gaps between them were short enough to
    bridge). The shift *span* therefore covers the brief undetected gaps too, making its duration a
    better time-on-surface estimate than summing only detected frames.
    """

    player: int
    index: int              # 0-based shift number within this player's game
    start_frame: int
    end_frame: int
    n_fragments: int

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def start_seconds(self, fps: float) -> float:
        return self.start_frame / fps if fps else 0.0

    def end_seconds(self, fps: float) -> float:
        # End of the last on-surface frame, i.e. the start of the frame after it.
        return (self.end_frame + 1) / fps if fps else 0.0

    def duration_seconds(self, fps: float) -> float:
        return self.n_frames / fps if fps else 0.0


def merge_spans(spans: list[tuple[int, int]], bridge_frames: int) -> list[tuple[int, int, int]]:
    """Merge inclusive ``(first, last)`` spans, bridging gaps no larger than ``bridge_frames``.

    Returns ``(start, end, n_fragments)`` tuples in time order. Two spans are joined when the idle
    gap between them — ``next.start - cur.end - 1`` frames — is ``<= bridge_frames`` (a brief
    occlusion / tracker re-acquire); a larger gap starts a new merged run (a bench trip). Overlapping
    spans (gap < 0) are always joined; ``bridge_frames < 0`` disables bridging (every span stands
    alone). The input is not mutated.
    """
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[list[int]] = []  # [start, end, n_fragments]
    for first, last in ordered:
        if merged and first - merged[-1][1] - 1 <= bridge_frames:
            merged[-1][1] = max(merged[-1][1], last)
            merged[-1][2] += 1
        else:
            merged.append([first, last, 1])
    return [(s, e, n) for s, e, n in merged]


def detect_shifts(
    track_spans: dict[int, list[tuple[int, int]]],
    fps: float,
    bridge_gap_seconds: float = 15.0,
    min_shift_seconds: float = 0.0,
) -> dict[int, list[Shift]]:
    """Segment each player's track fragments into true on-surface shifts.

    Args:
        track_spans: ``{player_id: [(first_frame, last_frame), ...]}`` — the inclusive frame spans
            of every track fragment belonging to each identity (re-ID guarantees these are disjoint
            in time within a player).
        fps: frames per second, to convert the second-based knobs to frames.
        bridge_gap_seconds: fragments separated by an on-surface gap no longer than this are the
            same shift (occlusion / tracker re-acquire); a longer gap is a bench trip → new shift.
            Default 15 s is a deliberately *physical* threshold — a player cannot get to the bench,
            sub off, and return in under ~15 s, so any shorter absence is treated as in-shift
            occlusion. (On real fisheye footage the inter-fragment gap distribution is **not**
            cleanly bimodal — short dropouts blur into longer occlusions — so this is a judgement
            call, not a learned boundary; an explicit entry/exit zone would resolve it properly.)
        min_shift_seconds: drop shifts shorter than this (debounces a stray one-frame fragment that
            didn't bridge into a neighbour). 0 keeps every shift.

    Returns:
        ``{player_id: [Shift, ...]}`` with shifts in time order, re-indexed 0..n-1 per player.
        Players with no spans are omitted.
    """
    bridge_frames = round(bridge_gap_seconds * fps) if fps else 0
    min_frames = max(1, round(min_shift_seconds * fps)) if (fps and min_shift_seconds) else 1

    out: dict[int, list[Shift]] = {}
    for player, spans in track_spans.items():
        runs = [(s, e, n) for s, e, n in merge_spans(spans, bridge_frames)
                if (e - s + 1) >= min_frames]
        # Re-index after the min-length filter so shift indices stay contiguous 0..n-1.
        shifts = [
            Shift(player=player, index=i, start_frame=s, end_frame=e, n_fragments=n)
            for i, (s, e, n) in enumerate(runs)
        ]
        if shifts:
            out[player] = shifts
    return out


@dataclass(frozen=True)
class ShiftSummary:
    """Per-player shift roll-up derived from a list of shifts."""

    player: int
    n_shifts: int
    n_fragments: int        # total track fragments stitched (>= n_shifts)
    shift_seconds: float    # total on-surface time across shifts (spans, incl. bridged gaps)
    longest_shift_s: float
    avg_shift_s: float
    first_frame: int
    last_frame: int


def summarize_player(player: int, shifts: list[Shift], fps: float) -> ShiftSummary:
    """Roll a player's shifts into headline numbers (counts + shift-length stats)."""
    durations = [sh.duration_seconds(fps) for sh in shifts]
    total = sum(durations)
    return ShiftSummary(
        player=player,
        n_shifts=len(shifts),
        n_fragments=sum(sh.n_fragments for sh in shifts),
        shift_seconds=round(total, 2),
        longest_shift_s=round(max(durations), 2) if durations else 0.0,
        avg_shift_s=round(total / len(shifts), 2) if shifts else 0.0,
        first_frame=min((sh.start_frame for sh in shifts), default=0),
        last_frame=max((sh.end_frame for sh in shifts), default=0),
    )


def shift_record(shift: Shift, fps: float, team: int | str = "") -> dict:
    """One shift as a plain dict (player + team + frame/second bounds + fragments).

    The single source of truth for the per-shift schema, used by the CSV writer.
    """
    return {
        "player": shift.player,
        "team": team,
        "shift": shift.index,
        "start_frame": shift.start_frame,
        "end_frame": shift.end_frame,
        "n_frames": shift.n_frames,
        "n_fragments": shift.n_fragments,
        "start_time_s": round(shift.start_seconds(fps), 2),
        "end_time_s": round(shift.end_seconds(fps), 2),
        "duration_s": round(shift.duration_seconds(fps), 2),
    }


def write_shifts_csv(
    path: str | Path,
    shifts_by_player: dict[int, list[Shift]],
    fps: float,
    teams: dict[int, int | str] | None = None,
) -> None:
    """Write one row per (player, shift): frame/second bounds, duration, fragments stitched.

    Rows are ordered by player then shift index. ``teams`` optionally maps player id → team for a
    ``team`` column (blank when unknown).
    """
    teams = teams or {}
    fields = ["player", "team", "shift", "start_frame", "end_frame", "n_frames",
              "n_fragments", "start_time_s", "end_time_s", "duration_s"]
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for player in sorted(shifts_by_player):
            for shift in shifts_by_player[player]:
                w.writerow(shift_record(shift, fps, team=teams.get(player, "")))
