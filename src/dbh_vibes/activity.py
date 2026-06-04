"""Active-play detection (Phase 2).

The real-footage test surfaced a practical problem: a game recording contains long dead
stretches (between games, breaks) where everyone clusters at the bench and the rink is empty.
Spending compute — and worse, accumulating "stats" — on those stretches is wrong.

This module derives a cheap per-frame "is the game live?" signal from detections we already
have. During play, many players are spread across the surface; during a lull, a few people
bunch up near the boards. So we gate on two signals: player count and horizontal spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActivitySummary:
    active_fraction: float          # fraction of frames classified as live play
    mean_players: float
    mean_spread: float              # mean normalized horizontal spread of foot points
    per_frame_active: list[bool]
    per_frame_players: list[int]


def foot_points(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Bottom-center of each box — the player's approximate ground contact point."""
    if len(boxes_xyxy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    y = boxes_xyxy[:, 3]
    return np.stack([x, y], axis=1).astype(np.float32)


def detect_activity(
    per_frame_feet: list[np.ndarray],
    frame_width: int,
    min_players: int = 5,
    min_spread: float = 0.15,
    smooth_window: int = 15,
) -> ActivitySummary:
    """Classify each frame as live play or idle from per-frame foot points.

    Args:
        per_frame_feet: list (one per frame) of (N,2) foot-point arrays.
        frame_width: pixel width, used to normalize horizontal spread.
        min_players: minimum people on the surface to consider it play.
        min_spread: minimum normalized std of foot x to consider it play (filters bench clumps).
        smooth_window: majority-smoothing window (frames) to debounce flicker.

    Returns:
        ActivitySummary with the per-frame decision and headline stats.
    """
    counts = []
    spreads = []
    raw_active = []
    for feet in per_frame_feet:
        n = len(feet)
        counts.append(n)
        spread = float(np.std(feet[:, 0]) / frame_width) if n >= 2 else 0.0
        spreads.append(spread)
        raw_active.append(n >= min_players and spread >= min_spread)

    active = _majority_smooth(raw_active, smooth_window)

    return ActivitySummary(
        active_fraction=float(np.mean(active)) if active else 0.0,
        mean_players=float(np.mean(counts)) if counts else 0.0,
        mean_spread=float(np.mean(spreads)) if spreads else 0.0,
        per_frame_active=active,
        per_frame_players=counts,
    )


def _majority_smooth(flags: list[bool], window: int) -> list[bool]:
    """Smooth a boolean series with a centered majority filter to debounce single-frame flips."""
    if window <= 1 or not flags:
        return list(flags)
    arr = np.asarray(flags, dtype=int)
    half = window // 2
    out = []
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out.append(arr[lo:hi].mean() >= 0.5)
    return out
