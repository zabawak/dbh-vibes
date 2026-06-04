"""Playing-surface detection — separate on-court players from bench/spectators (Phase 2).

The team classifier was polluted by spectators and bench players: anyone detected who isn't a
red-pinnie player fell into the "other" cluster, and they inflated time-on-surface totals. We
fix that by only counting detections whose ground-contact point lands on the **playing surface**.

Design choice — why not a fixed polygon: a hand-drawn pixel polygon of the rink would be the
simplest mask, but it is tied to one exact camera pose and breaks if the camera is moved or
swapped. Instead we *derive* the surface from the footage every run: take a per-pixel time median
of sampled frames (moving players average out, the static court remains), segment the court's
distinctive color, and fill it. Because the mask is recomputed from each video's own pixels, it
follows the camera if its position changes — no recalibration — as long as the court is visible.

Assumption: the playing surface is a large, distinctively-colored region (here, vivid blue). A
very differently colored court or extreme lighting would need the HSV range retuned, or the
optional manual-polygon override. If detection fails (mask too small), we fall back to no
filtering and warn, so the pipeline still runs.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

# Court blue in OpenCV HSV (H 0-179). The dek surface is vividly saturated, while background
# blues (sky, houses, cars) are pale, so a high saturation floor rejects them.
DEFAULT_HSV_LOW = (95, 120, 60)
DEFAULT_HSV_HIGH = (128, 255, 255)

# Outward margin added to the detected court, as a fraction of the larger frame dimension.
# This catches goalies whose feet sit in the crease/net right at the boundary, while staying
# smaller than the white-board band that separates the court from spectators behind it.
DEFAULT_DILATE_FRAC = 0.012


def estimate_surface_mask(
    video_path: str,
    n_samples: int = 20,
    hsv_low: tuple[int, int, int] = DEFAULT_HSV_LOW,
    hsv_high: tuple[int, int, int] = DEFAULT_HSV_HIGH,
    min_area_frac: float = 0.05,
    dilate_frac: float = DEFAULT_DILATE_FRAC,
) -> np.ndarray | None:
    """Estimate a filled playing-surface mask from a video.

    Returns a uint8 HxW mask (255 = surface) or None if the surface couldn't be found.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(0, total - 1), num=min(n_samples, total), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        return None

    # Time median removes transient players/objects, leaving the static court.
    median = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return surface_mask_from_frame(median, hsv_low, hsv_high, min_area_frac, dilate_frac)


def surface_mask_from_frame(
    frame: np.ndarray,
    hsv_low: tuple[int, int, int] = DEFAULT_HSV_LOW,
    hsv_high: tuple[int, int, int] = DEFAULT_HSV_HIGH,
    min_area_frac: float = 0.05,
    dilate_frac: float = DEFAULT_DILATE_FRAC,
) -> np.ndarray | None:
    """Segment the court color in a single frame and fill it into a solid surface mask.

    We keep the largest connected court-colored region and fill only its *internal* holes
    (painted markings, players standing on it). We deliberately do not fill to the external
    contour, which would balloon over the boards into the sky/trees and defeat the filter.
    A small outward dilation then includes edge goalies (feet in the crease/net) without
    reaching spectators behind the boards.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))

    # Despeckle, then a light close to smooth edges. We keep the close small on purpose: a heavy
    # close bridges the thin white boards and merges the court with background blue. Interior gaps
    # from markings/players are recovered by the hole-fill below, not by closing.
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Largest connected component = the court (not scattered blue specks in the background).
    num, labels, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
    if num <= 1:
        warnings.warn("Playing surface not found; skipping surface filter.")
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < min_area_frac * h * w:
        warnings.warn("Playing-surface region too small; skipping surface filter.")
        return None
    comp = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Fill only enclosed holes: flood the exterior from a corner, then OR in the un-flooded holes.
    flood = comp.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(comp, holes)

    # Grow the boundary outward a touch so edge goalies (feet in the crease/net) count as on-court.
    margin = int(dilate_frac * max(h, w))
    if margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        filled = cv2.dilate(filled, k)
    return filled


def on_surface(foot_points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Boolean array: is each (x, y) foot point inside the surface mask?"""
    if len(foot_points) == 0:
        return np.empty((0,), dtype=bool)
    h, w = mask.shape[:2]
    xs = np.clip(foot_points[:, 0].astype(int), 0, w - 1)
    ys = np.clip(foot_points[:, 1].astype(int), 0, h - 1)
    return mask[ys, xs] > 0
