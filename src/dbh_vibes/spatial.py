"""Spatial stats — player position heatmaps (Phase 2).

Where do players spend time? We accumulate each player's ground-contact point (the bottom-center
of the detection box) over the clip into a density map, then render it over a reference frame.

This is deliberately done in **image space**, not a top-down rink projection. The camera is a
fixed fisheye with the near boards occluded by the mounting rail, so a single planar homography
to a clean overhead view would be unreliable — and a spatial stat we can't trust is worse than
none. An image-space heatmap is honest: it shows exactly where on the captured frame activity
concentrates (near the goals, along the boards, the bench), which is already a useful stat and a
solid base for a properly calibrated top-down map later (see docs/architecture.md).
"""

from __future__ import annotations

import cv2
import numpy as np


class PositionHeatmap:
    """Accumulates foot points into a Gaussian-splatted density map over the frame."""

    def __init__(self, frame_height: int, frame_width: int, sigma: int = 16) -> None:
        self.h = frame_height
        self.w = frame_width
        self.sigma = sigma
        self.accum = np.zeros((frame_height, frame_width), dtype=np.float32)

    def add(self, foot_points: np.ndarray) -> None:
        """Add this frame's foot points (N,2 array of x,y) to the accumulator."""
        for x, y in foot_points:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < self.w and 0 <= yi < self.h:
                self.accum[yi, xi] += 1.0

    def render(self, base_frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Blend the smoothed density as a colored overlay on a reference frame."""
        density = cv2.GaussianBlur(self.accum, (0, 0), self.sigma)
        if density.max() > 0:
            density = density / density.max()
        heat = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # Only blend where there is signal, so empty rink keeps the real image.
        mask = (density > 0.04)[:, :, None]
        blended = np.where(mask, cv2.addWeighted(base_frame, 1 - alpha, heat, alpha, 0), base_frame)
        return blended.astype(np.uint8)
