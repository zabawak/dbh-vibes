"""Lightweight 2-team classification by jersey color.

This is the *lightest-lift* fallback for the MVP: it crops each player's torso region, takes a
robust average color, and clusters all crops in a frame batch into two groups with KMeans.

The heavier, more accurate approach used by roboflow/sports is SigLIP image embeddings ->
UMAP -> KMeans on player crops. We deliberately avoid pulling those models for the MVP; this
can be swapped in later (see docs/architecture.md, Phase 2) without changing the call site.
"""

from __future__ import annotations

import cv2
import numpy as np


class TorsoColorTeamClassifier:
    """Assigns each detection to team 0 or 1 based on torso color.

    Stateful by design: the first time it sees enough players it fits a 2-means model on torso
    colors and reuses those cluster centers for the rest of the clip, so a given jersey color
    keeps the same team id frame to frame.
    """

    def __init__(self, min_samples_to_fit: int = 8) -> None:
        self.min_samples_to_fit = min_samples_to_fit
        self._centers: np.ndarray | None = None  # shape (2, 3), float32 BGR
        self._pending: list[np.ndarray] = []

    def predict(self, frame: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Return an int array (one team id per box) for the given frame."""
        colors = np.array([self._torso_color(frame, box) for box in boxes_xyxy], dtype=np.float32)

        if self._centers is None:
            self._pending.extend(colors)
            if len(self._pending) >= self.min_samples_to_fit:
                self._fit(np.array(self._pending, dtype=np.float32))
            else:
                # Not enough data yet: provisional split by brightness so output is still usable.
                return self._brightness_split(colors)

        return self._assign(colors)

    def _torso_color(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Average color of the upper-center (torso) region of a bounding box."""
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(3, dtype=np.float32)

        box_h, box_w = y2 - y1, x2 - x1
        # Upper third vertically, central 60% horizontally — where the jersey usually is.
        ty1 = y1 + int(0.15 * box_h)
        ty2 = y1 + int(0.45 * box_h)
        tx1 = x1 + int(0.20 * box_w)
        tx2 = x1 + int(0.80 * box_w)
        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            crop = frame[y1:y2, x1:x2]
        return crop.reshape(-1, 3).mean(axis=0).astype(np.float32)

    def _fit(self, samples: np.ndarray) -> None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(
            samples, 2, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
        )
        self._centers = centers.astype(np.float32)

    def _assign(self, colors: np.ndarray) -> np.ndarray:
        # Nearest cluster center in BGR space.
        dists = np.linalg.norm(colors[:, None, :] - self._centers[None, :, :], axis=2)
        return dists.argmin(axis=1).astype(int)

    @staticmethod
    def _brightness_split(colors: np.ndarray) -> np.ndarray:
        brightness = colors.mean(axis=1)
        threshold = brightness.mean()
        return (brightness > threshold).astype(int)
