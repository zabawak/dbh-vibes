"""Export a per-track labeling set so a human can produce ground truth in ~2 minutes.

This is the data-side half of priority #1 (the eval harness is ``evaluate.py``). To *measure*
team/identity accuracy we need labels, and the cheapest way to get them is to show one montage per
track and let a person tag it. The pipeline already samples a few crops per track for team
clustering; here we tile those into ``crops/track_XXXX.jpg`` and write a ``labels.csv`` template
pre-filled with the pipeline's own predictions and stats, leaving the truth columns blank.

Crucially the crops and the template come from the **same** detect/track pass that writes
``tracks.csv``, so track ids line up exactly — the labels can be scored against the predictions
with no id remapping. The montage is plain OpenCV image tiling; the template is stdlib ``csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

# Label columns a human fills in; the rest are pre-filled hints (predictions + stats) for context.
LABEL_FIELDS = ["team", "role", "player"]
HINT_FIELDS = ["pred_team", "pred_role", "frames_seen", "seconds_on_surface", "median_area_px"]


def montage(crops: list[np.ndarray], cell_w: int = 80, cell_h: int = 160, max_cells: int = 6,
            pad: int = 2) -> np.ndarray | None:
    """Tile up to ``max_cells`` crops into a single horizontal strip, each resized to a fixed cell.

    Fixed cells keep the strip readable when crops vary wildly in size (near vs far players). Returns
    ``None`` if there are no usable crops.
    """
    usable = [c for c in crops if c is not None and c.size and c.shape[0] >= 2 and c.shape[1] >= 2]
    if not usable:
        return None
    usable = usable[:max_cells]
    cells = []
    for c in usable:
        resized = cv2.resize(c, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cells.append(resized)
        cells.append(np.full((cell_h, pad, 3), 255, dtype=np.uint8))  # white separator
    strip = np.hstack(cells[:-1]) if len(cells) > 1 else cells[0]
    return strip


def export_labeling_set(
    out_dir: str | Path,
    track_crops: dict[int, list[np.ndarray]],
    track_rows: dict[int, dict],
    order: list[int] | None = None,
) -> tuple[Path, Path, int]:
    """Write per-track crop montages and a ``labels.csv`` template into ``out_dir``.

    ``track_rows[tid]`` supplies the pre-filled hint columns (pred_team/pred_role/stats). ``order``
    optionally fixes row order (e.g. most-active first); otherwise tracks are sorted by id. Only
    tracks with at least one usable crop are written, since a montage-less row can't be labeled by
    sight. Returns (crops_dir, labels_csv_path, n_tracks_written).
    """
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"

    ids = order if order is not None else sorted(track_crops)
    fieldnames = ["track_id", *LABEL_FIELDS, *HINT_FIELDS, "crop"]
    n_written = 0
    with labels_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for tid in ids:
            strip = montage(track_crops.get(tid, []))
            if strip is None:
                continue
            crop_name = f"track_{tid:04d}.jpg"
            cv2.imwrite(str(crops_dir / crop_name), strip)
            hints = track_rows.get(tid, {})
            row = {"track_id": tid, "crop": f"crops/{crop_name}"}
            for fld in LABEL_FIELDS:
                row[fld] = ""  # human fills these by viewing the montage
            for fld in HINT_FIELDS:
                row[fld] = hints.get(fld, "")
            w.writerow(row)
            n_written += 1
    return crops_dir, labels_path, n_written
