"""Auto-clip / dead-time skip (quick win, built on the done activity signal).

A full game recording is mostly *not* live play: warmups, between-game downtime, breaks at the
bench. Running the full Phase 2 pipeline over all 38 minutes wastes compute and — worse — would
accrue "stats" during dead time. This module finds the live-play stretches *first*, so the
expensive analysis only ever runs on real action.

How it works:

1. A **cheap detection-only pre-pass** streams YOLO over the video at a coarse frame stride
   (``vid_stride``) — no tracking, so it's much faster than the analysis pipeline. We reuse the
   same playing-surface filter so bench clumps can't masquerade as play.
2. The existing :mod:`dbh_vibes.activity` signal classifies each sampled frame as live/idle.
3. :func:`segments_from_activity` turns that boolean series into contiguous live **segments**,
   bridging short idle gaps (a whistle), dropping blips too short to be real play, and padding
   each segment so we don't clip the first/last second of action.

Outputs a ``segments.json`` / ``segments.csv`` manifest (start/end frame + seconds per segment,
plus a compute-savings estimate). With ``cut=True`` it also writes each live segment to its own
``.mp4`` via ffmpeg, ready to feed back into the analysis pipeline. This same segmentation is the
basis for shift detection later (see docs/architecture.md).
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dbh_vibes.activity import ActivitySummary, detect_activity, foot_points
from dbh_vibes.surface import estimate_surface_mask, on_surface

if TYPE_CHECKING:
    import supervision as sv

PERSON_CLASS_ID = 0


@dataclass
class Segment:
    """One contiguous stretch of live play, in both frame and second coordinates."""

    index: int
    start_frame: int
    end_frame: int          # exclusive
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass
class AutoClipResult:
    segments: list[Segment]
    fps: float
    total_frames: int
    total_seconds: float
    live_seconds: float
    savings_frac: float            # fraction of the video skippable as dead time
    activity: ActivitySummary
    surface_found: bool
    clip_paths: list[Path]


def segments_from_activity(
    active: list[bool],
    *,
    fps: float,
    stride: int,
    total_frames: int,
    min_segment_seconds: float = 3.0,
    merge_gap_seconds: float = 2.0,
    pad_seconds: float = 1.0,
) -> list[Segment]:
    """Turn a per-sampled-frame live/idle boolean series into padded live segments.

    ``active[j]`` is the decision for the sampled frame at original index ``j * stride``; each
    sampled frame is treated as covering the ``stride`` original frames up to the next sample.

    Args:
        active: per-sampled-frame live-play flags (from :func:`detect_activity`).
        fps: original video frame rate, for the frame<->second mapping.
        stride: frame stride used during the detection pre-pass.
        total_frames: total frames in the source video (segments are clamped to it).
        min_segment_seconds: drop live runs shorter than this (debounces stray detections).
        merge_gap_seconds: merge two live runs separated by an idle gap no longer than this
            (so a brief whistle/stoppage doesn't split one shift into two).
        pad_seconds: extend each segment by this much on both ends (catch the run-up/run-out).

    Returns:
        Live segments in chronological order, re-indexed from 0.
    """
    # 1. Raw runs of consecutive live samples, expressed as half-open frame intervals.
    intervals: list[list[int]] = []
    n = len(active)
    j = 0
    while j < n:
        if active[j]:
            k = j
            while k + 1 < n and active[k + 1]:
                k += 1
            start_f = j * stride
            end_f = min(total_frames, (k + 1) * stride)
            intervals.append([start_f, end_f])
            j = k + 1
        else:
            j += 1

    # 2. Bridge short idle gaps between consecutive live runs.
    merge_gap_frames = merge_gap_seconds * fps
    merged: list[list[int]] = []
    for iv in intervals:
        if merged and iv[0] - merged[-1][1] <= merge_gap_frames:
            merged[-1][1] = iv[1]
        else:
            merged.append(list(iv))

    # 3. Drop runs too short to be real play (measured *before* padding, so min_segment refers
    #    to detected play length, not the padded clip length).
    min_segment_frames = min_segment_seconds * fps
    kept = [iv for iv in merged if (iv[1] - iv[0]) >= min_segment_frames]

    # 4. Pad both ends, then re-merge any intervals padding pushed into overlap.
    pad_frames = int(round(pad_seconds * fps))
    padded: list[list[int]] = []
    for a, b in kept:
        a, b = max(0, a - pad_frames), min(total_frames, b + pad_frames)
        if padded and a <= padded[-1][1]:
            padded[-1][1] = max(padded[-1][1], b)
        else:
            padded.append([a, b])

    # 5. Materialize Segments.
    segments: list[Segment] = []
    for a, b in padded:
        duration = (b - a) / fps if fps else 0.0
        segments.append(
            Segment(
                index=len(segments),
                start_frame=a,
                end_frame=b,
                start_sec=round(a / fps, 3) if fps else 0.0,
                end_sec=round(b / fps, 3) if fps else 0.0,
                duration_sec=round(duration, 3),
            )
        )
    return segments


def detect_live_segments(
    source: str | Path,
    *,
    model_name: str = "yolo11n.pt",
    conf: float = 0.25,
    stride: int = 15,
    filter_to_surface: bool = True,
    min_players: int = 5,
    min_spread: float = 0.15,
    smooth_seconds: float = 2.0,
    min_segment_seconds: float = 3.0,
    merge_gap_seconds: float = 2.0,
    pad_seconds: float = 1.0,
) -> "tuple[list[Segment], ActivitySummary, sv.VideoInfo, bool]":
    """Run the cheap detection pre-pass and derive live-play segments.

    Returns ``(segments, activity_summary, video_info, surface_found)``.
    """
    import supervision as sv
    from ultralytics import YOLO

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    info = sv.VideoInfo.from_video_path(str(source))
    fps = float(info.fps)
    model = YOLO(model_name)

    # Same per-run surface derivation as the analysis pipeline, so off-court clumps (the bench)
    # don't get counted as play. None => couldn't find it; fall back to no filtering.
    surface = estimate_surface_mask(str(source)) if filter_to_surface else None

    # Detection only (no tracker) at a coarse stride — this is the whole speed win over Phase 2.
    per_frame_feet = []
    results = model.predict(
        source=str(source), classes=[PERSON_CLASS_ID], conf=conf,
        stream=True, verbose=False, vid_stride=stride,
    )
    for r in results:
        det = sv.Detections.from_ultralytics(r)
        feet = foot_points(det.xyxy)
        if surface is not None and len(feet):
            feet = feet[on_surface(feet, surface)]
        per_frame_feet.append(feet)

    # smooth_window is in sampled frames; convert the desired debounce window from seconds.
    sampled_fps = fps / stride if stride else fps
    smooth_window = max(1, int(round(smooth_seconds * sampled_fps)))
    activity = detect_activity(
        per_frame_feet, info.width,
        min_players=min_players, min_spread=min_spread, smooth_window=smooth_window,
    )

    total_frames = info.total_frames or (len(per_frame_feet) * stride)
    segments = segments_from_activity(
        activity.per_frame_active,
        fps=fps, stride=stride, total_frames=total_frames,
        min_segment_seconds=min_segment_seconds,
        merge_gap_seconds=merge_gap_seconds,
        pad_seconds=pad_seconds,
    )
    return segments, activity, info, surface is not None


def cut_segments(
    source: str | Path, segments: list[Segment], out_dir: str | Path, reencode: bool = False
) -> list[Path]:
    """Cut each live segment to its own mp4 via ffmpeg. Returns the written paths.

    Uses fast stream-copy by default (cuts land on the nearest keyframe, which the segment
    padding absorbs); pass ``reencode=True`` for frame-accurate but slower cuts.
    """
    if shutil.which("ffmpeg") is None:
        warnings.warn("ffmpeg not found on PATH; skipping clip cutting (manifest still written).")
        return []
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for seg in segments:
        out = out_dir / f"segment_{seg.index:03d}_{int(seg.start_sec)}s-{int(seg.end_sec)}s.mp4"
        cmd = ["ffmpeg", "-y", "-ss", f"{seg.start_sec:.3f}", "-i", str(source),
               "-t", f"{seg.duration_sec:.3f}"]
        cmd += (["-c:v", "libx264", "-preset", "veryfast", "-an"] if reencode else ["-c", "copy"])
        cmd.append(str(out))
        subprocess.run(cmd, check=True, capture_output=True)
        paths.append(out)
    return paths


def run_autoclip(
    source: str | Path,
    out_dir: str | Path,
    *,
    model_name: str = "yolo11n.pt",
    conf: float = 0.25,
    stride: int = 15,
    filter_to_surface: bool = True,
    min_segment_seconds: float = 3.0,
    merge_gap_seconds: float = 2.0,
    pad_seconds: float = 1.0,
    cut: bool = False,
    reencode_clips: bool = False,
) -> AutoClipResult:
    """Find live-play segments in a clip and write a manifest (and optionally cut the clips)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments, activity, info, surface_found = detect_live_segments(
        source, model_name=model_name, conf=conf, stride=stride,
        filter_to_surface=filter_to_surface, min_segment_seconds=min_segment_seconds,
        merge_gap_seconds=merge_gap_seconds, pad_seconds=pad_seconds,
    )

    fps = float(info.fps)
    total_frames = info.total_frames or 0
    total_seconds = total_frames / fps if fps else 0.0
    live_seconds = sum(s.duration_sec for s in segments)
    savings_frac = 1.0 - (live_seconds / total_seconds) if total_seconds else 0.0

    clip_paths = cut_segments(source, segments, out_dir, reencode=reencode_clips) if cut else []

    result = AutoClipResult(
        segments=segments, fps=fps, total_frames=total_frames, total_seconds=total_seconds,
        live_seconds=live_seconds, savings_frac=savings_frac, activity=activity,
        surface_found=surface_found, clip_paths=clip_paths,
    )
    _write_manifest(out_dir, source, result)
    return result


def _write_manifest(out_dir: Path, source: str | Path, result: AutoClipResult) -> None:
    """Write segments.json (rich) and segments.csv (spreadsheet-friendly)."""
    manifest = {
        "source": str(source),
        "fps": round(result.fps, 3),
        "total_frames": result.total_frames,
        "total_seconds": round(result.total_seconds, 2),
        "live_seconds": round(result.live_seconds, 2),
        "savings_frac": round(result.savings_frac, 4),
        "surface_found": result.surface_found,
        "n_segments": len(result.segments),
        "segments": [asdict(s) for s in result.segments],
    }
    (out_dir / "segments.json").write_text(json.dumps(manifest, indent=2))

    with (out_dir / "segments.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["index", "start_frame", "end_frame", "start_sec", "end_sec",
                           "duration_sec"]
        )
        w.writeheader()
        for s in result.segments:
            w.writerow(asdict(s))
