"""Auto-clip / dead-time skip — a cheap pre-pass that finds live play before the full analysis.

The ``--phase2 --clips`` path already exports per-segment clips, but only *after* running the
whole expensive detect+track+team pipeline over every frame. A full game is mostly dead time
(warmups, between-game downtime, bench breaks), so that's a lot of wasted compute. This module
is the fast front-door: a **detection-only pre-pass** that locates the live-play stretches first,
so the heavy pipeline (or a human) only ever looks at real action.

How it works:

1. Stream YOLO over the video at a coarse frame stride (``stride``) with **no tracker** — much
   cheaper than the Phase 2 pass. Reuse the same playing-surface filter so bench clumps can't
   masquerade as play.
2. Feed the sampled detections through the existing :mod:`dbh_vibes.activity` signal.
3. Reuse :func:`dbh_vibes.segments.segment_play` (main's segmentation core) on the sampled signal
   expanded back to full-frame resolution, then pad each segment so we don't clip the run-up /
   run-out of a play.

Outputs a ``segments.json`` + ``segments.csv`` manifest (frame/second bounds per segment plus a
**compute-savings estimate** — how much of the video is skippable dead time). With ``cut=True`` it
also writes each live segment to its own ``.mp4`` via ffmpeg, ready to feed back in. This is the
same segmentation that underpins shift detection later (see docs/architecture.md).
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dbh_vibes.activity import ActivitySummary, detect_activity, foot_points
from dbh_vibes.segments import (
    PlaySegment,
    pad_segments,
    segment_play,
    total_live_seconds,
    write_segments_csv,
    write_segments_json,
)
from dbh_vibes.surface import estimate_surface_mask, on_surface

if TYPE_CHECKING:
    import supervision as sv

PERSON_CLASS_ID = 0


@dataclass
class AutoClipResult:
    segments: list[PlaySegment]
    fps: float
    total_frames: int
    total_seconds: float
    live_seconds: float
    savings_frac: float            # fraction of the video skippable as dead time
    activity: ActivitySummary
    surface_found: bool
    segments_json: Path
    segments_csv: Path
    clip_paths: list[Path] = field(default_factory=list)


def segments_from_sampled_active(
    sampled_active: list[bool],
    *,
    fps: float,
    stride: int,
    total_frames: int,
    min_segment_seconds: float = 3.0,
    merge_gap_seconds: float = 2.0,
    pad_seconds: float = 1.0,
) -> list[PlaySegment]:
    """Turn a per-*sampled*-frame live/idle series into padded full-resolution live segments.

    ``sampled_active[j]`` is the decision for the sampled frame at original index ``j * stride``;
    each sample is treated as covering the ``stride`` original frames up to the next one. We expand
    that back to a full-frame boolean and hand it to the shared :func:`segment_play` core so the
    pre-pass and the in-pipeline path segment with identical logic, then pad the result.

    Pure (no video/model) so it can be unit-tested directly.
    """
    full_active: list[bool] = []
    for a in sampled_active:
        full_active.extend([bool(a)] * stride)
    full_active = full_active[:total_frames]
    if len(full_active) < total_frames:
        full_active.extend([False] * (total_frames - len(full_active)))

    segments = segment_play(
        full_active, fps,
        min_segment_seconds=min_segment_seconds,
        bridge_gap_seconds=merge_gap_seconds,
    )
    return pad_segments(segments, fps, pad_seconds, frame_count=total_frames)


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
) -> "tuple[list[PlaySegment], ActivitySummary, sv.VideoInfo, bool]":
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
    # aren't counted as play. None => couldn't find it; fall back to no filtering.
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

    # detect_activity's smooth_window is in (sampled) frames; convert the debounce from seconds.
    sampled_fps = fps / stride if stride else fps
    smooth_window = max(1, int(round(smooth_seconds * sampled_fps)))
    activity = detect_activity(
        per_frame_feet, info.width,
        min_players=min_players, min_spread=min_spread, smooth_window=smooth_window,
    )

    total_frames = info.total_frames or (len(per_frame_feet) * stride)
    segments = segments_from_sampled_active(
        activity.per_frame_active,
        fps=fps, stride=stride, total_frames=total_frames,
        min_segment_seconds=min_segment_seconds,
        merge_gap_seconds=merge_gap_seconds,
        pad_seconds=pad_seconds,
    )
    return segments, activity, info, surface is not None


def cut_segments(
    source: str | Path, segments: list[PlaySegment], fps: float, out_dir: str | Path,
    reencode: bool = False,
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
        start_s = seg.start_seconds(fps)
        dur_s = seg.duration_seconds(fps)
        out = out_dir / f"segment_{seg.index:03d}_{int(start_s)}s-{int(seg.end_seconds(fps))}s.mp4"
        cmd = ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(source), "-t", f"{dur_s:.3f}"]
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
    live_seconds = total_live_seconds(segments, fps)
    savings_frac = 1.0 - (live_seconds / total_seconds) if total_seconds else 0.0

    clip_paths = (
        cut_segments(source, segments, fps, out_dir, reencode=reencode_clips) if cut else []
    )

    segments_json = out_dir / "segments.json"
    segments_csv = out_dir / "segments.csv"
    write_segments_json(segments_json, segments, fps, extra={
        "source": str(source),
        "total_frames": total_frames,
        "total_seconds": round(total_seconds, 2),
        "savings_frac": round(savings_frac, 4),
        "surface_found": surface_found,
        "stride": stride,
    })
    write_segments_csv(segments_csv, segments, fps)

    return AutoClipResult(
        segments=segments, fps=fps, total_frames=total_frames, total_seconds=total_seconds,
        live_seconds=live_seconds, savings_frac=savings_frac, activity=activity,
        surface_found=surface_found, segments_json=segments_json, segments_csv=segments_csv,
        clip_paths=clip_paths,
    )
