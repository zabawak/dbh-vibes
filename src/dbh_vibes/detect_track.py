"""Detect and track players in a single-camera clip.

Pipeline (Phase 1 MVP):
    video -> YOLO11 person detection -> ByteTrack -> annotated video + per-track presence CSV

Everything here uses pretrained COCO weights (class 0 = "person"), so no training or labeled
ball-hockey data is required to get a first result. Sport-specific detection (ball, goalie,
referee) and identity (jersey numbers) come in later phases — see docs/architecture.md.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# COCO class id for "person". The pretrained detector knows nothing about ball hockey
# specifically, but a player on the surface is just a person to it.
PERSON_CLASS_ID = 0


@dataclass
class TrackPresence:
    """Accumulates how long a single track id was visible on the surface."""

    track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    # Per-frame team votes (only populated when team clustering is enabled). Majority wins.
    team_votes: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def seconds(self, fps: float) -> float:
        return self.frames_seen / fps if fps > 0 else 0.0

    def team(self) -> int | None:
        if not self.team_votes:
            return None
        return max(self.team_votes, key=self.team_votes.get)


@dataclass
class AnalysisResult:
    """Summary of one processed clip."""

    video_path: Path
    annotated_path: Path
    csv_path: Path
    fps: float
    frame_count: int
    tracks: dict[int, TrackPresence]
    # people-on-surface count per frame index, for sanity-checking / later shift detection.
    people_per_frame: list[int]


def analyze_video(
    source: str | Path,
    out_dir: str | Path,
    model_name: str = "yolo11n.pt",
    conf: float = 0.3,
    assign_teams: bool = False,
    tracker: str = "bytetrack.yaml",
) -> AnalysisResult:
    """Run detection + tracking over a clip and write outputs.

    Args:
        source: path to the input video.
        out_dir: directory for ``annotated.mp4`` and ``tracks.csv`` (created if missing).
        model_name: Ultralytics weights to use (``yolo11n.pt`` is the lightest, CPU-friendly).
        conf: detection confidence threshold.
        assign_teams: if True, cluster players into two teams by torso color (see team_cluster).
        tracker: Ultralytics tracker config (``bytetrack.yaml`` or ``botsort.yaml``).

    Returns:
        AnalysisResult with per-track presence and the people-per-frame series.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = out_dir / "annotated.mp4"
    csv_path = out_dir / "tracks.csv"

    video_info = sv.VideoInfo.from_video_path(str(source))
    fps = float(video_info.fps)

    model = YOLO(model_name)

    # Lazy import so team clustering stays optional and its deps are only touched when used.
    team_classifier = None
    if assign_teams:
        from dbh_vibes.team_cluster import TorsoColorTeamClassifier

        team_classifier = TorsoColorTeamClassifier()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    tracks: dict[int, TrackPresence] = {}
    people_per_frame: list[int] = []

    # stream=True yields one Results per frame without buffering the whole video in memory.
    results = model.track(
        source=str(source),
        classes=[PERSON_CLASS_ID],
        conf=conf,
        tracker=tracker,
        persist=True,
        stream=True,
        verbose=False,
    )

    with sv.VideoSink(target_path=str(annotated_path), video_info=video_info) as sink:
        for frame_idx, result in enumerate(results):
            frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)

            # Drop detections the tracker couldn't assign a stable id to.
            if detections.tracker_id is None:
                detections = detections[[]]
            else:
                detections = detections[detections.tracker_id != None]  # noqa: E711

            people_per_frame.append(len(detections))

            team_ids = None
            if team_classifier is not None and len(detections) > 0:
                team_ids = team_classifier.predict(frame, detections.xyxy)

            # Update presence bookkeeping.
            for i, track_id in enumerate(detections.tracker_id):
                track_id = int(track_id)
                tp = tracks.get(track_id)
                if tp is None:
                    tp = TrackPresence(
                        track_id=track_id, first_frame=frame_idx, last_frame=frame_idx
                    )
                    tracks[track_id] = tp
                tp.last_frame = frame_idx
                tp.frames_seen += 1
                if team_ids is not None:
                    tp.team_votes[int(team_ids[i])] += 1

            labels = _build_labels(detections, team_ids)
            annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )
            sink.write_frame(annotated)

    _write_tracks_csv(csv_path, tracks, fps, has_teams=assign_teams)

    return AnalysisResult(
        video_path=source,
        annotated_path=annotated_path,
        csv_path=csv_path,
        fps=fps,
        frame_count=len(people_per_frame),
        tracks=tracks,
        people_per_frame=people_per_frame,
    )


def _build_labels(detections: sv.Detections, team_ids: np.ndarray | None) -> list[str]:
    labels = []
    for i, track_id in enumerate(detections.tracker_id):
        if team_ids is not None:
            labels.append(f"#{int(track_id)} T{int(team_ids[i])}")
        else:
            labels.append(f"#{int(track_id)}")
    return labels


def _write_tracks_csv(
    csv_path: Path, tracks: dict[int, TrackPresence], fps: float, has_teams: bool
) -> None:
    fieldnames = [
        "track_id",
        "first_frame",
        "last_frame",
        "frames_seen",
        "seconds_on_surface",
    ]
    if has_teams:
        fieldnames.append("team")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tp in sorted(tracks.values(), key=lambda t: t.seconds(fps), reverse=True):
            row = {
                "track_id": tp.track_id,
                "first_frame": tp.first_frame,
                "last_frame": tp.last_frame,
                "frames_seen": tp.frames_seen,
                "seconds_on_surface": round(tp.seconds(fps), 2),
            }
            if has_teams:
                row["team"] = tp.team()
            writer.writerow(row)
