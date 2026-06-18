"""Phase 2 pipeline: detection + tracking + team ID + spatial stats + activity gating.

Builds on the Phase 1 detect/track core but adds the capabilities validated on real footage:

- SigLIP team classification (per track, majority vote) instead of torso-color.
- A position heatmap of where players spend time.
- Active-play gating so "time on surface" only accrues during live play, not bench downtime.

Structure is two-pass to keep SigLIP affordable on CPU:
  Pass A: stream YOLO+ByteTrack once, buffering lightweight per-frame detections, sampled crops
          per track, the heatmap, and per-frame foot points. (No frames kept in memory.)
  -> fit the team model on sampled crops, assign one team per track, classify activity.
  Pass B: re-decode the video and render the annotated output from buffered boxes + team colors
          (no YOLO), which is fast.
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

from dbh_vibes.activity import ActivitySummary, detect_activity, foot_points
from dbh_vibes.boxscore import PlayerLine, build_boxscore, write_boxscore_json
from dbh_vibes.identity import assign_identities
from dbh_vibes.segments import (
    PlaySegment,
    frame_segment_index,
    segment_play,
    total_live_seconds,
    write_segments_csv,
)
from dbh_vibes.shifts import detect_shifts, summarize_player, write_shifts_csv
from dbh_vibes.spatial import PositionHeatmap
from dbh_vibes.surface import estimate_surface_mask, on_surface
from dbh_vibes.team_siglip import SiglipTeamClassifier, assign_teams, crop_box, embed_tracks

PERSON_CLASS_ID = 0
TEAM_COLORS_BGR = {0: (60, 220, 60), 1: (60, 60, 240)}  # green / red
SPECTATOR_COLOR_BGR = (130, 130, 130)  # gray for off-surface detections
UNKNOWN_COLOR_BGR = (200, 200, 200)


@dataclass
class TrackStat:
    track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    active_frames: int = 0           # frames seen while play was live
    on_surface_frames: int = 0       # frames whose foot point was on the playing surface
    areas: list[float] = field(default_factory=list)
    team: int | None = None
    team_conf: float | None = None   # confidence in the team assignment ([0,1]); label-free
    player: int | None = None        # Phase 3 identity: tracks of one person share this id
    player_conf: float | None = None # confidence in the identity assignment ([0,1]); label-free

    def on_surface_frac(self) -> float:
        return self.on_surface_frames / self.frames_seen if self.frames_seen else 0.0

    def is_player(self, min_frames: int, min_frac: float) -> bool:
        """A player spends most of its time on the surface; spectators/bench do not."""
        return self.frames_seen >= min_frames and self.on_surface_frac() >= min_frac


@dataclass
class TeamQuality:
    """Label-free clustering quality, so team ID can be judged without ground truth.

    silhouette: separation of the two-team split ([-1,1]; higher is cleaner).
    team_sizes: (T0, T1) track counts — a wildly lopsided split flags a bad fit.
    n_micro: micro-clusters used before merging to two (>2 means outliers like goalies were
        peeled off rather than allowed to tip the split).
    """

    silhouette: float
    team_sizes: tuple[int, int]
    n_micro: int
    method: str = "siglip"           # which path produced the split: "kit-color" or "siglip"


@dataclass
class IdentityQuality:
    """Label-free quality of the Phase 3 identity clustering.

    n_identities: distinct people the player tracks were stitched into.
    n_tracks: player tracks fed into the clustering (fragments before stitching).
    silhouette: separation of the identity labelling ([-1,1]; NaN when undefined).
    n_blocked_merges: merges the temporal cannot-link constraint vetoed (look-alikes kept apart).
    """

    n_identities: int
    n_tracks: int
    silhouette: float
    n_blocked_merges: int


@dataclass
class Phase2Result:
    annotated_path: Path
    heatmap_path: Path
    csv_path: Path
    segments_path: Path
    boxscore_path: Path
    fps: float
    frame_count: int
    activity: ActivitySummary
    tracks: dict[int, TrackStat]
    team_seconds: dict[int, float]
    n_players: int
    n_spectators: int
    surface_found: bool
    segments: list[PlaySegment]
    boxscore: dict
    clips_dir: Path | None = None
    team_quality: TeamQuality | None = None
    labels_path: Path | None = None
    identity_quality: IdentityQuality | None = None
    players_path: Path | None = None
    shifts_path: Path | None = None
    n_shifts: int | None = None
    report_path: Path | None = None
    shift_chart_path: Path | None = None


def run_phase2(
    source: str | Path,
    out_dir: str | Path,
    model_name: str = "yolo11s.pt",
    conf: float = 0.25,
    tracker: str = "bytetrack.yaml",
    use_siglip_teams: bool = True,
    filter_to_surface: bool = True,
    min_surface_frac: float = 0.5,
    min_track_frames: int = 15,
    min_player_area: float = 1500.0,
    crops_per_track: int = 6,
    write_clips: bool = False,
    export_labels: bool = False,
    suppress_background: bool = True,
    reid: bool = False,
    roster_size: int | None = None,
    reid_distance: float = 0.35,
    shift_gap_seconds: float = 15.0,
) -> Phase2Result:
    """Run the Phase 2 pipeline over a clip and write annotated video, heatmap, and stats."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = out_dir / "annotated.mp4"
    heatmap_path = out_dir / "heatmap.jpg"
    csv_path = out_dir / "tracks.csv"
    segments_path = out_dir / "segments.csv"
    boxscore_path = out_dir / "boxscore.json"

    info = sv.VideoInfo.from_video_path(str(source))
    fps = float(info.fps)
    model = YOLO(model_name)

    # Auto-derive the playing surface from this video's pixels (re-derived per run, so it follows
    # the camera if its position changes). None => detection failed, fall back to no filtering.
    surface = estimate_surface_mask(str(source)) if filter_to_surface else None

    # ---- Pass A: detect + track, buffer everything except frames ----
    frame_boxes: list[list[tuple[int, np.ndarray]]] = []  # per frame: [(track_id, xyxy), ...]
    per_frame_feet: list[np.ndarray] = []
    track_crops: dict[int, list[np.ndarray]] = defaultdict(list)
    tracks: dict[int, TrackStat] = {}
    heat = PositionHeatmap(info.height, info.width)
    base_frame = None

    results = model.track(
        source=str(source), classes=[PERSON_CLASS_ID], conf=conf, tracker=tracker,
        persist=True, stream=True, verbose=False,
    )
    for fi, r in enumerate(results):
        if base_frame is None:
            base_frame = r.orig_img.copy()
        det = sv.Detections.from_ultralytics(r)
        boxes_this: list[tuple[int, np.ndarray]] = []
        if det.tracker_id is not None:
            feet = foot_points(det.xyxy)
            surf = on_surface(feet, surface) if surface is not None else np.ones(len(feet), bool)
            # Activity + heatmap consider only on-surface players, so the bench can't trigger them.
            on_feet = feet[surf]
            per_frame_feet.append(on_feet)
            heat.add(on_feet)
            for i, (box, tid) in enumerate(zip(det.xyxy, det.tracker_id)):
                tid = int(tid)
                boxes_this.append((tid, box))
                area = float((box[2] - box[0]) * (box[3] - box[1]))
                ts = tracks.get(tid)
                if ts is None:
                    ts = TrackStat(track_id=tid, first_frame=fi, last_frame=fi)
                    tracks[tid] = ts
                ts.last_frame = fi
                ts.frames_seen += 1
                ts.areas.append(area)
                if surf[i]:
                    ts.on_surface_frames += 1
                # Collect team-classification crops only from on-surface detections, so the team
                # model isn't trained on spectators/bench.
                if (surf[i] and area >= min_player_area
                        and len(track_crops[tid]) < crops_per_track and ts.frames_seen % 10 == 1):
                    c = crop_box(r.orig_img, box)
                    if c is not None and c.shape[0] > 20 and c.shape[1] > 10:
                        track_crops[tid].append(c)
        else:
            per_frame_feet.append(np.empty((0, 2), dtype=np.float32))
        frame_boxes.append(boxes_this)

    frame_count = len(frame_boxes)
    players = {t for t, ts in tracks.items() if ts.is_player(min_track_frames, min_surface_frac)}

    # ---- Activity classification ----
    activity = detect_activity(per_frame_feet, info.width)
    for fi, boxes_this in enumerate(frame_boxes):
        if fi < len(activity.per_frame_active) and activity.per_frame_active[fi]:
            for tid, _ in boxes_this:
                tracks[tid].active_frames += 1

    # ---- Auto-clip: collapse the live/idle signal into live-play segments ----
    segments = segment_play(activity.per_frame_active, fps)
    write_segments_csv(segments_path, segments, fps)

    # ---- Team assignment + Phase 3 identity (per track, hardened clustering) — players only ----
    # Both consume the *same* per-track SigLIP embedding, so the expensive embedding pass is run at
    # most once: identity always needs it; team needs it only when the kit-colour prior declines.
    team_quality: TeamQuality | None = None
    identity_quality: IdentityQuality | None = None
    player_crops = {t: track_crops[t] for t in players if track_crops.get(t)}
    if (use_siglip_teams or reid) and player_crops:
        embedder = SiglipTeamClassifier()
        precomputed = None
        if reid:  # identity can't be done by colour alone — embed up front and share with team
            present_ids, track_emb = embed_tracks(
                embedder, player_crops, suppress_background=suppress_background
            )
            precomputed = (present_ids, track_emb)

        if use_siglip_teams:
            assignment = assign_teams(
                embedder, player_crops, suppress_background=suppress_background,
                precomputed=precomputed,
            )
            for tid, team in assignment.track_team.items():
                tracks[tid].team = team
                tracks[tid].team_conf = assignment.track_conf.get(tid)
            team_quality = TeamQuality(
                silhouette=assignment.silhouette,
                team_sizes=assignment.team_sizes,
                n_micro=assignment.info.n_micro,
                method=assignment.info.method,
            )

        if reid and present_ids:
            spans = {t: (tracks[t].first_frame, tracks[t].last_frame) for t in present_ids}
            id_assignment = assign_identities(
                track_emb, present_ids, spans,
                n_identities=roster_size, distance_threshold=reid_distance,
            )
            for tid, pid in id_assignment.track_identity.items():
                tracks[tid].player = pid
                tracks[tid].player_conf = id_assignment.track_conf.get(tid)
            identity_quality = IdentityQuality(
                n_identities=id_assignment.n_identities,
                n_tracks=len(present_ids),
                silhouette=id_assignment.info.silhouette,
                n_blocked_merges=id_assignment.info.n_blocked_merges,
            )

    # ---- Labeling set (optional): per-track crop montages + a labels.csv template ----
    # Same detect/track pass that writes tracks.csv, so the labels line up with the predictions by
    # track id and can be scored directly (see evaluate.py / docs priority #1).
    labels_path: Path | None = None
    if export_labels:
        from dbh_vibes.labeling import export_labeling_set

        order = sorted(players, key=lambda t: tracks[t].active_frames, reverse=True)
        track_rows = {
            tid: {
                "pred_team": tracks[tid].team if tracks[tid].team is not None else "",
                "pred_role": "player",
                "frames_seen": tracks[tid].frames_seen,
                "seconds_on_surface": round(tracks[tid].frames_seen / fps, 2) if fps else 0.0,
                "median_area_px": int(np.median(tracks[tid].areas)) if tracks[tid].areas else 0,
            }
            for tid in order
        }
        _, labels_path, _ = export_labeling_set(
            out_dir, {t: track_crops.get(t, []) for t in order}, track_rows, order=order
        )

    # ---- Heatmap output ----
    cv2.imwrite(str(heatmap_path), heat.render(base_frame))

    # ---- Auto-clip: per-segment raw clip writers (optional) ----
    clips_dir: Path | None = None
    clip_sinks: dict[int, sv.VideoSink] = {}
    frame_seg = frame_segment_index(segments, frame_count) if write_clips else []
    if write_clips and segments:
        clips_dir = out_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        for seg in segments:
            sink = sv.VideoSink(target_path=str(clips_dir / f"segment_{seg.index:02d}.mp4"),
                                video_info=info)
            sink.__enter__()
            clip_sinks[seg.index] = sink

    # ---- Pass B: render annotated video from buffered boxes ----
    with sv.VideoSink(target_path=str(annotated_path), video_info=info) as sink:
        cap = cv2.VideoCapture(str(source))
        for fi in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            # Raw (un-annotated) frame goes to the clip for its segment, before we draw on it.
            if write_clips and fi < len(frame_seg) and frame_seg[fi] is not None:
                clip_sinks[frame_seg[fi]].write_frame(frame.copy())
            live = fi < len(activity.per_frame_active) and activity.per_frame_active[fi]
            for tid, box in frame_boxes[fi]:
                ts = tracks[tid]
                is_player = tid in players
                if is_player:
                    color = TEAM_COLORS_BGR.get(ts.team, UNKNOWN_COLOR_BGR)
                    label = f"#{tid}" + (f" T{ts.team}" if ts.team is not None else "")
                else:
                    color = SPECTATOR_COLOR_BGR  # off-surface: bench / spectator
                    label = "spec"
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if not is_player else 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 if not is_player else 2)
            banner = "LIVE PLAY" if live else "IDLE"
            bcolor = (60, 220, 60) if live else (160, 160, 160)
            cv2.putText(frame, banner, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bcolor, 2)
            sink.write_frame(frame)
        cap.release()
    for s in clip_sinks.values():
        s.__exit__(None, None, None)

    # ---- Stats output ----
    team_seconds = _write_csv(csv_path, tracks, players, fps)
    players_path: Path | None = None
    shifts_path: Path | None = None
    n_shifts: int | None = None
    if reid:
        players_path = out_dir / "players.csv"
        shifts_path = out_dir / "shifts.csv"
        n_shifts = _write_players_csv(
            players_path, shifts_path, tracks, players, fps, shift_gap_seconds
        )
    boxscore = build_boxscore(
        [_player_line(tracks[t]) for t in players],
        fps=fps,
        frame_count=frame_count,
        n_spectators=len(tracks) - len(players),
        live_seconds=total_live_seconds(segments, fps),
        n_segments=len(segments),
        active_fraction=activity.active_fraction,
        surface_found=surface is not None,
    )
    write_boxscore_json(boxscore_path, boxscore)

    # ---- Per-game report + shift chart (priority #5) ----
    # Pure rendering over the artifacts just written; only meaningful once re-ID has produced
    # per-player identities + shifts (otherwise there is no shift chart to draw).
    report_path: Path | None = None
    shift_chart_path: Path | None = None
    if reid and players_path is not None:
        from dbh_vibes.report import write_report

        paths = write_report(out_dir)
        report_path = paths.html
        shift_chart_path = paths.chart_png

    return Phase2Result(
        annotated_path=annotated_path, heatmap_path=heatmap_path, csv_path=csv_path,
        segments_path=segments_path, boxscore_path=boxscore_path, fps=fps,
        frame_count=frame_count, activity=activity,
        tracks=tracks, team_seconds=team_seconds, n_players=len(players),
        n_spectators=len(tracks) - len(players), surface_found=surface is not None,
        segments=segments, boxscore=boxscore, clips_dir=clips_dir,
        team_quality=team_quality, labels_path=labels_path,
        identity_quality=identity_quality, players_path=players_path,
        shifts_path=shifts_path, n_shifts=n_shifts,
        report_path=report_path, shift_chart_path=shift_chart_path,
    )


def _player_line(ts: TrackStat) -> PlayerLine:
    """Project a pipeline ``TrackStat`` onto the box-score's plain-int input."""
    return PlayerLine(
        track_id=ts.track_id,
        team=ts.team,
        frames_seen=ts.frames_seen,
        active_frames=ts.active_frames,
        on_surface_frames=ts.on_surface_frames,
        median_area_px=int(np.median(ts.areas)) if ts.areas else 0,
    )


def _write_csv(
    csv_path: Path, tracks: dict[int, TrackStat], players: set[int], fps: float
) -> dict[int, float]:
    """Write per-track stats; return per-team active-play seconds aggregate (players only)."""
    team_seconds: dict[int, float] = defaultdict(float)
    fields = ["track_id", "role", "team", "team_conf", "player", "player_conf", "first_frame",
              "last_frame", "frames_seen", "seconds_on_surface", "active_seconds",
              "on_surface_frac", "median_area_px"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ts in sorted(tracks.values(), key=lambda t: t.active_frames, reverse=True):
            is_player = ts.track_id in players
            active_s = round(ts.active_frames / fps, 2) if fps else 0.0
            if is_player and ts.team is not None:
                team_seconds[ts.team] += active_s
            w.writerow({
                "track_id": ts.track_id,
                "role": "player" if is_player else "spectator",
                "team": ts.team if (is_player and ts.team is not None) else "",
                "team_conf": round(ts.team_conf, 2) if (is_player and ts.team_conf is not None) else "",
                "player": ts.player if (is_player and ts.player is not None) else "",
                "player_conf": round(ts.player_conf, 2)
                if (is_player and ts.player_conf is not None) else "",
                "first_frame": ts.first_frame,
                "last_frame": ts.last_frame,
                "frames_seen": ts.frames_seen,
                "seconds_on_surface": round(ts.frames_seen / fps, 2) if fps else 0.0,
                "active_seconds": active_s,
                "on_surface_frac": round(ts.on_surface_frac(), 2),
                "median_area_px": int(np.median(ts.areas)) if ts.areas else 0,
            })
    return dict(team_seconds)


def _write_players_csv(
    players_path: Path, shifts_path: Path, tracks: dict[int, TrackStat],
    players: set[int], fps: float, shift_gap_seconds: float,
) -> int:
    """Roll fragmented tracks up to per-player identities — the true per-player time-on-surface.

    Each identity sums the time of all its track fragments. ``n_shifts`` is the count of **true
    on-surface shifts** — contiguous stretches found by stitching the identity's track fragments and
    splitting only on a bench-length temporal gap (``shift.detect_shifts``), *not* the raw fragment
    count, which over-counts every time the tracker briefly drops a still-on-surface player.
    ``n_fragments`` keeps the old raw count alongside for transparency, and a per-shift breakdown is
    written to ``shifts_path``. ``team`` is the majority vote of the identity's fragments (weighted
    by frames). Rows are most-active first; tracks without an identity (re-ID off, or unembeddable)
    are skipped. Returns the total number of shifts across all players.
    """
    by_player: dict[int, list[TrackStat]] = defaultdict(list)
    for tid in players:
        ts = tracks[tid]
        if ts.player is not None:
            by_player[ts.player].append(ts)

    def team_vote(frags: list[TrackStat]) -> int | str:
        votes: dict[int, int] = defaultdict(int)
        for f in frags:
            if f.team is not None:
                votes[f.team] += f.frames_seen
        return max(votes, key=votes.get) if votes else ""

    # True shifts: stitch each identity's fragment spans, splitting only on a bench-length gap.
    track_spans = {
        pid: [(f.first_frame, f.last_frame) for f in frags] for pid, frags in by_player.items()
    }
    shifts_by_player = detect_shifts(track_spans, fps, bridge_gap_seconds=shift_gap_seconds)
    teams = {pid: team_vote(frags) for pid, frags in by_player.items()}
    write_shifts_csv(shifts_path, shifts_by_player, fps, teams=teams)

    fields = ["player", "team", "n_shifts", "n_fragments", "track_ids", "frames_seen",
              "seconds_on_surface", "shift_seconds", "active_seconds", "longest_shift_s",
              "avg_shift_s", "first_frame", "last_frame", "mean_conf"]
    rows = []
    total_shifts = 0
    for pid, frags in by_player.items():
        active_s = round(sum(f.active_frames for f in frags) / fps, 2) if fps else 0.0
        confs = [f.player_conf for f in frags if f.player_conf is not None]
        summary = summarize_player(pid, shifts_by_player.get(pid, []), fps)
        total_shifts += summary.n_shifts
        rows.append({
            "player": pid,
            "team": teams[pid],
            "n_shifts": summary.n_shifts,
            "n_fragments": len(frags),
            "track_ids": " ".join(str(f.track_id) for f in sorted(frags, key=lambda f: f.first_frame)),
            "frames_seen": sum(f.frames_seen for f in frags),
            "seconds_on_surface": round(sum(f.frames_seen for f in frags) / fps, 2) if fps else 0.0,
            "shift_seconds": summary.shift_seconds,
            "active_seconds": active_s,
            "longest_shift_s": summary.longest_shift_s,
            "avg_shift_s": summary.avg_shift_s,
            "first_frame": min(f.first_frame for f in frags),
            "last_frame": max(f.last_frame for f in frags),
            "mean_conf": round(sum(confs) / len(confs), 2) if confs else "",
        })
    rows.sort(key=lambda r: r["active_seconds"], reverse=True)
    with players_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return total_shifts
