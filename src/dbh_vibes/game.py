"""Full-game mode — the end-to-end path from a raw game recording to one game report.

Every piece existed but the user still had to chain them by hand: ``--autoclip --cut`` to find and
slice the live play, ``--phase2 --reid`` per clip, and then *nothing* merged the per-clip outputs
back into per-player numbers for the game. This module is that missing orchestration + merge:

1. **Pre-pass** (``autoclip.detect_live_segments``): cheap detection-only scan finds the live-play
   segments — on the reference recording the middle third is between-games downtime, so this is
   the difference between processing ~38 min and ~20 min.
2. **Cut** (``autoclip.cut_segments``): each live segment becomes its own clip. Game mode re-encodes
   (frame-accurate) so clip frame 0 really is the segment's start frame — the merge below depends
   on that mapping.
3. **Analyze**: the full Phase 2+3 pipeline (``run_phase2 --reid``) runs per segment, exactly as it
   would on a hand-cut clip.
4. **Merge** — the genuinely new part:
   - **Cross-segment identity stitching.** Identities are clustered per segment, so player #3 of
     segment 0 and player #7 of segment 4 may be the same person. Each segment saved one mean
     embedding per identity (``identities.npz``, in the shared embedding space); those centroids
     are clustered with the same constrained-agglomerative machinery, under a hard cannot-link for
     identities *from the same segment* (the within-segment clustering already ruled them different
     people). ``--roster`` pins the game-level count.
   - **Shift stitching across stoppages, in live time.** A stoppage (idle gap between segments) is
     dead time, not a bench trip: a player on the surface at the end of one segment and the start
     of the next never left. Track spans are therefore mapped to a **live-time axis** (dead time
     compressed out) before gap-based shift detection, then mapped back to game time for
     reporting — so bench gaps are measured in *live* seconds and stoppages don't split shifts.
5. **Report**: merged ``players.csv``/``shifts.csv``/``boxscore.json`` land in the game directory in
   the standard schema, so the existing report renderer (and ``--apply-labels``) work unchanged on
   the game as a whole.

Honest caveats: team ids are per-segment (anchored to kit colour, which is designed to be
run-invariant, but low-contrast kits can flip an anchor between segments — the merge takes a
frame-weighted majority per game identity); appearance drifts over a long game (lighting, layers),
which stresses the cross-segment stitch. Both are surfaced in the printed summary rather than
hidden.

The mapping + merge cores (``LiveTimeline``, ``merge_segment_identities``) are pure
numpy/stdlib — unit-testable with no video or model.
"""

from __future__ import annotations

import csv
import json
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dbh_vibes.segments import PlaySegment


# --------------------------------------------------------------------------------------------
# Live-time axis: game frames with the dead time between segments compressed out.
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LiveTimeline:
    """Bidirectional map between game frames and a compressed "live" frame axis.

    Built from the (disjoint, time-ordered) live-play segments. ``to_live`` maps a game frame
    inside a segment to its index on the live axis, where segment k starts exactly where segment
    k-1 ended — so the idle gap between two segments has zero live width, and a temporal-gap
    threshold measured on this axis counts only *live* seconds.
    """

    starts: tuple[int, ...]        # game start frame per segment
    ends: tuple[int, ...]          # game end frame per segment (inclusive)
    live_starts: tuple[int, ...]   # live-axis offset per segment

    @classmethod
    def from_segments(cls, segments: list[PlaySegment]) -> "LiveTimeline":
        ordered = sorted(segments, key=lambda s: s.start_frame)
        starts, ends, live_starts = [], [], []
        acc = 0
        for s in ordered:
            starts.append(s.start_frame)
            ends.append(s.end_frame)
            live_starts.append(acc)
            acc += s.end_frame - s.start_frame + 1
        return cls(tuple(starts), tuple(ends), tuple(live_starts))

    def _segment_of(self, game_frame: int) -> int:
        i = bisect_right(self.starts, game_frame) - 1
        if i < 0 or game_frame > self.ends[i]:
            raise ValueError(f"game frame {game_frame} is not inside any live segment")
        return i

    def to_live(self, game_frame: int) -> int:
        i = self._segment_of(game_frame)
        return self.live_starts[i] + (game_frame - self.starts[i])

    def to_game(self, live_frame: int) -> int:
        i = bisect_right(self.live_starts, live_frame) - 1
        i = max(0, min(i, len(self.starts) - 1))
        offset = live_frame - self.live_starts[i]
        return min(self.starts[i] + offset, self.ends[i])


# --------------------------------------------------------------------------------------------
# Cross-segment identity stitching (pure — synthetic-testable).
# --------------------------------------------------------------------------------------------

@dataclass
class GameIdentityMerge:
    """Outcome of stitching per-segment identities into game-level players."""

    game_id: dict[tuple[int, int], int]   # (segment_index, local_identity) -> game player id
    n_game_players: int
    n_segment_identities: int
    n_blocked_merges: int


def merge_segment_identities(
    seg_identities: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    roster: int | None = None,
    distance_threshold: float = 0.35,
) -> GameIdentityMerge:
    """Cluster per-segment identity centroids into game-level players.

    ``seg_identities`` is ``[(segment_index, local_ids, centroids), ...]`` where ``centroids`` rows
    align to ``local_ids`` and live in the *shared* embedding space (all segments must have used the
    same embedder). Two identities from the **same segment** carry a hard cannot-link — the
    within-segment clustering (with its temporal constraint) already decided they are different
    people — so the game-level count can never fall below the busiest segment's roster.
    """
    from dbh_vibes.identity import _reduce_identity, constrained_agglomerative

    owners: list[tuple[int, int]] = []
    rows: list[np.ndarray] = []
    for seg_idx, local_ids, centroids in seg_identities:
        for lid, vec in zip(local_ids, centroids):
            owners.append((int(seg_idx), int(lid)))
            rows.append(vec)
    n = len(rows)
    if n == 0:
        return GameIdentityMerge({}, 0, 0, 0)

    reduced = _reduce_identity(np.vstack(rows), 24, 42)
    seg_of = np.array([o[0] for o in owners])
    cannot = seg_of[:, None] == seg_of[None, :]
    np.fill_diagonal(cannot, False)

    labels, blocked = constrained_agglomerative(
        reduced, cannot, n_identities=roster, distance_threshold=distance_threshold
    )
    # Relabel to contiguous ids ordered by first appearance (earliest segment first) for stable,
    # human-friendly game player ids.
    order: dict[int, int] = {}
    for lab in labels:
        if int(lab) not in order:
            order[int(lab)] = len(order)
    game_id = {owners[i]: order[int(labels[i])] for i in range(n)}
    return GameIdentityMerge(
        game_id=game_id, n_game_players=len(order),
        n_segment_identities=n, n_blocked_merges=blocked,
    )


# --------------------------------------------------------------------------------------------
# Orchestration (video + model + filesystem).
# --------------------------------------------------------------------------------------------

@dataclass
class GameResult:
    out_dir: Path
    segments: list[PlaySegment]
    n_segments_analyzed: int
    n_game_players: int
    n_segment_identities: int
    total_seconds: float
    live_seconds: float
    players_csv: Path
    shifts_csv: Path
    report_html: Path | None
    segment_dirs: list[Path] = field(default_factory=list)


def run_game(
    source: str | Path,
    out_dir: str | Path,
    *,
    model_name: str = "yolo11s.pt",
    prepass_model: str = "yolo11n.pt",
    conf: float = 0.25,
    stride: int = 15,
    min_segment_seconds: float = 3.0,
    merge_gap_seconds: float = 2.0,
    pad_seconds: float = 1.0,
    embedder: str = "siglip",
    reid_weights: str | None = None,
    roster: int | None = None,
    reid_distance: float | None = None,
    shift_gap_seconds: float = 15.0,
    max_segments: int | None = None,
) -> GameResult:
    """Process a full raw game video end to end. See the module docstring for the stages.

    ``max_segments`` optionally caps how many live segments get the heavy analysis (longest first
    is *not* used — segments run in time order and the cap truncates), which keeps a first CPU run
    of a long game bounded; the manifest still lists everything found.
    """
    from dbh_vibes.autoclip import cut_segments, detect_live_segments
    from dbh_vibes.pipeline import REID_DISTANCE_DEFAULTS, run_phase2
    from dbh_vibes.segments import total_live_seconds, write_segments_csv, write_segments_json

    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if reid_distance is None:
        reid_distance = REID_DISTANCE_DEFAULTS.get(embedder, 0.35)

    # ---- 1. cheap pre-pass: find live play ----
    segments, activity, info, surface_found = detect_live_segments(
        source, model_name=prepass_model, conf=conf, stride=stride,
        min_segment_seconds=min_segment_seconds, merge_gap_seconds=merge_gap_seconds,
        pad_seconds=pad_seconds,
    )
    fps = float(info.fps)
    total_frames = info.total_frames or 0
    total_seconds = total_frames / fps if fps else 0.0
    live_seconds = total_live_seconds(segments, fps)
    write_segments_json(out_dir / "segments.json", segments, fps, extra={
        "source": str(source), "total_frames": total_frames,
        "total_seconds": round(total_seconds, 2), "surface_found": surface_found,
    })
    write_segments_csv(out_dir / "segments.csv", segments, fps)
    if not segments:
        raise RuntimeError("no live-play segments found — nothing to analyze "
                           "(check --clip-stride / surface detection)")

    analyzed = segments if max_segments is None else segments[:max_segments]

    # ---- 2. frame-accurate cuts (the identity/shift merge relies on exact frame mapping) ----
    clip_paths = cut_segments(source, analyzed, fps, out_dir / "clips", reencode=True)
    if len(clip_paths) != len(analyzed):
        raise RuntimeError("ffmpeg unavailable or cutting failed — game mode needs the cut clips")

    # ---- 3. heavy per-segment analysis ----
    seg_dirs: list[Path] = []
    seg_identities: list[tuple[int, np.ndarray, np.ndarray]] = []
    for seg, clip in zip(analyzed, clip_paths):
        seg_dir = out_dir / f"seg_{seg.index:03d}"
        # The game roster pins the per-segment clustering too: data-driven counts over-segment
        # heavily on fragmented footage (measured 27+ identities/segment for a ~13-person game),
        # which balloons the centroid pool and puts the game-level roster out of reach of the
        # constrained merge. Pinning is safe downward — a segment where fewer people appear simply
        # stops merging early at its track count.
        run_phase2(
            clip, seg_dir, model_name=model_name, conf=conf,
            reid=True, embedder=embedder, reid_weights=reid_weights,
            roster_size=roster, reid_distance=reid_distance,
            shift_gap_seconds=shift_gap_seconds,
        )
        seg_dirs.append(seg_dir)
        npz_path = seg_dir / "identities.npz"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as z:
                seg_identities.append((seg.index, z["ids"].copy(), z["centroids"].copy()))

    # ---- 4. merge ----
    merge = merge_segment_identities(
        seg_identities, roster=roster, distance_threshold=reid_distance
    )
    players_csv, shifts_csv = _write_game_stats(
        out_dir, analyzed, seg_dirs, merge, fps, shift_gap_seconds,
        total_seconds=total_seconds, live_seconds=live_seconds,
        active_fraction=activity.active_fraction,
    )

    # ---- 5. one game report over the merged artifacts ----
    report_html: Path | None = None
    try:
        from dbh_vibes.report import write_report

        report_html = write_report(out_dir, title="Ball Hockey — Full Game Report").html
    except FileNotFoundError:
        pass

    return GameResult(
        out_dir=out_dir, segments=segments, n_segments_analyzed=len(analyzed),
        n_game_players=merge.n_game_players, n_segment_identities=merge.n_segment_identities,
        total_seconds=total_seconds, live_seconds=live_seconds,
        players_csv=players_csv, shifts_csv=shifts_csv, report_html=report_html,
        segment_dirs=seg_dirs,
    )


def _write_game_stats(
    out_dir: Path,
    segments: list[PlaySegment],
    seg_dirs: list[Path],
    merge: GameIdentityMerge,
    fps: float,
    shift_gap_seconds: float,
    *,
    total_seconds: float,
    live_seconds: float,
    active_fraction: float,
) -> tuple[Path, Path]:
    """Merge per-segment tracks into game-level players.csv + shifts.csv (+ boxscore header).

    Track spans move to game frames (clip frame + segment start), shifts are detected on the
    live-time axis (stoppages compressed out, so an idle gap can't split a shift), and shift bounds
    are mapped back to game frames for the chart. Output schemas match the per-clip artifacts so
    ``report.py`` and ``--apply-labels`` work on the game directory unchanged.
    """
    from dbh_vibes.shifts import detect_shifts, shift_record, summarize_player

    timeline = LiveTimeline.from_segments(segments)
    seg_by_index = {s.index: s for s in segments}

    # Collect per-game-player track spans (game frames) + roll-up ingredients.
    spans_live: dict[int, list[tuple[int, int]]] = defaultdict(list)
    frames_seen: dict[int, int] = defaultdict(int)
    active_s: dict[int, float] = defaultdict(float)
    n_fragments: dict[int, int] = defaultdict(int)
    track_ids: dict[int, list[str]] = defaultdict(list)
    team_votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    first_game: dict[int, int] = {}
    last_game: dict[int, int] = {}

    for seg_dir in seg_dirs:
        seg_index = int(seg_dir.name.split("_")[-1])
        seg = seg_by_index[seg_index]
        with (seg_dir / "tracks.csv").open(newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("player") or "") == "" or r.get("role") != "player":
                    continue
                local = (seg_index, int(float(r["player"])))
                gid = merge.game_id.get(local)
                if gid is None:
                    continue
                # Clamp to the segment: an ffmpeg cut can run a frame or two past the manifest.
                g_first = min(seg.start_frame + int(float(r["first_frame"])), seg.end_frame)
                g_last = min(seg.start_frame + int(float(r["last_frame"])), seg.end_frame)
                spans_live[gid].append((timeline.to_live(g_first), timeline.to_live(g_last)))
                fs = int(float(r.get("frames_seen") or 0))
                frames_seen[gid] += fs
                active_s[gid] += float(r.get("active_seconds") or 0.0)
                n_fragments[gid] += 1
                track_ids[gid].append(f"s{seg_index}:{r['track_id']}")
                if (r.get("team") or "") != "":
                    team_votes[gid][r["team"].strip()] += fs
                first_game[gid] = min(first_game.get(gid, g_first), g_first)
                last_game[gid] = max(last_game.get(gid, g_last), g_last)

    # Shifts on the live axis, then back to game frames for reporting.
    shifts_by_player = detect_shifts(dict(spans_live), fps, bridge_gap_seconds=shift_gap_seconds)
    teams = {
        gid: (sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if v else "")
        for gid, v in ((g, team_votes.get(g, {})) for g in spans_live)
    }

    shifts_csv = out_dir / "shifts.csv"
    fields = ["player", "team", "shift", "start_frame", "end_frame", "n_frames",
              "n_fragments", "start_time_s", "end_time_s", "duration_s"]
    with shifts_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for player in sorted(shifts_by_player):
            for sh in shifts_by_player[player]:
                rec = shift_record(sh, fps, team=teams.get(player, ""))
                # Map the live-axis bounds back to game time for the chart/table.
                g_start = timeline.to_game(sh.start_frame)
                g_end = timeline.to_game(sh.end_frame)
                rec.update({
                    "start_frame": g_start, "end_frame": g_end,
                    "start_time_s": round(g_start / fps, 2) if fps else 0.0,
                    "end_time_s": round((g_end + 1) / fps, 2) if fps else 0.0,
                })
                w.writerow(rec)

    players_csv = out_dir / "players.csv"
    pfields = ["player", "team", "n_shifts", "n_fragments", "track_ids", "frames_seen",
               "seconds_on_surface", "shift_seconds", "active_seconds", "longest_shift_s",
               "avg_shift_s", "first_frame", "last_frame", "mean_conf"]
    rows = []
    for gid in sorted(spans_live):
        summary = summarize_player(gid, shifts_by_player.get(gid, []), fps)
        rows.append({
            "player": gid,
            "team": teams.get(gid, ""),
            "n_shifts": summary.n_shifts,
            "n_fragments": n_fragments[gid],
            "track_ids": " ".join(track_ids[gid]),
            "frames_seen": frames_seen[gid],
            "seconds_on_surface": round(frames_seen[gid] / fps, 2) if fps else 0.0,
            "shift_seconds": summary.shift_seconds,
            "active_seconds": round(active_s[gid], 2),
            "longest_shift_s": summary.longest_shift_s,
            "avg_shift_s": summary.avg_shift_s,
            "first_frame": first_game.get(gid, 0),
            "last_frame": last_game.get(gid, 0),
            "mean_conf": "",
        })
    rows.sort(key=lambda r: r["active_seconds"], reverse=True)
    with players_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        w.writerows(rows)

    # Minimal box-score header so the report's game cards populate.
    (out_dir / "boxscore.json").write_text(json.dumps({
        "game": {
            "duration_s": round(total_seconds, 2),
            "live_play_s": round(live_seconds, 2),
            "n_segments": len(segments),
            "n_players": len(rows),
            "active_fraction": round(live_seconds / total_seconds, 4) if total_seconds else 0.0,
            "fps": fps,
        },
        "note": "game-mode merge over per-segment runs; see seg_*/boxscore.json for details",
    }, indent=2))
    return players_csv, shifts_csv


def format_game_summary(result: GameResult) -> str:
    lines = [
        f"Full game: {result.total_seconds:.0f}s of video, {result.live_seconds:.0f}s live play "
        f"across {len(result.segments)} segment(s) "
        f"({result.n_segments_analyzed} analyzed)",
        f"Identity stitch: {result.n_segment_identities} per-segment identities -> "
        f"{result.n_game_players} game players",
        f"Game stats: {result.players_csv}  +  {result.shifts_csv}",
    ]
    if result.report_html is not None:
        lines.append(f"Game report: {result.report_html}")
    return "\n".join(lines)
