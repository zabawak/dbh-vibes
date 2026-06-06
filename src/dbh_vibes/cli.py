"""Command-line entry point.

    python -m dbh_vibes data/sample.mp4 --out runs/sample [--teams] [--model yolo11n.pt]
"""

from __future__ import annotations

import argparse
import sys

from dbh_vibes.detect_track import analyze_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbh_vibes",
        description="Detect and track ball hockey players in a single-camera clip.",
    )
    parser.add_argument(
        "video", nargs="?",
        help="Path to the input video (e.g. data/sample.mp4). Optional with --evaluate.",
    )
    parser.add_argument(
        "--out", default="runs/output", help="Output directory (default: runs/output)"
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics weights. yolo11n.pt (fastest/CPU) ... yolo11x.pt (most accurate).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.3, help="Detection confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracker config (default: bytetrack.yaml)",
    )
    parser.add_argument(
        "--teams",
        action="store_true",
        help="Cluster players into two teams by jersey color (Phase 1, lightweight).",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Run the Phase 2 pipeline: SigLIP team ID + position heatmap + active-play gating.",
    )
    parser.add_argument(
        "--no-siglip",
        action="store_true",
        help="With --phase2, skip SigLIP team classification (faster, no team colors).",
    )
    parser.add_argument(
        "--no-surface-filter",
        action="store_true",
        help="With --phase2/--autoclip, don't filter off-court spectators/bench by playing surface.",
    )
    parser.add_argument(
        "--no-bg-suppress",
        action="store_true",
        help="With --phase2, embed raw crops instead of background-suppressed ones (ablation; "
             "the suppressed crops mask the rink before SigLIP so it keys on the kit, not the rink).",
    )
    parser.add_argument(
        "--clips",
        action="store_true",
        help="With --phase2, also write each live-play segment as a raw clip to <out>/clips/.",
    )
    parser.add_argument(
        "--label-crops",
        action="store_true",
        help="With --phase2, export a per-track crop montage + labels.csv template (for the "
             "eval harness) to <out>/crops/ and <out>/labels.csv.",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="With --phase2, run Phase 3 appearance re-ID: stitch fragmented tracks into per-player "
             "identities (adds a `player` column + players.csv with true per-player time-on-surface).",
    )
    parser.add_argument(
        "--roster",
        type=int,
        default=None,
        help="With --reid, pin the number of distinct players (roster size). Default: data-driven "
             "from --reid-distance and the max number of players on the surface at once.",
    )
    parser.add_argument(
        "--reid-distance",
        type=float,
        default=0.35,
        help="With --reid (and no --roster), cosine-distance threshold for merging track fragments "
             "into one identity (lower = more, smaller identities; default: 0.35).",
    )
    parser.add_argument(
        "--shift-gap",
        type=float,
        default=3.0,
        help="With --reid, on-surface gap (seconds) that separates two shifts. Fragments of one "
             "player closer than this are stitched into one shift (occlusion / tracker re-acquire); "
             "a longer gap is a bench trip → a new shift (default: 3).",
    )
    parser.add_argument(
        "--evaluate",
        metavar="LABELS_CSV",
        help="Score predictions against a filled-in labels CSV (team/role/player accuracy) and "
             "exit. Compares to <out>/tracks.csv unless --tracks is given. Needs no video.",
    )
    parser.add_argument(
        "--tracks",
        metavar="TRACKS_CSV",
        help="With --evaluate, the predictions CSV to score (default: <out>/tracks.csv).",
    )
    parser.add_argument(
        "--autoclip",
        action="store_true",
        help="Auto-clip mode: a cheap detection-only pre-pass that finds live-play segments "
             "(skipping dead time) and writes a segments.json/.csv manifest.",
    )
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=15,
        help="With --autoclip, frame stride for the cheap detection pre-pass (default: 15).",
    )
    parser.add_argument(
        "--min-segment",
        type=float,
        default=3.0,
        help="With --autoclip, drop live segments shorter than this many seconds (default: 3).",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=2.0,
        help="With --autoclip, merge live segments separated by <= this many idle seconds "
             "(default: 2).",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=1.0,
        help="With --autoclip, pad each segment by this many seconds on both ends (default: 1).",
    )
    parser.add_argument(
        "--cut",
        action="store_true",
        help="With --autoclip, also cut each live segment to its own mp4 via ffmpeg.",
    )
    parser.add_argument(
        "--reencode-clips",
        action="store_true",
        help="With --autoclip --cut, re-encode (frame-accurate) instead of fast stream-copy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.evaluate:
        return _run_evaluate(args)

    if args.video is None:
        print("error: a video path is required (or use --evaluate LABELS_CSV)", file=sys.stderr)
        return 1

    if args.autoclip:
        return _run_autoclip(args)

    if args.phase2:
        return _run_phase2(args)

    try:
        result = analyze_video(
            source=args.video,
            out_dir=args.out,
            model_name=args.model,
            conf=args.conf,
            assign_teams=args.teams,
            tracker=args.tracker,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total = sum(tp.seconds(result.fps) for tp in result.tracks.values())
    avg_on_surface = (
        sum(result.people_per_frame) / len(result.people_per_frame)
        if result.people_per_frame
        else 0
    )

    print(f"Processed {result.frame_count} frames @ {result.fps:.1f} fps")
    print(f"Distinct track ids: {len(result.tracks)}")
    print(f"Avg people on surface per frame: {avg_on_surface:.1f}")
    print(f"Total tracked player-seconds: {total:.1f}")
    print(f"Annotated video: {result.annotated_path}")
    print(f"Track summary:   {result.csv_path}")
    return 0


def _run_phase2(args) -> int:
    from dbh_vibes.pipeline import run_phase2

    try:
        result = run_phase2(
            source=args.video,
            out_dir=args.out,
            model_name=args.model,
            conf=args.conf,
            tracker=args.tracker,
            use_siglip_teams=not args.no_siglip,
            filter_to_surface=not args.no_surface_filter,
            write_clips=args.clips,
            export_labels=args.label_crops,
            suppress_background=not args.no_bg_suppress,
            reid=args.reid,
            roster_size=args.roster,
            reid_distance=args.reid_distance,
            shift_gap_seconds=args.shift_gap,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    a = result.activity
    print(f"Processed {result.frame_count} frames @ {result.fps:.1f} fps")
    surf = "auto-detected" if result.surface_found else "NOT found (filter off)"
    print(f"Playing surface: {surf}")
    print(f"Tracks: {len(result.tracks)} total -> "
          f"{result.n_players} players, {result.n_spectators} spectators/bench (filtered out)")
    print(f"Active play: {a.active_fraction*100:.0f}% of frames "
          f"(mean {a.mean_players:.1f} on-surface players/frame, spread {a.mean_spread:.3f})")
    from dbh_vibes.segments import total_live_seconds
    live_s = total_live_seconds(result.segments, result.fps)
    print(f"Live-play segments: {len(result.segments)} ({live_s:.0f}s of live play total)")
    if result.team_seconds:
        q = result.team_quality
        if q is not None:
            method = "kit-colour prior" if q.method == "kit-color" else "SigLIP embeddings"
            print(f"Team clustering [{method}]: {q.team_sizes[0]} vs {q.team_sizes[1]} tracks, "
                  f"silhouette {q.silhouette:.2f}, {q.n_micro} micro-cluster(s) "
                  f"(higher silhouette = cleaner split)")
        for team, secs in sorted(result.team_seconds.items()):
            print(f"  Team {team}: {secs:.0f} active player-seconds")
    iq = result.identity_quality
    if iq is not None:
        sil = f"{iq.silhouette:.2f}" if iq.silhouette == iq.silhouette else "n/a"  # NaN-safe
        print(f"Identity re-ID: {iq.n_tracks} player tracks -> {iq.n_identities} identities "
              f"(silhouette {sil}, {iq.n_blocked_merges} concurrent-overlap merge(s) blocked)")
        if result.n_shifts is not None:
            print(f"Shifts: {iq.n_tracks} track fragments -> {result.n_shifts} true on-surface "
                  f"shifts across {iq.n_identities} players "
                  f"(short tracker gaps stitched; bench-length gaps split)")
    from dbh_vibes.boxscore import format_boxscore
    print(format_boxscore(result.boxscore))
    print(f"Annotated video: {result.annotated_path}")
    print(f"Position heatmap: {result.heatmap_path}")
    print(f"Track summary:   {result.csv_path}")
    print(f"Play segments:   {result.segments_path}")
    print(f"Box score:       {result.boxscore_path}")
    if result.clips_dir is not None:
        print(f"Live-play clips: {result.clips_dir}")
    if result.players_path is not None:
        print(f"Per-player:      {result.players_path} (identities w/ true per-player time-on-surface)")
    if result.shifts_path is not None:
        print(f"Per-shift:       {result.shifts_path} (one row per on-surface shift)")
    if result.labels_path is not None:
        print(f"Labeling set:    {result.labels_path} (+ crops/) — fill in team/role/player, "
              f"then: python -m dbh_vibes --evaluate {result.labels_path}")
    return 0


def _run_evaluate(args) -> int:
    from pathlib import Path

    from dbh_vibes.evaluate import evaluate, format_report

    labels_csv = Path(args.evaluate)
    if not labels_csv.exists():
        print(f"error: labels CSV not found: {labels_csv}", file=sys.stderr)
        return 1
    tracks_csv = Path(args.tracks) if args.tracks else Path(args.out) / "tracks.csv"
    if not tracks_csv.exists():
        print(f"error: predictions CSV not found: {tracks_csv} "
              f"(run --phase2 first, or pass --tracks)", file=sys.stderr)
        return 1

    report = evaluate(labels_csv, tracks_csv)
    print(format_report(report))
    return 0


def _run_autoclip(args) -> int:
    from dbh_vibes.autoclip import run_autoclip

    try:
        result = run_autoclip(
            source=args.video,
            out_dir=args.out,
            model_name=args.model,
            conf=args.conf,
            stride=args.clip_stride,
            filter_to_surface=not args.no_surface_filter,
            min_segment_seconds=args.min_segment,
            merge_gap_seconds=args.merge_gap,
            pad_seconds=args.pad,
            cut=args.cut,
            reencode_clips=args.reencode_clips,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    surf = "auto-detected" if result.surface_found else "NOT found (filter off)"
    print(f"Scanned {result.total_frames} frames @ {result.fps:.1f} fps "
          f"({result.total_seconds:.0f}s) every {args.clip_stride} frames")
    print(f"Playing surface: {surf}")
    print(f"Live play: {result.activity.active_fraction*100:.0f}% of sampled frames "
          f"(mean {result.activity.mean_players:.1f} on-surface players/frame)")
    print(f"Found {len(result.segments)} live segment(s), {result.live_seconds:.0f}s of play "
          f"-> skip {result.savings_frac*100:.0f}% of the video as dead time")
    for s in result.segments:
        print(f"  [{s.index:02d}] {s.start_seconds(result.fps):7.1f}s - "
              f"{s.end_seconds(result.fps):7.1f}s ({s.duration_seconds(result.fps):.1f}s)  "
              f"frames {s.start_frame}-{s.end_frame}")
    if result.clip_paths:
        print(f"Cut {len(result.clip_paths)} clip(s) into {args.out}/")
    print(f"Segment manifest: {result.segments_json}  (+ {result.segments_csv.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
