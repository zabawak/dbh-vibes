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
    parser.add_argument("video", help="Path to the input video (e.g. data/sample.mp4)")
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
        "--clips",
        action="store_true",
        help="With --phase2, also write each live-play segment as a raw clip to <out>/clips/.",
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
            print(f"Team clustering: {q.team_sizes[0]} vs {q.team_sizes[1]} tracks, "
                  f"silhouette {q.silhouette:.2f}, {q.n_micro} micro-cluster(s) "
                  f"(higher silhouette = cleaner split)")
        for team, secs in sorted(result.team_seconds.items()):
            print(f"  Team {team}: {secs:.0f} active player-seconds")
    print(f"Annotated video: {result.annotated_path}")
    print(f"Position heatmap: {result.heatmap_path}")
    print(f"Track summary:   {result.csv_path}")
    print(f"Play segments:   {result.segments_path}")
    if result.clips_dir is not None:
        print(f"Live-play clips: {result.clips_dir}")
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
