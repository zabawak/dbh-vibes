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
        help="With --phase2, don't filter out off-court spectators/bench by playing surface.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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
    if result.team_seconds:
        for team, secs in sorted(result.team_seconds.items()):
            print(f"  Team {team}: {secs:.0f} active player-seconds")
    print(f"Annotated video: {result.annotated_path}")
    print(f"Position heatmap: {result.heatmap_path}")
    print(f"Track summary:   {result.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
