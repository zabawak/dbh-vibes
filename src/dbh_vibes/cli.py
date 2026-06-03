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
        help="Cluster players into two teams by jersey color.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
