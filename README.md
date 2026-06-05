# dbh-vibes — Ball Hockey Video Analysis

Feed video of ball hockey games into open-source computer vision tooling, then attribute
player stats — time on surface, positioning, and (eventually) per-player events.

This repo is an early **proof-of-concept**. It currently does the *lightest-lift* slice of the
full vision: detect people in a clip from a single fixed camera, track them with stable IDs,
and emit a per-track **presence table** as a first proxy for "time on ice."

See [`docs/research-landscape.md`](docs/research-landscape.md) for the survey of open-source
tooling, [`docs/architecture.md`](docs/architecture.md) for the phased roadmap, and
[`docs/feature-ideas.md`](docs/feature-ideas.md) for the broader prioritized feature menu.

## What works today (Phase 1 MVP)

- **Detection**: pretrained [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
  `person` class — **no training required**.
- **Tracking**: ByteTrack (built into Ultralytics) gives each player a persistent track ID
  within a continuous segment.
- **Output**: an annotated video (boxes + track IDs, team colors if enabled) and a
  `tracks.csv` summarizing how long each track was on the surface.
- **Optional**: lightweight torso-color KMeans clustering into two teams.

> This is offline batch processing for a **single stationary camera**. Ball detection, rink
> mapping/heatmaps, and jersey-number identification are deferred — see the roadmap.

## What works today (Phase 2)

Validated on real ball hockey footage. Adds three capabilities on top of detection + tracking
(`src/dbh_vibes/pipeline.py`):

- **SigLIP team classification** (`team_siglip.py`) — appearance embeddings clustered **per
  track** (one mean embedding per player, not per frame, so it's CPU-affordable) into two teams.
  Hardened against the run-to-run instability that plagued the first version (deterministic PCA — no
  UMAP, over-segment-then-merge so goalies/refs can't form a team, colour-anchored stable labels,
  crop-scale decorrelation) and now **run-to-run stable (validated 100% on real footage)**. Team
  *accuracy* is still weak on low-contrast kits (white-vs-dark) — see
  [`docs/team-clustering.md`](docs/team-clustering.md) for the validation results and next steps.
- **Position heatmap** (`spatial.py`) — where players spend time, as a density overlay.
- **Active-play detection** (`activity.py`) — separates live play from bench downtime, so
  time-on-surface only accrues during real play.
- **Auto-clip / dead-time skip** (`segments.py`) — collapses the active-play signal into
  contiguous **live-play segments** (`segments.csv`), bridging brief gaps and dropping blips.
  With `--clips`, also writes each segment as a raw clip under `<out>/clips/` — the basis for
  shift detection and a big compute saving on a mostly-idle full game.

```bash
pip install -e ".[phase2]"     # adds transformers + scikit-learn
python -m dbh_vibes data/game.mp4 --out runs/game --phase2
```

Outputs `annotated.mp4` (team-colored boxes + LIVE/IDLE banner), `heatmap.jpg`, an enriched
`tracks.csv` (`team`, `team_conf`, `active_seconds`, `median_area_px`), and `segments.csv` (live-play
spans). Add `--no-siglip` to skip team classification for a faster run, or `--clips` to also
export per-segment raw clips.

### Auto-clip pre-pass (`--autoclip`) — skip dead time *before* the heavy pass

A full game is mostly dead time, so running the whole pipeline over all 38 minutes is wasteful.
`--autoclip` (`autoclip.py`) is a **cheap detection-only pre-pass** — YOLO at a coarse frame
stride, no tracker — that locates the live-play stretches first and writes a manifest you can
act on, instead of paying for the full analysis everywhere.

```bash
python -m dbh_vibes data/game.mp4 --out runs/scan --autoclip            # just the manifest
python -m dbh_vibes data/game.mp4 --out runs/scan --autoclip --cut      # + cut each clip (ffmpeg)
```

Writes `segments.json` (frame/second bounds per segment **plus a compute-savings estimate** —
how much of the video is skippable dead time) and `segments.csv`. With `--cut` it also slices
each live segment to its own `.mp4`. Tunables: `--clip-stride` (pre-pass sampling), `--min-segment`,
`--merge-gap`, `--pad`. On the reference footage a bench-break clip reports *skip 100%* (zero
segments) while live gameplay reports ~*skip 3%*.

## Quickstart

Requires Python 3.11.

```bash
# 1. Install (editable)
pip install -e .

# 2. Drop a game clip into data/ (gitignored), then run:
python -m dbh_vibes data/sample.mp4 --out runs/sample

# Lighter/faster on CPU? use the nano model (default):
python -m dbh_vibes data/sample.mp4 --out runs/sample --model yolo11n.pt

# Enable 2-team jersey-color clustering:
python -m dbh_vibes data/sample.mp4 --out runs/sample --teams
```

Outputs land in the `--out` directory:

- `annotated.mp4` — the input with detection boxes + persistent track IDs overlaid.
- `tracks.csv` — one row per track ID: first/last frame, frames seen, seconds on surface,
  and (if `--teams`) the assigned team.

The first run downloads the YOLO weights (~5 MB for `yolo11n`). Runs on CPU; an NVIDIA GPU
makes it much faster.

## Compute

- **This MVP**: runs on **CPU** with `yolo11n`/`yolo11s` for offline processing. A Colab or
  local NVIDIA GPU (≥4 GB VRAM) makes it near real-time.
- **Later (fine-tuning, full games)**: a local NVIDIA GPU (≥8 GB) or rented cloud/Colab GPU.

## License note

Ultralytics YOLO11 is **AGPL-3.0**. `supervision`, OpenCV, NumPy, and pandas are permissive
(MIT/BSD/Apache). The AGPL obligation matters if this is ever distributed as a product —
swap in an Apache/MIT detector at that point if needed.
