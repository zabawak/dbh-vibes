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

- **Team classification** (`team_siglip.py`) — two auto-selected paths. A **kit-colour prior**
  splits on background-suppressed torso chroma when one team wears a vivid kit (the "pinnies vs
  none" case) — strong, scale-immune, and it skips SigLIP. Otherwise it falls back to **SigLIP
  appearance embeddings** clustered **per track** (one mean embedding per player). The embedding
  path was hardened against the run-to-run instability that plagued the first version (deterministic
  PCA — no UMAP, over-segment-then-merge so goalies/refs can't form a team, colour-anchored stable
  labels, crop-scale decorrelation) and is now **run-to-run stable (validated 100% on real
  footage)**. Crops are also **background-suppressed before embedding** (torso-crop + mask the
  rink-coloured pixels to grey) so SigLIP keys on the kit not the blue rink — measured to lift team
  accuracy **52.2% → 56.5%** on the reference clip (`--no-bg-suppress` ablates it). *Accuracy* on
  low-contrast kits (white-vs-dark) is improved but still weak when the colour prior can't fire —
  see [`docs/team-clustering.md`](docs/team-clustering.md) for validation + next steps.
- **Position heatmap** (`spatial.py`) — where players spend time, as a density overlay.
- **Active-play detection** (`activity.py`) — separates live play from bench downtime, so
  time-on-surface only accrues during real play.
- **Auto-clip / dead-time skip** (`segments.py`) — collapses the active-play signal into
  contiguous **live-play segments** (`segments.csv`), bridging brief gaps and dropping blips.
  With `--clips`, also writes each segment as a raw clip under `<out>/clips/` — the basis for
  shift detection and a big compute saving on a mostly-idle full game.
- **Box-score / stats export** (`boxscore.py`) — rolls the scattered per-track numbers into one
  consumable `boxscore.json` (game header + per-team totals + a per-player table) and prints a
  compact text table. Per-track today, not yet per-player (no jersey numbers → true shift counts
  wait on Phase 3 identity); team totals are robust to the track fragmentation.
- **Labeled eval set + harness** (`labeling.py` + `evaluate.py`) — the way we now *measure*
  accuracy instead of eyeballing it. `--label-crops` exports one crop montage per track plus a
  pre-filled `labels.csv` template (same pass as `tracks.csv`, so ids line up); a human tags
  team/role/identity by sight in ~2 minutes; `python -m dbh_vibes --evaluate <labels.csv>` scores
  the predictions with optimal cluster-label alignment (an arbitrary team `0`/`1` aligns to
  "white"/"dark"). Measured numbers on the reference clip: **team 52.2% (raw crops) → 56.5%
  (background-suppressed crops), role 100%** — see [`eval/README.md`](eval/README.md).

```bash
pip install -e ".[phase2]"     # adds transformers + scikit-learn
python -m dbh_vibes data/game.mp4 --out runs/game --phase2
```

Outputs `annotated.mp4` (team-colored boxes + LIVE/IDLE banner), `heatmap.jpg`, an enriched
`tracks.csv` (`team`, `team_conf`, `active_seconds`, `median_area_px`), `segments.csv` (live-play
spans), and `boxscore.json` (per-game roll-up). Add `--no-siglip` to skip team classification for a
faster run, `--clips` to also export per-segment raw clips, or `--label-crops` to export the
labeling set for the eval harness.

## What works today (Phase 3 — per-player identity)

Detection + ByteTrack give a track id that survives only one continuous on-surface stretch, so one
person fragments into many track ids (27 tracks for ~13 people on the reference clip). **Appearance
re-ID** (`identity.py`, `--reid`) stitches the fragments back into per-player **identities** so we
get *true per-player* time-on-surface and shift counts — the headline goal.

- Reuses the **same per-track SigLIP embedding** as team clustering (background-suppressed crops),
  shared so SigLIP runs once, then clusters tracks with **constrained agglomerative clustering**
  under a hard **temporal cannot-link** (two tracks overlapping in time can't be one person — which
  also blocks the look-alike failure mode and floors the identity count near the roster).
- Adds a `player`/`player_conf` column to `tracks.csv` and writes **`players.csv`**: one row per
  identity with summed time-on-surface, `n_shifts`, the constituent track ids, and team.
- **Validated on real footage:** deterministic; 0 temporal violations; forced to roster size
  (`--roster 13`) every team-checkable merge is same-team (15/15, 0 cross-team, vs ~49% chance) —
  real identity signal. A clean per-individual accuracy number still waits on identity ground truth
  (hard to label by sight at this crop resolution). See [`docs/identity-reid.md`](docs/identity-reid.md).

```bash
python -m dbh_vibes data/game.mp4 --out runs/game --phase2 --reid            # data-driven count
python -m dbh_vibes data/game.mp4 --out runs/game --phase2 --reid --roster 13  # pin roster size
```

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
