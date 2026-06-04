# Architecture & Roadmap

How dbh-vibes is built and where it's going. Background and citations live in
[research-landscape.md](research-landscape.md).

## Design constraints

- **Single stationary camera** to start (open to multiple later).
- **Offline batch processing**, not real-time — we process recorded clips after the game.
- Favor **pretrained, no-label** capabilities first; defer anything that needs us to label
  ball-hockey-specific data.

## Current pipeline (Phase 1 — implemented)

```
data/sample.mp4
      │
      ▼
YOLO11 person detection (pretrained COCO, class 0)      [detect_track.py]
      │
      ▼
ByteTrack  →  persistent track id per player            [Ultralytics built-in]
      │
      ├─▶ (optional) torso-color KMeans → team 0/1       [team_cluster.py]
      ▼
annotated.mp4   +   tracks.csv (per-track presence)      [detect_track.py]
```

`tracks.csv` columns: `track_id, first_frame, last_frame, frames_seen, seconds_on_surface[, team]`.
`seconds_on_surface` is the first proxy for **time on ice** — it's per *track*, so a player who
leaves the frame and returns currently counts as two tracks. Stable cross-shift identity (Phase
3) is what turns this into true per-player TOI.

### Key modules
- `src/dbh_vibes/detect_track.py` — `analyze_video()` orchestrates detection→tracking→output.
- `src/dbh_vibes/team_cluster.py` — `TorsoColorTeamClassifier`, the lightweight team split.
- `src/dbh_vibes/cli.py` — argument parsing + summary printout.

We lean on **[supervision](https://github.com/roboflow/supervision)** for detection wrangling,
annotation (`BoxAnnotator`, `LabelAnnotator`), and video I/O (`VideoSink`, `VideoInfo`), and on
**Ultralytics** for both detection and the bundled ByteTrack/BoT-SORT trackers.

## Phase 2 — team ID, spatial stats, activity gating (implemented)

Validated on real footage from a single fixed fisheye camera. Run with:

```bash
python -m dbh_vibes data/game.mp4 --out runs/game --phase2
```

Outputs `annotated.mp4` (team-colored boxes + a LIVE/IDLE banner), `heatmap.jpg`, and an enriched
`tracks.csv` (adds `team`, `active_seconds`, `median_area_px`). Pipeline lives in
`src/dbh_vibes/pipeline.py` (two-pass: detect/track once → fit teams + activity → render).

- **SigLIP team classification** (`team_siglip.py`) — embeds player crops with the SigLIP vision
  tower, reduces with UMAP, clusters with KMeans, mirroring roboflow/sports. Replaces the MVP
  torso-color split, which collapsed on real footage. Classified **per track** (majority vote over
  a few sampled crops), so a clip costs a few hundred embeds, not tens of thousands — practical
  even on CPU (~2 min/clip). Validated: cleanly isolates the red-pinnie team with zero
  contamination.
- **Position heatmap** (`spatial.py`) — accumulates foot-point density into a colored overlay.
  Kept in **image space**: a single planar homography to a top-down view is unreliable on this
  fixed fisheye with the near boards occluded, so an honest image-space map is the base for a
  properly calibrated top-down view later.
- **Active-play detection** (`activity.py`) — gates on on-surface player count + horizontal
  spread to separate live play from bench downtime. Validated: gameplay 100% live vs. a break
  0% live. `time_on_surface` accrues only during live frames.
- **Playing-surface filter** (`surface.py`) — separates on-court players from bench/spectators
  by keeping only detections whose foot point lands on the playing surface. The surface is
  **auto-derived per run** from a time-median of the footage + court-color segmentation, so it
  follows the camera if its position changes — no fixed polygon, no recalibration. Validated to
  re-derive correctly under a simulated camera move. Disable with `--no-surface-filter`. Tracks
  are tagged `player`/`spectator` in `tracks.csv`; only players get teams and time totals.

### Still open in Phase 2
- **Fine-tune a ball-hockey detector** for `ball`, `goalie`, `referee` classes (needs labeled
  clips; reuse [MHPTD](https://github.com/grant81/hockeyTrackingDataset) where it transfers).
- **Calibrated top-down rink map** once camera intrinsics/keypoints are available (fisheye
  undistort + homography) → zone time, possession %, shots-on-net.

### Phase 3 — player identity + event attribution
- **Jersey-number OCR + appearance re-ID** to stitch track ids across occlusions and line
  changes into stable player identities → **true per-player time on surface, shift/line-change
  detection, goals/assists**.
- **Shift detection**: model the bench / entry-exit zones (a `supervision` line/polygon zone) so
  a player crossing on/off the surface starts/stops a shift cleanly.

### Phase 4 — scale & UX
- Multi-camera capture + fusion for full surface coverage and fewer occlusions.
- A simple report/dashboard per game; possibly near-real-time processing on a GPU.

## Compute guidance
- **Phase 1 & 2**: validated end-to-end on **CPU** (4 cores) with `yolo11s`. A 30s 720p clip takes
  a few minutes; SigLIP team fitting adds ~2 min. The per-track team trick keeps SigLIP affordable
  without a GPU. A Colab or local NVIDIA GPU (≥4 GB VRAM) makes it near real-time.
- **Fine-tuning (open Phase 2 / Phase 3)**: local NVIDIA GPU (≥8 GB) or rented cloud/Colab GPU.
  Defer the buy/rent decision until we actually start labeling and training.
