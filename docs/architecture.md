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

## Roadmap

### Phase 2 — sport-specific detection + spatial stats
- **Fine-tune a ball-hockey detector**: label clips for `ball`, `goalie`, `referee` (+ player);
  train YOLO11 (Roboflow/Label Studio → Ultralytics). Reuse [MHPTD](https://github.com/grant81/hockeyTrackingDataset)
  where it transfers.
- **Rink homography**: build a top-down ball-hockey rink template, label court keypoints, train a
  keypoint model, project tracks to rink coordinates → **positions, heatmaps, zone time,
  team-level aggregates** (possession %, shots-on-net counts).
- Upgrade team classification to the SigLIP→UMAP→KMeans approach from roboflow/sports if the
  color fallback proves fragile.

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
- **Phase 1**: CPU is fine with `yolo11n`/`yolo11s` for offline clips (slow but works). A Colab
  or local NVIDIA GPU (≥4 GB VRAM) makes it near real-time.
- **Phase 2+**: local NVIDIA GPU (≥8 GB) or rented cloud/Colab GPU for fine-tuning and full-game
  processing. Defer the buy/rent decision until Phase 2.
