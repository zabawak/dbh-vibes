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

Outputs `annotated.mp4` (team-colored boxes + a LIVE/IDLE banner), `heatmap.jpg`, an enriched
`tracks.csv` (adds `team`, `active_seconds`, `median_area_px`), `segments.csv` (live-play
spans; `--clips` also exports per-segment raw clips), and `boxscore.json` (a consumable per-game
roll-up: game header + per-team totals + per-player table). Pipeline lives in
`src/dbh_vibes/pipeline.py` (two-pass: detect/track once → fit teams + activity + segments → render).

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
- **Auto-clip / dead-time skip** (`segments.py`) — collapses the per-frame active signal into
  contiguous **live-play segments** (written to `segments.csv` with frame/second bounds),
  bridging brief idle gaps and dropping sub-second blips. `--clips` re-uses the render pass to
  also write each segment as a raw clip under `<out>/clips/`. This is the compute-saving
  "process only live play" lever and the scaffolding the Phase 3 shift detector builds on. Pure
  stdlib core, unit-tested in `tests/test_segments.py`.
  - **Auto-clip pre-pass** (`autoclip.py`, `--autoclip`) — runs the *same* `segment_play` core,
    but fed by a **detection-only pre-pass** (YOLO at a coarse `--clip-stride`, no tracker) so it
    finds live play *before* paying for the full Phase 2 pass. Writes a `segments.json` manifest
    with frame/second bounds **and a compute-savings estimate** (fraction skippable as dead time),
    plus `segments.csv`; `--cut` slices each segment to its own mp4 via ffmpeg. Knobs:
    `--min-segment`, `--merge-gap`, `--pad`. The expand-to-full-resolution + segment + pad logic is
    pure and unit-tested in `tests/test_autoclip.py`. Validated on the reference footage: a
    bench-break clip → 0 segments (skip 100%), live gameplay → ~skip 3%.
- **Box-score / stats export** (`boxscore.py`) — rolls the scattered per-track numbers into one
  consumable `boxscore.json` (game header + per-team totals + a per-player table, most-active
  first) and a compact text table in the console summary. Deliberately **per-track**, not per-
  *player*: with no jersey numbers a re-entering player is still two tracks (same caveat as
  `tracks.csv`), so true per-player shift counts wait on Phase 3 identity; team totals sum over
  tracks and are robust to the fragmentation. Pure-stdlib core, unit-tested in
  `tests/test_boxscore.py`.
- **Playing-surface filter** (`surface.py`) — separates on-court players from bench/spectators
  by keeping only detections whose foot point lands on the playing surface. The surface is
  **auto-derived per run** from a time-median of the footage + court-color segmentation, so it
  follows the camera if its position changes — no fixed polygon, no recalibration. Validated to
  re-derive correctly under a simulated camera move. Disable with `--no-surface-filter`. Tracks
  are tagged `player`/`spectator` in `tracks.csv`; only players get teams and time totals.

### Still open in Phase 2
- **Harden team clustering (highest priority).** The current SigLIP→UMAP→KMeans(k=2) team split
  is **unstable run to run** (e.g., 18-vs-15 one run, 28-vs-6 the next). It is the weakest link
  and undermines every team-level stat. Full analysis, root causes, and the plan to fix it are in
  **[team-clustering.md](team-clustering.md)** — deferred to a dedicated session.
- **Fine-tune a ball-hockey detector** for `ball`, `goalie`, `referee` classes (needs labeled
  clips; reuse [MHPTD](https://github.com/grant81/hockeyTrackingDataset) where it transfers).
- **Calibrated top-down rink map** once camera intrinsics/keypoints are available (fisheye
  undistort + homography) → zone time, possession %, shots-on-net.

### Phase 3 — player identity + event attribution

**Constraint: no jersey numbers.** This is pickup ball hockey — players generally won't have
readable numbers (no numbers at all, or too low-res / blurred / facing away to OCR). So the
classic jersey-number-OCR path does **not** apply. Instead we identify players by **appearance**:
each player wears distinct gear (shirt, shorts, socks, helmet, build, skin tone) that is
**consistent within a single game**, even if it changes between games.

- **Appearance-based re-identification.** Build a per-player appearance signature (a re-ID
  embedding — e.g. OSNet/torchreid, or per-track aggregated SigLIP features) and cluster all
  tracks in a game into ~roster-size identities. Each identity = the set of fragmented tracks
  belonging to one person. This stitches the fragmented track ids (we saw ~100+ ids for ~13
  people) into **stable per-player identities** → true per-player time-on-surface and shift
  counts. Same embedding machinery as team clustering, but at finer (per-individual) granularity.
- **How this differs from team clustering.** Team = coarse (2 groups by kit); identity = fine
  (one cluster per person, using the *per-player* gear differences as the signal). Identity is
  the harder, more valuable target.
- **Known failure mode to document.** If two players wear near-identical gear, appearance alone
  can't separate them — fall back on spatiotemporal continuity (motion/position across short
  gaps) and, where it exists, any distinguishing cue. A real league with matching uniforms would
  break this entirely and would need numbers or positional tracking; pickup with varied gear is
  the favorable case.
- **Other constraints to respect.** Appearance is consistent only *within* a game (re-build the
  gallery per game); lighting drifts over a long game; players add/remove layers. Let roster size
  be configurable or auto-determined from clustering quality.
- **Shift detection.** Once identities are stable, model the bench / entry-exit zones (a
  `supervision` line/polygon zone) so a player crossing on/off the surface starts/stops a shift
  cleanly → line changes, goals/assists attribution.

### Phase 4 — scale & UX
- Multi-camera capture + fusion for full surface coverage and fewer occlusions.
- A simple report/dashboard per game; possibly near-real-time processing on a GPU.

For the broader, prioritized menu of stats/pipeline/UX features (ball detection, possession,
shots/goals, movement load, dashboards, quick wins like auto-clipping and human-in-the-loop
identity), see **[feature-ideas.md](feature-ideas.md)**.

## Compute guidance
- **Phase 1 & 2**: validated end-to-end on **CPU** (4 cores) with `yolo11s`. A 30s 720p clip takes
  a few minutes; SigLIP team fitting adds ~2 min. The per-track team trick keeps SigLIP affordable
  without a GPU. A Colab or local NVIDIA GPU (≥4 GB VRAM) makes it near real-time.
- **Fine-tuning (open Phase 2 / Phase 3)**: local NVIDIA GPU (≥8 GB) or rented cloud/Colab GPU.
  Defer the buy/rent decision until we actually start labeling and training.
