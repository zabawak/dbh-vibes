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
`tracks.csv` (adds `team`, `team_conf`, `active_seconds`, `median_area_px`), `segments.csv` (live-play
spans; `--clips` also exports per-segment raw clips), and `boxscore.json` (a consumable per-game
roll-up: game header + per-team totals + per-player table). Pipeline lives in
`src/dbh_vibes/pipeline.py` (two-pass: detect/track once → fit teams + activity + segments → render).

- **SigLIP team classification** (`team_siglip.py`) — embeds player crops with the SigLIP vision
  tower, then clusters appearance into two teams. Replaces the MVP torso-color split, which
  collapsed on real footage. Classified **per track** (one mean embedding per player), so a clip
  costs a few hundred embeds, not tens of thousands — practical even on CPU (~2 min/clip). The
  clusterer was hardened to fix run-to-run instability (deterministic PCA not UMAP, over-segment then
  merge by size so goalies/refs can't form a team, colour-anchored stable labels, scale-decorrelation)
  and is now **run-to-run stable (validated 100% on real footage)**. Crops are now
  **background-suppressed before embedding** (`background_suppressed_crop` torso-crops each box and
  masks the rink-coloured pixels to grey, so SigLIP keys on the kit not the blue rink), which lifts
  measured accuracy **52.2% → 56.5%** on the reference clip and balances the split — still short of
  clean on the low-contrast white-vs-dark kits here; see [team-clustering.md](team-clustering.md).
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
- **Labeled eval set + harness** (`labeling.py` + `evaluate.py`) — the way accuracy is now
  *measured* rather than eyeballed (this was the project's stated binding constraint). `--label-crops`
  exports one crop montage per track plus a pre-filled `labels.csv` template from the *same*
  detect/track pass that writes `tracks.csv`, so track ids line up. A human tags team/role/identity
  by sight in ~2 minutes; `python -m dbh_vibes --evaluate <labels.csv>` scores the predictions —
  **optimal cluster-label alignment** for team/identity (an arbitrary `0`/`1` aligns to
  "white"/"dark"), direct equality for role — over the labeled∩predicted overlap. The committed
  `eval/sample_labels.csv` gives the first measured numbers on natural footage: **team 52.2%
  (~chance), role 100%**. Pure-numpy metric core, unit-tested in `tests/test_evaluate.py`; see
  [`../eval/README.md`](../eval/README.md).

### Still open in Phase 2
- **Team clustering — stability fixed and validated; accuracy still limited on low-contrast kits.**
  The old SigLIP→UMAP→KMeans(k=2) team split was **unstable run to run** (18-vs-15 one run, 28-vs-6
  the next). It was reworked to cluster **per track** (mean embedding), reduce with **deterministic
  PCA** (no UMAP), **over-segment then merge by size** (goalies/refs fold into a team instead of
  *becoming* one), anchor **stable T0/T1 labels** to kit colour, and **decorrelate from crop scale**
  — plus a label-free quality signal (silhouette, balance, per-track `team_conf`). Validated on the
  real clip: **100% identical assignments across repeated runs** (instability fixed). But the same
  validation showed the split was driven by **crop near/far scale, not kit** (top PC ~0.86 correlated
  with crop area); decorrelation removes that confound, yet the low-contrast white/dark kits on this
  footage still don't separate by appearance. The eval harness (above) **measures** this: raw-crop
  embeddings score **team accuracy 52.2% (~chance)** on the reference clip. The kit-colour/pinnie
  prior and **background-suppressed crops** (mask the rink/legs/skin before embedding) are both in;
  suppression lifts the embedding path to **56.5%** and balances the split — a real but partial gain,
  white-vs-dark stays hard. Full write-up + numbers in **[team-clustering.md](team-clustering.md)**.
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

- **Appearance-based re-identification — implemented (`identity.py`, `--reid`).** Builds a
  per-player appearance signature (per-track aggregated SigLIP features on background-suppressed
  crops — the *same* embedding pass as team clustering, shared so SigLIP runs once) and clusters all
  player tracks in a clip into ~roster-size identities. Each identity = the set of fragmented tracks
  belonging to one person, so the fragmented track ids (27 tracks for ~13 people on the reference
  clip) stitch into **stable per-player identities** → true per-player time-on-surface and shift
  counts, emitted as a `player` column in `tracks.csv` and a per-player `players.csv` roll-up.
- **Constrained clustering with a temporal prior.** The clusterer is constrained agglomerative
  (average-linkage cosine) with a hard **temporal cannot-link**: two tracks whose frame spans
  overlap in time cannot be the same person. This both prevents the look-alike failure mode (below)
  and makes the *max number of players on the surface at once* a natural floor on the identity count
  (~12 in 5-on-5 + goalies), so the count lands near the roster even without `--roster`.
- **How this differs from team clustering.** Team = coarse (2 groups by kit); identity = fine
  (one cluster per person, using the *per-player* gear differences as the signal). Identity is
  the harder, more valuable target.
- **Validated on real footage.** Deterministic across runs; 0 temporal violations; forced to roster
  size (`--roster 13`) every team-checkable merge respects the team ground truth (15/15 same-team, 0
  cross-team, vs ~49% chance) — real identity signal. A clean per-individual *accuracy* number still
  needs identity ground truth, which is hard to label by sight at this crop resolution (same root
  cause as the 56.5% team ceiling). The eval harness (`evaluate.py`) already scores the `player`
  column with the same optimal-alignment metric used for teams, ready the moment such labels exist.
  See **[identity-reid.md](identity-reid.md)**.
- **Known failure mode (handled).** Two players in near-identical gear can't be separated by
  appearance alone — the temporal cannot-link constraint stops them collapsing *whenever they share
  the surface*, falling back on spatiotemporal continuity. A real league in matching uniforms would
  still need numbers or positional tracking; pickup with varied gear is the favorable case.
- **Other constraints to respect.** Appearance is consistent only *within* a game (re-build the
  gallery per game); lighting drifts over a long game; players add/remove layers. Let roster size
  be configurable or auto-determined from clustering quality.
- **Shift detection — implemented (`shifts.py`, emitted with `--reid`).** Once identities are
  stable, a player's *true shifts* are the contiguous on-surface stretches of their identity.
  `detect_shifts` stitches each identity's fragmented track spans, **bridging short temporal gaps**
  (an occlusion / tracker re-acquire — still the same shift) and **splitting on a bench-length gap**
  (the player went to the bench and came back — a new shift). The Phase 2 surface filter already
  drops off-surface (bench) detections, so a bench trip shows up as a long dark stretch in the
  identity's on-surface timeline — i.e. the *temporal gap is the bench signal*, riding on the
  surface filter with no hand-drawn bench polygon. This replaces the prior `n_shifts = fragment
  count`, which over-counted every time the tracker briefly lost a still-on-surface player. Outputs
  `shifts.csv` (one row per on-surface shift) and adds `n_shifts` (true) / `n_fragments` (raw) /
  `shift_seconds` / `longest_shift_s` / `avg_shift_s` to `players.csv`. `--shift-gap` sets the
  bench-vs-occlusion threshold; the **default is 15 s — a physical floor on a real bench change** (a
  player can't get to the bench, sub off, and return in less), so shorter absences are treated as
  in-shift occlusion. Pure-stdlib core, unit-tested in `tests/test_shifts.py`.
  - **Validated on real footage** (deterministic; shifts non-overlapping within each player;
    `n_shifts ≤ n_fragments` always). On the 30 s reference clip (`data/sample.mp4`, `--reid
    --roster 13`): 28 track fragments → 13 identities → **13 shifts = exactly 1.0/player** — the
    right answer for a 30 s window, since nobody completes a bench change that fast, so every
    player's dropouts collapse into one continuous shift. On a **3-minute clip with real line
    changes** (`-ss 1430 -t 180`): 141 fragments → 20 identities → **61 shifts (3.0/player, avg
    32 s)** — heavily-fragmented players collapse sensibly (12 fragments → 2 shifts) while a rotating
    bench's per-player shift structure is preserved.
  - **Honest limitation.** Without an explicit bench zone the threshold is the *only* thing
    separating "off the surface" from "on the surface but undetected", and on this fisheye footage
    the measured inter-fragment gap distribution is **not cleanly bimodal** (sub-3 s dropouts blur
    into 5–10 s occlusions into 20–60 s bench trips), so 15 s is a defensible judgement call, not a
    learned boundary. An earlier 3 s default over-split (it counted occlusions as bench trips: 5.2
    shifts/player on the 3-min clip); 15 s is grounded in bench-change physics and tunable per game.
  - *Next (deferred):* an explicit `supervision` entry/exit zone to sharpen the exact on/off instant
    and replace the gap heuristic — the principled fix once camera geometry is plumbed through.

### Phase 4 — scale & UX
- **Per-game report + shift chart — implemented (`report.py`, priority #5).** Identity + shifts
  produce `players.csv`/`shifts.csv`; this *surfaces* them. Emits a self-contained **`report.html`**
  (game header + per-player stat table — TOI, shifts, avg/longest — + per-team totals over true
  identities + the heatmap + the shift chart, every image inlined as a `data:` URI) and a **shift
  chart** **`shift_chart.png`** — the classic time-on-ice Gantt, one row per player, one bar per
  shift, **rows grouped by team and ordered most-time-on-surface first**. No new model/GPU/labels —
  pure rendering over the already-written artifacts. The **chart layout is a pure-stdlib core**
  (`build_shift_chart`: rows = players, bars = shifts — ordering + coordinates, no drawing),
  unit-tested in `tests/test_report.py` like `segments.py`/`shifts.py`, with a thin matplotlib PNG +
  HTML-assembly shell. Emits on `--phase2 --reid` and runs **standalone** over a finished run dir
  (`--report <run-dir>`, no video). Validated on real footage: the 30 s reference clip → 13 identities
  at 1 shift each (correct for a window too short to bench in); the 3-min line-change clip → a
  multi-shift Gantt with the two teams cleanly grouped.
- **Re-ID embedding upgrade — implemented (`--embedder osnet`, priority #6).** OSNet-AIN person
  re-ID network (vendored `osnet.py`, checkpoints fetched + safe-unpickled by `reid_embedder.py`)
  as a drop-in alternative to SigLIP for the shared team/identity embedding. **Measured: team
  accuracy 57.1% → 100.0% on the reference clip's fresh labels (identical tracks)** — the
  low-contrast kit ceiling was the borrowed embedding, not the footage. Identity improves (2× the
  fragment merges, all team-consistent) but still over-segments; per-embedder distance defaults in
  `pipeline.REID_DISTANCE_DEFAULTS`. ~10× cheaper per crop than SigLIP on CPU.
- **Human-in-the-loop naming — implemented (`--apply-labels`, `roster.py`).** A filled-in
  `labels.csv` propagates back through the pipeline's clusters: tag one track → the whole
  identity/team is named in `tracks.csv`/`players.csv` and the re-rendered report; clustering
  conflicts (over-merge / over-segmentation) are surfaced, not hidden.
- **Full-game mode — implemented (`--game`, `game.py`).** Autoclip pre-pass → frame-accurate cuts →
  per-segment `--phase2 --reid` → cross-segment identity stitch (per-segment centroids from
  `identities.npz`, same constrained clustering under a same-segment cannot-link) → shifts stitched
  across stoppages on a compressed **live-time axis** (`LiveTimeline`) → merged
  `players.csv`/`shifts.csv`/`boxscore.json` + one game report in the standard schema.
- Multi-camera capture + fusion for full surface coverage and fewer occlusions.
- Near-real-time processing on a GPU; a richer dashboard once event/spatial stats (ball, homography)
  exist to populate it.

For the broader, prioritized menu of stats/pipeline/UX features (ball detection, possession,
shots/goals, movement load, dashboards, quick wins like auto-clipping and human-in-the-loop
identity), see **[feature-ideas.md](feature-ideas.md)**.

## Compute guidance
- **Phase 1 & 2**: validated end-to-end on **CPU** (4 cores) with `yolo11s`. A 30s 720p clip takes
  a few minutes; SigLIP team fitting adds ~2 min. The per-track team trick keeps SigLIP affordable
  without a GPU. A Colab or local NVIDIA GPU (≥4 GB VRAM) makes it near real-time.
- **Fine-tuning (open Phase 2 / Phase 3)**: local NVIDIA GPU (≥8 GB) or rented cloud/Colab GPU.
  Defer the buy/rent decision until we actually start labeling and training.
