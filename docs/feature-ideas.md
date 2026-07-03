# Feature Ideas — what would help this use case

A prioritized catalog of features beyond the current pipeline, aimed at the real goal: **per-player
and per-team stats (time on surface, events, positioning) from single-camera pickup ball hockey
video, with no jersey numbers.**

This is a menu, not a commitment. Each item notes *why it helps*, rough *difficulty*, and *what it
depends on*. See [architecture.md](architecture.md) for the phased roadmap and
[team-clustering.md](team-clustering.md) for the team-ID work.

## Where we are & what's next (priorities)

Phase 1–2 are in: detection + tracking, surface filter, activity gating, auto-clip, position
heatmap, team clustering, and a **labeled eval set + harness**. **Phase 3 is in**: appearance re-ID
(priority #3) and **shift detection** (priority #4). **Priority #5 (per-game report + shift chart)
is in.** **Priority #6 (OSNet re-ID embedder) is in and was decisive: team accuracy 57.1% → 100%
on the reference clip's fresh labels** — the appearance-accuracy ceiling is broken for teams;
identity improves but still over-segments at safe thresholds. The labeling loop closed with
**`--apply-labels` tag propagation** (#7) and the whole thing now runs end-to-end on a raw game
recording via **`--game` full-game mode** (#8): autoclip → cut → per-segment analysis →
cross-segment identity stitch → one merged game report. The open frontier is **identity recall**
(merging more of each player's fragments without false merges) and the event/spatial stats that
gate on harder pieces (ball detection, homography). History of the completed priorities:

1. **Labeled eval set + harness** — ✅ **done** *(was the binding constraint).* `evaluate.py` +
   `labeling.py` + the `eval/` set close it: `--label-crops` exports one crop montage per track and
   a pre-filled `labels.csv` template; a human tags team/role/identity by sight in ~2 minutes;
   `--evaluate` scores `tracks.csv` against the labels with optimal cluster-label alignment (so an
   arbitrary team `0`/`1` aligns to "white"/"dark"). **First measured result on the reference clip:
   team accuracy 52.2% (12/23) — ~chance — vs role 100% (27/27)**, hard-confirming the suspected
   accuracy gap. This unblocks principled iteration on everything below. *(Done; see `eval/README.md`.)*
2. **Background-suppressed crops** for team/identity — ✅ **done.** `background_suppressed_crop`
   (`team_siglip.py`) torso-crops each box and neutralises the saturated rink-coloured (border-hue)
   pixels to flat grey *before* SigLIP, so blue rink + legs + skin stop dominating the embedding —
   the same border-hue suppression the colour path already used, now applied to the embedding.
   **Measured on the reference clip it beats the 52.2% baseline: team 56.5% (13/23)**, and the split
   gets markedly more balanced (sizes 18-vs-9 → 15-vs-12; active-seconds 1.83:1 → 1.12:1). An honest
   gain, not a fix: the low-contrast white/dark kit on this footage is still the hard case (56.5% is
   well short of clean), so appearance alone remains weak here. *(Done; `--no-bg-suppress` ablates
   it. See docs/team-clustering.md.)*
3. **Phase 3 appearance re-ID (per-player identity)** — ✅ **implemented & validated on real footage.**
   The biggest value unlock (true per-player time-on-surface, shifts, +/-). `identity.py` reuses the
   hardened per-track SigLIP embedding machinery at finer granularity, then stitches fragmented
   tracks into identities with **constrained agglomerative clustering** under a hard *temporal
   cannot-link* constraint (two tracks overlapping in time can't be one person — which also blocks
   the look-alike failure mode and floors the identity count near the roster). `--reid` adds a
   `player` column + **`players.csv`** (true per-player time-on-surface + shift counts).
   **Validated on the reference clip:** deterministic; 0 temporal violations; forced to roster size
   (`--roster 13`) every team-checkable merge is same-team (15/15, 0 cross-team, vs ~49% chance) —
   real identity signal — though a clean per-individual accuracy number still waits on identity
   ground truth (hard to label by sight at this crop resolution). *(Done; see
   [docs/identity-reid.md](identity-reid.md).)*
4. **Shift detection (per-player time-on-surface broken into shifts)** — ✅ **implemented & validated
   on real footage.** The headline "time on ice" stat, now unblocked by Phase 3 identity. `shifts.py`
   (`detect_shifts`, emitted with `--reid`) segments each identity's timeline into **true on-surface
   shifts**: it stitches the identity's fragmented track spans, **bridging short temporal gaps** (an
   occlusion / tracker re-acquire → same shift) and **splitting on a bench-length gap** (→ a new
   shift). Because the Phase 2 surface filter already drops off-surface (bench) detections, a bench
   trip is a long dark stretch in the identity's on-surface timeline — so the *temporal gap is the
   bench signal*, no hand-drawn bench polygon needed. This **replaces the old `n_shifts = fragment
   count`**, which over-counted on every brief tracker dropout of a still-on-surface player. Writes
   `shifts.csv` (one row per shift) and adds `n_shifts` (true) / `n_fragments` (raw) / `shift_seconds`
   / `longest_shift_s` / `avg_shift_s` to `players.csv`; `--shift-gap` tunes the bench-vs-occlusion
   threshold (default **15 s**, a physical floor on a real bench change). Validated: 30 s clip → 1.0
   shift/player (correct — too short to bench in), 3-min line-change clip → 3.0/player (avg 32 s).
   Honest caveat: the inter-fragment gap distribution on this fisheye footage isn't cleanly bimodal,
   so the threshold is a tunable judgement call — an explicit entry/exit zone is the deferred
   principled fix. Pure-stdlib core, unit-tested in `tests/test_shifts.py`. *(Done; see
   [docs/architecture.md](architecture.md) Phase 3.)*
5. **Per-game report + shift chart (make the stats consumable)** — ✅ **implemented & validated on
   real footage.** The pipeline *computed* the headline per-player stats (`players.csv`,
   `shifts.csv`), per-team totals (`boxscore.json`) and a `heatmap.jpg`, but a user still had to read
   CSV/JSON to see them. `report.py` turns those existing artifacts into the thing a coach/player
   actually looks at: a self-contained **`report.html`** (game header + per-player stat table — TOI,
   shifts, avg/longest shift — + per-team totals over *true identities* + heatmap + shift chart, every
   image inlined as a `data:` URI) and a **shift chart** (**`shift_chart.png`**) — the classic
   "time-on-ice" Gantt, one row per player, one bar per shift, **rows grouped by team, ordered most
   time-on-surface first**. It adds **no model, no GPU, no labels** — pure rendering over the artifacts
   — so it emits automatically on `--phase2 --reid` *and* runs standalone over a finished run dir
   (`--report <run-dir>`, no video). The **chart layout is a pure-stdlib core** (rows = players, bars =
   shifts — ordering + coordinates, no drawing), unit-tested in `tests/test_report.py` like
   `segments.py`/`shifts.py`, with the matplotlib PNG render + HTML assembly as thin shells.
   **Validated:** the 30 s reference clip renders 13 identities at 1 shift each (correct — too short to
   bench); the 3-min line-change clip renders a multi-shift Gantt with the two teams cleanly grouped
   (green T0 / red T1). *(Done; see "Output & UX" below.)*
6. **Real re-ID/embedding upgrade (lift the appearance ceiling)** — ✅ **implemented & validated on
   real footage; the gain was decisive.** `--embedder osnet` (`reid_embedder.py` + vendored
   `osnet.py`) swaps the repurposed SigLIP embedding for **OSNet-AIN**, a purpose-built person re-ID
   network (domain-generalized torchreid checkpoint, safe-unpickled, fetched once). Same `embed()`
   interface → team clustering and identity reuse it unchanged. **Measured on the reference clip
   with fresh labels (identical tracks): team accuracy 57.1% (SigLIP) → 100.0% (OSNet), 21/21** —
   the low-contrast white/dark ceiling was the borrowed embedding, not the footage. Identity
   improves too (2× the same-person merges, all team-consistent, 0 temporal violations; measured
   same/different distance distributions in docs/identity-reid.md) but still over-segments — the
   distributions overlap, so the data-driven threshold stays conservative
   (`REID_DISTANCE_DEFAULTS`: siglip 0.35, osnet 0.45) and `--roster` remains the accuracy path.
   Bonus: ~10× cheaper per crop than SigLIP on CPU. *(Done; see eval/README.md +
   docs/team-clustering.md.)*
7. **Human-in-the-loop tag propagation (`--apply-labels`)** — ✅ **implemented.** The "what's left"
   half of the labeling loop: apply a filled-in `labels.csv` to a finished run and the human tags
   flow back into the product — a player tag names every track of that identity, a team tag names
   the side (frame-weighted majority; human tags win on their own tracks); `tracks.csv` gains
   `team_name`/`player_name`, `players.csv` gains `name`, and the report/shift chart re-render with
   real names. Over-merge / over-segmentation conflicts are printed, not hidden. Pure
   `propagate_labels` core, unit-tested (`roster.py`, `tests/test_roster.py`).
8. **Full-game mode (`--game`)** — ✅ **implemented.** The missing end-to-end orchestration for a
   raw game recording: autoclip pre-pass → frame-accurate cuts → `--phase2 --reid` per segment →
   **cross-segment identity stitch** (per-segment identity centroids from `identities.npz`,
   clustered with the same constrained core under a same-segment cannot-link) → **shift stitching
   across stoppages in live time** (`LiveTimeline` compresses dead time out before gap-based shift
   detection, so a stoppage never splits a shift) → merged `players.csv`/`shifts.csv`/
   `boxscore.json` + one game report, standard schema so `--report`/`--apply-labels` work on the
   game dir unchanged (`game.py`, pure cores unit-tested in `tests/test_game.py`).

### Next priorities (proposed, in order)

1. **Identity recall on real rosters** — the remaining quality gap. OSNet fixed teams outright but
   identity still over-segments at safe thresholds. Three compounding levers, in increasing cost:
   (a) **more crops per track** now that embedding is ~10× cheaper (`--crops-per-track` knob is in;
   effect unmeasured pending identity ground truth); (b) ✅ **spatiotemporal handoff linking** —
   **done & validated** (`detect_handoffs`, on by default): exit/entry continuity pre-merges
   tracker-dropout fragments before appearance clustering — on the reference clip, known
   same-person pairs merged went **1/6 → 4/6 with 0 cross-team merges** (1 s gap + ambiguity
   rejection; see docs/identity-reid.md); (c) a **frame-level review tool** to produce
   per-individual ground truth so identity accuracy is a measured number, not a proxy — now the
   main open piece, since bench-length gaps are beyond what handoffs may touch.
2. **A real pinnie/vivid-kit clip** to confirm the kit-colour prior end-to-end on untouched footage
   (validated synthetically only), and a second camera setup to broaden the eval set.
3. **Ball detection (fine-tuned small-object detector)** — the long-pole unlock for every event
   stat (possession, shots, goals, +/- context). Needs labeled frames + a GPU; start with a small
   labeling pass on this recording.
4. **Rink homography / top-down mapping** — unlocks speed/distance/zones; blocked on fisheye
   calibration, which one checkerboard-style capture session (or court-line keypoint annotation)
   would resolve.
5. **Performance for full games** — ONNX/OpenVINO export of YOLO + OSNet, adaptive stride during
   idle stretches, and parallel per-segment analysis in `--game` (segments are independent).

## Dependency map (what unlocks what)

```
detection+tracking (done) ─┬─ activity gating (done) ── auto-clip (done) ── shift segmentation (done)
                           ├─ surface filter (done) ─── zone stats (needs homography)
                           ├─ eval harness (done) ──────── measures team/role/identity accuracy
                           ├─ team clustering (stable; kit prior; bg-suppressed crops; acc 56.5%) ─ team stats ─┐
                           ├─ appearance re-ID (Phase 3, done) ─ per-player TOI + shifts (done) ────────────────┼─▶ per-game report + shift chart (done)
                           ├─ ball detection (new) ───── possession, shots, passes
                           └─ rink homography (new) ──── speed/distance, heatmaps, zones
```
Most **per-player** stats gate on Phase 3 identity (done) and are surfaced by the per-game report +
shift chart (done). The **re-ID/embedding upgrade (priority #6, done — OSNet)** lifted the
appearance ceiling: team clustering is now measured at 100% on the reference clip; identity carries
real signal but still under-merges. **`--game` (done)** chains everything over a full raw recording
with cross-segment identity stitching. Most **event/spatial** stats still gate on **ball
detection** and/or **homography** — the two remaining "new model / new calibration" long poles.

## Quick wins (cheap, high leverage)

- **Auto-clip / dead-time skip** *(done — `segments.py` + `autoclip.py`).* Collapses the
  `activity.py` signal into contiguous live-play segments (`segments.csv`), bridging brief gaps and
  dropping blips. `--phase2 --clips` exports each segment as a raw clip from the full pass;
  `--autoclip` is a cheap **detection-only pre-pass** that finds live play *before* the heavy pass
  and writes a `segments.json` manifest with a **compute-savings estimate** (`--cut` slices clips via
  ffmpeg). Big compute savings on a mostly-idle full game and the scaffolding for shift detection.
  *Was: Low difficulty; depended on done pieces.*
- **Human-in-the-loop identity.** *(Labeling half done — `--label-crops` + `labeling.py`.)* Given
  no jersey numbers, a tiny labeling step beats perfect automation: `--label-crops` already shows
  one crop montage per track and takes a `team`/`role`/`player` tag in ~2 minutes, and those tags
  double as the labeled set the eval harness scores against. *What's left:* **propagate** the tags
  back into the pipeline output (tag a track cluster → apply to all its tracks) rather than only
  scoring them. *Low–medium; pairs with Phase 3.*
- **Box-score / stats export** *(done — `boxscore.py`).* Rolls the scattered per-track numbers into
  one consumable artifact: a `boxscore.json` (game header + per-team totals + a per-player table,
  most-active first) plus a compact text table in the `--phase2` console summary. Honest scope:
  **per-track**, not yet per-*player* — with no jersey numbers a player who re-enters is still two
  tracks (same caveat as `tracks.csv`), so true shift counts wait on Phase 3 identity; team totals
  sum over tracks and are robust to that. Pure-stdlib core, unit-tested in `tests/test_boxscore.py`.
  *Was: Low difficulty; already partway there in `tracks.csv`.*
- **Capture-side levers (no code).** Distinct **colored pinnies** are now the single biggest team-ID
  lever: the kit-colour prior (`team_siglip.detect_kit_split`) splits cleanly on a vivid kit and
  skips SigLIP entirely, where low-contrast white/dark kits still defeat the embedding. A higher,
  wider, fixed mount and a marked game-start help everything downstream. Document as recommended
  recording practice. *Trivial, large payoff — and now code-backed.*

## Stats & analytics (the end product)

- **Shift / time-on-surface per player** — the headline "time on ice." *(Done — `shifts.py`,
  `--reid`.)* Identity (Phase 3) plus gap-based shift segmentation: the surface filter makes a bench
  trip a long temporal gap in a player's on-surface timeline, so shifts split on bench-length gaps
  and stitch across brief tracker dropouts (`shifts.csv` + `n_shifts`/`shift_seconds` in
  `players.csv`). *Next:* an explicit entry/exit zone to sharpen the exact on/off instant.*
- **Plus/minus & on-surface context** — who was on when goals happened. *Depends on identity + goal
  detection.*
- **Ball detection & possession** — track the ball; attribute possession to nearest player/team,
  compute possession %. *Hard: the ball is small and fast; needs a fine-tuned detector + temporal
  smoothing/interpolation.*
- **Shots & goals** — shot attempts (ball toward net + windup), shots on net, goals (ball crosses
  goal line / enters net region). *Depends on ball detection + goal-mouth localization.*
- **Assists & passing networks** — ball possession passing from player to player. *Depends on ball +
  identity.*
- **Movement load** — distance covered, top speed, sprint count per player. *Depends on homography
  (pixel→metric) + identity.*
- **Per-player & per-team heatmaps / zone time** — offensive/defensive/neutral-zone occupancy.
  *Depends on homography; per-player needs identity.*
- **Goalie stats** — shots faced, saves, goals against. *Depends on ball + goal events + goalie ID
  (spatial, near net).*

## Pipeline & calibration

- **Ball detector (fine-tuned).** Label ball instances; train a small-object-tuned YOLO; add
  temporal interpolation for frames where it's occluded/blurred. *The unlock for all event stats.*
- **Rink homography / top-down map.** Fisheye undistort + court-keypoint correspondences → metric
  top-down coordinates. Enables speed, distance, zones, a clean minimap. Deferred because the fixed
  fisheye + occluded near boards make a naive planar homography unreliable. *Medium–hard.*
- **Referee / non-player handling.** Detect and exclude refs (often striped / distinct), beyond the
  current surface filter. *Low–medium.*
- **Performance.** GPU batch inference, ONNX/TensorRT export, adaptive frame sampling for a full
  38-min game. *Medium; mostly matters at full-game scale.*

## Output & UX

- **Per-game report / dashboard.** ✅ **Done (priority #5) — `report.py`.** A self-contained
  `report.html` (game header + per-player stat table — TOI + shift counts from
  `players.csv`/`shifts.csv` — + per-team totals over *true identities* + the `heatmap.jpg`) and a
  **shift chart** `shift_chart.png` (Gantt of who's on the surface when, rows grouped by team and
  ordered most-TOI-first), with every image inlined as a `data:` URI. Built exactly as planned: a
  **pure-logic layout core** (rows = players, bars = shifts) unit-tested like `shifts.py`
  (`tests/test_report.py`), with a thin matplotlib PNG + HTML rendering shell. Emits on
  `--phase2 --reid` **and** runs standalone over a finished run dir (`--report <run-dir>`, no video).
  No new model / GPU / labels — pure rendering — and validated on the reference footage (30 s clip:
  13 identities × 1 shift; 3-min line-change clip: a multi-shift Gantt, teams cleanly grouped).
- **Event-indexed video / highlights.** Auto-extract goals and big plays; a scrubbable timeline of
  events over the annotated video. *Medium; depends on event detection.*
- **Stat overlays burned into video** — scoreboard, possession bar, player labels. *Low–medium.*

## Evaluation & data (was the binding constraint — priority #1, now ✅ done for track-level fields)

- **Labeled dataset + eval harness — done (track-level).** `labeling.py` exports a per-track crop
  montage + a pre-filled `labels.csv` template from the same detect/track pass that writes
  `tracks.csv` (so ids line up); a human tags team/role/identity by sight; `evaluate.py` scores the
  predictions with optimal cluster-label alignment (team/identity) or direct equality (role), over
  the labeled∩predicted overlap. The committed `eval/sample_labels.csv` (23 of 27 player tracks
  labeled) gives the **first measured numbers on natural footage: team 52.2% (~chance), role 100%** —
  finally telling a good split from a plausible-looking bad one. Run: `python -m dbh_vibes
  --evaluate eval/sample_labels.csv --tracks runs/sample/tracks.csv`.
  - *Still open:* **box-level** ground truth (detection mAP, MOT/IDF1) needs per-frame boxes, not
    just per-track tags — a heavier labeling step deferred until a detector fine-tune needs it. The
    harness's `player` column is the **identity** slot, ready to score Phase 3 re-ID the moment it
    predicts identities. More clips / camera setups would broaden coverage beyond this one clip.
