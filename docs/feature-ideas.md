# Feature Ideas — what would help this use case

A prioritized catalog of features beyond the current pipeline, aimed at the real goal: **per-player
and per-team stats (time on surface, events, positioning) from single-camera pickup ball hockey
video, with no jersey numbers.**

This is a menu, not a commitment. Each item notes *why it helps*, rough *difficulty*, and *what it
depends on*. See [architecture.md](architecture.md) for the phased roadmap and
[team-clustering.md](team-clustering.md) for the team-ID work.

## Where we are & what's next (priorities)

Phase 1–2 are in: detection + tracking, surface filter, activity gating, auto-clip, position
heatmap, team clustering, and a **labeled eval set + harness**. Team clustering's run-to-run
*instability is fixed* (deterministic; 100% stable on real footage), a **kit-colour prior** handles
the common pinnie case, and **background-suppressed crops** nudged accuracy 52.2% → 56.5%; what's
left there is *accuracy on low-contrast kits*. **Phase 3 appearance re-ID is now in too** (priority
#3 below — per-player identity via constrained agglomerative clustering with a temporal constraint).
The near-term priorities are:

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
   threshold. Pure-stdlib core, unit-tested in `tests/test_shifts.py`. *(Done; see
   [docs/architecture.md](architecture.md) Phase 3.)*

## Dependency map (what unlocks what)

```
detection+tracking (done) ─┬─ activity gating (done) ── auto-clip (done) ── shift segmentation (done)
                           ├─ surface filter (done) ─── zone stats (needs homography)
                           ├─ eval harness (done) ──────── measures team/role/identity accuracy
                           ├─ team clustering (stable; kit prior; bg-suppressed crops; acc 56.5%) ─ team stats
                           ├─ appearance re-ID (Phase 3, done) ─ per-player TOI + shifts (done: players.csv + shifts.csv)
                           ├─ ball detection (new) ───── possession, shots, passes
                           └─ rink homography (new) ──── speed/distance, heatmaps, zones
```
Most **per-player** stats gate on Phase 3 identity; most **event/spatial** stats gate on **ball
detection** and/or **homography**.

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

- **Per-game report / dashboard.** Stat tables + heatmaps + a shift chart (Gantt of who's on when).
  *Medium; depends on the stats it summarizes.*
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
