# Feature Ideas — what would help this use case

A prioritized catalog of features beyond the current pipeline, aimed at the real goal: **per-player
and per-team stats (time on surface, events, positioning) from single-camera pickup ball hockey
video, with no jersey numbers.**

This is a menu, not a commitment. Each item notes *why it helps*, rough *difficulty*, and *what it
depends on*. See [architecture.md](architecture.md) for the phased roadmap and
[team-clustering.md](team-clustering.md) for the team-ID work.

## Where we are & what's next (priorities)

Phase 1–2 are in: detection + tracking, surface filter, activity gating, auto-clip, position
heatmap, team clustering, and now a **labeled eval set + harness**. Team clustering was the prior
top priority — its run-to-run *instability is fixed* (deterministic; 100% stable on real footage)
and a **kit-colour prior** handles the common pinnie case. What's left there is *accuracy on
low-contrast kits* — and that gap is now **measured, not guessed** (see below). The near-term
priorities are:

1. **Labeled eval set + harness** — ✅ **done** *(was the binding constraint).* `evaluate.py` +
   `labeling.py` + the `eval/` set close it: `--label-crops` exports one crop montage per track and
   a pre-filled `labels.csv` template; a human tags team/role/identity by sight in ~2 minutes;
   `--evaluate` scores `tracks.csv` against the labels with optimal cluster-label alignment (so an
   arbitrary team `0`/`1` aligns to "white"/"dark"). **First measured result on the reference clip:
   team accuracy 52.2% (12/23) — ~chance — vs role 100% (27/27)**, hard-confirming the suspected
   accuracy gap. This unblocks principled iteration on everything below. *(Done; see `eval/README.md`.)*
2. **Background-suppressed crops** for team/identity — *now the top open lever.* Person-segment (or
   tight-torso + rink masking) *before* embedding, so blue rink + legs + skin stop dominating SigLIP.
   The colour path already masks the rink; do the same for the embedding. The harness above gives it
   a number to beat (52.2%). *Low–medium.*
3. **Phase 3 appearance re-ID (per-player identity)** — the biggest value unlock (true per-player
   time-on-surface, shifts, +/-). Reuses the now-hardened per-track embedding machinery at finer
   (per-individual) granularity; the harness already has a `player` (identity) slot ready to score it.
   *Hard; see architecture.md Phase 3.*

## Dependency map (what unlocks what)

```
detection+tracking (done) ─┬─ activity gating (done) ── auto-clip (done) ── shift segmentation
                           ├─ surface filter (done) ─── zone stats (needs homography)
                           ├─ eval harness (done) ──────── measures team/role/identity accuracy
                           ├─ team clustering (stable; kit-colour prior; accuracy 52% measured) ─ team stats
                           ├─ appearance re-ID (Phase 3) ─ per-player stats, +/-, shifts
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
- **Box-score / stats export.** Emit per-game JSON/CSV (per player: shifts, seconds, team; per
  team: totals) — already partway there in `tracks.csv` (now also `team_conf`). Makes outputs
  consumable. *Low.*
- **Capture-side levers (no code).** Distinct **colored pinnies** are now the single biggest team-ID
  lever: the kit-colour prior (`team_siglip.detect_kit_split`) splits cleanly on a vivid kit and
  skips SigLIP entirely, where low-contrast white/dark kits still defeat the embedding. A higher,
  wider, fixed mount and a marked game-start help everything downstream. Document as recommended
  recording practice. *Trivial, large payoff — and now code-backed.*

## Stats & analytics (the end product)

- **Shift / time-on-surface per player** — the headline "time on ice." *Depends on Phase 3 identity
  + bench-zone detection.*
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
