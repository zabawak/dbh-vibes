# Feature Ideas — what would help this use case

A prioritized catalog of features beyond the current pipeline, aimed at the real goal: **per-player
and per-team stats (time on surface, events, positioning) from single-camera pickup ball hockey
video, with no jersey numbers.**

This is a menu, not a commitment. Each item notes *why it helps*, rough *difficulty*, and *what it
depends on*. See [architecture.md](architecture.md) for the phased roadmap and
[team-clustering.md](team-clustering.md) for the team-ID work.

## Dependency map (what unlocks what)

```
detection+tracking (done) ─┬─ activity gating (done) ── auto-clip (done) ── shift segmentation
                           ├─ surface filter (done) ─── zone stats (needs homography)
                           ├─ team clustering (stable; accuracy WIP) ─ team-level stats
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
- **Human-in-the-loop identity.** Given no jersey numbers, a tiny labeling step beats perfect
  automation: show one crop per track cluster, let the user tag "that's player A / team X," then
  propagate. Turns a hard CV problem into a 2-minute review. *Low–medium; pairs with Phase 3.*
- **Box-score / stats export.** Emit per-game JSON/CSV (per player: shifts, seconds, team; per
  team: totals) — already partway there in `tracks.csv`. Makes outputs consumable. *Low.*
- **Capture-side levers (no code).** Distinct **colored pinnies** make team clustering far more
  reliable; a higher, wider, fixed mount and a marked game-start make everything downstream
  easier. Document these as recommended recording practice. *Trivial, large payoff.*

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

## Evaluation & data (the foundation we're missing)

- **Labeled dataset + eval harness.** Hand-label a few clips (boxes, team, identity, ball, key
  events) to (a) fine-tune detectors and (b) *measure* accuracy — detection mAP, MOT/IDF1 for
  tracking+re-ID, team accuracy, identity accuracy. Without this we tune by eye. *Medium; unblocks
  principled iteration on everything above.* (See also the validation needs in team-clustering.md.)
