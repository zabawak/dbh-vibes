# Team Clustering — Robustness Needs

Status: **known-fragile, deferred to a future session.** This note captures what we observed and
the concrete ideas to harden it, so we can pick it up cold.

## The problem we saw

Team assignment uses SigLIP crop embeddings → UMAP → KMeans(k=2), classified per track by
majority vote (`src/dbh_vibes/team_siglip.py`, called from `pipeline.py`). On the real gameplay
clip it is **unstable run to run**:

| Run | Players | Team split (count) | Team split (active-seconds) |
|---|---|---|---|
| Pre-goalie-polish | 39 | 18 vs 15 | 96 vs 87 (balanced) |
| Post-goalie-polish | 45 | 28 vs 6 | 66 vs 140 (lopsided) |

Adding just ~6 edge tracks (the two goalies + boards-huggers) was enough to tip KMeans from a
balanced split into a degenerate one — almost every skater in one cluster, a handful in the
other. The grouping itself is sometimes clean (an earlier crop montage perfectly isolated the
red-pinnie team), but it is not *reliable*.

## Root causes

1. **More than two visual groups.** The scene contains red pinnies, dark jerseys, white shirts,
   distinct goalie gear, and residual spectators/refs. Forcing exactly 2 clusters onto 3+ groups
   makes the boundary arbitrary and input-dependent.
2. **Goalies skew the split.** Their gear (pads, different colors) is visually distinct from
   skaters and pulls cluster centers.
3. **Sensitivity to the crop set.** Even with fixed `random_state`, changing *which* crops are
   fed (different players, different sampled frames) changes the UMAP manifold and the KMeans
   result. The pipeline currently clusters per crop, not per track.
4. **Crop quality.** Motion blur, occlusion, loose boxes, and foot/edge crops add noise.
5. **Arbitrary, flippable labels.** Team `0`/`1` are assigned by KMeans ordering, so the same
   team can be `T0` one run and `T1` the next — no stable identity.
6. **No ground truth.** We have no labeled set, so we can't measure accuracy or tune thresholds —
   we're judging by eye.

## Ideas to harden (for the next session)

- **Cluster per track, not per crop.** Aggregate each track's crops into one mean embedding, then
  cluster tracks. One vote per player → far less sensitive to crop counts and blur.
- **Exclude goalies from team clustering.** Identify goalies separately (they sit near a net most
  of the clip — a spatial/positional cue) and assign each to the team defending the net they
  occupy, rather than by appearance.
- **Restrict to well-observed skaters.** Drop short, small, low-`on_surface_frac`, or occluded
  tracks before fitting; cluster only confident player crops.
- **Over-segment then merge.** Cluster into N>2 appearance groups, then map groups to the two
  teams (e.g., by size and inter-cluster separation), which tolerates extra visual groups and
  outliers better than a hard k=2.
- **Use a kit prior / config hint.** Many games are "pinnies vs none" or two known colors. Allow a
  config hint (or seed cluster centers from the two dominant kit hues) to anchor the split.
- **Stable, deterministic team labels.** Anchor `T0`/`T1` to something physical — kit hue, or
  which team defends which net — so labels don't flip between runs.
- **Temporal lock.** A track's team is constant; enforce it (already majority-voted, but make the
  *fit* per-track too).

## What we need to actually validate it

- **A small labeled set**: hand-label team for ~20–40 tracks across 2–3 clips (and a couple of
  different camera setups). Without this we can't measure or tune.
- **Metrics**: clustering accuracy vs labels; team-balance sanity (counts within reason);
  run-to-run stability (same clip, repeated runs, should agree).
- **Robustness checks**: different lighting, a clip where both teams wear similar colors (expected
  failure case — document it), and the goalie-heavy frames that tipped this run.
