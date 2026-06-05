# Team Clustering — Robustness Needs

Status: **hardened (algorithm), pending real-footage re-validation.** The clusterer has been
reworked to remove the instability described below; the new logic is unit-tested on synthetic
embeddings (`tests/test_team_cluster.py`). It has **not yet been re-run on the real gameplay clip**
(that needs the video + SigLIP weights, which aren't in this environment) — see *What changed* and
*Still to validate* at the bottom. The original analysis is kept for context.

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

## What changed (the hardening, all label-free)

Implemented in `src/dbh_vibes/team_siglip.py`; each item maps to a root cause above:

1. **Cluster per track, not per crop** (root cause #3). `aggregate_track_embeddings` pools each
   track's L2-normalised crop embeddings into one mean vector, so we cluster ~one point per player.
   Unit-tested to be invariant to how many crops a track contributed (1 vs 50 → same embedding).
2. **Deterministic reduction** (root cause #3). UMAP (stochastic, version-sensitive, noisy on a few
   dozen points) is replaced by exact PCA (`svd_solver='full'`) with fixed seeds and a fixed track
   order, so identical crops give identical teams every run. This also **drops the `umap-learn`
   dependency** (`pyproject.toml` `phase2` extras).
3. **Over-segment then merge by size** (root causes #1, #2 — the degenerate split). We cluster into
   K∈[2,4] micro-clusters, pick K by silhouette (biased toward smaller K so two genuine teams stay
   K=2), then take the **two largest** micro-clusters as team anchors and fold every smaller outlier
   (goalies, refs) into the nearest anchor by appearance. A small, visually distinct goalie cluster
   can no longer *become* a team — the regression test reproduces the 28-vs-6 scenario (20 skaters +
   2 distinct goalies) and confirms the skaters now split 10/10 with the goalies folded in.
4. **Colour-anchored stable labels** (root cause #5). `order_labels_by_color` assigns T0/T1 by kit
   colour — the more saturated (e.g. pinnie) team is T0 — so labels don't flip between runs even
   when the sampled crop set shifts.
5. **Label-free quality signal** (root cause #6, partial). Clustering returns a silhouette score,
   team-balance counts, micro-cluster count, and a per-track confidence margin (surfaced as the
   `team_conf` column in `tracks.csv` and printed by the CLI), so separation and run-to-run
   stability can be *measured* without ground truth.

Goalies are still merged by *appearance*, not the spatial cue the analysis preferred — they no
longer tip the split, but a goalie whose gear resembles team A's will fold into team A. Spatial
goalie handling (near-net position) remains a follow-up once positions are plumbed through.

## Still to validate (needs real footage / a labeled set)

- **Re-run on the gameplay clip** end-to-end with SigLIP and confirm the split is balanced and
  stable across repeated runs (the failure table above should no longer reproduce).
- **A small labeled set**: hand-label team for ~20–40 tracks across 2–3 clips (and a couple of
  different camera setups) to measure *accuracy*, not just internal separation.
- **Robustness checks**: different lighting, a clip where both teams wear similar colors (expected
  failure case — document it), and the goalie-heavy frames that tipped the original run.
