# Team Clustering — Robustness Needs

Status: **stability hardened and validated on real footage; team _accuracy_ now *measured* and
confirmed weak on low-contrast kits (52.2% — ~chance — via the new eval harness, see the bottom of
this doc and `eval/README.md`).** The run-to-run instability that was the stated top priority is fixed — the
reworked clusterer is deterministic and gave **100% identical team assignments across repeated
runs** on the real gameplay clip. But validating on that clip also surfaced a deeper problem the
original note only speculated about: on this fisheye footage the split is driven by **crop scale
(near vs far players), not kit**, and the two kits here (white vs dark shirts on a blue rink) are
too weak a signal for SigLIP to separate. We added a principled fix for the scale confound and a
**kit-colour prior** that splits on colour (and skips SigLIP) when a vivid kit is present — the
common "pinnies vs none" case — falling back to embeddings otherwise. On this white/dark recording
the prior correctly declines and the residual accuracy gap is real; both are documented in
*Real-footage validation* below. The original analysis and the stability rework are kept for context.

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
6. **Scale-decorrelation** (new root cause found in validation — see below). When per-track crop
   sizes are supplied, `_reduce` drops the principal components whose score correlates with
   log-crop-area above a threshold, so the near/far scale axis can't drive the split. Safe for the
   easy case: a vivid pinnie kit dominates its own PC, which doesn't correlate with size, so nothing
   kit-relevant is dropped. Unit-tested (`test_scale_does_not_hijack_split_when_sizes_given`).
7. **Kit-colour prior with auto-selection** (`detect_kit_split`, `torso_kit_chroma`). Since the
   embedding fails on low-contrast kits but the project's common case is "pinnies vs none", we try a
   colour split first: a background-suppressed torso **chroma** per track (rink pixels removed by
   dropping the crop's border hue), then accept a *vivid-vs-plain* split only when the saturated
   cluster is genuinely vivid, hue-coherent, near a neutral other cluster, and balanced. When it
   fires it splits on colour (scale-immune, and it **skips SigLIP entirely** — a compute win);
   otherwise it falls back to the embedding path. The selected path is reported as
   `team_quality.method` and printed by the CLI.

Goalies are still merged by *appearance*, not the spatial cue the analysis preferred — they no
longer tip the split, but a goalie whose gear resembles team A's will fold into team A. Spatial
goalie handling (near-net position) remains a follow-up once positions are plumbed through.

## Real-footage validation (what we actually measured)

Validated on the reference clip cut per `data/README.md` (active gameplay, `-ss 1490`, 30s @
720p/30fps — 5-on-5 + goalies), `yolo11s` + SigLIP, run twice.

**Stability — fixed.** Both runs were identical: 41 players / 55 spectators, team sizes (18, 11),
and **29/29 teamed tracks got the same team in both runs (100% agreement)**. The reworked clusterer
is deterministic; the run-to-run failure table above no longer reproduces. This was the stated top
priority and it is done.

**Accuracy — the harder, still-open problem.** Eyeballing per-team crop montages and the stats
showed the split was **not along kit lines**:

- The two clusters separated almost perfectly by **crop size**: team0 median box ≈ 5600 px, team1
  ≈ 1500 px, with team1 made up *entirely* of small/short (far-from-camera) tracks. The active-play
  seconds came out wildly lopsided (≈188 s vs 21 s) as a result.
- Diagnosing the embedding directly: the **top principal component correlated 0.86 with crop area**,
  and the overall cluster–area correlation was **0.73**. The dominant axis of SigLIP-embedding
  variation on this fisheye footage is near-vs-far crop detail, *not* kit.
- Adding the scale-decorrelation (#6 above) drops the cluster–area correlation to **0.08** — the
  scale artifact is gone — and the end-to-end split becomes much more **balanced**: team sizes
  (13, 16) and active-play seconds **80 s vs 129 s** (vs the pre-fix size-driven (18, 11) and
  **188 s vs 21 s**, a 9:1 → 1.6:1 improvement), still 100% stable across runs.
- But this is balance, not correctness: silhouette is only ~0.13 and the montages **still mix white
  and dark shirts in both clusters**. A direct torso-brightness test didn't separate the kits either
  (crops are contaminated by the bright blue rink, legs, skin). On *this* clip SigLIP simply does
  not encode the white-vs-dark kit contrast strongly enough to cluster on.

So removing the scale confound was necessary but not sufficient: the earlier "red-pinnie cleanly
isolated" success was a **high-contrast** kit; this clip's low-contrast white/dark kits are the hard
case and remain unsolved by appearance embeddings alone.

**Kit-colour prior (#7) on this footage.** The vivid-kit path was added for exactly the case the
embedding can't handle. This recording, however, is white-vs-dark across all of its games — no
natural vivid two-team scene exists (scanned the full 38 min) — so the prior correctly **declines
and falls back to embeddings** here (`method = "siglip"`), which is the right behaviour and means no
regression on the hard clip. It was validated *positively* two ways: on synthetic chroma, and on the
real crops with half of them tinted to a vivid kit, where `detect_kit_split` / `assign_teams`
isolate the tinted team with **100% accuracy, silhouette 0.90, confidence 0.98, and without invoking
SigLIP**. A natural pinnie clip is still wanted to confirm end-to-end on untouched footage.

### Concrete next steps for accuracy (still needs a labeled set to measure)

- **Kit-colour prior — done (#7), needs a real pinnie clip to confirm end-to-end.** Currently
  "vivid vs plain"; extend to two *distinct* vivid kits (e.g. red vs blue) — today that falls back
  to embeddings.
- **Background-suppressed crops.** The colour path already removes the rink via border-hue masking;
  do the same before *embedding* (person-segment or tight torso box) so blue rink + legs + skin
  stop dominating SigLIP — the most promising lever for the white-vs-dark case.
- **A small labeled set — done.** `eval/sample_labels.csv` hand-labels 23 of 27 player tracks on
  the reference clip; the harness (`evaluate.py`, `python -m dbh_vibes --evaluate`) now reports the
  **first true team accuracy on natural footage: 52.2% (12/23) — ~chance**, vs role 100% (27/27).
  This is the hard confirmation of the gap that internal signals (silhouette / scale-decorrelation /
  tinted-kit accuracy) could only hint at: the white/dark split here is no better than a coin flip.
  52.2% is the baseline the background-suppressed-crop lever must beat. More clips / camera setups
  would broaden coverage. See `eval/README.md` for the recipe.
- **Robustness checks**: different lighting, a both-teams-similar-colours clip (expected failure —
  document it), and the goalie-heavy frames that tipped the original run.

The validation scripts used (cut clip, run twice, montage, tinted-crop colour test, embedding
diagnostics) are ad-hoc and were kept out of the repo; re-create them from the recipe in
`data/README.md` if needed.
