# Phase 3 — Appearance Re-ID (per-player identity)

Status: **implemented and validated on real footage.** The clustering machinery runs end-to-end,
is deterministic, and is *sound* (the temporal constraint guarantees no impossible identities). The
real-footage finding is encouraging: at a safe threshold the clusterer is conservative
(under-merges), but **when stitched down to roster size every team-checkable merge respects the
team ground truth (15/15 same-team, 0 cross-team) and zero merges violate the temporal
constraint** — well above the ~49% same-team rate expected by chance, so the appearance embeddings
*do* carry real identity signal even on this low-contrast footage. Numbers below.

## Why this is the headline lever

Detection + ByteTrack give a **track id** that is stable only within one continuous on-surface
stretch. A player who leaves the frame, is occluded, or is simply lost by the tracker returns as a
**new** track id. On the 30 s reference clip ~13 people produced **27 player tracks** (and a full
game fragments far worse). Every per-track stat — time on surface, shifts, +/- — is therefore
split across several ids for one person. Phase 3 stitches the fragments back into **identities** so
we get *true per-player* time-on-surface and a shift count (one contiguous fragment ≈ one shift).

## How it works (`src/dbh_vibes/identity.py`)

Same embedding machinery as team clustering, at finer granularity:

1. **One mean SigLIP embedding per track**, on background-suppressed crops — the *exact* pass team
   clustering uses (`team_siglip.embed_tracks`), shared so SigLIP is paid for **once** when both run.
2. **Deterministic PCA** denoise (full SVD, fixed seed) — keeps more components than the 2-team
   split because every player is its own direction.
3. **Constrained agglomerative clustering** (average-linkage cosine) with a hard
   **temporal cannot-link** constraint: two tracks whose frame spans *overlap in time* cannot be the
   same person (`temporal_overlap_matrix`). The clusterer greedily merges the closest *permissible*
   pair until it hits the roster size (`--roster`) or the nearest permissible merge exceeds a cosine
   `--reid-distance` threshold (data-driven count otherwise).

The temporal constraint is the key idea that makes identity tractable where naive appearance
clustering fails:

- It **blocks the documented failure mode** — two players in similar gear collapsing into one
  identity — whenever they are on the surface together.
- The **maximum number of players on the surface at once becomes a hard floor** on the identity
  count. In 5-on-5 + goalies that is ~12, very near the true roster, so the clusterer lands near the
  right number of people *without being told the roster size*.

The clustering core is pure numpy + stdlib (no torch / model / video), unit-tested on synthetic
embeddings + frame spans in `tests/test_identity.py` (12 tests) — same discipline as the team core.

## Outputs

With `--phase2 --reid`:

- `tracks.csv` gains a `player` (identity id) and `player_conf` column.
- **`players.csv`** — the per-player roll-up: one row per identity with summed `seconds_on_surface`
  / `active_seconds`, `n_shifts` (**true on-surface shift count** from `shifts.py`, not the raw
  fragment count — see below), `n_fragments` (the raw count, kept for transparency), `shift_seconds`
  / `longest_shift_s` / `avg_shift_s`, the constituent `track_ids`, a majority-vote `team`, and span.
  **This is the true per-player time-on-surface the project set out to produce.**
- **`shifts.csv`** — one row per on-surface shift (player, team, shift index, frame/second bounds,
  duration, fragments stitched). `shifts.detect_shifts` stitches each identity's fragmented track
  spans, bridging short temporal gaps (occlusion / tracker re-acquire → same shift) and splitting on
  a bench-length gap (→ new shift); the surface filter already drops bench detections, so a bench
  trip *is* that long temporal gap. Replaces the over-counting fragment-count shift estimate.
  Validated on the reference clip (28 fragments → 13 identities → **23 true shifts**) and on a
  3-minute clip with real line changes (141 fragments → 20 identities → **104 true shifts**, mean
  5.2 shifts/player) — deterministic, shifts non-overlapping within each player, `n_shifts ≤
  n_fragments` always.
- The console prints `tracks -> identities`, silhouette, and how many concurrent-overlap merges the
  constraint blocked.

```bash
python -m dbh_vibes data/sample.mp4 --out runs/reid --phase2 --model yolo11s.pt --reid
python -m dbh_vibes data/sample.mp4 --out runs/reid --phase2 --reid --roster 13     # pin roster
```

## Real-footage validation (reference clip, `data/sample.mp4`)

Run on `data/sample.mp4` (the same 30 s active-gameplay clip the rest of the project validates on),
`yolo11s` + background-suppressed SigLIP embeddings:

| check | result |
|---|---|
| runs end-to-end on real footage | ✅ ~2.5 min on 4 CPU cores (shares the team SigLIP pass) |
| deterministic (two full runs, same partition) | ✅ **identical** `player` assignments |
| **temporal soundness** (no identity contains a time-overlapping pair) | ✅ **0 violations** (both configs) |
| identities found (default `--reid-distance 0.35`) | 27 tracks → **25 identities**, 2 merges (both same-team) |
| identities found (`--roster 13`, forced to roster size) | 27 tracks → 13 identities, 22 merged pairs |
| **team purity at roster=13** (a merge must not span two GT teams) | ✅ **15/15 team-checkable pairs same-team, 0 cross-team** |
| chance baseline for same-team (from the GT white/dark mix) | ~49% — so 15/15 is highly significant (p≈3e-5) |

The honest read, consistent with [team-clustering.md](team-clustering.md): the embedding silhouette
is ~0 on this footage, so at a *safe* threshold the clusterer **under-merges** — it makes only the
two merges it is most confident in (both temporally sound *and* same-team) and leaves the rest
fragmented (high precision, low recall). But forcing it down to roster size with `--roster 13` does
**not** descend into noise: all 22 merged pairs are temporally valid and every one of the 15 that
can be checked against the committed team labels is **same-team** (0 cross-team), versus the ~49%
same-team rate random merges would give. Several look right by eye too — e.g. the only two
orange-shorts crops (tracks 7 & 132) land in one identity, and the four-fragment white identity
`[29, 3, 307, 253]` is all-white. So the **machinery is sound and carries real signal**; what's
missing for a clean *identity* accuracy number is per-individual ground truth, which is hard to
label by sight in these low-resolution crops (the same root cause that caps team accuracy at 56.5%).

> Caveat: team-consistency is a *proxy* for identity correctness, not a substitute — two different
> white players merged would still be "team-pure". It bounds the error from above, not below. A true
> identity-accuracy number waits on per-individual labels (see *What would move identity accuracy*).

### Generalisation — four more clips from across the game

To check it isn't tuned to one clip, four more 30 s clips were cut from different points of the same
38-min game (`-ss 420 / 900 / 1700 / 2050`, all live play) and run with the default `--reid`. The
label-free checks (`eval/validate_reid.py`: temporal soundness, count sanity, and merge
team-consistency against the *predicted* teams since these clips have no hand labels):

| clip | tracks → identities | concurrency floor | temporal viol. | merged pairs (same / cross-team) |
|---|---|---|---|---|
| `clip_420`  | 39 → 34 | 12 | 0 ✅ | 7 / **7 same**, 0 cross |
| `clip_900`  | 43 → 34 | 12 | 0 ✅ | 11 / 10 same, **1 cross\*** |
| `clip_1700` | 40 → 36 | 10 | 0 ✅ | 4 / **4 same**, 0 cross |
| `clip_2050` | 33 → 28 | 10 | 0 ✅ | 6 / **6 same**, 0 cross |

Across all four (plus the reference clip): **0 temporal violations**, every identity count sits
between its concurrency floor and the track count, and **30 of 31 team-checkable merges are
same-team**. The lone "cross-team" merge (`clip_900`) is itself instructive: it joined two
disjoint-in-time white-shirted fragments (tracks 90 & 151) that *look like the same player* by eye —
the "cross-team" flag comes from the **team classifier** mislabelling track 90 (its `team_conf` was
0.49, a coin flip), not from a re-ID error. So re-ID was actually *more* robust than the weak team
head on that borderline track. The behaviour is consistent clip-to-clip: sound, conservative, and
carrying real signal — re-run cheaply with `python eval/validate_reid.py runs/<clip>/tracks.csv`.

\* a team-label error on a `team_conf 0.49` fragment, not an identity error (see above).

### What would move identity accuracy

- **A real re-ID embedding** (OSNet/torchreid) instead of repurposed SigLIP — trained to separate
  *individuals*, with part-based pooling, where SigLIP keys on coarse appearance.
- **Higher-resolution capture** (the capture-side lever already noted for teams) — bigger, sharper
  player crops carry the per-player gear detail re-ID needs.
- **Spatiotemporal motion priors** beyond the cannot-link: link fragments whose exit/entry points
  and timing are continuous, not appearance alone.
- **Identity ground truth.** The eval harness already scores a `player` column with the same
  optimal-alignment metric used for teams (`evaluate.py`); the blocker is that individuals are hard
  to tell apart by sight in these crops, so a confident per-track identity labelling needs sharper
  footage or a frame-level review tool. Until then, validation leans on the label-free soundness +
  team-purity checks above.
