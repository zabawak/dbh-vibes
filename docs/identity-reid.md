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
  trip *is* that long temporal gap. Replaces the over-counting fragment-count shift estimate. The
  gap threshold defaults to **15 s** — a physical floor on a real bench change (shorter absences are
  treated as occlusion). Validated: 30 s reference clip → 28 fragments → 13 identities → **13 shifts
  (exactly 1.0/player**, correct for a window too short to bench in); 3-minute line-change clip →
  141 fragments → 20 identities → **61 shifts (3.0/player, avg 32 s)** — deterministic, shifts
  non-overlapping within each player, `n_shifts ≤ n_fragments` always. (The inter-fragment gap
  distribution on this fisheye footage is not cleanly bimodal, so the threshold is a tunable
  judgement call; an explicit entry/exit zone would replace it — see architecture.md.)
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

## Update (2026-07): the OSNet embedder — priority #6 landed

The first item under *What would move identity accuracy* is now in: `--embedder osnet` replaces the
repurposed SigLIP embedding with OSNet-AIN, a real person re-ID network (`reid_embedder.py`,
vendored architecture in `osnet.py`; the same embedding is shared with team clustering, where it
took team accuracy 57.1% → **100%** on fresh labels — see team-clustering.md).

For **identity**, measured on the reference clip with known same-person track pairs (labeled by
sight from the fresh montages — distinctive gear only, so a partial but confident set):

- **SigLIP**: same-person pair distances `[0.18, 0.23, 0.54, 0.67, 0.87, 1.29]` vs same-team
  different-person pairs `[0.30, 0.48, 0.51, 0.65, ...]` — fully interleaved. No threshold can
  separate them; SigLIP genuinely cannot do identity on this footage.
- **OSNet**: same-person `[0.03, 0.08, 0.30, 0.51, 0.60, 0.79]` vs different-person
  `[0.04, 0.29, 0.31, 0.66, 0.70, ...]` — clearly better ordered (the closest pairs are true
  same-person) but still overlapping, so the data-driven threshold remains a precision/recall dial,
  not a clean cut. Two structural mitigations apply: the **temporal cannot-link** vetoes the most
  dangerous near-duplicates (concurrent look-alikes — the 0.04 different-person pair is exactly
  that, two white shirts on the surface together), and `--roster` sidesteps the threshold entirely.
- Per-embedder defaults now live in `pipeline.REID_DISTANCE_DEFAULTS` (siglip 0.35, osnet 0.45 —
  both deliberately conservative). On the reference clip at defaults: SigLIP merges 2 fragment
  pairs, OSNet 4; all label-free checks pass for both (0 temporal violations, every merge
  team-consistent — `eval/validate_reid.py`).

The honest summary: OSNet lifts identity from "no signal beyond the temporal constraint" to "real
but imperfect appearance signal". The remaining levers are unchanged — higher-resolution capture,
motion priors linking exit/entry continuity, and per-individual ground truth for a true accuracy
number.

### 3-minute-clip threshold sweep (156 fragments, ~15–20 people)

Sweeping the data-driven `--reid-distance` on the line-change clip (490 raw tracks → 208 players →
156 with crops; heavier fragmentation than the numbers earlier in this doc — dependency-version
drift changes the tracker's behaviour, another argument for the identity-recall work):

| threshold | SigLIP identities | OSNet identities | OSNet silhouette |
|---|---|---|---|
| 0.25 | 118 | 87 | 0.10 |
| 0.35 (siglip default) | 88 | 70 | 0.11 |
| 0.45 (osnet default) | 61 | 56 | 0.11 |
| 0.55 | 47 | 38 | 0.08 |
| 0.70 | 34 | 32 | 0.06 |

OSNet dominates at every threshold (fewer, cleaner identities; silhouette peaks in the 0.35–0.50
band, which brackets its 0.45 default) — but neither embedder shows an elbow at roster size. Forcing
the merge down to 15 identities accepts merge distances with median 0.25 and p90 0.64 (OSNet) —
well past any safe threshold — so **the data-driven count remains a precision dial, and `--roster`
remains the accuracy path** on heavily fragmented footage. Reproduce with the recipe in
`data/README.md` (the sweep script lives in the session notes; embeddings + spans are saved by the
pipeline's shared embedding pass).
