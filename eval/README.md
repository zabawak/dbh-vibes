# eval/ — labeled set + harness (priority #1)

The "binding constraint": team clustering is *stable* but we couldn't measure its true **accuracy**
on natural footage without ground-truth labels. This directory holds the labels; the harness is
`src/dbh_vibes/evaluate.py` (run via `python -m dbh_vibes --evaluate`).

## What's here

- **`sample_labels.csv`** — per-track ground truth for the reference active-gameplay clip
  (`data/sample.mp4`, cut per `data/README.md`: `-ss 1490 -t 30`, 720p/30fps). Columns:
  - `track_id` — matches the pipeline's `tracks.csv` (same detect/track pass).
  - `team` — `white` / `dark` (the two kits), hand-labeled from the per-track crop montages.
    Genuinely ambiguous tracks (a lone red-shirt player, tiny/occluded crops) are left **blank**
    on purpose — the harness scores only the labeled∩predicted overlap rather than guessing.
  - `role` — `player` / `spectator` (all exported crops here are on-court players).
  - `player` — identity slot for Phase 3 re-ID. **Phase 3 (`--reid`) now exists and predicts a
    `player` id**, but this column is still **blank**: the individuals are hard to tell apart by
    sight in these low-resolution crops, so a confident per-track identity labelling isn't possible
    from the montages alone (some same-colour tracks even overlap in time, i.e. are different people
    in matching gear). Until sharper footage or a frame-level review tool exists, Phase 3 is
    validated by the label-free **temporal-soundness + team-purity** checks in
    [`../docs/identity-reid.md`](../docs/identity-reid.md) rather than a `player`-column accuracy.
  - `note` — why a row was left blank.

23 of 27 player tracks are team-labeled; 4 are intentionally blank.

## Label-free identity validation (`validate_reid.py`)

Because per-individual identity labels are hard to get from these crops, `validate_reid.py` checks
the Phase 3 (`--reid`) output without any ground truth: **temporal soundness** (no identity contains
two time-overlapping tracks — a hard guarantee), **count sanity** (identities between the peak
on-surface concurrency and the track count), and **merge team-consistency** (a merge shouldn't join
two tracks the team head called different teams). Validated across five clips of the reference game —
0 temporal violations, 30/31 merges same-team — see [`../docs/identity-reid.md`](../docs/identity-reid.md).

```bash
python eval/validate_reid.py runs/<clip>/tracks.csv [...more tracks.csv]
```

## Regenerate predictions + score

```bash
# 1. Cut the same clip (see data/README.md for fetching the full game)
ffmpeg -y -ss 1490 -i data/_full_game.mp4 -t 30 -vf "scale=1280:-2,fps=30" -an data/sample.mp4

# 2. Run the pipeline, exporting the labeling set (crops/ + a labels.csv template)
python -m dbh_vibes data/sample.mp4 --out runs/sample --phase2 --model yolo11s.pt --label-crops

# 3. Score the committed labels against the fresh predictions
python -m dbh_vibes --evaluate eval/sample_labels.csv --tracks runs/sample/tracks.csv
```

To label a **new** clip, run step 2, view each `runs/<out>/crops/track_*.jpg` montage, fill the
`team`/`role`/`player` columns of `runs/<out>/labels.csv`, and copy it here.

> Track ids come from YOLO + ByteTrack and are reproducible for a given clip + model, but can shift
> if the Ultralytics/model version changes. If ids drift, re-label from the fresh montages.

## First measured result (the point of all this)

On `data/sample.mp4`, `yolo11s` + SigLIP embeddings (the kit-colour prior correctly declines on
this white/dark footage):

| field | accuracy | n |
|---|---|---|
| team (raw crops) | 52.2% | 12/23 |
| **team (bg-suppressed crops)** | **56.5%** | 13/23 |
| role | 100.0% | 27/27 |

Raw-crop team accuracy is ~chance — the **first hard confirmation** of the long-suspected accuracy
gap (docs/team-clustering.md): the embedding split is driven by crop scale, not the low-contrast
white/dark kits. **Background-suppressed crops** (the next lever, now implemented — torso-crop and
mask the rink before embedding) beat that baseline at **56.5% (13/23)** and balance the split, but
white-vs-dark stays hard. Reproduce both with `--no-bg-suppress` (raw) vs the default (suppressed):

```bash
python -m dbh_vibes data/sample.mp4 --out runs/raw --phase2 --model yolo11s.pt --no-bg-suppress
python -m dbh_vibes data/sample.mp4 --out runs/sup --phase2 --model yolo11s.pt
python -m dbh_vibes --evaluate eval/sample_labels.csv --tracks runs/raw/tracks.csv   # 52.2%
python -m dbh_vibes --evaluate eval/sample_labels.csv --tracks runs/sup/tracks.csv   # 56.5%
```

Role is perfect: the surface filter classified every on-court player correctly.
