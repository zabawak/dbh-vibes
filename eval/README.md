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
  - `player` — identity slot for Phase 3 re-ID; blank until that exists.
  - `note` — why a row was left blank.

23 of 27 player tracks are team-labeled; 4 are intentionally blank.

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
| **team** | **52.2%** | 12/23 |
| role | 100.0% | 27/27 |

Team accuracy is ~chance — the **first hard confirmation** of the long-suspected accuracy gap
(docs/team-clustering.md): the embedding split is driven by crop scale, not the low-contrast
white/dark kits. Role is perfect: the surface filter classified every on-court player correctly.
This number is the baseline that the next lever — **background-suppressed crops** — must beat.
