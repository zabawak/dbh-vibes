# data/

Drop your ball hockey game clips here (e.g. `data/sample.mp4`). Video files are **gitignored**
— they never get committed.

Then run, for example:

```bash
python -m dbh_vibes data/sample.mp4 --out runs/sample
```

Tips for good results from a single fixed camera:

- Mount the camera high and wide enough to see the whole playing surface.
- Keep it stationary — the current pipeline assumes a fixed viewpoint.
- 1080p at 30 fps is plenty; higher resolution mostly just slows processing.

## Reference test footage (for future sessions)

The clip used to develop and validate Phases 1–2. It's a full pickup ball hockey game on a blue
outdoor dek rink, single fixed fisheye camera: **~38 min, 1910×1080 @ 60 fps, ~1.09 GB**.

- **Google Drive**: https://drive.google.com/file/d/1IvZEHVKtGJV6onzUwpQVAchnFuhQS_wr/view
- **File ID**: `1IvZEHVKtGJV6onzUwpQVAchnFuhQS_wr`

### Fetching it in this environment

YouTube downloads are blocked here (bot wall / DRM), but Google Drive works via `gdown`. The
sandbox proxy uses a self-signed cert, so pass `--no-check-certificate`:

```bash
pip install -q gdown
gdown --no-check-certificate 1IvZEHVKtGJV6onzUwpQVAchnFuhQS_wr -O data/_full_game.mp4
```

### Segments used for validation

Processing the full 38 min on CPU is slow, so we cut short 720p/30fps clips with ffmpeg
(`-ss <start>` seconds, `-t 30` for 30s). The two segments referenced throughout the docs:

```bash
# Active gameplay (~24:50) — the main test clip (5-on-5 + goalies)
ffmpeg -y -ss 1490 -i data/_full_game.mp4 -t 30 -vf "scale=1280:-2,fps=30" -an data/sample.mp4

# Lull / "middle" (~19:00) — a break in play, everyone at the bench (idle-detection test)
ffmpeg -y -ss 1137 -i data/_full_game.mp4 -t 30 -vf "scale=1280:-2,fps=30" -an data/sample_mid.mp4

# 3 minutes with real line changes (~23:50) — the shift-detection / re-ID stress clip
ffmpeg -y -ss 1430 -i data/_full_game.mp4 -t 180 -vf "scale=1280:-2,fps=30" -an data/sample_3min.mp4

# 10-minute slice with gameplay + stoppages (~23:20) — the --game full-game-mode validation slice
ffmpeg -y -ss 1400 -i data/_full_game.mp4 -t 600 -vf "scale=1280:-2,fps=30" -an data/game_10min.mp4
```

Note: the middle third of this recording is downtime between games — pick `-ss` around the
gameplay timestamps above (or sample frames first) to land on live action.

