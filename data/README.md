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
