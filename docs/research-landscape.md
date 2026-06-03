# Open-Source Landscape for Sports Video Analysis

A survey of the open-source building blocks for turning game video into player stats, and how
they apply to **ball hockey** specifically. This is the research that backs the
[architecture/roadmap](architecture.md).

## The standard pipeline

Nearly every open-source sports-analysis project composes the same stages. You rarely build
these from scratch — you wire together mature components:

```
video ─▶ player detection ─▶ multi-object tracking ─▶ team classification ─┐
                                                                           ▼
        stats  ◀─ identity (jersey #) ◀─ court mapping (homography) ◀───────┘
```

## Component-by-component

### 1. Player detection
- **[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)** (AGPL-3.0) — the
  practical default. The pretrained COCO model already detects the `person` class, so you get
  player boxes with **zero training**. Fine-tune later to add ball / goalie / referee classes.
- **[YOLO26](https://blog.roboflow.com/how-to-train-yolo26-custom-data/)** and earlier YOLO
  versions are drop-in alternatives.
- Training a custom detector needs labeled frames — tools:
  [Roboflow](https://blog.roboflow.com/yolov11-how-to-train-custom-data/) (SAM-2 assisted
  labeling), [Label Studio](https://labelstud.io/), CVAT.

### 2. Multi-object tracking (persistent IDs)
- **[ByteTrack](https://github.com/FoundationVision/ByteTrack)** (ECCV 2022) — associates every
  detection box, recovering low-confidence ones; strong default, **built into Ultralytics**.
- **[BoT-SORT](https://github.com/NirAharon/BoT-SORT)** — adds appearance features + camera-
  motion compensation; better through occlusions, also built into Ultralytics.
- These maintain a stable track id *within a continuous segment*. Bridging ids across occlusions
  / camera cuts / line changes is the **re-identification** problem (stages 5).

### 3. Team classification
- **[roboflow/sports](https://github.com/roboflow/sports)** (MIT) demonstrates the reference
  approach: **SigLIP** image embeddings → **UMAP** dimensionality reduction → **KMeans** (k=2)
  on player crops. Robust to lighting, no labels needed.
- **Lightweight fallback** (what this repo's MVP uses): KMeans on torso-region jersey colors.
  Cheaper, no extra models, good enough for visibly different kits.

### 4. Court / rink mapping (homography)
- Maps camera pixels → top-down rink coordinates, unlocking positions, distances, heatmaps,
  and zone-based stats.
- Done via **keypoint detection** of court markings + homography. roboflow/sports ships
  soccer-pitch and basketball-court keypoint examples and a
  [camera-calibration writeup](https://blog.roboflow.com/camera-calibration-sports-computer-vision/).
- For hockey specifically: [Multi Player Tracking in Ice Hockey with Homographic
  Projections](https://arxiv.org/html/2405.13397v1) maps broadcast frames onto a top-view rink
  template. Needs a **ball-hockey rink template + labeled keypoints** we'd create.

### 5. Player identity (jersey numbers / re-ID) — the hard part
- The most distinctive per-player feature is the **jersey number** (same-team kits look alike).
  Treated as a scene-text-recognition problem; pipelines add legibility filtering + torso
  localization before OCR.
- Hockey-specific research: [Player Tracking and Identification in Ice
  Hockey](https://arxiv.org/pdf/2110.03090) (~91% jersey accuracy on hockey images);
  [Towards long-term player tracking with graph hierarchies](https://arxiv.org/pdf/2502.21242)
  (SportsSUSHI — folds team + jersey number into tracking association).
- This stage gates **true per-player stat attribution**. It's hard: blur, occlusion, players
  facing away. Deferred past the MVP.

## End-to-end references worth studying
- **[roboflow/sports](https://github.com/roboflow/sports)** + **[supervision](https://github.com/roboflow/supervision)**
  (MIT) — closest off-the-shelf reference for the whole pipeline (soccer/basketball). This
  repo borrows its patterns.
- **[soccer-multi-object-tracking](https://github.com/Anudeep007-hub/soccer-multi-object-tracking)**
  — YOLOv8 + DeepSORT/ByteTrack/BoT-SORT with team assignment & re-ID, modular.
- **[Spicing up Ice Hockey with AI](https://towardsdatascience.com/spicing-up-ice-hockey-with-ai-player-tracking-with-computer-vision-ce9ceec9122a/)**
  — narrative walkthrough of a hockey tracking pipeline.
- **[SoccerNet Game State Reconstruction](https://arxiv.org/pdf/2504.06357)** — the
  single-camera → top-down minimap pattern, directly relevant to our single-camera setup.

## Datasets reusable for ball hockey
- **[Montreal Hockey Player Tracking Dataset (MHPTD)](https://github.com/grant81/hockeyTrackingDataset)**
  — ice-hockey tracking in MOT format with team IDs + jersey numbers. Closest existing labeled
  data; useful for fine-tuning/eval even though ice ≠ ball hockey.
- Roboflow Universe has many soccer/basketball player-detection and jersey-OCR datasets to
  bootstrap from.

## Ball-hockey reality check
Ball hockey is niche, so **no pretrained ball-hockey weights exist**. Consequences:
- Player **detection works immediately** via the generic COCO person class — that's our MVP.
- Anything sport-specific — the **ball**, **rink keypoints**, **jersey numbers** — requires our
  own labeled clips. That labeling is the main cost driver, which is why the roadmap stages it.
- A **single fixed camera** simplifies tracking and (later) homography vs. broadcast footage.

## License snapshot
| Component | License | Implication |
|---|---|---|
| Ultralytics YOLO11 | **AGPL-3.0** | Copyleft — matters only if distributed as a product; swap for MIT/Apache detector then. |
| supervision, OpenCV, NumPy, pandas | MIT / BSD / Apache | Permissive. |
| roboflow/sports | MIT | Permissive. |
