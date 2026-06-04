"""SigLIP-embedding team classifier (Phase 2).

This is the upgrade from the MVP's torso-color KMeans, which collapsed on real footage
(dark-vs-white jerseys plus bench/spectator clothing all landed in one cluster). It mirrors the
roboflow/sports approach: embed each player crop with the SigLIP vision tower, reduce with UMAP,
and cluster into two teams with KMeans. Appearance embeddings separate kits far more robustly
than mean color.

Efficiency note: SigLIP on CPU is ~240ms/crop, far too slow to embed every detection in every
frame. We lean on the tracker instead — a track's team is constant, so we embed only a handful
of crops per track id and majority-vote (see assign_teams_by_track). That turns tens of
thousands of embeds into a few hundred for a whole clip.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")


class SiglipTeamClassifier:
    """Cluster player crops into two teams via SigLIP embeddings -> UMAP -> KMeans."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        batch_size: int = 16,
        n_components: int = 3,
        n_clusters: int = 2,
        device: str = "cpu",
    ) -> None:
        # Heavy imports are local so importing this module stays cheap and the color fallback
        # has no transformers/umap dependency.
        import torch
        import umap
        from sklearn.cluster import KMeans
        from transformers import AutoImageProcessor, SiglipVisionModel

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        # use_fast keeps preprocessing off the slow Python path.
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = SiglipVisionModel.from_pretrained(model_name).eval().to(device)
        torch.set_num_threads(4)

        self.reducer = umap.UMAP(n_components=n_components, random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self._fitted = False

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed BGR crops (as from OpenCV) into SigLIP feature vectors."""
        from PIL import Image

        if not crops:
            return np.empty((0, 768), dtype=np.float32)

        pil = [Image.fromarray(c[:, :, ::-1]) for c in crops]  # BGR -> RGB
        feats = []
        with self._torch.no_grad():
            for i in range(0, len(pil), self.batch_size):
                batch = pil[i : i + self.batch_size]
                inp = self.processor(images=batch, return_tensors="pt").to(self.device)
                out = self.model(**inp).pooler_output
                feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def fit(self, crops: list[np.ndarray]) -> "SiglipTeamClassifier":
        """Fit the UMAP+KMeans team model on a representative set of player crops."""
        emb = self.embed(crops)
        reduced = self.reducer.fit_transform(emb)
        self.kmeans.fit(reduced)
        self._fitted = True
        return self

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return a team id (0/1) per crop. Must call fit() first."""
        if not self._fitted:
            raise RuntimeError("SiglipTeamClassifier.predict called before fit()")
        if not crops:
            return np.empty((0,), dtype=int)
        reduced = self.reducer.transform(self.embed(crops))
        return self.kmeans.predict(reduced).astype(int)


def assign_teams_by_track(
    classifier: SiglipTeamClassifier,
    track_crops: dict[int, list[np.ndarray]],
) -> dict[int, int]:
    """Assign one team id per track id by majority vote over that track's sampled crops.

    Args:
        classifier: a fitted SiglipTeamClassifier.
        track_crops: track_id -> list of sampled BGR crops for that track.

    Returns:
        track_id -> team id (0/1).
    """
    # Flatten to a single batch so we pay the embedding cost once, then scatter back.
    flat_crops: list[np.ndarray] = []
    owners: list[int] = []
    for track_id, crops in track_crops.items():
        for crop in crops:
            flat_crops.append(crop)
            owners.append(track_id)

    if not flat_crops:
        return {}

    preds = classifier.predict(flat_crops)

    votes: dict[int, list[int]] = defaultdict(list)
    for track_id, team in zip(owners, preds):
        votes[track_id].append(int(team))

    return {tid: int(round(np.mean(v))) for tid, v in votes.items()}


def crop_box(frame: np.ndarray, box_xyxy: np.ndarray, pad: float = 0.0) -> np.ndarray | None:
    """Crop a bounding box from a frame, with optional fractional padding. None if degenerate."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(w, x2 + pad * bw))
    y2 = int(min(h, y2 + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
