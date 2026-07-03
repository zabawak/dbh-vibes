"""OSNet person re-ID embedder — the purpose-built appearance model (priority #6).

Both team clustering and Phase 3 identity re-ID were built on a repurposed **SigLIP** embedding.
That was the pragmatic choice (already needed, zero-shot), but SigLIP is a general image-text model,
not a person re-identification model — and the measured ceiling shows it: team accuracy 56.5% on
low-contrast kits, identity over-segmentation on the 3-min clip. This module swaps in **OSNet-AIN**
(`osnet.py`, vendored from torchreid), a network *trained for exactly our problem*: embed a person
crop such that the same person is close and different people are far, across camera domains.

Interface-compatible with ``team_siglip.SiglipTeamClassifier`` (an ``embed(crops) -> (N, D)``
method), so the whole downstream stack — per-track pooling, team clustering, constrained identity
clustering — is reused unchanged via the ``--embedder`` switch.

Practical differences from the SigLIP path, handled by the caller (``pipeline.py``):

* **Full-body crops, no background suppression.** OSNet was trained on whole person detections
  (256x128, background and all); its instance normalisation is designed to absorb domain/background
  shift. The torso-crop + rink-masking that helps SigLIP would *break* OSNet's input distribution,
  so background suppression defaults off for this embedder.
* **Cheap.** 2.2M params vs SigLIP's ~93M — ~10x faster per crop on CPU, so it can embed more crops
  per track for a *better* per-track mean, at lower cost.

Weights are the published torchreid checkpoints, fetched once from the torchreid model zoo (Google
Drive, via ``gdown``) into ``~/.cache/dbh_vibes/`` and loaded with ``weights_only`` safe
unpickling (tensors + numpy scalars only — no arbitrary code execution from a downloaded file).
The default checkpoint is the **domain-generalisation** one (trained on MSMT17+DukeMTMC+CUHK03,
cosine metric) — the right prior for footage that looks nothing like the training rinks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EMBED_DIM = 512
CROP_HW = (256, 128)  # (height, width) every OSNet checkpoint was trained at
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Published torchreid checkpoints (Google Drive file ids from the torchreid MODEL_ZOO).
# "msdc" = multi-source domain generalisation (MSMT17 + DukeMTMC + CUHK03, cosine metric) — the
# default because our footage is far outside any single re-ID training domain.
WEIGHT_SOURCES: dict[str, dict] = {
    "msdc": {
        "variant": "osnet_ain_x1_0",
        "gdrive_id": "1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB",
        "filename": "osnet_ain_x1_0_msdc.pth",
    },
    "msmt17": {
        "variant": "osnet_ain_x1_0",
        "gdrive_id": "1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
        "filename": "osnet_ain_x1_0_msmt17.pth",
    },
}
DEFAULT_WEIGHTS = "msdc"


def cache_dir() -> Path:
    return Path.home() / ".cache" / "dbh_vibes"


def fetch_weights(name: str = DEFAULT_WEIGHTS) -> Path:
    """Return a local path to the named checkpoint, downloading it on first use."""
    spec = WEIGHT_SOURCES[name]
    path = cache_dir() / spec["filename"]
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            f"OSNet weights '{name}' are not cached at {path} and gdown is not installed. "
            f"Run `pip install gdown` (or download Google Drive file id {spec['gdrive_id']} "
            f"to that path manually)."
        ) from exc
    # no-check-certificate mirrors the sandbox-proxy guidance in data/README.md; harmless elsewhere.
    gdown.download(id=spec["gdrive_id"], output=str(path), quiet=True)
    if not path.exists():
        raise RuntimeError(f"failed to download OSNet weights '{name}' (Drive id {spec['gdrive_id']})")
    return path


def adapt_state_dict(raw: dict, model_keys: set[str]) -> tuple[dict, list[str]]:
    """Adapt a torchreid checkpoint state dict to the vendored model. Pure — unit-testable.

    Strips the DataParallel ``module.`` prefix and drops keys the embedder doesn't have (the
    training-time ``classifier`` head sized to the training identity count). Returns
    ``(adapted, dropped_keys)``; every remaining key must exist in ``model_keys``.
    """
    adapted: dict = {}
    dropped: list[str] = []
    for k, v in raw.items():
        k2 = k[len("module."):] if k.startswith("module.") else k
        if k2 in model_keys and not k2.startswith("classifier."):
            adapted[k2] = v
        else:
            dropped.append(k2)
    return adapted, dropped


def _safe_load_checkpoint(path: Path):
    """Load a downloaded checkpoint without arbitrary-code pickle execution.

    ``weights_only=True`` restricts unpickling to tensors/containers; torchreid checkpoints
    additionally carry a few numpy scalars (epoch counters, rank1 floats), so those specific numpy
    globals are allowlisted — data, not code.
    """
    import torch

    allow = []
    try:
        import numpy._core.multiarray as _ma  # numpy >= 2
    except ImportError:  # pragma: no cover - numpy 1.x
        import numpy.core.multiarray as _ma
    # The pickled global is spelled `numpy.core.multiarray.scalar` regardless of numpy version.
    allow.append((_ma.scalar, "numpy.core.multiarray.scalar"))
    allow.append((np.dtype, "numpy.dtype"))
    for dt in ("Float64DType", "Float32DType", "Int64DType", "Int32DType"):
        obj = getattr(getattr(np, "dtypes", None), dt, None)
        if obj is not None:
            allow.append(obj)
    with torch.serialization.safe_globals(allow):
        return torch.load(path, map_location="cpu", weights_only=True)


def preprocess_crops(crops: list[np.ndarray]) -> np.ndarray:
    """BGR crops (any size) -> float32 NCHW batch at 256x128, ImageNet-normalised. Pure numpy+cv2."""
    import cv2

    h, w = CROP_HW
    batch = np.empty((len(crops), 3, h, w), dtype=np.float32)
    for i, c in enumerate(crops):
        rgb = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        x = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        batch[i] = x.transpose(2, 0, 1)
    return batch


class OsnetEmbedder:
    """Embed BGR player crops with OSNet-AIN. Same ``embed`` interface as the SigLIP embedder."""

    embed_dim = EMBED_DIM
    # OSNet is trained on full-body detections with background; suppression would hurt it.
    default_suppress_background = False

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        import torch

        from dbh_vibes.osnet import osnet_ain_x1_0

        self._torch = torch
        self.device = device
        self.batch_size = batch_size

        path = Path(weights) if isinstance(weights, (str, Path)) and str(weights).endswith(
            (".pth", ".pt")
        ) else fetch_weights(str(weights))
        ck = _safe_load_checkpoint(path)
        raw = ck.get("state_dict", ck) if isinstance(ck, dict) else ck

        self.model = osnet_ain_x1_0()
        model_keys = set(self.model.state_dict().keys())
        adapted, _dropped = adapt_state_dict(raw, model_keys)
        missing = [k for k in model_keys if k not in adapted and not k.startswith("classifier.")]
        if missing:
            raise RuntimeError(
                f"OSNet checkpoint {path} did not cover the model "
                f"({len(missing)} missing keys, e.g. {missing[:3]})"
            )
        self.model.load_state_dict(adapted, strict=False)
        self.model.eval().to(device)
        torch.set_num_threads(4)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed BGR crops (as from OpenCV) into 512-d OSNet re-ID features."""
        if not crops:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        feats = []
        with self._torch.no_grad():
            for i in range(0, len(crops), self.batch_size):
                batch = preprocess_crops(crops[i : i + self.batch_size])
                out = self.model(self._torch.from_numpy(batch).to(self.device))
                feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)
