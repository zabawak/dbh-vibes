"""Tests for the OSNet re-ID embedder (priority #6) — the parts that need no download.

The checkpoint-download path is exercised by the real-footage validation runs; here we test the
pure pieces (state-dict adaptation, preprocessing) and — when torch is installed — that the vendored
architecture produces the right embedding shape and accepts its own state dict end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from dbh_vibes.reid_embedder import (
    CROP_HW,
    DEFAULT_WEIGHTS,
    EMBED_DIM,
    WEIGHT_SOURCES,
    adapt_state_dict,
    preprocess_crops,
)

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------------------------
# adapt_state_dict (pure)
# --------------------------------------------------------------------------------------------

class TestAdaptStateDict:
    def test_strips_dataparallel_prefix(self):
        raw = {"module.conv1.weight": 1, "module.fc.0.bias": 2}
        adapted, dropped = adapt_state_dict(raw, {"conv1.weight", "fc.0.bias"})
        assert adapted == {"conv1.weight": 1, "fc.0.bias": 2}
        assert dropped == []

    def test_drops_training_classifier_head(self):
        raw = {"module.conv1.weight": 1, "module.classifier.weight": 2,
               "module.classifier.bias": 3}
        adapted, dropped = adapt_state_dict(
            raw, {"conv1.weight", "classifier.weight", "classifier.bias"}
        )
        assert adapted == {"conv1.weight": 1}
        assert set(dropped) == {"classifier.weight", "classifier.bias"}

    def test_drops_keys_absent_from_model(self):
        raw = {"conv1.weight": 1, "optimizer_state": 2}
        adapted, dropped = adapt_state_dict(raw, {"conv1.weight"})
        assert adapted == {"conv1.weight": 1}
        assert dropped == ["optimizer_state"]

    def test_unprefixed_checkpoint_passes_through(self):
        raw = {"conv1.weight": 1}
        adapted, _ = adapt_state_dict(raw, {"conv1.weight"})
        assert adapted == {"conv1.weight": 1}


# --------------------------------------------------------------------------------------------
# preprocess_crops (pure numpy + cv2)
# --------------------------------------------------------------------------------------------

class TestPreprocess:
    def test_shape_and_dtype(self):
        crops = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                 for h, w in [(200, 80), (37, 21), (400, 400)]]
        batch = preprocess_crops(crops)
        assert batch.shape == (3, 3, *CROP_HW)
        assert batch.dtype == np.float32

    def test_imagenet_normalisation(self):
        # A crop of pure ImageNet-mean colour must normalise to ~zero everywhere.
        mean_bgr = np.array([0.406, 0.456, 0.485]) * 255  # BGR order of the RGB mean
        crop = np.full((64, 32, 3), mean_bgr, dtype=np.uint8)
        batch = preprocess_crops([crop])
        assert np.abs(batch).max() < 0.05

    def test_bgr_to_rgb(self):
        # Pure blue in BGR must land in the (RGB) blue channel, not red.
        crop = np.zeros((64, 32, 3), dtype=np.uint8)
        crop[:, :, 0] = 255  # BGR blue
        batch = preprocess_crops([crop])
        blue = batch[0, 2].mean()   # RGB channel 2
        red = batch[0, 0].mean()    # RGB channel 0
        assert blue > red


# --------------------------------------------------------------------------------------------
# Vendored architecture (torch, random weights — no download)
# --------------------------------------------------------------------------------------------

class TestOsnetArchitecture:
    def test_embedding_shape_and_determinism(self):
        from dbh_vibes.osnet import osnet_ain_x0_25

        model = osnet_ain_x0_25().eval()
        x = torch.randn(2, 3, *CROP_HW)
        with torch.no_grad():
            v1 = model(x)
            v2 = model(x)
        assert v1.shape == (2, 512)
        assert torch.allclose(v1, v2)

    def test_own_state_dict_roundtrips_through_adapt(self):
        # Simulate a torchreid checkpoint: DataParallel prefix + a sized classifier head.
        from dbh_vibes.osnet import osnet_ain_x0_25

        model = osnet_ain_x0_25()
        raw = {f"module.{k}": v for k, v in model.state_dict().items()}
        adapted, dropped = adapt_state_dict(raw, set(model.state_dict().keys()))
        missing = [k for k in model.state_dict() if k not in adapted
                   and not k.startswith("classifier.")]
        assert missing == []
        assert all(k.startswith("classifier.") for k in dropped)
        model.load_state_dict(adapted, strict=False)  # must not raise


# --------------------------------------------------------------------------------------------
# Registry sanity
# --------------------------------------------------------------------------------------------

def test_default_weights_registered():
    assert DEFAULT_WEIGHTS in WEIGHT_SOURCES
    for spec in WEIGHT_SOURCES.values():
        assert spec["variant"] == "osnet_ain_x1_0"
        assert spec["gdrive_id"] and spec["filename"].endswith(".pth")


def test_embed_dim_constant():
    assert EMBED_DIM == 512
