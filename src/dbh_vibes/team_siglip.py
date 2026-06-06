"""SigLIP-embedding team clustering (Phase 2, hardened).

The upgrade from the MVP torso-color KMeans, which collapsed on real footage. We embed each
player crop with the SigLIP vision tower and cluster appearance into two teams. Appearance
embeddings separate kits far more robustly than mean color.

This module is the hardening of that clusterer. The earlier version (per-crop SigLIP -> UMAP ->
KMeans(k=2)) was *unstable run to run* (e.g. an 18-vs-15 split one run, a degenerate 28-vs-6 the
next) — see docs/team-clustering.md. The instability had concrete causes, each addressed here and
**all without any labelled data**:

1. Clustering per *crop* made the result sensitive to how many/which crops each track contributed
   and to motion blur. -> We aggregate each track into one mean embedding and cluster *tracks*:
   one vote per player (`aggregate_track_embeddings`).
2. UMAP is stochastic across versions and noisy on the few dozen points we have. -> We reduce with
   PCA, which is deterministic, and fix every seed and the track ordering, so the same crops give
   identical teams every run (`_reduce`). This also drops the `umap-learn` dependency.
3. Forcing exactly two clusters onto 3+ visual groups (two kits + goalies + the odd ref) let a
   handful of edge tracks tip the boundary into a degenerate split. -> We *over-segment* into K>=2
   micro-clusters (K chosen by silhouette) and then merge **by size**: the two largest groups are
   the team anchors, and every smaller outlier (goalies, refs) folds into the nearest anchor by
   appearance. A goalie cluster can no longer *become* a team (`cluster_team_embeddings`).
4. KMeans labels are arbitrary and flip between runs, so the same team was T0 one run and T1 the
   next. -> We anchor T0/T1 to a physical, run-invariant signal — kit colour — so the more
   saturated (e.g. pinnie) team is consistently T0 (`order_labels_by_color`).
5. We have no ground truth, so we cannot tune by accuracy. -> Clustering returns a label-free
   quality signal (silhouette, team balance, per-track confidence margin) so separation and
   run-to-run stability can be *measured* rather than eyeballed (`ClusterInfo`, `team_confidence`).

Efficiency note: SigLIP on CPU is ~240ms/crop, far too slow to embed every detection in every
frame. We lean on the tracker — a track's team is constant — so we embed only a handful of crops
per track id and pool them. That turns tens of thousands of embeds into a few hundred per clip.

The clustering core (`aggregate_track_embeddings`, `cluster_team_embeddings`,
`order_labels_by_color`, `team_confidence`) operates on plain numpy embedding matrices, with no
torch/transformers dependency, so it is unit-testable on synthetic embeddings without any video.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

warnings.filterwarnings("ignore")

EMBED_DIM = 768


# --------------------------------------------------------------------------------------------
# SigLIP embedder (the only torch/transformers-dependent piece)
# --------------------------------------------------------------------------------------------

class SiglipTeamClassifier:
    """Embed player crops with the SigLIP vision tower.

    Despite the historical name this is now purely an *embedder*: the clustering lives in the pure
    functions below so it can be tested and reasoned about without the heavy model. Kept as a class
    so the model is loaded once and reused across the crops of a clip.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        # Heavy imports are local so importing this module stays cheap and the pure clustering
        # core (and its tests) need no transformers dependency.
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        # use_fast keeps preprocessing off the slow Python path.
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = SiglipVisionModel.from_pretrained(model_name).eval().to(device)
        torch.set_num_threads(4)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed BGR crops (as from OpenCV) into SigLIP feature vectors."""
        from PIL import Image

        if not crops:
            return np.empty((0, EMBED_DIM), dtype=np.float32)

        pil = [Image.fromarray(c[:, :, ::-1]) for c in crops]  # BGR -> RGB
        feats = []
        with self._torch.no_grad():
            for i in range(0, len(pil), self.batch_size):
                batch = pil[i : i + self.batch_size]
                inp = self.processor(images=batch, return_tensors="pt").to(self.device)
                out = self.model(**inp).pooler_output
                feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)


# --------------------------------------------------------------------------------------------
# Pure clustering core (numpy/sklearn only — no torch, no video)
# --------------------------------------------------------------------------------------------

@dataclass
class ClusterInfo:
    """Label-free quality signal for a team clustering, so we can measure rather than eyeball.

    silhouette: separation of the final two-team labelling in reduced space ([-1, 1]; higher is
        cleaner). Computed on the *track* points, so it is comparable run to run on the same clip.
    team_sizes: (n_team0, n_team1) track counts — a balance sanity check (a wildly lopsided split
        is the classic failure mode).
    n_micro: how many micro-clusters were used before merging to two (>2 means outlier groups such
        as goalies were peeled off rather than allowed to tip the split).
    """

    silhouette: float
    team_sizes: tuple[int, int]
    n_micro: int
    reduced: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    centroids: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    method: str = "siglip"           # "kit-color" (vivid-kit prior) or "siglip" (embedding fallback)
    conf: np.ndarray | None = field(repr=False, default=None)  # per-track confidence, if precomputed


def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def aggregate_track_embeddings(
    crop_embeddings: np.ndarray, owners: list[int], track_ids: list[int]
) -> np.ndarray:
    """Pool per-crop embeddings into one mean embedding per track, in `track_ids` order.

    Each crop embedding is L2-normalised before averaging so a few high-magnitude crops can't
    dominate a track, and the per-track mean is renormalised. Clustering tracks (not crops) is what
    makes the result insensitive to how many crops each player happened to contribute.
    """
    if len(track_ids) == 0:
        return np.empty((0, crop_embeddings.shape[1] if crop_embeddings.ndim == 2 else EMBED_DIM))
    normed = _l2norm(crop_embeddings.astype(np.float64))
    owners_arr = np.asarray(owners)
    rows = []
    for tid in track_ids:
        mask = owners_arr == tid
        if not mask.any():
            rows.append(np.zeros(normed.shape[1]))
        else:
            rows.append(normed[mask].mean(axis=0))
    return _l2norm(np.vstack(rows))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, 0 if either side is constant."""
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _reduce(
    track_emb: np.ndarray,
    n_components: int,
    random_state: int,
    sizes: np.ndarray | None = None,
    decorr_threshold: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Deterministic PCA denoise, optional scale-decorrelation, then L2-normalise.

    Returns (reduced points, n_dropped). On real fisheye footage the dominant axis of variation in
    SigLIP crop embeddings is *crop scale* (near players are large and detailed, far players small
    and blurry) — measured at ~0.86 correlation between the top PC and crop area — so a naive split
    separates near-vs-far rather than the two kits. When per-track `sizes` are supplied we drop the
    principal components whose score correlates with log-size above `decorr_threshold`, removing that
    confound. This is safe for the easy case (a vivid pinnie kit dominates its own PC, which does not
    correlate with size, so nothing kit-relevant is dropped); it only bites when scale would
    otherwise hijack the split. At least two components are always kept.
    """
    from sklearn.decomposition import PCA

    n, d = track_emb.shape
    k = max(1, min(n_components, n - 1, d))
    # svd_solver='full' is exact/deterministic (n is tiny, so this is cheap) — no randomised SVD,
    # so the reduction does not wobble between runs the way UMAP did.
    scores = PCA(n_components=k, svd_solver="full", random_state=random_state).fit_transform(
        track_emb
    )
    n_dropped = 0
    if sizes is not None and n >= 4 and scores.shape[1] >= 3:
        lsz = np.log(np.asarray(sizes, dtype=np.float64) + 1e-6)
        corr = np.array([abs(_safe_corr(scores[:, i], lsz)) for i in range(scores.shape[1])])
        keep = corr < decorr_threshold
        if keep.sum() >= 2:
            n_dropped = int((~keep).sum())
            scores = scores[:, keep]
        else:  # almost everything tracks size — keep the two least size-correlated components
            order = np.argsort(corr)
            n_dropped = scores.shape[1] - 2
            scores = scores[:, order[:2]]
    return _l2norm(scores), n_dropped


def _merge_micro_to_two(reduced: np.ndarray, micro_labels: np.ndarray) -> np.ndarray:
    """Merge K>=2 micro-clusters into two teams, size-first.

    The two *largest* micro-clusters are the team anchors (the skater groups); every smaller
    outlier cluster (goalies, refs, boards-huggers) folds into whichever anchor its centroid is
    most cosine-similar to. This is the fix for the degenerate split: a small, visually distinct
    goalie cluster attaches to a team instead of being allowed to *become* one.
    """
    micro_ids, counts = np.unique(micro_labels, return_counts=True)
    centroids = _l2norm(
        np.vstack([reduced[micro_labels == m].mean(axis=0) for m in micro_ids])
    )
    # Anchors: the two biggest micro-clusters (ties broken by lower micro id for determinism).
    order = sorted(range(len(micro_ids)), key=lambda i: (-counts[i], micro_ids[i]))
    anchor_a, anchor_b = order[0], order[1]

    team_of_micro: dict[int, int] = {}
    for i, m in enumerate(micro_ids):
        if i == anchor_a:
            team_of_micro[m] = 0
        elif i == anchor_b:
            team_of_micro[m] = 1
        else:
            sim_a = float(centroids[i] @ centroids[anchor_a])
            sim_b = float(centroids[i] @ centroids[anchor_b])
            team_of_micro[m] = 0 if sim_a >= sim_b else 1
    return np.array([team_of_micro[m] for m in micro_labels], dtype=int)


def cluster_team_embeddings(
    track_emb: np.ndarray,
    *,
    sizes: np.ndarray | None = None,
    max_micro: int = 4,
    n_components: int = 16,
    random_state: int = 42,
    silhouette_margin: float = 0.03,
) -> tuple[np.ndarray, ClusterInfo]:
    """Cluster per-track embeddings into two teams. Returns (labels, ClusterInfo).

    Pipeline: deterministic PCA (with optional scale-decorrelation if per-track `sizes` are given,
    to stop crop near/far scale from hijacking the split) -> choose K in [2, max_micro] micro-clusters
    by silhouette (preferring smaller K unless a larger one is clearly cleaner) -> merge to two teams
    by size. Fully deterministic given the input row order, so repeated runs on the same crops are
    identical.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(track_emb)
    if n == 0:
        return np.empty((0,), dtype=int), ClusterInfo(0.0, (0, 0), 0)
    if n == 1:
        return np.zeros(1, dtype=int), ClusterInfo(0.0, (1, 0), 1)
    if n == 2:
        labels = np.array([0, 1], dtype=int)
        return labels, ClusterInfo(0.0, (1, 1), 2)

    reduced, _ = _reduce(track_emb, n_components, random_state, sizes=sizes)

    # Over-segment: try K micro-clusters and keep the cleanest by silhouette, biased toward the
    # smallest K (so two genuine teams stay K=2 and we only peel outliers off when it clearly helps).
    best_k, best_labels, best_sil = 2, None, -1.0
    for k in range(2, min(max_micro, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels_k = km.fit_predict(reduced)
        if len(np.unique(labels_k)) < k:  # degenerate (empty cluster) — skip
            continue
        sil = float(silhouette_score(reduced, labels_k))
        if best_labels is None or sil > best_sil + silhouette_margin:
            best_k, best_labels, best_sil = k, labels_k, sil
    if best_labels is None:  # silhouette never computable — fall back to a plain 2-way split
        best_labels = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(reduced)
        best_k = 2

    labels = _merge_micro_to_two(reduced, best_labels)

    # Quality of the *final* two-team labelling (what downstream stats actually use).
    if len(np.unique(labels)) == 2:
        final_sil = float(silhouette_score(reduced, labels))
        centroids = _l2norm(np.vstack([reduced[labels == t].mean(axis=0) for t in (0, 1)]))
    else:
        final_sil = 0.0
        centroids = _l2norm(reduced.mean(axis=0, keepdims=True))
    sizes = (int((labels == 0).sum()), int((labels == 1).sum()))
    return labels, ClusterInfo(final_sil, sizes, best_k, reduced, centroids)


def team_confidence(info: ClusterInfo, labels: np.ndarray) -> np.ndarray:
    """Per-track confidence in [0, 1]: how much closer a track is to its team than the other.

    Margin = (cos to own centroid - cos to other centroid) mapped from [-2, 2] to [0, 1]. Low
    values flag borderline tracks (blurry crops, ambiguous kit) without needing any labels.
    """
    if info.centroids.shape[0] < 2 or len(labels) == 0:
        return np.full(len(labels), 0.5)
    sim0 = info.reduced @ info.centroids[0]
    sim1 = info.reduced @ info.centroids[1]
    own = np.where(labels == 0, sim0, sim1)
    other = np.where(labels == 0, sim1, sim0)
    return np.clip((own - other + 2.0) / 4.0, 0.0, 1.0)


# --------------------------------------------------------------------------------------------
# Stable team labels from kit colour (anchors T0/T1 to something physical)
# --------------------------------------------------------------------------------------------

def torso_color_hsv(crop: np.ndarray) -> np.ndarray:
    """Mean HSV of a crop's torso region (upper-centre), where the kit colour is clearest.

    Returns [hue (0-179), saturation (0-255), value (0-255)]. The torso window avoids the head and
    legs/floor, which carry less kit signal. Pure numpy so it needs no OpenCV.
    """
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        region = crop
    else:
        region = crop[int(0.15 * h) : int(0.55 * h), int(0.2 * w) : int(0.8 * w)]
        if region.size == 0:
            region = crop
    return _bgr_to_hsv_mean(region)


def _bgr_to_hsv_mean(region: np.ndarray) -> np.ndarray:
    """Mean HSV (OpenCV ranges: H 0-179, S/V 0-255) of a BGR region, without an OpenCV dependency."""
    bgr = region.reshape(-1, 3).astype(np.float64) / 255.0
    b, g, r = bgr[:, 0], bgr[:, 1], bgr[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    val = mx
    sat = np.where(mx > 1e-8, diff / (mx + 1e-8), 0.0)
    hue = np.zeros_like(mx)
    nz = diff > 1e-8
    # Standard piecewise hue, in degrees [0, 360).
    rmax = nz & (mx == r)
    gmax = nz & (mx == g) & ~rmax
    bmax = nz & (mx == b) & ~rmax & ~gmax
    hue[rmax] = (60 * ((g[rmax] - b[rmax]) / diff[rmax]) + 360) % 360
    hue[gmax] = 60 * ((b[gmax] - r[gmax]) / diff[gmax]) + 120
    hue[bmax] = 60 * ((r[bmax] - g[bmax]) / diff[bmax]) + 240
    return np.array([hue.mean() / 2.0, sat.mean() * 255.0, val.mean() * 255.0])


def order_labels_by_color(
    labels: np.ndarray, colors: np.ndarray
) -> dict[int, int]:
    """Return a {raw_label -> stable_team} remap anchoring T0 to the more saturated (e.g. pinnie) kit.

    KMeans cluster ids are arbitrary and the sampled crop set shifts run to run, so the raw label
    of a team is not stable. We anchor it to a physical signal: the team whose tracks have the
    higher median saturation becomes T0 (ties broken by brightness, then hue). With two kits where
    one is a coloured pinnie and the other plain, T0 is consistently the pinnie team across runs.
    """
    uniq = sorted(set(int(x) for x in labels))
    if len(uniq) < 2:
        return {lab: 0 for lab in uniq}

    def key(lab: int) -> tuple[float, float, float]:
        sel = colors[labels == lab]
        med = np.median(sel, axis=0)  # [hue, sat, val]
        # Sort descending by saturation, then brightness, then hue -> negate for ascending sort.
        return (-med[1], -med[2], med[0])

    ranked = sorted(uniq, key=key)
    return {lab: team for team, lab in enumerate(ranked)}


# --------------------------------------------------------------------------------------------
# Kit-colour prior: when one team wears a vivid kit (pinnies), colour beats SigLIP outright
# --------------------------------------------------------------------------------------------
#
# Real-footage validation showed SigLIP embeddings split by crop scale and don't separate
# low-contrast (white-vs-dark) kits. But the project's common case is "pinnies vs none" — a vivid
# kit on one team. There, a colour split is far stronger *and* immune to the scale confound. So we
# compute a robust per-track kit chroma and, when a coherent vivid group exists, split on colour;
# otherwise we fall back to the embedding path. The chroma is background-suppressed per crop (the
# blue rink would otherwise dominate the hue) by dropping pixels near the crop *border* hue, which
# is almost always the rink/surroundings the player stands against.


def torso_kit_chroma(crop: np.ndarray) -> np.ndarray:
    """Background-suppressed torso chroma vector [cx, cy] (size-invariant kit-colour signal).

    cx/cy are the saturation-weighted mean of the torso pixels' hue direction, after removing
    pixels whose hue matches the crop border (the rink/background). A vivid coherent kit gives a
    large vector in a consistent direction; a plain white/dark kit gives a near-zero vector.
    """
    h, w = crop.shape[:2]
    region = crop[int(0.15 * h):int(0.55 * h), int(0.20 * w):int(0.80 * w)] if (h >= 6 and w >= 6) else crop
    if region.size == 0:
        region = crop
    hsv_full = cv2_cvt_hsv(crop)
    border = np.concatenate([hsv_full[0], hsv_full[-1], hsv_full[:, 0], hsv_full[:, -1]])
    border = border.reshape(-1, 3).astype(np.float64)
    bg_hue = float(np.median(border[:, 0]))
    rh = cv2_cvt_hsv(region).reshape(-1, 3).astype(np.float64)
    hue, sat = rh[:, 0], rh[:, 1] / 255.0
    dh = np.minimum(np.abs(hue - bg_hue), 180.0 - np.abs(hue - bg_hue))
    keep = ~((dh < 12.0) & (sat > 0.25))         # drop rink-like (near-border hue & saturated)
    if keep.sum() < 10:
        keep = np.ones_like(keep, dtype=bool)
    ang = hue[keep] * 2.0 * np.pi / 180.0         # OpenCV hue is 0-179 == degrees/2
    return np.array([np.mean(sat[keep] * np.cos(ang)), np.mean(sat[keep] * np.sin(ang))])


def cv2_cvt_hsv(bgr: np.ndarray) -> np.ndarray:
    """BGR->HSV via OpenCV if available, else a numpy fallback (keeps the core import-light)."""
    try:
        import cv2
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    except Exception:  # pragma: no cover - exercised only without OpenCV
        flat = _bgr_to_hsv_array(bgr.reshape(-1, 3))
        return flat.reshape(bgr.shape)


def _bgr_to_hsv_array(bgr: np.ndarray) -> np.ndarray:
    """Vectorised BGR->HSV in OpenCV ranges (H 0-179, S/V 0-255) for an (N,3) uint8 array."""
    x = bgr.astype(np.float64) / 255.0
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    hue = np.zeros_like(mx)
    nz = diff > 1e-8
    rm = nz & (mx == r); gm = nz & (mx == g) & ~rm; bm = nz & (mx == b) & ~rm & ~gm
    hue[rm] = (60 * ((g[rm] - b[rm]) / diff[rm]) + 360) % 360
    hue[gm] = 60 * ((b[gm] - r[gm]) / diff[gm]) + 120
    hue[bm] = 60 * ((r[bm] - g[bm]) / diff[bm]) + 240
    sat = np.where(mx > 1e-8, diff / (mx + 1e-8), 0.0)
    return np.stack([hue / 2.0, sat * 255.0, mx * 255.0], axis=1)


def track_kit_chroma(crops: list[np.ndarray]) -> np.ndarray:
    """Median torso chroma over a track's crops -> one [cx, cy] kit vector per track."""
    if not crops:
        return np.zeros(2)
    return np.median(np.vstack([torso_kit_chroma(c) for c in crops]), axis=0)


def background_suppressed_crop(
    crop: np.ndarray,
    *,
    neutral: int = 128,
    hue_tol: float = 12.0,
    sat_floor: float = 0.25,
    min_keep: int = 30,
) -> np.ndarray:
    """Return a torso crop with rink/background pixels neutralised, for SigLIP embedding.

    The embedding path otherwise keys on the bright blue rink (and legs/skin) rather than the kit —
    on the reference fisheye footage the top principal component correlated 0.86 with crop scale,
    i.e. near-vs-far *background detail*, not the white-vs-dark shirts (docs/team-clustering.md). We
    (1) crop to the torso window (drops head, legs, and most floor) and (2) replace pixels whose hue
    matches the crop *border* — the rink/surroundings the player stands against — and are saturated,
    with a flat neutral grey, so SigLIP sees the kit against a uniform field instead of the rink.

    This mirrors the background suppression already used by ``torso_kit_chroma`` for the colour
    prior, now applied *before* embedding (the lever logged as priority #2 in docs/feature-ideas.md).
    Low-saturation kit pixels (white/dark shirts) survive — only saturated rink-coloured pixels are
    neutralised — and it falls back to the unmasked torso crop if suppression would remove almost
    everything (a tiny/far crop that is mostly background).
    """
    h, w = crop.shape[:2]
    if h < 6 or w < 6:
        return crop
    hsv_full = cv2_cvt_hsv(crop)
    border = np.concatenate([hsv_full[0], hsv_full[-1], hsv_full[:, 0], hsv_full[:, -1]])
    bg_hue = float(np.median(border.reshape(-1, 3)[:, 0]))
    r0, r1, c0, c1 = int(0.15 * h), int(0.55 * h), int(0.20 * w), int(0.80 * w)
    torso = crop[r0:r1, c0:c1].copy()
    if torso.size == 0:
        return crop
    thsv = hsv_full[r0:r1, c0:c1].reshape(-1, 3).astype(np.float64)
    hue, sat = thsv[:, 0], thsv[:, 1] / 255.0
    dh = np.minimum(np.abs(hue - bg_hue), 180.0 - np.abs(hue - bg_hue))
    bg = (dh < hue_tol) & (sat > sat_floor)            # rink-coloured & saturated -> background
    if int((~bg).sum()) < min_keep:                    # almost all background -> don't gut the crop
        return torso
    flat = torso.reshape(-1, 3)
    flat[bg] = neutral
    return flat.reshape(torso.shape)


def _euclid_margin_conf(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Per-point confidence in [0,1] from the relative distance to own vs other centroid."""
    d0 = np.linalg.norm(points - centroids[0], axis=1)
    d1 = np.linalg.norm(points - centroids[1], axis=1)
    own = np.where(labels == 0, d0, d1)
    other = np.where(labels == 0, d1, d0)
    return np.clip((other - own) / (other + own + 1e-9) * 0.5 + 0.5, 0.0, 1.0)


def detect_kit_split(
    chroma: np.ndarray,
    *,
    vivid_bar: float = 0.35,
    plain_bar: float = 0.22,
    coherence_bar: float = 0.6,
    min_frac: float = 0.15,
) -> tuple[np.ndarray, ClusterInfo] | None:
    """Split tracks into two teams by kit colour iff a coherent *vivid* kit group exists.

    Returns (labels, ClusterInfo with method='kit-color') when accepted, else None (caller falls
    back to embeddings). The structure we accept is one *vivid* team vs one *plain* team, so we
    require the more-saturated cluster to be genuinely vivid (`vivid_bar`) and hue-coherent
    (`coherence_bar`), the other cluster to be near-neutral (`plain_bar`, which rejects two
    competing colourful groups / scattered accessories), and a non-degenerate share of tracks
    (`min_frac`). Validated: real white/dark footage falls back; a pinnie team is cleanly isolated.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(chroma)
    if n < 6:
        return None
    mags = np.linalg.norm(chroma, axis=1)
    if np.std(mags) < 1e-6:
        return None
    labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(chroma)
    if len(np.unique(labels)) < 2:
        return None
    vivid = 0 if mags[labels == 0].mean() >= mags[labels == 1].mean() else 1
    vmask = labels == vivid
    vivid_mag = float(mags[vmask].mean())
    plain_mag = float(mags[~vmask].mean())
    units = chroma[vmask] / (mags[vmask][:, None] + 1e-9)
    coherence = float(np.linalg.norm(units.mean(axis=0)))
    balance = float(min(vmask.mean(), (~vmask).mean()))
    if not (vivid_mag >= vivid_bar and plain_mag <= plain_bar and coherence >= coherence_bar
            and balance >= min_frac and vmask.sum() >= 2 and (~vmask).sum() >= 2):
        return None

    sil = float(silhouette_score(chroma, labels)) if n > 2 else 0.0
    centroids = np.vstack([chroma[labels == 0].mean(axis=0), chroma[labels == 1].mean(axis=0)])
    info = ClusterInfo(sil, (int((labels == 0).sum()), int((labels == 1).sum())), 2,
                       chroma, centroids, method="kit-color")
    info.conf = _euclid_margin_conf(chroma, labels, centroids)
    return labels, info


# --------------------------------------------------------------------------------------------
# Shared embedding pass (one mean SigLIP embedding per track) — reused by team + identity
# --------------------------------------------------------------------------------------------

def embed_tracks(
    embedder: SiglipTeamClassifier,
    track_crops: dict[int, list[np.ndarray]],
    *,
    suppress_background: bool = True,
) -> tuple[list[int], np.ndarray]:
    """Embed a track's sampled crops and pool them into one mean embedding per track.

    Returns ``(present_ids, track_emb)`` where ``track_emb`` rows align to ``present_ids`` (the
    track ids that had at least one crop). With ``suppress_background`` each crop is torso-cropped
    and its rink-coloured background neutralised before embedding, so SigLIP keys on the kit, not the
    blue rink. Factored out so the pipeline runs the (expensive) SigLIP pass *once* and shares the
    per-track embeddings between team clustering and Phase 3 identity re-ID.
    """
    present_ids = [t for t in sorted(track_crops) if track_crops.get(t)]
    flat_crops: list[np.ndarray] = []
    owners: list[int] = []
    for tid in present_ids:
        for crop in track_crops[tid]:
            flat_crops.append(crop)
            owners.append(tid)
    if not flat_crops:
        return [], np.empty((0, EMBED_DIM), dtype=np.float64)
    embed_crops = (
        [background_suppressed_crop(c) for c in flat_crops] if suppress_background else flat_crops
    )
    crop_emb = embedder.embed(embed_crops)
    return present_ids, aggregate_track_embeddings(crop_emb, owners, present_ids)


# --------------------------------------------------------------------------------------------
# Orchestration: crops -> per-track team assignment (used by the pipeline)
# --------------------------------------------------------------------------------------------

@dataclass
class TeamAssignment:
    track_team: dict[int, int]              # track_id -> stable team id (0/1)
    track_conf: dict[int, float]            # track_id -> confidence in [0, 1]
    info: ClusterInfo

    @property
    def team_sizes(self) -> tuple[int, int]:
        return self.info.team_sizes

    @property
    def silhouette(self) -> float:
        return self.info.silhouette


def assign_teams(
    embedder: SiglipTeamClassifier,
    track_crops: dict[int, list[np.ndarray]],
    *,
    suppress_background: bool = True,
    precomputed: tuple[list[int], np.ndarray] | None = None,
) -> TeamAssignment:
    """Assign one stable team id per track from its sampled crops.

    Two label-free paths, auto-selected:
      - **Kit-colour prior** (preferred when it fires): if a coherent *vivid* kit group exists
        (e.g. pinnies), split on background-suppressed torso chroma. Strong, scale-immune, and
        skips the SigLIP embedding entirely — much cheaper.
      - **Embedding fallback**: otherwise embed crops -> pool per track -> scale-decorrelated
        clustering (over-segment + size-merge). With ``suppress_background`` (default), each crop is
        torso-cropped and its rink-coloured background neutralised before embedding, so SigLIP keys
        on the kit rather than the blue rink / legs / skin (docs/feature-ideas.md priority #2).
    Either way T0/T1 are anchored to kit colour and a per-track confidence is reported.

    ``precomputed`` optionally supplies ``(present_ids, track_emb)`` from a shared
    ``embed_tracks`` pass (e.g. when Phase 3 identity also runs), so SigLIP is paid for once. It is
    only consulted on the embedding path — the kit-colour prior still wins when it fires.
    """
    track_ids = sorted(track_crops)
    colors_per_track: list[np.ndarray] = []
    sizes_per_track: list[float] = []
    chroma_per_track: list[np.ndarray] = []
    for tid in track_ids:
        crops = track_crops[tid]
        if not crops:
            continue
        colors_per_track.append(np.median([torso_color_hsv(c) for c in crops], axis=0))
        # Crop pixel area is our scale proxy: the near/far confound we decorrelate the split from.
        sizes_per_track.append(float(np.median([c.shape[0] * c.shape[1] for c in crops])))
        chroma_per_track.append(track_kit_chroma(crops))

    present_ids = [t for t in track_ids if track_crops.get(t)]
    if not present_ids:
        return TeamAssignment({}, {}, ClusterInfo(0.0, (0, 0), 0))

    # Cheap colour prior first — if a vivid kit team is present, we never pay for SigLIP.
    kit = detect_kit_split(np.vstack(chroma_per_track))
    if kit is not None:
        labels, info = kit
        conf = info.conf
    else:
        if precomputed is not None:
            present_ids, track_emb = precomputed
        else:
            present_ids, track_emb = embed_tracks(
                embedder, track_crops, suppress_background=suppress_background
            )
        labels, info = cluster_team_embeddings(track_emb, sizes=np.array(sizes_per_track))
        conf = team_confidence(info, labels)

    remap = order_labels_by_color(labels, np.vstack(colors_per_track))
    stable = np.array([remap[int(l)] for l in labels], dtype=int)
    # Re-derive sizes/silhouette stay valid under the relabel (it's just a permutation of 0/1).
    info.team_sizes = (int((stable == 0).sum()), int((stable == 1).sum()))

    track_team = {tid: int(stable[i]) for i, tid in enumerate(present_ids)}
    track_conf = {tid: float(conf[i]) for i, tid in enumerate(present_ids)}
    return TeamAssignment(track_team, track_conf, info)


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
