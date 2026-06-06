"""Phase 3 appearance re-identification — stitch fragmented tracks into per-player identities.

The headline value unlock of the project. Detection + ByteTrack give a *track id* that is stable
only within one continuous on-surface stretch: a player who leaves the frame, is occluded, or whom
the tracker simply loses comes back as a **new** track id. On the reference footage ~13 people
produced ~27+ player tracks (and far more across a full game). Every per-track stat — time on
surface, shifts, +/- — is therefore fragmented across several ids for the same person.

This module clusters the per-track appearance signatures into ~roster-size **identities**, so the
fragments of one person fold back together into a single identity → true per-player time-on-surface
and a shift count (one contiguous track fragment ≈ one shift).

It reuses the *same* embedding machinery as team clustering (per-track mean SigLIP embedding on
background-suppressed crops — ``team_siglip.embed_tracks``), just at a finer granularity: team is a
coarse 2-way split by kit, identity is a fine ~K-way split by each player's individual gear
(shirt + shorts + socks + helmet + build + skin tone), which is consistent *within a game*.

What makes identity tractable where naive appearance clustering would fail is a hard, reliable
**spatiotemporal constraint**: a person cannot be in two places at once, so two tracks whose frame
spans *overlap in time* cannot be the same identity. We enforce that as a **cannot-link**
constraint in a constrained agglomerative clusterer. This does two things:

1. It stops two different players who happen to wear similar gear (the documented failure mode) from
   collapsing into one identity whenever they are on the surface together.
2. The maximum number of players simultaneously on the surface becomes a natural *floor* on the
   identity count — in 5-on-5 + goalies that is ~12, very close to the true roster — so the
   clusterer lands near the right number of people even without being told the roster size.

The clustering core (``temporal_overlap_matrix``, ``constrained_agglomerative``,
``cluster_identities``, ``identity_confidence``) is pure numpy + stdlib, so it is unit-testable on
synthetic embeddings + frame spans with no torch, no model, and no video — same discipline as the
team-clustering core in ``team_siglip``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dbh_vibes.team_siglip import _l2norm


@dataclass
class IdentityInfo:
    """Label-free quality signal for an identity clustering, so it can be judged without truth.

    n_identities: how many distinct people the tracks were stitched into.
    sizes: per-identity track-fragment counts (a person with many fragments was re-acquired often).
    silhouette: separation of the final labelling in reduced space ([-1, 1]; higher is cleaner).
        ``nan`` when undefined (a single identity, or one fragment each).
    n_blocked_merges: merges the temporal cannot-link constraint vetoed — how often appearance
        alone *would* have fused two people who were on the surface at the same time.
    """

    n_identities: int
    sizes: list[int]
    silhouette: float
    n_blocked_merges: int = 0
    method: str = "agglomerative"
    reduced: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    centroids: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    conf: np.ndarray | None = field(repr=False, default=None)


def temporal_overlap_matrix(spans: list[tuple[int, int]], min_gap: int = 0) -> np.ndarray:
    """Boolean (n×n) cannot-link matrix: ``True`` where tracks i,j overlap in time.

    ``spans[i] = (first_frame, last_frame)`` inclusive. Two tracks conflict — cannot be the same
    person — when their frame ranges intersect. ``min_gap`` additionally forbids merging tracks
    separated by *fewer* than ``min_gap`` frames (a real player needs a moment to leave and
    re-enter; two ids that hand off within a frame or two are usually one person mid-track-switch,
    but if you want to be conservative about near-simultaneous ids raise this). The diagonal is
    ``False`` (a track never conflicts with itself).
    """
    n = len(spans)
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(n):
        fi, li = spans[i]
        for j in range(i + 1, n):
            fj, lj = spans[j]
            # Overlap (with a min_gap cushion) iff the ranges, each widened by min_gap, intersect.
            if max(fi, fj) <= min(li, lj) + min_gap:
                conflict[i, j] = conflict[j, i] = True
    return conflict


def _reduce_identity(
    track_emb: np.ndarray, n_components: int, random_state: int
) -> np.ndarray:
    """Deterministic PCA denoise + L2-normalise, for the fine identity split.

    Identity needs to preserve *more* structure than the 2-team split (every player is its own
    direction), so we keep more components than team clustering. PCA with ``svd_solver='full'`` is
    exact and deterministic, so repeated runs on the same embeddings give identical identities.
    """
    import warnings

    from sklearn.decomposition import PCA

    n, d = track_emb.shape
    k = max(1, min(n_components, n - 1, d))
    if k <= 0 or n <= 1:
        return _l2norm(track_emb.astype(np.float64))
    with warnings.catch_warnings():
        # Degenerate inputs (e.g. identical embeddings) make explained-variance 0/0; harmless here.
        warnings.simplefilter("ignore", RuntimeWarning)
        scores = PCA(n_components=k, svd_solver="full", random_state=random_state).fit_transform(
            track_emb
        )
    return _l2norm(scores)


def constrained_agglomerative(
    points: np.ndarray,
    cannot_link: np.ndarray,
    *,
    n_identities: int | None = None,
    distance_threshold: float = 0.35,
) -> tuple[np.ndarray, int]:
    """Average-linkage agglomerative clustering with hard cannot-link constraints.

    Greedily merges the two closest clusters whose members contain **no** cannot-link pair, using
    average-linkage cosine distance (``1 - cos`` on the already-L2-normalised ``points``). Stops at
    ``n_identities`` clusters when given; otherwise stops when the closest *permissible* merge is
    farther than ``distance_threshold`` (a data-driven identity count). The cannot-link veto is what
    keeps two simultaneously-on-surface players apart even if their gear looks alike.

    Returns ``(labels, n_blocked_merges)`` where ``labels`` are contiguous ids ``0..K-1`` and
    ``n_blocked_merges`` counts how often the closest candidate merge was vetoed by a constraint.
    """
    n = len(points)
    if n == 0:
        return np.empty((0,), dtype=int), 0
    if n == 1:
        return np.zeros(1, dtype=int), 0

    # Pairwise cosine distance (points are unit vectors → 1 - dot). Clip tiny negatives from fp.
    dist = np.clip(1.0 - points @ points.T, 0.0, 2.0)

    members: list[list[int]] = [[i] for i in range(n)]
    alive = list(range(n))
    blocked = 0

    def clusters_conflict(a: int, b: int) -> bool:
        # Any member pair across the two clusters that cannot be linked vetoes the whole merge.
        return bool(cannot_link[np.ix_(members[a], members[b])].any())

    def avg_linkage(a: int, b: int) -> float:
        return float(dist[np.ix_(members[a], members[b])].mean())

    target = n_identities if n_identities is not None else 1
    while len(alive) > target:
        # Find the closest permissible pair; track whether the *global* closest was blocked.
        best_pair: tuple[int, int] | None = None
        best_d = np.inf
        global_best_d = np.inf
        global_blocked = False
        for ii in range(len(alive)):
            for jj in range(ii + 1, len(alive)):
                a, b = alive[ii], alive[jj]
                d = avg_linkage(a, b)
                conflict = clusters_conflict(a, b)
                if d < global_best_d:
                    global_best_d, global_blocked = d, conflict
                if not conflict and d < best_d:
                    best_d, best_pair = d, (a, b)
        if global_blocked:
            blocked += 1
        if best_pair is None:
            break  # every remaining merge is constraint-blocked → can't reduce further
        if n_identities is None and best_d > distance_threshold:
            break  # nearest permissible identities are too far apart → stop merging
        a, b = best_pair
        members[a].extend(members[b])
        members[b] = []
        alive.remove(b)

    labels = np.empty(n, dtype=int)
    for new_id, c in enumerate(alive):
        for m in members[c]:
            labels[m] = new_id
    return labels, blocked


def identity_confidence(
    reduced: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Per-track confidence in [0,1]: how much closer a fragment sits to its own identity centroid.

    Margin = (cos to own centroid − cos to nearest *other* centroid), mapped to [0,1]. A singleton
    identity (its own and only centroid) gets a neutral 0.5 — there is nothing to be confident
    against. Flags fragments that sit between two people (a blurry/ambiguous crop) without labels.
    """
    n = len(labels)
    if n == 0 or centroids.shape[0] == 0:
        return np.full(n, 0.5)
    sims = reduced @ centroids.T                       # (n, K) cos to every identity centroid
    out = np.full(n, 0.5)
    for i in range(n):
        own = sims[i, labels[i]]
        others = np.delete(sims[i], labels[i])
        if others.size == 0:
            continue
        out[i] = float(np.clip((own - others.max()) * 0.5 + 0.5, 0.0, 1.0))
    return out


def cluster_identities(
    track_emb: np.ndarray,
    spans: list[tuple[int, int]],
    *,
    n_identities: int | None = None,
    distance_threshold: float = 0.35,
    min_gap: int = 0,
    n_components: int = 24,
    random_state: int = 42,
) -> tuple[np.ndarray, IdentityInfo]:
    """Cluster per-track embeddings into per-player identities. Returns ``(labels, IdentityInfo)``.

    Pipeline: deterministic PCA denoise → constrained agglomerative clustering with a temporal
    cannot-link constraint built from the track frame ``spans``. Fully deterministic given the input
    row order, so repeated runs on the same tracks produce identical identities.

    ``n_identities`` pins the roster size if you know it; otherwise the count is data-driven from
    ``distance_threshold`` (and floored by the max number of mutually-overlapping tracks). Rows of
    ``track_emb`` and entries of ``spans`` must be aligned to the same track order.
    """
    n = len(track_emb)
    if n == 0:
        return np.empty((0,), dtype=int), IdentityInfo(0, [], float("nan"))
    if n == 1:
        return np.zeros(1, dtype=int), IdentityInfo(1, [1], float("nan"))

    reduced = _reduce_identity(track_emb, n_components, random_state)
    conflict = temporal_overlap_matrix(spans, min_gap=min_gap)
    labels, blocked = constrained_agglomerative(
        reduced, conflict, n_identities=n_identities, distance_threshold=distance_threshold
    )

    ids, counts = np.unique(labels, return_counts=True)
    centroids = _l2norm(np.vstack([reduced[labels == k].mean(axis=0) for k in ids]))
    # Relabel so identity ids are contiguous 0..K-1 in centroid order (already are, but be explicit).
    sil = _silhouette(reduced, labels)
    info = IdentityInfo(
        n_identities=len(ids),
        sizes=[int(c) for c in counts],
        silhouette=sil,
        n_blocked_merges=blocked,
        reduced=reduced,
        centroids=centroids,
    )
    info.conf = identity_confidence(reduced, labels, centroids)
    return labels, info


def _silhouette(reduced: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette of the identity labelling, or ``nan`` when it is undefined (need 2..n-1 clusters)."""
    k = len(np.unique(labels))
    if k < 2 or k >= len(labels):
        return float("nan")
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(reduced, labels))
    except Exception:  # pragma: no cover - silhouette undefined for degenerate inputs
        return float("nan")


# --------------------------------------------------------------------------------------------
# Orchestration: crops -> per-track identity assignment (used by the pipeline)
# --------------------------------------------------------------------------------------------

@dataclass
class IdentityAssignment:
    track_identity: dict[int, int]          # track_id -> identity id (0..K-1), stable per person
    track_conf: dict[int, float]            # track_id -> confidence in [0, 1]
    info: IdentityInfo

    @property
    def n_identities(self) -> int:
        return self.info.n_identities


def assign_identities(
    track_emb: np.ndarray,
    present_ids: list[int],
    spans: dict[int, tuple[int, int]],
    *,
    n_identities: int | None = None,
    distance_threshold: float = 0.35,
    min_gap: int = 0,
) -> IdentityAssignment:
    """Assign one identity id per track from precomputed per-track embeddings + frame spans.

    ``track_emb`` rows align to ``present_ids`` (as returned by ``team_siglip.embed_tracks``);
    ``spans[tid]`` is each track's ``(first_frame, last_frame)``. Embeddings are precomputed so the
    pipeline can share the single SigLIP pass with team classification rather than paying for it
    twice.
    """
    if len(present_ids) == 0:
        return IdentityAssignment({}, {}, IdentityInfo(0, [], float("nan")))
    span_list = [spans[t] for t in present_ids]
    labels, info = cluster_identities(
        track_emb, span_list, n_identities=n_identities,
        distance_threshold=distance_threshold, min_gap=min_gap,
    )
    conf = info.conf if info.conf is not None else np.full(len(present_ids), 0.5)
    track_identity = {tid: int(labels[i]) for i, tid in enumerate(present_ids)}
    track_conf = {tid: float(conf[i]) for i, tid in enumerate(present_ids)}
    return IdentityAssignment(track_identity, track_conf, info)
