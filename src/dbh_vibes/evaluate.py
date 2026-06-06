"""Labeled-set evaluation harness — priority #1, "the binding constraint".

Team clustering was hardened to be **stable** (deterministic, 100% run-to-run agreement on real
footage), but its true **accuracy** stayed *unmeasurable*: with no ground truth we could only
report label-free internal signals (silhouette, balance, scale-decorrelation, kit-accuracy on
*tinted* crops) — never "what fraction of players got the right team on natural footage". As
docs/team-clustering.md put it, we "still can't tell a good split from a plausible-looking bad
one." That is the binding constraint on iterating everything downstream.

This module closes the gap with the cheapest thing that works: a tiny **per-track label file**
(the pipeline exports a pre-filled template — see ``labeling.py``), and a harness that scores the
pipeline's predictions against it. Two things make the scoring honest:

* **Cluster labels are arbitrary.** The pipeline calls a team ``0``/``1`` (anchored to kit colour,
  but still just an id), and a human labels teams however they like ("white"/"dark", "A"/"B").
  So team accuracy is computed under the **optimal label alignment** between predicted clusters and
  true classes (the standard clustering-accuracy / Hungarian match), not naive equality.
* **Only the overlap is scored.** Tracks that are unlabeled, or labeled but absent from the
  prediction, are reported separately rather than silently counted as wrong.

The metric core (``best_map_accuracy``, ``confusion_matrix``, the contingency helpers) is pure
numpy + stdlib, so it is unit-testable without any video, model, or labeled data. The same
``best_map_accuracy`` generalises from the 2-team split to the many-cluster **identity** accuracy
Phase 3 will need, so the harness grows with the project rather than being team-only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import numpy as np

# Sentinel for "this track has no value in this column" (blank cell in the label/prediction file).
_UNSET = ""


# --------------------------------------------------------------------------------------------
# Pure metric core (numpy + stdlib only — no video, no model, no labeled data needed)
# --------------------------------------------------------------------------------------------

def contingency(
    pred: list, true: list
) -> tuple[np.ndarray, list, list]:
    """Build the (pred-label x true-label) co-occurrence count matrix.

    Returns (matrix, pred_labels, true_labels) with both label lists sorted for determinism, so
    ``matrix[i, j]`` is how many items got predicted label ``pred_labels[i]`` while truly being
    ``true_labels[j]``.
    """
    pred_labels = sorted(set(pred), key=str)
    true_labels = sorted(set(true), key=str)
    pi = {lab: i for i, lab in enumerate(pred_labels)}
    ti = {lab: i for i, lab in enumerate(true_labels)}
    m = np.zeros((len(pred_labels), len(true_labels)), dtype=int)
    for p, t in zip(pred, true):
        m[pi[p], ti[t]] += 1
    return m, pred_labels, true_labels


def _optimal_assignment(weight: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight one-to-one assignment of rows to columns. Returns [(row, col), ...].

    Uses scipy's Hungarian solver when available; otherwise brute-forces, which is exact and fast
    for the small label counts here (2 teams, roster-size identities). One-to-one matching is the
    standard definition of clustering accuracy — each predicted cluster maps to at most one true
    class — so an over-segmenting predictor can't claim credit by mapping two clusters to one class.
    """
    n_rows, n_cols = weight.shape
    rows = range(n_rows)
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(weight, maximize=True)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        # Exhaustive over the smaller dimension. k! with k<=~8 is trivial; guard larger cases.
        if min(n_rows, n_cols) > 8:
            # Greedy fallback (rarely hit; identity eval with many clusters). Pick best cell
            # repeatedly. Not provably optimal but a sane degrade, and flagged by the guard above.
            taken_r, taken_c, pairs = set(), set(), []
            for r, c in sorted(
                ((r, c) for r in range(n_rows) for c in range(n_cols)),
                key=lambda rc: -weight[rc[0], rc[1]],
            ):
                if r not in taken_r and c not in taken_c:
                    taken_r.add(r); taken_c.add(c); pairs.append((r, c))
            return pairs
        best_pairs, best_score = [], -1
        if n_cols <= n_rows:
            for cols in permutations(range(n_rows), n_cols):
                score = sum(weight[cols[j], j] for j in range(n_cols))
                if score > best_score:
                    best_score, best_pairs = score, [(cols[j], j) for j in range(n_cols)]
        else:
            for cols in permutations(range(n_cols), n_rows):
                score = sum(weight[r, cols[r]] for r in rows)
                if score > best_score:
                    best_score, best_pairs = score, [(r, cols[r]) for r in rows]
        return best_pairs


@dataclass
class MatchResult:
    """Outcome of scoring predicted cluster labels against true labels under optimal alignment."""

    accuracy: float
    n: int                                   # items scored (those with both a pred and a true label)
    n_correct: int
    mapping: dict                            # predicted label -> aligned true label
    pred_labels: list = field(default_factory=list)
    true_labels: list = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), int), repr=False)


def best_map_accuracy(pred: list, true: list) -> MatchResult:
    """Clustering accuracy under the best one-to-one map from predicted clusters to true classes.

    ``pred`` and ``true`` are aligned per item (same length). Predicted cluster ids are arbitrary,
    so we find the label map that maximises agreement (2 permutations for a 2-team split; Hungarian
    in general) and report the resulting accuracy. Items are assumed already filtered to those that
    have both a prediction and a truth value.
    """
    n = len(true)
    if n == 0:
        return MatchResult(0.0, 0, 0, {})
    matrix, pred_labels, true_labels = contingency(pred, true)
    pairs = _optimal_assignment(matrix.astype(float))
    mapping = {pred_labels[r]: true_labels[c] for r, c in pairs}
    n_correct = int(sum(matrix[r, c] for r, c in pairs))
    return MatchResult(
        accuracy=n_correct / n,
        n=n,
        n_correct=n_correct,
        mapping=mapping,
        pred_labels=pred_labels,
        true_labels=true_labels,
        matrix=matrix,
    )


def direct_accuracy(pred: list, true: list) -> MatchResult:
    """Accuracy when labels are *not* arbitrary (e.g. role: ``player``/``spectator`` mean the same
    in both files). Compared by direct equality, with a confusion matrix for the breakdown.
    """
    n = len(true)
    matrix, pred_labels, true_labels = contingency(pred, true)
    n_correct = int(sum(1 for p, t in zip(pred, true) if p == t))
    mapping = {lab: lab for lab in pred_labels}
    return MatchResult(
        accuracy=n_correct / n if n else 0.0,
        n=n, n_correct=n_correct, mapping=mapping,
        pred_labels=pred_labels, true_labels=true_labels, matrix=matrix,
    )


# --------------------------------------------------------------------------------------------
# I/O: read the prediction (tracks.csv) and the human labels, line them up by track id
# --------------------------------------------------------------------------------------------

def _read_csv_column(path: Path, value_col: str, key_col: str = "track_id") -> dict[int, str]:
    """Read ``{track_id: value}`` from a CSV, skipping rows whose value cell is blank.

    Tolerant of either file (predictions or labels): a missing column yields an empty mapping
    rather than an error, so e.g. evaluating ``team`` against a label file that only labeled roles
    just reports "0 labeled" instead of crashing.
    """
    out: dict[int, str] = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or value_col not in reader.fieldnames:
            return out
        for row in reader:
            raw = (row.get(value_col) or _UNSET).strip()
            key = (row.get(key_col) or _UNSET).strip()
            if raw == _UNSET or key == _UNSET:
                continue
            try:
                out[int(float(key))] = raw
            except ValueError:
                continue
    return out


@dataclass
class FieldEval:
    """Evaluation of one labeled field (team / role / player) against the prediction."""

    field: str
    result: MatchResult
    n_labeled: int                 # tracks the human labeled for this field
    n_predicted: int               # tracks the pipeline predicted a value for
    n_overlap: int                 # labeled AND predicted -> the ones actually scored
    arbitrary_labels: bool         # True => optimal alignment (team/identity); False => direct (role)
    missing_pred: list = field(default_factory=list)   # labeled but the pipeline gave no value


def evaluate_field(
    pred_by_track: dict[int, str],
    true_by_track: dict[int, str],
    field_name: str,
    arbitrary_labels: bool = True,
) -> FieldEval:
    """Score one field: align predictions and labels by track id, then accuracy on the overlap."""
    overlap = sorted(set(pred_by_track) & set(true_by_track))
    pred = [pred_by_track[t] for t in overlap]
    true = [true_by_track[t] for t in overlap]
    result = (best_map_accuracy if arbitrary_labels else direct_accuracy)(pred, true)
    missing = sorted(set(true_by_track) - set(pred_by_track))
    return FieldEval(
        field=field_name, result=result,
        n_labeled=len(true_by_track), n_predicted=len(pred_by_track),
        n_overlap=len(overlap), arbitrary_labels=arbitrary_labels, missing_pred=missing,
    )


# Which prediction column in tracks.csv each labelable field is scored against, and whether its
# labels are arbitrary (clusters needing optimal alignment) or fixed-meaning (direct equality).
_FIELD_SPECS = {
    "team": ("team", True),     # arbitrary 0/1 cluster id vs the human's team naming
    "role": ("role", False),    # player/spectator mean the same in both files
    "player": ("player", True), # Phase 3 identity: arbitrary cluster id vs the human's player naming
}


@dataclass
class EvalReport:
    tracks_csv: Path
    labels_csv: Path
    fields: dict[str, FieldEval]


def evaluate(
    labels_csv: str | Path,
    tracks_csv: str | Path,
    fields: list[str] | None = None,
) -> EvalReport:
    """Score the pipeline's ``tracks.csv`` against a human ``labels.csv``, per labelable field.

    Only fields the human actually labeled are reported. ``team`` is the headline number (the one
    the team-clustering work needs); ``role`` validates the surface filter for free; ``player`` is
    the Phase 3 identity slot, scored the moment identity labels exist.
    """
    labels_csv = Path(labels_csv)
    tracks_csv = Path(tracks_csv)
    requested = fields or list(_FIELD_SPECS)
    out: dict[str, FieldEval] = {}
    for fld in requested:
        if fld not in _FIELD_SPECS:
            continue
        pred_col, arbitrary = _FIELD_SPECS[fld]
        true_by_track = _read_csv_column(labels_csv, fld)
        if not true_by_track:
            continue  # nothing labeled for this field; don't manufacture a 0-track metric
        pred_by_track = _read_csv_column(tracks_csv, pred_col)
        out[fld] = evaluate_field(pred_by_track, true_by_track, fld, arbitrary)
    return EvalReport(tracks_csv=tracks_csv, labels_csv=labels_csv, fields=out)


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------

def format_report(report: EvalReport) -> str:
    """Render an EvalReport as a human-readable block for the CLI."""
    lines: list[str] = []
    lines.append(f"Evaluation: {report.labels_csv}  vs  {report.tracks_csv}")
    if not report.fields:
        lines.append("  (no labeled fields found — fill in team/role/player in the labels CSV)")
        return "\n".join(lines)
    for fld, fe in report.fields.items():
        r = fe.result
        how = "optimal-aligned" if fe.arbitrary_labels else "direct"
        lines.append(
            f"  {fld:7s}: {r.accuracy*100:5.1f}%  ({r.n_correct}/{r.n} {how}; "
            f"{fe.n_labeled} labeled, {fe.n_overlap} scored)"
        )
        if fe.arbitrary_labels and r.mapping:
            mapped = ", ".join(f"{k}->{v}" for k, v in sorted(r.mapping.items(), key=str))
            lines.append(f"           cluster map: {mapped}")
        if fe.missing_pred:
            shown = ", ".join(str(t) for t in fe.missing_pred[:10])
            extra = f" (+{len(fe.missing_pred)-10} more)" if len(fe.missing_pred) > 10 else ""
            lines.append(f"           labeled but unpredicted: {shown}{extra}")
    return "\n".join(lines)
