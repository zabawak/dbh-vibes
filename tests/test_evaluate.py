"""Tests for the labeled-set eval harness (pure numpy/stdlib — no video, model, or labels needed).

These cover the scoring logic priority #1 hinges on: arbitrary cluster labels aligned optimally to
true classes (so a team called "0" that is really "white" still scores correctly), direct equality
for fixed-meaning fields (role), and the CSV plumbing that lines labels up with predictions by
track id and scores only the overlap.
"""

from __future__ import annotations

import csv

from dbh_vibes.evaluate import (
    best_map_accuracy,
    direct_accuracy,
    evaluate,
    evaluate_field,
)


# ---- best_map_accuracy: arbitrary cluster ids aligned to true classes ----

def test_perfect_split_with_flipped_labels():
    # Predicted cluster "0" is truly "white", "1" is truly "dark" — accuracy must be 100% despite
    # the names never matching by equality.
    pred = ["0", "0", "0", "1", "1", "1"]
    true = ["white", "white", "white", "dark", "dark", "dark"]
    r = best_map_accuracy(pred, true)
    assert r.accuracy == 1.0
    assert r.n == 6 and r.n_correct == 6
    assert r.mapping == {"0": "white", "1": "dark"}


def test_alignment_picks_the_better_of_two_permutations():
    # 5 of 6 consistent with 0->A,1->B; the other mapping would score 1/6. Optimal picks 5/6.
    pred = ["0", "0", "0", "1", "1", "1"]
    true = ["A", "A", "A", "B", "B", "A"]
    r = best_map_accuracy(pred, true)
    assert r.n_correct == 5 and r.n == 6
    assert abs(r.accuracy - 5 / 6) < 1e-9
    assert r.mapping == {"0": "A", "1": "B"}


def test_over_segmenting_is_one_to_one():
    # Three predicted clusters, two true classes: one-to-one matching means the third cluster can't
    # also claim a class, so it scores as misses (clustering-accuracy convention, not purity).
    pred = ["0", "0", "1", "1", "2", "2"]
    true = ["A", "A", "B", "B", "A", "A"]
    r = best_map_accuracy(pred, true)
    # Best one-to-one: 0->A (2), 1->B (2); cluster 2's items (truly A, but A is taken) are misses.
    assert r.n_correct == 4 and r.n == 6


def test_empty_input():
    r = best_map_accuracy([], [])
    assert r.accuracy == 0.0 and r.n == 0


# ---- direct_accuracy: fixed-meaning labels compared by equality ----

def test_direct_accuracy_does_not_realign():
    pred = ["player", "player", "spectator", "spectator"]
    true = ["player", "spectator", "spectator", "spectator"]
    r = direct_accuracy(pred, true)
    assert r.n_correct == 3 and r.n == 4
    assert abs(r.accuracy - 0.75) < 1e-9


# ---- evaluate_field: overlap handling ----

def test_field_scores_only_overlap_and_reports_missing():
    pred = {1: "0", 2: "0", 3: "1"}          # track 4 has no prediction
    true = {1: "w", 2: "w", 3: "d", 4: "w"}  # track 4 is labeled but unpredicted
    fe = evaluate_field(pred, true, "team", arbitrary_labels=True)
    assert fe.n_labeled == 4
    assert fe.n_overlap == 3
    assert fe.result.accuracy == 1.0
    assert fe.missing_pred == [4]


# ---- evaluate(): end-to-end over CSV files ----

def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_evaluate_reads_csvs_and_scores_team_and_role(tmp_path):
    tracks = tmp_path / "tracks.csv"
    labels = tmp_path / "labels.csv"
    _write_csv(tracks, ["track_id", "role", "team"], [
        {"track_id": 1, "role": "player", "team": "0"},
        {"track_id": 2, "role": "player", "team": "0"},
        {"track_id": 3, "role": "player", "team": "1"},
        {"track_id": 4, "role": "spectator", "team": ""},
    ])
    _write_csv(labels, ["track_id", "team", "role", "player"], [
        {"track_id": 1, "team": "white", "role": "player", "player": ""},
        {"track_id": 2, "team": "white", "role": "player", "player": ""},
        {"track_id": 3, "team": "dark", "role": "player", "player": ""},
        {"track_id": 4, "team": "", "role": "spectator", "player": ""},
    ])
    report = evaluate(labels, tracks)
    assert report.fields["team"].result.accuracy == 1.0
    assert report.fields["team"].n_overlap == 3       # spectator has no team label/prediction
    assert report.fields["role"].result.accuracy == 1.0
    assert report.fields["role"].n_overlap == 4
    assert "player" not in report.fields              # no identity labels -> field omitted


def test_evaluate_omits_unlabeled_fields(tmp_path):
    tracks = tmp_path / "tracks.csv"
    labels = tmp_path / "labels.csv"
    _write_csv(tracks, ["track_id", "role", "team"], [
        {"track_id": 1, "role": "player", "team": "0"},
    ])
    _write_csv(labels, ["track_id", "team", "role", "player"], [
        {"track_id": 1, "team": "", "role": "", "player": ""},  # nothing actually labeled
    ])
    report = evaluate(labels, tracks)
    assert report.fields == {}
