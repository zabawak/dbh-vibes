"""Tests for label propagation (roster.py) — the human-in-the-loop tag write-back.

All through the pure ``propagate_labels`` core (plain dict rows, no filesystem), plus one
end-to-end apply over a synthetic run directory.
"""

from __future__ import annotations

import csv
from pathlib import Path

from dbh_vibes.roster import apply_labels_to_run, propagate_labels


def _track(tid, team="", player="", frames=100):
    return {"track_id": str(tid), "team": str(team), "player": str(player),
            "frames_seen": str(frames)}


def _label(tid, team="", player=""):
    return {"track_id": str(tid), "team": team, "player": player}


class TestDirectTags:
    def test_human_tag_lands_on_its_track(self):
        applied = propagate_labels([_track(1, team="0")], [_label(1, team="white")])
        assert applied.track_team_name[1] == "white"
        assert applied.n_direct == 1

    def test_blank_labels_do_nothing(self):
        applied = propagate_labels([_track(1, team="0")], [_label(1)])
        assert applied.track_team_name == {}
        assert applied.n_direct == 0

    def test_unparseable_track_ids_skipped(self):
        applied = propagate_labels(
            [_track(1, team="0"), {"track_id": "", "team": "0"}],
            [_label(1, team="white"), {"track_id": "n/a", "team": "dark"}],
        )
        assert applied.track_team_name == {1: "white"}


class TestPropagation:
    def test_player_tag_spreads_through_identity_cluster(self):
        # Tracks 1,2,3 are one identity (player cluster 7); only track 1 is tagged.
        tracks = [_track(1, player="7"), _track(2, player="7"), _track(3, player="7")]
        applied = propagate_labels(tracks, [_label(1, player="Sarah")])
        assert applied.track_player_name == {1: "Sarah", 2: "Sarah", 3: "Sarah"}
        assert applied.player_names == {"7": "Sarah"}
        assert applied.n_direct == 1
        assert applied.n_propagated == 2

    def test_team_tag_spreads_through_team_cluster(self):
        tracks = [_track(1, team="0"), _track(2, team="0"), _track(3, team="1")]
        applied = propagate_labels(tracks, [_label(1, team="white")])
        assert applied.track_team_name == {1: "white", 2: "white"}
        assert 3 not in applied.track_team_name  # other team unlabeled -> stays unnamed

    def test_majority_vote_is_frame_weighted(self):
        # Two labels disagree; the heavier track wins the cluster name.
        tracks = [_track(1, team="0", frames=500), _track(2, team="0", frames=10),
                  _track(3, team="0")]
        applied = propagate_labels(
            tracks, [_label(1, team="white"), _label(2, team="dark")]
        )
        assert applied.team_names["0"] == "white"
        assert applied.track_team_name[3] == "white"
        assert any("team cluster 0" in c for c in applied.conflicts)

    def test_human_tag_beats_propagation_on_own_track(self):
        # Track 2's own tag ('dark') survives even though its cluster majority says 'white'.
        tracks = [_track(1, team="0", frames=500), _track(2, team="0", frames=10)]
        applied = propagate_labels(
            tracks, [_label(1, team="white"), _label(2, team="dark")]
        )
        assert applied.track_team_name[1] == "white"
        assert applied.track_team_name[2] == "dark"

    def test_tracks_without_cluster_get_nothing(self):
        applied = propagate_labels([_track(1), _track(2)], [_label(1, player="Sam")])
        assert applied.track_player_name == {1: "Sam"}   # direct only; no cluster to ride
        assert applied.n_propagated == 0


class TestConflictReporting:
    def test_over_merge_reported(self):
        # One identity carries two different human names -> over-merge conflict.
        tracks = [_track(1, player="4", frames=100), _track(2, player="4", frames=90)]
        applied = propagate_labels(
            tracks, [_label(1, player="Ana"), _label(2, player="Ben")]
        )
        assert any("over-merge" in c for c in applied.conflicts)
        assert applied.player_names["4"] == "Ana"  # frame-weighted majority

    def test_over_segmentation_reported(self):
        # The same human name lands in two identities -> the clustering split one person.
        tracks = [_track(1, player="4"), _track(2, player="9")]
        applied = propagate_labels(
            tracks, [_label(1, player="Ana"), _label(2, player="Ana")]
        )
        assert any("over-segmentation" in c for c in applied.conflicts)

    def test_deterministic_tie_break(self):
        tracks = [_track(1, player="4", frames=100), _track(2, player="4", frames=100)]
        applied = propagate_labels(
            tracks, [_label(1, player="Ben"), _label(2, player="Ana")]
        )
        assert applied.player_names["4"] == "Ana"  # equal weight -> lexicographic


class TestApplyToRun:
    def test_end_to_end_writes_name_columns_and_report(self, tmp_path: Path):
        run = tmp_path / "run"
        run.mkdir()
        with (run / "tracks.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["track_id", "role", "team", "player", "frames_seen"])
            w.writeheader()
            w.writerow({"track_id": "1", "role": "player", "team": "0", "player": "0",
                        "frames_seen": "100"})
            w.writerow({"track_id": "2", "role": "player", "team": "0", "player": "0",
                        "frames_seen": "50"})
            w.writerow({"track_id": "3", "role": "player", "team": "1", "player": "1",
                        "frames_seen": "80"})
        with (run / "players.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["player", "team", "n_shifts", "shift_seconds",
                                              "active_seconds"])
            w.writeheader()
            w.writerow({"player": "0", "team": "0", "n_shifts": "1", "shift_seconds": "5.0",
                        "active_seconds": "5.0"})
            w.writerow({"player": "1", "team": "1", "n_shifts": "1", "shift_seconds": "2.7",
                        "active_seconds": "2.7"})
        with (run / "shifts.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["player", "team", "shift", "start_time_s",
                                              "end_time_s", "duration_s"])
            w.writeheader()
            w.writerow({"player": "0", "team": "0", "shift": "0", "start_time_s": "0.0",
                        "end_time_s": "5.0", "duration_s": "5.0"})
            w.writerow({"player": "1", "team": "1", "shift": "0", "start_time_s": "1.0",
                        "end_time_s": "3.7", "duration_s": "2.7"})

        labels = tmp_path / "labels.csv"
        with labels.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["track_id", "team", "role", "player"])
            w.writeheader()
            w.writerow({"track_id": "1", "team": "white", "role": "player", "player": "Sarah"})

        result = apply_labels_to_run(run, labels)

        with (run / "tracks.csv").open(newline="") as f:
            rows = {r["track_id"]: r for r in csv.DictReader(f)}
        assert rows["1"]["player_name"] == "Sarah"
        assert rows["2"]["player_name"] == "Sarah"      # propagated through identity 0
        assert rows["2"]["team_name"] == "white"        # propagated through team 0
        assert rows["3"]["player_name"] == ""           # other identity untouched

        with (run / "players.csv").open(newline="") as f:
            prows = {r["player"]: r for r in csv.DictReader(f)}
        assert prows["0"]["name"] == "Sarah"
        assert prows["1"]["name"] == ""

        assert result.report_html is not None and result.report_html.exists()
        html = result.report_html.read_text()
        assert "Sarah" in html                          # named row in table + chart labels

    def test_reapply_is_idempotent(self, tmp_path: Path):
        run = tmp_path / "run"
        run.mkdir()
        with (run / "tracks.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["track_id", "team", "player", "frames_seen"])
            w.writeheader()
            w.writerow({"track_id": "1", "team": "0", "player": "0", "frames_seen": "10"})
        labels = tmp_path / "labels.csv"
        with labels.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["track_id", "team", "role", "player"])
            w.writeheader()
            w.writerow({"track_id": "1", "team": "white", "role": "", "player": "Ana"})

        apply_labels_to_run(run, labels)
        apply_labels_to_run(run, labels)  # second apply must not duplicate columns
        with (run / "tracks.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames.count("team_name") == 1
            row = next(reader)
        assert row["team_name"] == "white"
        assert row["player_name"] == "Ana"
