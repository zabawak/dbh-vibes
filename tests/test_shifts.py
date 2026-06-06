"""Tests for shift detection (pure stdlib, no model/video/heavy deps).

The headline behaviour: a player's fragmented track ids stitch into *true* on-surface shifts —
short tracker gaps (occlusion / re-acquire) bridge into one shift, bench-length gaps split into
separate shifts — so ``n_shifts`` stops over-counting the way the raw fragment count did.
"""

from __future__ import annotations

import csv

from dbh_vibes.shifts import (
    Shift,
    detect_shifts,
    merge_spans,
    shift_record,
    summarize_player,
    write_shifts_csv,
)

FPS = 10.0  # 1 frame = 0.1s, keeps the second-knobs easy to reason about


# ---- merge_spans -------------------------------------------------------------------------------

def test_merge_empty():
    assert merge_spans([], 10) == []


def test_short_gap_merges_into_one_run():
    # gap between [0,9] and [12,20] is 12-9-1 = 2 frames <= bridge(5) -> one run.
    assert merge_spans([(0, 9), (12, 20)], bridge_frames=5) == [(0, 20, 2)]


def test_long_gap_stays_separate():
    # gap of 50 frames > bridge(5) -> two runs, each one fragment.
    assert merge_spans([(0, 9), (60, 80)], bridge_frames=5) == [(0, 9, 1), (60, 80, 1)]


def test_unsorted_input_is_ordered():
    assert merge_spans([(60, 80), (0, 9)], bridge_frames=5) == [(0, 9, 1), (60, 80, 1)]


def test_overlapping_spans_always_merge_even_with_no_bridge():
    assert merge_spans([(0, 20), (10, 30)], bridge_frames=-1) == [(0, 30, 2)]


def test_negative_bridge_keeps_disjoint_spans_apart():
    # gap of exactly 1 frame; bridge_frames<0 disables bridging -> two runs.
    assert merge_spans([(0, 9), (11, 20)], bridge_frames=-1) == [(0, 9, 1), (11, 20, 1)]


# ---- detect_shifts -----------------------------------------------------------------------------

def test_fragments_of_one_shift_collapse():
    # One player, tracker dropped twice mid-shift (0.2s gaps) -> one true shift, three fragments.
    spans = {7: [(0, 30), (33, 60), (63, 90)]}
    shifts = detect_shifts(spans, FPS, bridge_gap_seconds=1.0)
    assert len(shifts[7]) == 1
    sh = shifts[7][0]
    assert (sh.start_frame, sh.end_frame, sh.n_fragments) == (0, 90, 3)


def test_bench_trip_splits_shifts():
    # On 0..30, off ~7s (bench), back 100..150 -> two shifts.
    spans = {7: [(0, 30), (100, 150)]}
    shifts = detect_shifts(spans, FPS, bridge_gap_seconds=3.0)
    assert len(shifts[7]) == 2
    assert [(s.start_frame, s.end_frame) for s in shifts[7]] == [(0, 30), (100, 150)]
    assert [s.index for s in shifts[7]] == [0, 1]


def test_corrects_fragment_overcount():
    # A player whose single continuous shift fragmented into 4 track ids should count as 1 shift,
    # not 4 — the whole point of the feature.
    spans = {1: [(0, 20), (22, 40), (41, 60), (62, 100)]}
    shifts = detect_shifts(spans, FPS, bridge_gap_seconds=2.0)
    assert len(shifts[1]) == 1
    assert shifts[1][0].n_fragments == 4


def test_min_shift_filter_drops_blips_and_reindexes():
    # A 0.2s blip (frames 200..201) is dropped by min_shift_seconds=1.0; remaining shift re-indexes.
    spans = {5: [(0, 30), (200, 201)]}
    shifts = detect_shifts(spans, FPS, bridge_gap_seconds=1.0, min_shift_seconds=1.0)
    assert len(shifts[5]) == 1
    assert shifts[5][0].index == 0
    assert shifts[5][0].start_frame == 0


def test_empty_player_omitted():
    assert detect_shifts({9: []}, FPS) == {}


def test_zero_fps_is_safe():
    shifts = detect_shifts({1: [(0, 10), (20, 30)]}, fps=0.0)
    # No fps -> bridge=0, so only overlapping/adjacent spans merge; here a 9-frame gap splits them.
    assert len(shifts[1]) == 2


# ---- Shift second/frame conversions ------------------------------------------------------------

def test_shift_inclusive_bounds_and_seconds():
    sh = Shift(player=1, index=0, start_frame=0, end_frame=19, n_fragments=1)
    assert sh.n_frames == 20
    assert sh.start_seconds(FPS) == 0.0
    assert sh.end_seconds(FPS) == 2.0       # frame after the last on-surface frame
    assert sh.duration_seconds(FPS) == 2.0


# ---- summarize_player --------------------------------------------------------------------------

def test_summary_counts_and_stats():
    spans = {3: [(0, 30), (33, 60), (200, 209)]}  # two shifts (3.1s + 1.0s blip... actually merged)
    shifts = detect_shifts(spans, FPS, bridge_gap_seconds=1.0)
    summary = summarize_player(3, shifts[3], FPS)
    # First two fragments merge (0..60 = 6.1s), the third is its own shift (200..209 = 1.0s).
    assert summary.n_shifts == 2
    assert summary.n_fragments == 3
    assert summary.longest_shift_s == 6.1
    assert summary.shift_seconds == round(6.1 + 1.0, 2)
    assert summary.first_frame == 0 and summary.last_frame == 209


def test_summary_empty_is_zeroed():
    summary = summarize_player(1, [], FPS)
    assert summary.n_shifts == 0 and summary.shift_seconds == 0.0


# ---- CSV output --------------------------------------------------------------------------------

def test_shift_record_schema():
    sh = Shift(player=2, index=1, start_frame=10, end_frame=29, n_fragments=2)
    rec = shift_record(sh, FPS, team=0)
    assert rec == {
        "player": 2, "team": 0, "shift": 1, "start_frame": 10, "end_frame": 29,
        "n_frames": 20, "n_fragments": 2, "start_time_s": 1.0, "end_time_s": 3.0,
        "duration_s": 2.0,
    }


def test_write_shifts_csv_round_trip(tmp_path):
    spans = {2: [(0, 30)], 1: [(0, 10), (100, 120)]}
    shifts = detect_shifts(spans, FPS, bridge_gap_seconds=1.0)
    path = tmp_path / "shifts.csv"
    write_shifts_csv(path, shifts, FPS, teams={1: 0, 2: 1})
    rows = list(csv.DictReader(path.open()))
    # Ordered by player then shift: player 1 has two shifts, player 2 has one.
    assert [r["player"] for r in rows] == ["1", "1", "2"]
    assert [r["shift"] for r in rows] == ["0", "1", "0"]
    assert rows[0]["team"] == "0" and rows[2]["team"] == "1"
    assert rows[2]["duration_s"] == "3.1"
