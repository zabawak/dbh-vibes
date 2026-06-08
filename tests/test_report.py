"""Tests for the per-game report's pure layout core (no matplotlib/numpy/video).

The headline behaviour: shifts loaded from ``shifts.csv`` lay out into a player-by-time Gantt —
one row per player, one bar per shift, rows grouped by team and ordered most-time-on-surface first
— and the surrounding loaders/HTML assembly round-trip the run artifacts. The matplotlib PNG render
is a thin shell and is exercised by the real-footage validation, not unit-tested here.
"""

from __future__ import annotations

import json

from dbh_vibes.report import (
    ShiftBar,
    build_report,
    build_shift_chart,
    format_report_text,
    load_players_csv,
    load_shifts_csv,
    render_report_html,
    team_color,
    team_totals_from_players,
)


# ---- ShiftBar ----------------------------------------------------------------------------------

def test_shiftbar_duration():
    assert ShiftBar(2.0, 5.0).duration_s == 3.0
    assert ShiftBar(5.0, 5.0).duration_s == 0.0
    assert ShiftBar(5.0, 2.0).duration_s == 0.0  # never negative


# ---- build_shift_chart -------------------------------------------------------------------------

def _shift(player, team, start, end):
    return {"player": player, "team": team, "start_time_s": start, "end_time_s": end}


def test_empty_chart():
    chart = build_shift_chart([])
    assert chart.n_rows == 0 and chart.duration_s == 0.0


def test_one_player_bars_grouped_and_sorted():
    rows = [_shift(3, 0, 10, 15), _shift(3, 0, 0, 4)]  # out of time order on purpose
    chart = build_shift_chart(rows)
    assert chart.n_rows == 1
    r = chart.rows[0]
    assert r.player == 3 and r.team == 0 and r.n_shifts == 2
    assert [b.start_s for b in r.bars] == [0.0, 10.0]  # bars sorted by start
    assert r.toi_s == 9.0                              # 4 + 5
    assert r.label == "P3 (T0)"


def test_rows_ordered_by_team_then_toi():
    rows = [
        _shift(1, 0, 0, 10),    # T0, 10s
        _shift(2, 0, 0, 30),    # T0, 30s  -> first within T0
        _shift(3, 1, 0, 20),    # T1, 20s
        _shift(4, None, 0, 5),  # unassigned -> last
    ]
    chart = build_shift_chart(rows)
    # T0 (most TOI first): player 2 then 1; then T1 player 3; then unassigned player 4.
    assert [r.player for r in chart.rows] == [2, 1, 3, 4]
    assert [r.row for r in chart.rows] == [0, 1, 2, 3]
    assert chart.rows[-1].team is None and chart.rows[-1].label == "P4"


def test_duration_defaults_to_latest_shift_end_or_override():
    rows = [_shift(1, 0, 0, 10), _shift(1, 0, 40, 55)]
    assert build_shift_chart(rows).duration_s == 55.0
    assert build_shift_chart(rows, duration_s=120.0).duration_s == 120.0


def test_team_color_mapping():
    assert team_color(0) != team_color(1)
    assert team_color(None) == team_color("")  # both unknown -> grey
    assert team_color("0") == team_color(0)     # string coerces


# ---- team_totals_from_players ------------------------------------------------------------------

def test_team_totals_aggregate_identities():
    players = [
        {"team": "0", "shift_seconds": "30.0", "active_seconds": "25.0"},
        {"team": "0", "shift_seconds": "20.0", "active_seconds": "18.0"},
        {"team": "1", "shift_seconds": "40.0", "active_seconds": "35.0"},
        {"team": "", "shift_seconds": "5.0", "active_seconds": "4.0"},
    ]
    totals = team_totals_from_players(players)
    # Ordered T0, T1, then unassigned last; counts are identity counts, seconds summed.
    assert [t["team"] for t in totals] == [0, 1, None]
    assert totals[0] == {"team": 0, "n_players": 2, "toi_s": 50.0, "active_s": 43.0}
    assert totals[2]["n_players"] == 1  # the unassigned identity


# ---- loaders + end-to-end build over a run dir -------------------------------------------------

def _write_run_dir(tmp_path):
    shifts = tmp_path / "shifts.csv"
    shifts.write_text(
        "player,team,shift,start_frame,end_frame,n_frames,n_fragments,"
        "start_time_s,end_time_s,duration_s\n"
        "1,0,0,0,300,301,1,0.0,10.0,10.0\n"
        "1,0,1,1200,1500,301,1,40.0,50.0,10.0\n"
        "2,1,0,0,600,601,2,0.0,20.0,20.0\n"
    )
    players = tmp_path / "players.csv"
    players.write_text(
        "player,team,n_shifts,n_fragments,track_ids,frames_seen,seconds_on_surface,"
        "shift_seconds,active_seconds,longest_shift_s,avg_shift_s,first_frame,last_frame,mean_conf\n"
        "2,1,1,2,3 5,600,20.0,20.0,20.0,20.0,20.0,0,600,0.8\n"
        "1,0,2,2,1 2,600,20.0,20.0,18.0,10.0,10.0,0,1500,0.7\n"
    )
    box = {
        "game": {"fps": 30.0, "duration_s": 60.0, "live_play_s": 55.0, "n_segments": 2,
                 "active_fraction": 0.9, "n_players": 2, "n_spectators": 1},
        "teams": [{"team": 0, "n_players": 1, "active_s": 18.0, "on_surface_s": 20.0},
                  {"team": 1, "n_players": 1, "active_s": 20.0, "on_surface_s": 20.0}],
    }
    (tmp_path / "boxscore.json").write_text(json.dumps(box))
    return tmp_path


def test_load_shifts_csv_coerces_types(tmp_path):
    _write_run_dir(tmp_path)
    rows = load_shifts_csv(tmp_path / "shifts.csv")
    assert len(rows) == 3
    assert rows[0]["player"] == 1 and rows[0]["team"] == 0
    assert rows[0]["start_time_s"] == 0.0 and rows[0]["end_time_s"] == 10.0


def test_build_report_assembles_from_run_dir(tmp_path):
    _write_run_dir(tmp_path)
    report = build_report(tmp_path)
    # Chart spans the full clip (from boxscore), not just the last shift end.
    assert report.chart.duration_s == 60.0
    assert report.chart.n_rows == 2
    # Two teams, one player each; players.csv preserved.
    assert len(report.players) == 2 and len(report.teams) == 2
    assert report.game["live_play_s"] == 55.0
    # Team totals are derived from the identities in players.csv (not boxscore's per-track teams).
    t0 = next(t for t in report.teams if t["team"] == 0)
    assert t0["n_players"] == 1 and t0["toi_s"] == 20.0 and t0["active_s"] == 18.0
    # No heatmap.jpg written -> heatmap_path stays None.
    assert report.heatmap_path is None


def test_build_report_requires_reid_artifacts(tmp_path):
    (tmp_path / "boxscore.json").write_text("{}")
    try:
        build_report(tmp_path)
    except FileNotFoundError as exc:
        assert "players.csv" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when players/shifts CSVs are missing")


def test_format_report_text_lists_players(tmp_path):
    _write_run_dir(tmp_path)
    text = format_report_text(build_report(tmp_path))
    assert "Game report" in text and "Players (most TOI first)" in text
    # Both players appear as rows in the table (bare ids under the `player` column).
    assert text.count("T0") >= 1 and text.count("T1") >= 1
    assert "2 segment(s)" in text


def test_render_html_is_self_contained(tmp_path):
    """HTML embeds the chart as a data URI and contains the player rows — no sidecar files."""
    _write_run_dir(tmp_path)
    report = build_report(tmp_path)
    # A tiny stand-in PNG so we don't need matplotlib in the pure test path.
    fake_png = tmp_path / "chart.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    html_path = render_report_html(report, tmp_path / "report.html", fake_png)
    html = html_path.read_text()
    assert "data:image/png;base64," in html       # chart embedded inline
    assert "Shift chart" in html and "Players" in html
    assert "Team totals" in html                   # boxscore teams rendered
