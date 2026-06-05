"""Tests for the per-game box-score export (pure stdlib, no heavy deps)."""

from __future__ import annotations

import json

from dbh_vibes.boxscore import (
    PlayerLine,
    build_boxscore,
    format_boxscore,
    write_boxscore_json,
)

FPS = 10.0  # 10 fps => 1 frame = 0.1s, keeps the second math easy to read


def _player(track_id, team, frames_seen, active_frames, on_surface_frames, area=2000):
    return PlayerLine(
        track_id=track_id,
        team=team,
        frames_seen=frames_seen,
        active_frames=active_frames,
        on_surface_frames=on_surface_frames,
        median_area_px=area,
    )


def _bs(players, **overrides):
    kwargs = dict(
        fps=FPS, frame_count=600, n_spectators=2, live_seconds=42.0,
        n_segments=3, active_fraction=0.7, surface_found=True,
    )
    kwargs.update(overrides)
    return build_boxscore(players, **kwargs)


def test_player_line_on_surface_frac():
    assert _player(1, 0, 100, 80, 75).on_surface_frac() == 0.75
    assert _player(2, 0, 0, 0, 0).on_surface_frac() == 0.0  # no divide-by-zero


def test_game_header_carries_headline_totals():
    bs = _bs([_player(1, 0, 100, 80, 95)])
    g = bs["game"]
    assert g["fps"] == 10.0
    assert g["frames"] == 600
    assert g["duration_s"] == 60.0          # 600 / 10
    assert g["live_play_s"] == 42.0
    assert g["n_segments"] == 3
    assert g["active_fraction"] == 0.7
    assert g["surface_found"] is True
    assert g["n_players"] == 1
    assert g["n_spectators"] == 2


def test_player_seconds_conversion():
    bs = _bs([_player(7, 1, 120, 90, 108)])
    p = bs["players"][0]
    assert p["track_id"] == 7
    assert p["team"] == 1
    assert p["on_surface_s"] == 12.0        # 120 / 10
    assert p["active_s"] == 9.0             # 90 / 10
    assert p["on_surface_frac"] == 0.9      # 108 / 120
    assert p["median_area_px"] == 2000


def test_players_sorted_most_active_first():
    bs = _bs([
        _player(1, 0, 100, 30, 90),
        _player(2, 1, 100, 90, 90),
        _player(3, 0, 100, 60, 90),
    ])
    assert [p["track_id"] for p in bs["players"]] == [2, 3, 1]


def test_team_totals_sum_over_tracks():
    bs = _bs([
        _player(1, 0, 100, 50, 100),
        _player(2, 0, 100, 30, 100),
        _player(3, 1, 100, 40, 100),
    ])
    teams = {t["team"]: t for t in bs["teams"]}
    assert teams[0]["n_players"] == 2
    assert teams[0]["active_s"] == 8.0       # (50 + 30) / 10
    assert teams[0]["on_surface_s"] == 20.0  # (100 + 100) / 10
    assert teams[1]["n_players"] == 1
    assert teams[1]["active_s"] == 4.0


def test_unassigned_team_sorts_last():
    bs = _bs([
        _player(1, None, 100, 50, 100),
        _player(2, 1, 100, 50, 100),
        _player(3, 0, 100, 50, 100),
    ])
    assert [t["team"] for t in bs["teams"]] == [0, 1, None]


def test_empty_players_is_valid():
    bs = _bs([])
    assert bs["game"]["n_players"] == 0
    assert bs["teams"] == []
    assert bs["players"] == []


def test_zero_fps_is_safe():
    bs = _bs([_player(1, 0, 100, 80, 90)], fps=0.0)
    assert bs["players"][0]["active_s"] == 0.0
    assert bs["game"]["duration_s"] == 0.0


def test_write_and_reload_json(tmp_path):
    bs = _bs([_player(1, 0, 100, 80, 90)])
    out = tmp_path / "boxscore.json"
    write_boxscore_json(out, bs)
    reloaded = json.loads(out.read_text())
    assert reloaded == bs
    assert reloaded["schema_version"] == 1


def test_format_boxscore_is_readable_text():
    bs = _bs([_player(1, 0, 100, 80, 90), _player(2, None, 50, 20, 40)])
    text = format_boxscore(bs)
    assert "Box score" in text
    assert "T0" in text
    assert "T?" in text          # unassigned team rendered distinctly
    assert "track" in text       # header row present
