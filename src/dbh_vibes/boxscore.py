"""Per-game box-score / stats export (Phase 2 quick win).

The pipeline already computes everything a coach actually wants to read — who was on the
surface, for how long, on which team, and how much live play there was — but it only ships it
as a wide `tracks.csv` plus numbers scattered across the console summary. This module rolls
those into one consumable artifact: a structured **box-score** (`boxscore.json`) with a game
header, per-team totals, and a per-player table, plus a compact text rendering for the CLI.

Honest scope: this is a **per-track** box-score, not yet per-*player*. With no jersey numbers and
no Phase 3 identity, a player who leaves the frame and returns still counts as two tracks (same
caveat as `tracks.csv`). So we report track-level time-on-surface and team totals — the numbers we
can stand behind today — and leave true per-player shift counts to Phase 3 identity. Team totals,
which sum over tracks, are robust to that fragmentation.

Pure stdlib (no numpy/cv2) so the core stays trivially testable and import-cheap, mirroring
`segments.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlayerLine:
    """One on-surface track's raw counts, the input to the box-score.

    Kept to plain ints so the core needs neither numpy nor the pipeline's ``TrackStat``; the
    caller computes ``median_area_px`` (e.g. via numpy) and hands it in.
    """

    track_id: int
    team: int | None
    frames_seen: int
    active_frames: int
    on_surface_frames: int
    median_area_px: int

    def on_surface_frac(self) -> float:
        return self.on_surface_frames / self.frames_seen if self.frames_seen else 0.0


def _seconds(frames: int, fps: float) -> float:
    return round(frames / fps, 2) if fps else 0.0


def _player_record(p: PlayerLine, fps: float) -> dict:
    """One player row as a plain dict — the single source of truth for the player schema."""
    return {
        "track_id": p.track_id,
        "team": p.team,
        "on_surface_s": _seconds(p.frames_seen, fps),
        "active_s": _seconds(p.active_frames, fps),
        "on_surface_frac": round(p.on_surface_frac(), 2),
        "median_area_px": p.median_area_px,
    }


def _team_totals(players: list[PlayerLine], fps: float) -> list[dict]:
    """Aggregate per-team counts. Teams sorted ascending; unassigned (team ``None``) last."""
    by_team: dict[int | None, list[PlayerLine]] = {}
    for p in players:
        by_team.setdefault(p.team, []).append(p)

    def sort_key(team: int | None) -> tuple[int, int]:
        return (1, 0) if team is None else (0, team)  # None sorts after all real teams

    rows: list[dict] = []
    for team in sorted(by_team, key=sort_key):
        members = by_team[team]
        rows.append({
            "team": team,
            "n_players": len(members),
            "active_s": round(sum(_seconds(p.active_frames, fps) for p in members), 2),
            "on_surface_s": round(sum(_seconds(p.frames_seen, fps) for p in members), 2),
        })
    return rows


def build_boxscore(
    players: list[PlayerLine],
    *,
    fps: float,
    frame_count: int,
    n_spectators: int,
    live_seconds: float,
    n_segments: int,
    active_fraction: float,
    surface_found: bool,
) -> dict:
    """Assemble the per-game box-score dict from already-computed per-track counts.

    Players are listed most-active first; the game header carries the headline totals that the
    console summary also prints, so the JSON is self-contained.
    """
    ordered = sorted(players, key=lambda p: (p.active_frames, p.frames_seen), reverse=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "game": {
            "fps": round(fps, 3),
            "frames": frame_count,
            "duration_s": _seconds(frame_count, fps),
            "live_play_s": round(live_seconds, 2),
            "n_segments": n_segments,
            "active_fraction": round(active_fraction, 3),
            "surface_found": bool(surface_found),
            "n_players": len(players),
            "n_spectators": n_spectators,
        },
        "teams": _team_totals(players, fps),
        "players": [_player_record(p, fps) for p in ordered],
    }


def write_boxscore_json(path: str | Path, boxscore: dict) -> None:
    """Write the box-score to ``path`` as pretty JSON."""
    Path(path).write_text(json.dumps(boxscore, indent=2))


def _team_label(team: int | None) -> str:
    return "T?" if team is None else f"T{team}"


def format_boxscore(boxscore: dict) -> str:
    """Render the box-score as a compact, fixed-width text table for the console."""
    g = boxscore["game"]
    lines = [
        f"Box score — {g['n_players']} players, {g['duration_s']:.0f}s clip, "
        f"{g['live_play_s']:.0f}s live play ({g['n_segments']} segment(s))",
    ]
    if boxscore["teams"]:
        lines.append("  Teams:")
        for t in boxscore["teams"]:
            lines.append(f"    {_team_label(t['team']):>3}: {t['n_players']:2d} players, "
                         f"{t['active_s']:6.0f} active player-s")
    lines.append("  Players (most active first):")
    lines.append(f"    {'track':>5} {'team':>4} {'active_s':>9} {'surface_s':>10} {'surf%':>6}")
    for p in boxscore["players"]:
        lines.append(f"    {p['track_id']:>5} {_team_label(p['team']):>4} "
                     f"{p['active_s']:>9.1f} {p['on_surface_s']:>10.1f} "
                     f"{p['on_surface_frac']*100:>5.0f}%")
    return "\n".join(lines)
