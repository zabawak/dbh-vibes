"""Propagate human labels back into the pipeline output — the second half of human-in-the-loop.

The labeling flow so far was one-directional: ``--label-crops`` exports per-track crop montages +
a ``labels.csv`` template, a human tags team/role/player by sight in ~2 minutes, and ``--evaluate``
*scores* the pipeline against those tags. But the tags themselves never flowed back into the
product: the report still says "P3 (T0)" instead of "Sarah (white)". This module closes the loop
(the "what's left" of the human-in-the-loop feature in docs/feature-ideas.md): apply a filled-in
``labels.csv`` to a finished run directory, and the human's names **propagate through the
pipeline's own clusters** —

* a *player* tag on one track spreads to every track the Phase 3 identity clustering stitched into
  the same person, so tagging one clean crop names the player's whole game;
* a *team* tag spreads to every track the team clustering put on the same side (majority vote over
  the labeled tracks of that side; frame-weighted, human tags win over propagation on conflicts).

That inversion is the point: the human supplies a handful of anchor tags, the clustering supplies
the coverage. Output: ``tracks.csv`` gains ``team_name``/``player_name`` columns, ``players.csv``
gains a ``name`` column, and the report re-renders with real names on the shift chart and stat
table. Conflicts (one identity carrying two different human names — an over-merge; one name split
across identities — an over-segmentation) are *reported*, not silently resolved, because they are
exactly the re-ID failure modes worth surfacing.

The propagation core (``propagate_labels``) is pure stdlib over plain dict rows — unit-testable
with no video, model, or filesystem — with thin CSV/report I/O shells around it.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppliedLabels:
    """Result of propagating human tags through the pipeline's clusters."""

    track_team_name: dict[int, str] = field(default_factory=dict)    # track_id -> team name
    track_player_name: dict[int, str] = field(default_factory=dict)  # track_id -> player name
    team_names: dict[str, str] = field(default_factory=dict)         # pred team id -> team name
    player_names: dict[str, str] = field(default_factory=dict)       # pred identity id -> player name
    conflicts: list[str] = field(default_factory=list)               # human-readable notes
    n_direct: int = 0        # tracks the human tagged themselves
    n_propagated: int = 0    # tracks that inherited a tag through a cluster


def _clean(value) -> str:
    return (value or "").strip()


def _majority(votes: dict[str, float]) -> str | None:
    """Highest-weight label; deterministic tie-break by label string."""
    if not votes:
        return None
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def propagate_labels(track_rows: list[dict], label_rows: list[dict]) -> AppliedLabels:
    """Spread human team/player tags through the pipeline's team + identity clusters. Pure.

    ``track_rows`` are ``tracks.csv`` rows (needs ``track_id``; uses ``team``/``player`` cluster
    ids and ``frames_seen`` when present). ``label_rows`` are filled-in ``labels.csv`` rows (needs
    ``track_id``; uses the human ``team``/``player`` strings; blank cells mean "not labeled").

    Per cluster the propagated name is the **frame-weighted majority** of its labeled tracks. A
    track the human labeled directly always keeps its own tag (human beats propagation). Conflicts
    are recorded on the result rather than raised: they are real signal about the clustering.
    """
    out = AppliedLabels()

    human_team: dict[int, str] = {}
    human_player: dict[int, str] = {}
    for r in label_rows:
        try:
            tid = int(float(_clean(r.get("track_id"))))
        except (TypeError, ValueError):
            continue
        if _clean(r.get("team")):
            human_team[tid] = _clean(r.get("team"))
        if _clean(r.get("player")):
            human_player[tid] = _clean(r.get("player"))

    track_team: dict[int, str] = {}     # track -> predicted team cluster id (as string)
    track_player: dict[int, str] = {}   # track -> predicted identity cluster id (as string)
    weight: dict[int, float] = {}
    track_ids: list[int] = []
    for r in track_rows:
        try:
            tid = int(float(_clean(r.get("track_id"))))
        except (TypeError, ValueError):
            continue
        track_ids.append(tid)
        if _clean(r.get("team")) != "":
            track_team[tid] = _clean(r.get("team"))
        if _clean(r.get("player")) != "":
            track_player[tid] = _clean(r.get("player"))
        try:
            weight[tid] = max(1.0, float(r.get("frames_seen") or 1.0))
        except (TypeError, ValueError):
            weight[tid] = 1.0

    # ---- cluster-level majority votes (frame-weighted) ----
    team_votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for tid, name in human_team.items():
        cl = track_team.get(tid)
        if cl is not None:
            team_votes[cl][name] += weight.get(tid, 1.0)
    for cl, votes in team_votes.items():
        name = _majority(votes)
        out.team_names[cl] = name
        if len(votes) > 1:
            out.conflicts.append(
                f"team cluster {cl}: labeled tracks disagree "
                f"({', '.join(f'{k}={v:.0f}f' for k, v in sorted(votes.items()))}) -> '{name}'"
            )

    player_votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for tid, name in human_player.items():
        cl = track_player.get(tid)
        if cl is not None:
            player_votes[cl][name] += weight.get(tid, 1.0)
    for cl, votes in player_votes.items():
        name = _majority(votes)
        out.player_names[cl] = name
        if len(votes) > 1:
            out.conflicts.append(
                f"identity {cl}: labeled tracks carry different names "
                f"({', '.join(sorted(votes))}) -> '{name}' (possible over-merge)"
            )
    # One human name landing in several identities = the clustering split one person.
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for cl, name in out.player_names.items():
        name_to_ids[name].append(cl)
    for name, ids in sorted(name_to_ids.items()):
        if len(ids) > 1:
            out.conflicts.append(
                f"player '{name}' spans identities {sorted(ids)} (re-ID over-segmentation; "
                f"their stats are split across those rows)"
            )

    # ---- per-track resolution: human tag wins, else inherit the cluster's name ----
    for tid in track_ids:
        direct = False
        if tid in human_team:
            out.track_team_name[tid] = human_team[tid]
            direct = True
        else:
            inherited = out.team_names.get(track_team.get(tid))
            if inherited:
                out.track_team_name[tid] = inherited
                out.n_propagated += 1
        if tid in human_player:
            out.track_player_name[tid] = human_player[tid]
            direct = True
        else:
            inherited = out.player_names.get(track_player.get(tid))
            if inherited:
                out.track_player_name[tid] = inherited
                out.n_propagated += 1
        if direct:
            out.n_direct += 1
    return out


# --------------------------------------------------------------------------------------------
# I/O shells: read the run artifacts, write them back with name columns, refresh the report.
# --------------------------------------------------------------------------------------------

def _read_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


@dataclass
class ApplyResult:
    applied: AppliedLabels
    tracks_csv: Path
    players_csv: Path | None
    report_html: Path | None


def apply_labels_to_run(run_dir: str | Path, labels_csv: str | Path) -> ApplyResult:
    """Apply a filled-in ``labels.csv`` to a finished run dir: name columns + refreshed report.

    Rewrites ``tracks.csv`` with ``team_name``/``player_name`` columns and ``players.csv`` (when
    present) with a ``name`` column, then re-renders ``report.html``/``shift_chart.png`` so the
    shift chart and stat table show the human names. Idempotent: re-applying overwrites the name
    columns in place.
    """
    run_dir = Path(run_dir)
    labels_csv = Path(labels_csv)
    tracks_path = run_dir / "tracks.csv"
    if not tracks_path.exists():
        raise FileNotFoundError(f"no tracks.csv in {run_dir} — run --phase2 first")
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels CSV not found: {labels_csv}")

    track_rows, track_fields = _read_rows(tracks_path)
    label_rows, _ = _read_rows(labels_csv)
    applied = propagate_labels(track_rows, label_rows)

    # tracks.csv: add/overwrite the name columns.
    for col in ("team_name", "player_name"):
        if col not in track_fields:
            track_fields.append(col)
    for r in track_rows:
        try:
            tid = int(float(r.get("track_id", "")))
        except (TypeError, ValueError):
            tid = None
        r["team_name"] = applied.track_team_name.get(tid, "")
        r["player_name"] = applied.track_player_name.get(tid, "")
    _write_rows(tracks_path, track_rows, track_fields)

    # players.csv: one name per identity row.
    players_path = run_dir / "players.csv"
    players_out: Path | None = None
    if players_path.exists():
        player_rows, player_fields = _read_rows(players_path)
        if "name" not in player_fields:
            player_fields.insert(1, "name")
        for r in player_rows:
            r["name"] = applied.player_names.get(_clean(r.get("player")), "")
        _write_rows(players_path, player_rows, player_fields)
        players_out = players_path

    # Refresh the report so the names actually show up where a coach looks.
    report_html: Path | None = None
    if players_out is not None and (run_dir / "shifts.csv").exists():
        from dbh_vibes.report import write_report

        report_html = write_report(run_dir).html
    return ApplyResult(
        applied=applied, tracks_csv=tracks_path, players_csv=players_out,
        report_html=report_html,
    )


def format_apply_summary(result: ApplyResult) -> str:
    """Console summary of what propagated where."""
    a = result.applied
    lines = [
        f"Applied labels: {a.n_direct} tracks tagged directly, "
        f"{a.n_propagated} inherited a tag through team/identity clusters",
    ]
    if a.player_names:
        named = ", ".join(f"{cl}->'{n}'" for cl, n in sorted(a.player_names.items(), key=str))
        lines.append(f"  identities named: {named}")
    if a.team_names:
        named = ", ".join(f"T{cl}->'{n}'" for cl, n in sorted(a.team_names.items(), key=str))
        lines.append(f"  teams named: {named}")
    for c in a.conflicts:
        lines.append(f"  conflict: {c}")
    lines.append(f"  wrote: {result.tracks_csv}"
                 + (f", {result.players_csv}" if result.players_csv else ""))
    if result.report_html is not None:
        lines.append(f"  report refreshed: {result.report_html}")
    return "\n".join(lines)
