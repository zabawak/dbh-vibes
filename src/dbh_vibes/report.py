"""Per-game report + shift chart — make the computed stats consumable (priority #5).

The pipeline already *computes* the headline per-player stats (``players.csv`` true per-player
time-on-surface + shift counts, ``shifts.csv`` one row per shift), the per-team totals
(``boxscore.json``) and a position ``heatmap.jpg`` — but a user still has to read CSV/JSON to see
any of it. This module turns those existing artifacts into the thing a coach/player actually looks
at: a single self-contained **per-game report** (``report.html``) carrying a per-player stat table,
per-team totals, the heatmap, and a **shift chart** — the classic "time-on-ice" Gantt of who is on
the surface when (``shift_chart.png``).

It adds **no model, no GPU, no labels**: it is pure rendering over already-written artifacts, so it
runs anywhere and is validatable on real footage today. Following the discipline of ``segments.py``
/ ``shifts.py`` / ``boxscore.py``, the **chart layout is a pure-stdlib core** (rows = players, bars =
shifts — coordinates and ordering, no drawing) that is trivially unit-testable, with the matplotlib
PNG render and the HTML assembly as thin shells on top. The whole thing also runs **standalone** over
a finished run directory (``--report <run-dir>``), reading only the CSV/JSON it wrote earlier.
"""

from __future__ import annotations

import base64
import csv
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

# Team colours, kept in lock-step with ``pipeline.TEAM_COLORS_BGR`` (green / red) but expressed as
# hex RGB for matplotlib + HTML. Team ``None``/unknown falls back to a neutral grey.
TEAM_COLORS = {0: "#3cdc3c", 1: "#f03c3c"}
UNKNOWN_COLOR = "#9c9c9c"


def team_color(team) -> str:
    """Hex colour for a team id (``0``/``1``), grey for unknown/unassigned."""
    try:
        return TEAM_COLORS.get(int(team), UNKNOWN_COLOR)
    except (TypeError, ValueError):
        return UNKNOWN_COLOR


# --------------------------------------------------------------------------------------------
# Pure layout core (no matplotlib, no numpy) — rows = players, bars = shifts.
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ShiftBar:
    """One shift as a time interval on the chart's x-axis (game seconds)."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class ChartRow:
    """One player's row in the shift chart: ordered position + the bars to draw."""

    player: int
    team: int | None
    row: int                 # 0-based row index from the top of the chart
    label: str               # e.g. "P3 (T0)"
    toi_s: float             # total on-surface time = sum of bar durations
    n_shifts: int
    bars: tuple[ShiftBar, ...]

    @property
    def color(self) -> str:
        return team_color(self.team)


@dataclass(frozen=True)
class ShiftChart:
    """Laid-out shift chart: player rows (top-to-bottom) over a shared ``duration_s`` x-axis."""

    rows: tuple[ChartRow, ...]
    duration_s: float

    @property
    def n_rows(self) -> int:
        return len(self.rows)


def _team_sort_key(team) -> tuple[int, int]:
    """Sort real teams ascending; unknown/unassigned (``None``/blank) last."""
    if team is None:
        return (1, 0)
    return (0, int(team))


def build_shift_chart(
    shift_rows: list[dict],
    *,
    duration_s: float | None = None,
) -> ShiftChart:
    """Lay out shifts into a player-by-time Gantt — the pure core of the shift chart.

    Args:
        shift_rows: per-shift dicts (as loaded from ``shifts.csv``), each carrying ``player``,
            ``team``, ``start_time_s`` and ``end_time_s``. Rows for the same ``player`` become that
            player's bars.
        duration_s: x-axis extent (game seconds). Defaults to the latest shift end, so the chart
            spans exactly the observed play; pass the box-score game duration to anchor it to the
            full clip instead.

    Returns:
        A ``ShiftChart`` whose rows are ordered by **team, then most time-on-surface first** (so the
        two teams cluster the way a real bench chart reads) and assigned contiguous row indices.
    """
    by_player: dict[int, list[ShiftBar]] = {}
    player_team: dict[int, int | None] = {}
    for r in shift_rows:
        pid = int(r["player"])
        bar = ShiftBar(start_s=float(r["start_time_s"]), end_s=float(r["end_time_s"]))
        by_player.setdefault(pid, []).append(bar)
        player_team.setdefault(pid, _parse_team(r.get("team")))

    max_end = max((b.end_s for bars in by_player.values() for b in bars), default=0.0)
    span = duration_s if duration_s is not None else max_end

    def toi(pid: int) -> float:
        return sum(b.duration_s for b in by_player[pid])

    ordered = sorted(
        by_player,
        key=lambda pid: (_team_sort_key(player_team[pid]), -toi(pid), pid),
    )

    rows: list[ChartRow] = []
    for i, pid in enumerate(ordered):
        bars = tuple(sorted(by_player[pid], key=lambda b: b.start_s))
        team = player_team[pid]
        rows.append(ChartRow(
            player=pid,
            team=team,
            row=i,
            label=_player_label(pid, team),
            toi_s=round(toi(pid), 2),
            n_shifts=len(bars),
            bars=bars,
        ))
    return ShiftChart(rows=tuple(rows), duration_s=float(span))


def _player_label(player: int, team: int | None) -> str:
    return f"P{player}" + (f" (T{team})" if team is not None else "")


def _parse_team(value) -> int | None:
    """Parse a team cell (``"0"``/``"1"``/``""``/``None``) to an int or ``None``."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------------
# Report model + loaders (read the already-written run artifacts).
# --------------------------------------------------------------------------------------------

@dataclass
class GameReport:
    """Everything the per-game report renders, assembled from a finished run directory."""

    game: dict                       # box-score game header (duration, live play, n_players, ...)
    teams: list[dict]                # per-team totals (from boxscore.json)
    players: list[dict]              # per-player rows (from players.csv), most active first
    chart: ShiftChart
    run_dir: Path
    heatmap_path: Path | None = None
    title: str = "Ball Hockey — Game Report"


def load_shifts_csv(path: str | Path) -> list[dict]:
    """Load ``shifts.csv`` rows, coercing the numeric fields the chart needs."""
    rows: list[dict] = []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "player": int(r["player"]),
                "team": _parse_team(r.get("team")),
                "shift": int(r["shift"]),
                "start_time_s": float(r["start_time_s"]),
                "end_time_s": float(r["end_time_s"]),
                "duration_s": float(r["duration_s"]),
            })
    return rows


def load_players_csv(path: str | Path) -> list[dict]:
    """Load ``players.csv`` rows as plain dicts (strings; rendering coerces as needed)."""
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def team_totals_from_players(players: list[dict]) -> list[dict]:
    """Aggregate per-team totals over true identities (rows of ``players.csv``).

    Returns one dict per team — ``team``, ``n_players`` (identity count), ``toi_s`` (summed
    on-surface shift time), ``active_s`` (summed active time) — teams ascending, unassigned last.
    """
    by_team: dict[int | None, dict] = {}
    for p in players:
        team = _parse_team(p.get("team"))
        agg = by_team.setdefault(team, {"team": team, "n_players": 0, "toi_s": 0.0, "active_s": 0.0})
        agg["n_players"] += 1
        agg["toi_s"] += _to_float(p.get("shift_seconds"))
        agg["active_s"] += _to_float(p.get("active_seconds"))
    rows = sorted(by_team.values(), key=lambda r: _team_sort_key(r["team"]))
    for r in rows:
        r["toi_s"] = round(r["toi_s"], 2)
        r["active_s"] = round(r["active_s"], 2)
    return rows


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_report(run_dir: str | Path, *, title: str | None = None) -> GameReport:
    """Assemble a ``GameReport`` from a finished run directory's artifacts.

    Reads ``players.csv`` + ``shifts.csv`` (per-player stats and the shift chart) and, when present,
    ``boxscore.json`` (game header + per-team totals) and ``heatmap.jpg``. ``players.csv``/
    ``shifts.csv`` come from a ``--phase2 --reid`` run; without them there are no per-player stats to
    surface, so this raises.
    """
    run_dir = Path(run_dir)
    players_csv = run_dir / "players.csv"
    shifts_csv = run_dir / "shifts.csv"
    if not players_csv.exists() or not shifts_csv.exists():
        raise FileNotFoundError(
            f"need players.csv + shifts.csv in {run_dir} — run `--phase2 --reid` first "
            f"(missing: {[p.name for p in (players_csv, shifts_csv) if not p.exists()]})"
        )

    players = load_players_csv(players_csv)
    shift_rows = load_shifts_csv(shifts_csv)

    game: dict = {}
    boxscore_path = run_dir / "boxscore.json"
    if boxscore_path.exists():
        box = json.loads(boxscore_path.read_text())
        game = box.get("game", {})

    # Team totals are computed from the *identities* in players.csv, not the box-score's per-track
    # teams: this is a per-player report, so a team's player count should be its true roster on the
    # surface, not its fragmented track count. (Active/TOI seconds sum identically either way.)
    teams = team_totals_from_players(players)

    duration_s = game.get("duration_s")
    chart = build_shift_chart(shift_rows, duration_s=duration_s)

    heatmap = run_dir / "heatmap.jpg"
    return GameReport(
        game=game,
        teams=teams,
        players=players,
        chart=chart,
        run_dir=run_dir,
        heatmap_path=heatmap if heatmap.exists() else None,
        title=title or "Ball Hockey — Game Report",
    )


# --------------------------------------------------------------------------------------------
# Text rendering (console summary).
# --------------------------------------------------------------------------------------------

def format_report_text(report: GameReport) -> str:
    """A compact console rendering of the report (mirrors what the HTML shows)."""
    g = report.game
    lines = [f"Game report — {report.chart.n_rows} players"]
    if g:
        lines[0] += (f", {g.get('duration_s', 0):.0f}s clip, "
                     f"{g.get('live_play_s', 0):.0f}s live play "
                     f"({g.get('n_segments', 0)} segment(s))")
    if report.teams:
        lines.append("  Teams:")
        for t in report.teams:
            tl = "T?" if t.get("team") is None else f"T{t['team']}"
            lines.append(f"    {tl:>3}: {t.get('n_players', 0):2d} players, "
                         f"{t.get('toi_s', 0):6.0f}s on surface, "
                         f"{t.get('active_s', 0):6.0f}s active")
    lines.append("  Players (most TOI first):")
    lines.append(f"    {'player':>6} {'team':>4} {'shifts':>6} {'TOI_s':>7} "
                 f"{'avg_s':>6} {'long_s':>7}")
    for row in sorted(report.chart.rows, key=lambda r: r.toi_s, reverse=True):
        tl = "T?" if row.team is None else f"T{row.team}"
        avg = row.toi_s / row.n_shifts if row.n_shifts else 0.0
        longest = max((b.duration_s for b in row.bars), default=0.0)
        lines.append(f"    {row.player:>6} {tl:>4} {row.n_shifts:>6} "
                     f"{row.toi_s:>7.1f} {avg:>6.1f} {longest:>7.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Shift-chart PNG (thin matplotlib shell over the pure layout).
# --------------------------------------------------------------------------------------------

def render_shift_chart_png(
    chart: ShiftChart,
    path: str | Path,
    *,
    dpi: int = 120,
    row_height: float = 0.36,
) -> Path:
    """Render the laid-out shift chart to a PNG (matplotlib ``broken_barh`` per player row).

    A thin shell: all ordering/positioning is already decided by ``build_shift_chart``; here we just
    draw one coloured bar per shift on each player's row, the team colour carrying who's who.
    matplotlib is imported lazily so the pure core (and its tests) need no plotting backend.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed, render straight to file
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path = Path(path)
    n = max(1, chart.n_rows)
    fig_h = max(2.0, 0.9 + n * row_height)
    fig, ax = plt.subplots(figsize=(11, fig_h), dpi=dpi)

    for row in chart.rows:
        spans = [(b.start_s, b.duration_s) for b in row.bars]
        # y grows downward so row 0 sits at the top, the way a bench chart reads.
        y = chart.n_rows - 1 - row.row
        if spans:
            ax.broken_barh(spans, (y - 0.4, 0.8), facecolors=row.color,
                           edgecolor="white", linewidth=0.5)

    ax.set_xlim(0, chart.duration_s if chart.duration_s > 0 else 1)
    ax.set_ylim(-0.6, chart.n_rows - 0.4)
    ax.set_yticks([chart.n_rows - 1 - r.row for r in chart.rows])
    ax.set_yticklabels([f"{r.label}  ·  {r.n_shifts} sh / {r.toi_s:.0f}s" for r in chart.rows],
                       fontsize=8)
    ax.set_xlabel("game time (s)")
    ax.set_title("Shift chart — time on surface")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    teams_present = sorted({r.team for r in chart.rows}, key=_team_sort_key)
    handles = [Patch(facecolor=team_color(t),
                     label=("unassigned" if t is None else f"Team {t}"))
               for t in teams_present]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------
# Self-contained HTML report (thin string-templating shell; embeds images inline).
# --------------------------------------------------------------------------------------------

def _data_uri(path: str | Path) -> str:
    """Base64 ``data:`` URI for an image, so the HTML is self-contained (no sidecar files)."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _player_table_html(report: GameReport) -> str:
    cols = [
        ("player", "Player"), ("team", "Team"), ("n_shifts", "Shifts"),
        ("shift_seconds", "TOI (s)"), ("avg_shift_s", "Avg shift (s)"),
        ("longest_shift_s", "Longest (s)"), ("active_seconds", "Active (s)"),
        ("n_fragments", "Fragments"),
    ]
    head = "".join(f"<th>{_esc(label)}</th>" for _, label in cols)
    body = []
    for p in report.players:
        team = _parse_team(p.get("team"))
        swatch = (f'<span class="swatch" style="background:{team_color(team)}"></span>'
                  f'{"—" if team is None else "T" + str(team)}')
        cells = []
        for key, _ in cols:
            cells.append(swatch if key == "team" else _esc(p.get(key, "")))
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _team_table_html(report: GameReport) -> str:
    if not report.teams:
        return ""
    rows = []
    for t in report.teams:
        team = t.get("team")
        swatch = (f'<span class="swatch" style="background:{team_color(team)}"></span>'
                  f'{"unassigned" if team is None else "Team " + str(team)}')
        rows.append(
            f"<tr><td>{swatch}</td><td>{_esc(t.get('n_players', 0))}</td>"
            f"<td>{_esc(t.get('toi_s', 0))}</td>"
            f"<td>{_esc(t.get('active_s', 0))}</td></tr>"
        )
    return ("<table><thead><tr><th>Team</th><th>Players</th><th>TOI (s)</th>"
            f"<th>Active (s)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")


def _game_header_html(report: GameReport) -> str:
    g = report.game
    if not g:
        return ""
    items = [
        ("Players", g.get("n_players", report.chart.n_rows)),
        ("Duration", f"{g.get('duration_s', 0):.0f} s"),
        ("Live play", f"{g.get('live_play_s', 0):.0f} s"),
        ("Live segments", g.get("n_segments", "")),
        ("Active fraction", f"{g.get('active_fraction', 0) * 100:.0f}%"),
        ("FPS", g.get("fps", "")),
    ]
    cards = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for k, v in items
    )
    return f'<div class="cards">{cards}</div>'


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          margin: 0; padding: 24px; line-height: 1.4; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 4px; }}
  h2 {{ font-size: 1.1rem; margin: 28px 0 8px; }}
  .sub {{ color: #888; margin: 0 0 16px; font-size: 0.9rem; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0 4px; }}
  .card {{ background: rgba(127,127,127,.12); border-radius: 10px; padding: 10px 14px; min-width: 92px; }}
  .card .k {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; color: #888; }}
  .card .v {{ font-size: 1.15rem; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .9rem; margin: 4px 0 8px; }}
  th, td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid rgba(127,127,127,.25); }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .03em; color: #888; }}
  .swatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px;
             margin-right: 6px; vertical-align: middle; }}
  img.chart, img.heatmap {{ max-width: 100%; border-radius: 10px;
             border: 1px solid rgba(127,127,127,.25); }}
  .grid2 {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
  footer {{ margin-top: 28px; color: #999; font-size: .78rem; }}
</style></head>
<body>
  <h1>{title}</h1>
  <p class="sub">Per-player time-on-surface, shift counts, and the who's-on-when shift chart.</p>
  {header}
  <h2>Shift chart</h2>
  <img class="chart" src="{chart_uri}" alt="Shift chart — time on surface per player">
  {teams_section}
  <h2>Players</h2>
  {player_table}
  {heatmap_section}
  <footer>Generated by dbh-vibes — pure rendering over players.csv / shifts.csv / boxscore.json.</footer>
</body></html>
"""


def render_report_html(
    report: GameReport,
    html_path: str | Path,
    chart_png_path: str | Path,
) -> Path:
    """Write the self-contained ``report.html``, embedding the shift-chart PNG + heatmap inline.

    Images are base64 ``data:`` URIs so the single HTML file is portable (no sidecar files needed),
    matching the "self-contained report" goal.
    """
    html_path = Path(html_path)
    teams_html = _team_table_html(report)
    teams_section = f"<h2>Team totals</h2>{teams_html}" if teams_html else ""
    heatmap_section = ""
    if report.heatmap_path is not None and Path(report.heatmap_path).exists():
        heatmap_section = (f'<h2>Position heatmap</h2>'
                           f'<img class="heatmap" src="{_data_uri(report.heatmap_path)}" '
                           f'alt="Player position heatmap">')

    html = _HTML_TEMPLATE.format(
        title=_esc(report.title),
        header=_game_header_html(report),
        chart_uri=_data_uri(chart_png_path),
        teams_section=teams_section,
        player_table=_player_table_html(report),
        heatmap_section=heatmap_section,
    )
    html_path.write_text(html)
    return html_path


@dataclass(frozen=True)
class ReportPaths:
    """Where the report artifacts landed."""

    html: Path
    chart_png: Path


def write_report(run_dir: str | Path, *, title: str | None = None) -> ReportPaths:
    """End-to-end: build the report from a run dir and write ``report.html`` + ``shift_chart.png``.

    The one call the pipeline and the standalone ``--report`` CLI both use, so both paths render the
    identical report.
    """
    run_dir = Path(run_dir)
    report = build_report(run_dir, title=title)
    chart_png = run_dir / "shift_chart.png"
    render_shift_chart_png(report.chart, chart_png)
    html = render_report_html(report, run_dir / "report.html", chart_png)
    return ReportPaths(html=html, chart_png=chart_png)
