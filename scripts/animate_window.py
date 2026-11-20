"""Animate the tracking window built for one make (play-by-play event).

    .venv/bin/python scripts/animate_window.py <game_id> <eventnum>

Replays the raw tracking frames for data/windows/index.parquet's window
around one make, in the same visual style as scripts/vendor/'s play
animation (team-colored circles, jersey numbers, a home/visitor roster
table, the same court.png) -- reusing scripts/vendor/Team.py for team
colors/names. Two reference times are annotated and highlighted once the
animation's current frame reaches them: the play-by-play's own recorded
clock_s, and this make's detected rim-crossing time
(data/derived/make_times.parquet). The title is the play's own
description from data/derived/makes.parquet.

Requires a build_windows.py-produced window (an index.parquet row with a
computed start/end) -- a make excluded before a window was ever built (not
detected, basket mismatch, unclear direction) has nothing to animate here.

Importable from a notebook: build_event_animation() returns the
matplotlib animation.FuncAnimation object for inline display, e.g.
``IPython.display.HTML(anim.to_jshtml())``.
"""

# pandas' stubs are known to be imprecise for this kind of code (see
# src/nba_assists/rim.py for the full explanation); verified by running
# this script against several games/events. Full type checking stays on
# elsewhere in the project -- this is scoped to this file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportReturnType=false

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import Circle

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(VENDOR_DIR))

from Team import Team  # noqa: E402

from nba_assists.config import Config, get_config  # noqa: E402

# Matches scripts/vendor/Constant.py exactly, for the same look.
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.0, 100.0, 0.0, 50.0
DIFF = 6.0
NORMALIZATION_COEF = 7.0
PLAYER_CIRCLE_SIZE = 12.0 / NORMALIZATION_COEF
INTERVAL_MS = 10
COL_WIDTH = 0.3
TABLE_SCALE = 1.65
TABLE_FONTSIZE = 6
X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35

BALL_COLOR = "#ff8c00"
MUTED_COLOR = "#888888"
PBP_COLOR = "#c0392b"
DETECTED_COLOR = "#1e8449"


def game_id_to_path(tracking_dir: Path) -> dict[str, Path]:
    """Canonical game_id -> its tracking Parquet path."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT game_id, filename FROM read_parquet('{tracking_dir}/*.parquet', filename=true)"
    ).fetchall()
    con.close()
    return {game_id: Path(filename) for game_id, filename in rows}


def _clock_text(quarter: int, game_clock: float, shot_clock: float) -> str:
    return "Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}".format(
        quarter, int(game_clock) % 3600 // 60, int(game_clock) % 60, shot_clock if pd.notna(shot_clock) else 0.0
    )


def _event_info(game_id: str, eventnum: int, derived_dir: Path, windows_dir: Path) -> dict:
    makes = pd.read_parquet(derived_dir / "makes.parquet")
    make_rows = makes[(makes["game_id"] == game_id) & (makes["eventnum"] == eventnum)]
    if make_rows.empty:
        raise ValueError(f"no make for game_id={game_id!r} eventnum={eventnum}")
    make = make_rows.iloc[0]

    make_times = pd.read_parquet(derived_dir / "make_times.parquet")
    mt_rows = make_times[(make_times["game_id"] == game_id) & (make_times["eventnum"] == eventnum)]
    mt = mt_rows.iloc[0] if len(mt_rows) else None

    index = pd.read_parquet(windows_dir / "index.parquet")
    idx_rows = index[(index["game_id"] == game_id) & (index["eventnum"] == eventnum)]
    if idx_rows.empty or pd.isna(idx_rows.iloc[0]["start_ts_ms"]):
        reason = idx_rows.iloc[0]["reason"] if len(idx_rows) else "no index row for this make"
        raise ValueError(
            f"no built window for game_id={game_id!r} eventnum={eventnum} (reason: {reason!r}) -- "
            "run scripts/build_windows.py first, or pick a make with a built window"
        )
    idx = idx_rows.iloc[0]

    return {
        "period": int(make["period"]),
        "clock_s": float(make["clock_s"]),
        "description": str(make["description"]),
        "detected": bool(mt["detected"]) if mt is not None else False,
        "cross_ts_ms": float(mt["cross_ts_ms"]) if mt is not None and pd.notna(mt["cross_ts_ms"]) else None,
        "cross_game_clock": float(mt["cross_game_clock"])
        if mt is not None and pd.notna(mt["cross_game_clock"])
        else None,
        "start_ts_ms": float(idx["start_ts_ms"]),
        "end_ts_ms": float(idx["end_ts_ms"]),
    }


def build_event_animation(
    game_id: str, eventnum: int, cfg: Config | None = None, display_hz: float = 10.0
) -> animation.FuncAnimation:
    """The FuncAnimation for one make's built window. Caller decides how to
    show it (``plt.show()``, or ``IPython.display.HTML(anim.to_jshtml())``
    in a notebook).

    ``display_hz`` subsamples the native 25 Hz frames for display only
    (every ``round(source_frame_rate_hz / display_hz)``th frame) -- each
    rendered frame costs tens of milliseconds (court image, ~20+
    circles/annotations, the roster table, no blitting), so a full-length
    (up to ``windows.max_window_s``) window at the native rate can take well
    over a minute to render. Pass the config's own
    ``tracking.source_frame_rate_hz`` for full fidelity.
    """
    cfg = cfg or get_config()
    derived_dir = cfg.get_path("derived")
    tracking_dir = cfg.get_path("tracking")
    roster_dir = cfg.get_path("rosters")
    windows_dir = cfg.get_path("windows")
    ball_team_id = int(cfg.sentinels.ball_team_id)
    ball_player_id = int(cfg.sentinels.ball_player_id)

    info = _event_info(game_id, eventnum, derived_dir, windows_dir)

    path = game_id_to_path(tracking_dir).get(game_id)
    if path is None:
        raise ValueError(f"no tracking file for game_id={game_id!r}")

    tracking = pd.read_parquet(
        path, columns=["ts_ms", "quarter", "game_clock", "shot_clock", "team_id", "player_id", "x", "y", "z"]
    )
    window = tracking[
        (tracking["quarter"] == info["period"])
        & (tracking["ts_ms"] >= info["start_ts_ms"])
        & (tracking["ts_ms"] <= info["end_ts_ms"])
    ]
    if window.empty:
        raise ValueError("no tracking frames found in this window")

    roster = pd.read_csv(roster_dir / (path.stem + ".csv"))
    name_jersey = {
        int(r.player_id): (f"{r.first_name} {r.last_name}", str(r.jersey)) for r in roster.itertuples(index=False)
    }
    side_by_team: dict[int, str] = {int(r.team_id): r.side for r in roster.itertuples(index=False)}

    native_hz = float(cfg.tracking.source_frame_rate_hz)
    stride = max(1, round(native_hz / display_hz))
    ts_values = np.array(sorted(window["ts_ms"].unique()))[::stride]
    players = window[window["team_id"] != ball_team_id]
    player_ids = sorted(int(p) for p in players["player_id"].unique())
    team_by_player = players.groupby("player_id")["team_id"].first()

    full_index = pd.MultiIndex.from_product([ts_values, player_ids], names=["ts_ms", "player_id"])
    piv = players.set_index(["ts_ms", "player_id"])[["x", "y"]].reindex(full_index)
    xy = piv.to_numpy().reshape(len(ts_values), len(player_ids), 2)
    present = ~np.isnan(xy[:, :, 0])

    ball = window[(window["team_id"] == ball_team_id) & (window["player_id"] == ball_player_id)]
    ball_xyz = ball.drop_duplicates("ts_ms").set_index("ts_ms")[["x", "y", "z"]].reindex(ts_values).to_numpy()

    clocks = (
        window.drop_duplicates("ts_ms").set_index("ts_ms")[["game_clock", "shot_clock"]].reindex(ts_values)
    )
    game_clock_arr = clocks["game_clock"].to_numpy()
    shot_clock_arr = clocks["shot_clock"].to_numpy()

    fig = plt.figure(figsize=(7, 5.2))
    ax = plt.axes(xlim=(X_MIN, X_MAX), ylim=(Y_MIN, Y_MAX))
    ax.axis("off")
    ax.grid(False)
    fig.suptitle(info["description"], fontsize=11)

    clock_info = ax.annotate(
        "", xy=(X_CENTER, Y_CENTER), color="black", horizontalalignment="center", verticalalignment="center"
    )
    pbp_text = f"play-by-play time: {info['clock_s']:.0f}s"
    detected_text = (
        f"detected crossing: {info['cross_game_clock']:.2f}s" if info["cross_game_clock"] is not None else "not detected"
    )
    # axes-fraction coordinates, above the court image, so these never get
    # obscured by a player or the ball passing underneath them
    pbp_note = ax.annotate(
        pbp_text, xy=(0.0, 1.08), xycoords="axes fraction", color=MUTED_COLOR, fontsize=8, horizontalalignment="left"
    )
    detected_note = ax.annotate(
        detected_text, xy=(0.0, 1.02), xycoords="axes fraction", color=MUTED_COLOR, fontsize=8, horizontalalignment="left"
    )

    player_circles = {
        pid: Circle((0, 0), PLAYER_CIRCLE_SIZE, color=Team(int(team_by_player[pid])).color) for pid in player_ids
    }
    player_annotations = {
        pid: ax.annotate(
            name_jersey.get(pid, ("?", "?"))[1],
            xy=(0, 0), color="w", horizontalalignment="center", verticalalignment="center", fontweight="bold",
        )
        for pid in player_ids
    }
    for circle in player_circles.values():
        ax.add_patch(circle)
    ball_circle = Circle((0, 0), PLAYER_CIRCLE_SIZE, color=BALL_COLOR)
    ax.add_patch(ball_circle)

    _add_roster_table(player_ids, team_by_player, side_by_team, name_jersey)

    court = plt.imread(str(VENDOR_DIR / "court.png"))
    plt.imshow(court, zorder=0, extent=(X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN))

    def update(i: int):
        for a, pid in enumerate(player_ids):
            circle = player_circles[pid]
            if present[i, a]:
                circle.center = (xy[i, a, 0], xy[i, a, 1])
                circle.set_alpha(1.0)
                player_annotations[pid].set_position(circle.center)
                player_annotations[pid].set_alpha(1.0)
            else:
                circle.set_alpha(0.0)
                player_annotations[pid].set_alpha(0.0)

        ball_circle.center = (ball_xyz[i, 0], ball_xyz[i, 1])
        ball_circle.radius = max(ball_xyz[i, 2], 0.0) / NORMALIZATION_COEF + PLAYER_CIRCLE_SIZE / 3
        clock_info.set_text(_clock_text(info["period"], game_clock_arr[i], shot_clock_arr[i]))

        if game_clock_arr[i] <= info["clock_s"]:
            pbp_note.set_color(PBP_COLOR)
            pbp_note.set_fontweight("bold")
        if info["cross_ts_ms"] is not None and ts_values[i] >= info["cross_ts_ms"]:
            detected_note.set_color(DETECTED_COLOR)
            detected_note.set_fontweight("bold")

        return list(player_circles.values()) + [ball_circle]

    return animation.FuncAnimation(fig, update, frames=len(ts_values), interval=INTERVAL_MS)


def _add_roster_table(
    player_ids: list[int],
    team_by_player: pd.Series,
    side_by_team: dict[int, str],
    name_jersey: dict[int, tuple[str, str]],
) -> None:
    home_ids = [pid for pid in player_ids if side_by_team.get(int(team_by_player[pid])) == "home"]
    visitor_ids = [pid for pid in player_ids if side_by_team.get(int(team_by_player[pid])) == "visitor"]
    n_rows = max(len(home_ids), len(visitor_ids), 1)

    def label(pid: int | None) -> str:
        if pid is None:
            return ""
        name, jersey = name_jersey.get(pid, ("?", "?"))
        return f"{name} #{jersey}"

    home_team_id = int(team_by_player[home_ids[0]]) if home_ids else None
    visitor_team_id = int(team_by_player[visitor_ids[0]]) if visitor_ids else None
    home_color = Team(home_team_id).color if home_team_id is not None else "#444444"
    visitor_color = Team(visitor_team_id).color if visitor_team_id is not None else "#444444"
    home_name = Team(home_team_id).name if home_team_id is not None else "home"
    visitor_name = Team(visitor_team_id).name if visitor_team_id is not None else "visitor"

    rows = [
        (label(home_ids[i] if i < len(home_ids) else None), label(visitor_ids[i] if i < len(visitor_ids) else None))
        for i in range(n_rows)
    ]
    table = plt.table(
        cellText=rows,
        colLabels=(home_name, visitor_name),
        colColours=(home_color, visitor_color),
        colWidths=[COL_WIDTH, COL_WIDTH],
        loc="bottom",
        cellColours=[(home_color, visitor_color) for _ in rows],
        fontsize=TABLE_FONTSIZE,
        cellLoc="center",
    )
    table.scale(1, TABLE_SCALE)
    for cell in table.get_celld().values():
        cell.get_text().set_color("white")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: animate_window.py <game_id> <eventnum>")
        return 1
    game_id, eventnum = sys.argv[1], int(sys.argv[2])
    anim = build_event_animation(game_id, eventnum)
    plt.show()
    del anim
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
