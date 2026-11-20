"""Cut a fixed-end tracking window leading up to each detected make.

Used by scripts/build_windows.py to build model-input sequences (position
tracks for the ball and both teams' players) ending at a make's rim
crossing. See PROJECT.md and config.yaml's windows: section.
"""

# pandas' stubs are known to be imprecise for this kind of code (see
# src/nba_assists/rim.py for the full explanation); verified against
# tests/test_windows.py and by running scripts/build_windows.py. Full type
# checking stays on elsewhere in the project -- this is scoped to this file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportReturnType=false

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_offset_used(offset_s: float, offset_nudge_s: float) -> float:
    """offset_s nudged toward 0 (never past it): the correction applied to a
    previous event's clock_s so the window errs toward starting later."""
    return min(offset_s + offset_nudge_s, 0.0)


def resolve_window_start(
    *,
    prev_same_period: bool,
    prev_msgtype: float | None,
    prev_clock_s: float | None,
    made_shot_code: int,
    prev_make_detected: bool | None,
    prev_make_cross_game_clock: float | None,
    offset_s: float,
    period_max_game_clock: float,
    start_margin_s: float,
    after_prev_make_s: float,
    offset_nudge_s: float,
) -> dict:
    """Where (in game_clock seconds) this make's window should start.

    Priority: (1) no usable previous event in this period -> the period's
    own start; (2) the previous event was a make with its own detected
    crossing -> anchor on that crossing instead of a corrected pbp time;
    (3) otherwise, correct the previous event's clock_s by this make's own
    offset (nudged toward 0) and back off by start_margin_s.

    Returns ``{"start_clock", "offset_used", "previous_event_problem"}``;
    ``offset_used`` is None where the offset-correction formula wasn't the
    branch taken.
    """
    if not prev_same_period or prev_clock_s is None or pd.isna(prev_clock_s):
        return {"start_clock": period_max_game_clock, "offset_used": None, "previous_event_problem": False}

    prev_is_make = prev_msgtype is not None and not pd.isna(prev_msgtype) and int(prev_msgtype) == made_shot_code

    if prev_is_make and prev_make_detected:
        assert prev_make_cross_game_clock is not None, (
            "prev_make_detected=True must carry its own cross_game_clock"
        )
        return {
            "start_clock": prev_make_cross_game_clock - after_prev_make_s,
            "offset_used": None,
            "previous_event_problem": False,
        }

    offset_used = compute_offset_used(offset_s, offset_nudge_s)
    start_clock = prev_clock_s - offset_used - start_margin_s
    return {
        "start_clock": start_clock,
        "offset_used": offset_used,
        "previous_event_problem": prev_is_make and not prev_make_detected,
    }


def rotate_and_normalize(
    x: float | np.ndarray,
    y: float | np.ndarray,
    attack_right: bool,
    length_x: float,
    width_y: float,
    basket_left_x: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """(u, v): court coordinates with every play attacking the left basket.

    Rotates 180 degrees when the team attacks the right basket, then shifts
    so the attacked basket sits at u=0 (v is centered on the court's width).
    """
    if attack_right:
        x = length_x - x
        y = width_y - y
    return x - basket_left_x, y - width_y / 2.0


def build_window(
    frame_table_q: pd.DataFrame,
    start_clock: float,
    end_ts_ms: float,
    max_window_s: float,
    max_gap_running_s: float,
    max_missing_frame_share: float,
    players_on_court: int,
) -> dict:
    """Resolve raw window bounds against one quarter's frame table, truncate
    to at most max_window_s, filter to clock-running frames, and compute
    length/coverage-gap/missing-frame diagnostics.

    ``frame_table_q`` is one quarter's output of
    ``nba_assists.direction.frame_table()`` with an added ``clock_running``
    column (``game_clock`` strictly decreasing vs. the previous native
    frame), sorted by ts_ms.

    ``coverage_gap`` is a genuine tracking gap during live play -- a gap
    between native frames whose transition itself was clock-running -- not
    the (expected, harmless) jump a dead-ball stretch leaves once its frozen
    frames are filtered out. ``many_missing_frames`` reuses
    ``inventory.max_missing_frame_share``'s exact definition (share of
    frames missing a player or the ball) over the frames actually in range.
    """
    ts = frame_table_q["ts_ms"].to_numpy()

    before_start = frame_table_q.loc[frame_table_q["game_clock"] <= start_clock, "ts_ms"]
    raw_start_ts = float(before_start.min()) if len(before_start) else float(ts.min())
    raw_start_ts = max(raw_start_ts, float(ts.min()))

    if end_ts_ms - raw_start_ts > max_window_s * 1000.0:
        raw_start_ts = end_ts_ms - max_window_s * 1000.0

    in_range = frame_table_q.loc[(frame_table_q["ts_ms"] >= raw_start_ts) & (frame_table_q["ts_ms"] <= end_ts_ms)]
    kept = in_range.loc[in_range["clock_running"].fillna(False)]
    kept_ts = kept["ts_ms"].to_numpy()

    if len(kept_ts) < 2:
        return {
            "kept_ts": kept_ts,
            "length_s": 0.0,
            "length_frames": int(len(kept_ts)),
            "coverage_gap": True,
            "many_missing_frames": True,
            "start_ts_ms": raw_start_ts,
            "end_ts_ms": end_ts_ms,
        }

    length_s = float((kept_ts[-1] - kept_ts[0]) / 1000.0)

    range_ts = in_range["ts_ms"].to_numpy()
    range_running = in_range["clock_running"].fillna(False).to_numpy()
    gap_ms = np.diff(range_ts)
    running_transition = range_running[1:]
    coverage_gap = bool(np.any(gap_ms[running_transition] > max_gap_running_s * 1000.0))

    missing = (in_range["n_player_rows"] < players_on_court) | (in_range["n_ball_rows"] != 1)
    many_missing_frames = bool(missing.mean() > max_missing_frame_share)

    return {
        "kept_ts": kept_ts,
        "length_s": length_s,
        "length_frames": int(len(kept_ts)),
        "coverage_gap": coverage_gap,
        "many_missing_frames": many_missing_frames,
        "start_ts_ms": float(kept_ts[0]),
        "end_ts_ms": float(kept_ts[-1]),
    }


def build_agent_tracks(
    player_positions_q: pd.DataFrame,
    ball_xyz_q: pd.DataFrame,
    kept_ts: np.ndarray,
    shooter_team_id: int,
    shooter_id: int,
) -> dict:
    """Per-window agent tracks: the ball, then every player_id appearing in
    the kept frames (offense first is not enforced here; callers may sort
    by ``roles`` if a fixed grouping is wanted).

    Returns ``pos`` [T, A, 3] (float32; u, v, z -- z is 0 for players),
    ``mask`` [T, A] (frame x agent presence), ``roles`` [A, 3] (is_ball,
    is_offense, is_shooter -- time-invariant per agent), and ``player_ids``
    (the A-1 player ids backing columns 1..A-1, for the caller's own
    bookkeeping; never written to the model-input arrays themselves).
    """
    t_count = len(kept_ts)
    players = player_positions_q[player_positions_q["ts_ms"].isin(kept_ts)]
    player_ids = sorted(int(p) for p in players["player_id"].unique())
    n_players = len(player_ids)
    n_agents = 1 + n_players

    pos = np.zeros((t_count, n_agents, 3), dtype=np.float32)
    mask = np.zeros((t_count, n_agents), dtype=bool)
    roles = np.zeros((n_agents, 3), dtype=bool)
    roles[0, 0] = True  # is_ball

    ball = ball_xyz_q.set_index("ts_ms").reindex(kept_ts)
    ball_mask = ball["x"].notna().to_numpy()
    pos[ball_mask, 0, 0] = ball.loc[ball_mask, "x"].to_numpy()
    pos[ball_mask, 0, 1] = ball.loc[ball_mask, "y"].to_numpy()
    pos[ball_mask, 0, 2] = ball.loc[ball_mask, "z"].to_numpy()
    mask[:, 0] = ball_mask

    if n_players:
        full_index = pd.MultiIndex.from_product([kept_ts, player_ids], names=["ts_ms", "player_id"])
        piv = players.set_index(["ts_ms", "player_id"])[["x", "y"]].reindex(full_index)
        xy = piv.to_numpy().reshape(t_count, n_players, 2)
        valid = ~np.isnan(xy[:, :, 0])
        pos[:, 1:, 0] = xy[:, :, 0]
        pos[:, 1:, 1] = xy[:, :, 1]
        mask[:, 1:] = valid

        team_by_player = players.groupby("player_id")["team_id"].first()
        for a, pid in enumerate(player_ids, start=1):
            roles[a, 1] = int(team_by_player.loc[pid]) == shooter_team_id
            roles[a, 2] = pid == shooter_id

    return {"pos": pos, "mask": mask, "roles": roles, "player_ids": player_ids}
