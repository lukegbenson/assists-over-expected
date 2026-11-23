"""Pass attribution: trace who last had the ball before the shooter.

Works on the real, per-player tracking frames inside a Step-4 window
(data/windows/index.parquet's start_ts_ms through the make's own rim
crossing) -- not the anonymized model-input tensors in data/windows/*.npz,
which discard player identity on purpose. See PROJECT.md ("Pass
attribution") and config.yaml's passer: section.

The pipeline, per make:

1. ``possession_geometry`` -- once per make, independent of any
   hold_dist/hold_height threshold: the nearest offensive player to the
   ball each frame, its distance, and the ball's height. Computed once and
   reused across a hold_dist/hold_height grid search
   (scripts/attribute_passers.py), since this step is the expensive one (it
   touches the raw tracking rows).
2. ``classify_held`` -- a frame is "held" by the nearest offensive player
   when it's within hold_dist and the ball is at or below hold_height;
   otherwise the ball is "free" (in flight, loose, or bouncing away). Cheap:
   just two threshold comparisons over the geometry from step 1.
3. ``possession_runs`` -- consecutive held-by-the-same-player frames,
   merged across gaps of at most dribble_gap_s (so a dribble's brief
   bounces don't split one possession into several), then dropped below
   min_run_s (too short to be a real possession rather than an incidental
   touch).
4. ``backward_trace`` -- the recorded shooter is never in doubt (it's a
   play-by-play fact, not something to detect), so it anchors on the
   *last* run instead of trusting that run's own player_id. When that run
   really is the shooter's, step back one run for the passer: a different
   offensive player there is the passer; nothing there, or the shooter's
   own run reaching back to the window's first frame, mean no passer can be
   traced ("NONE", represented as ``None``). When it *isn't* the shooter's
   -- the shooter's own catch never registered as a run at all (e.g. the
   ball was caught above hold_height on an alley-oop) -- that last run is,
   by elimination, the most recent passer, and ``trace_problem`` flags the
   mismatch for later review rather than silently trusting a geometric
   attribution that was never actually anchored on the shooter.
5. ``attribute_passer`` -- the play's assister when assisted (the ground
   truth), otherwise the traced passer (or ``None``).

Steps 2-4 operate on plain NumPy arrays (not DataFrames): this is the hot
path of scripts/attribute_passers.py's grid search over hold_dist,
hold_height, min_run_s and dribble_gap_s, run once per (make, parameter
combination) pair, so it avoids per-call pandas overhead on purpose.
"NONE" possession is represented as player_id -1 in these arrays (SportVU
player ids are never negative) and as ``None`` at the DataFrame/dict
boundary.
"""

# pandas' stubs are known to be imprecise for this kind of code (see
# src/nba_assists/rim.py for the full explanation); verified against
# tests/test_passer.py and by running scripts/attribute_passers.py. Full
# type checking stays on elsewhere in the project -- this is scoped to this
# file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportReturnType=false

from __future__ import annotations

import numpy as np
import pandas as pd

NO_PLAYER = -1  # sentinel for "nobody" in the NumPy-array hot path


def possession_geometry(
    player_positions_q: pd.DataFrame,
    ball_xyz_q: pd.DataFrame,
    offense_team_id: int,
    kept_ts: np.ndarray,
) -> dict:
    """Per-frame possession geometry for one make's window, independent of
    any hold_dist/hold_height threshold.

    ``player_positions_q`` is one quarter's ``nba_assists.direction.
    player_positions()`` output (ts_ms, team_id, player_id, x, y);
    ``ball_xyz_q`` is that quarter's ball x/y/z indexed by ts_ms (e.g.
    ``frame_table()``'s ball_x/ball_y/ball_z, renamed). ``kept_ts`` is the
    make's own kept frames (already restricted to clock-running frames in
    ``[start_ts_ms, cross_ts_ms]`` by the caller).

    Returns a dict of equal-length arrays over ``kept_ts``: ``ts``,
    ``off_player_id`` (NO_PLAYER if no offensive player was present that
    frame), ``off_dist`` (+inf where absent), ``ball_z`` (NaN where the
    ball itself is missing that frame).
    """
    n = len(kept_ts)
    ball = (
        ball_xyz_q.drop_duplicates("ts_ms", keep="first")
        .set_index("ts_ms")[["x", "y", "z"]]
        .reindex(kept_ts)
    )
    ball_xy = ball[["x", "y"]].to_numpy()
    ball_z = ball["z"].to_numpy()
    ball_missing = np.isnan(ball_xy[:, 0])

    players = player_positions_q.loc[
        (player_positions_q["team_id"] == offense_team_id) & (player_positions_q["ts_ms"].isin(kept_ts))
    ]
    if players.empty:
        off_player_id = np.full(n, NO_PLAYER, dtype=np.int64)
        off_dist = np.full(n, np.inf)
    else:
        player_ids = sorted(int(p) for p in players["player_id"].unique())
        m = len(player_ids)
        full_index = pd.MultiIndex.from_product([kept_ts, player_ids], names=["ts_ms", "player_id"])
        xy = players.set_index(["ts_ms", "player_id"])[["x", "y"]].reindex(full_index).to_numpy().reshape(n, m, 2)

        # ts is the slower-varying axis of full_index, so a plain reshape
        # already lines (n, m, 2) up with kept_ts row-by-row; broadcast the
        # ball's own (n, 2) against it directly, no row-repeating needed.
        d = np.hypot(xy[:, :, 0] - ball_xy[:, 0:1], xy[:, :, 1] - ball_xy[:, 1:2])
        d = np.where(np.isnan(d), np.inf, d)
        best = np.argmin(d, axis=1)
        row_idx = np.arange(n)
        pid_grid = np.tile(np.array(player_ids, dtype=np.int64), (n, 1))
        off_player_id, off_dist = pid_grid[row_idx, best], d[row_idx, best]

    off_player_id = np.where(ball_missing, NO_PLAYER, off_player_id)
    off_dist = np.where(ball_missing, np.inf, off_dist)

    return {"ts": kept_ts, "off_player_id": off_player_id, "off_dist": off_dist, "ball_z": ball_z}


def classify_held(geometry: dict, hold_dist: float, hold_height: float) -> np.ndarray:
    """``held_player_id`` per frame: ``geometry["off_player_id"]`` where
    within ``hold_dist`` and ``ball_z`` is at most ``hold_height``, else
    ``NO_PLAYER``. A missing ball (``ball_z`` is NaN) is never held."""
    ball_z = geometry["ball_z"]
    held_ok = (geometry["off_dist"] <= hold_dist) & (ball_z <= hold_height) & ~np.isnan(ball_z)
    return np.where(held_ok, geometry["off_player_id"], NO_PLAYER)


def possession_runs(ts: np.ndarray, held_player_id: np.ndarray, min_run_s: float, dribble_gap_s: float) -> dict:
    """Consecutive held-by-the-same-player stretches, merged across gaps of
    at most ``dribble_gap_s`` (same player on both sides), then dropped
    below ``min_run_s``. Returns ``{"player_id", "start_ts", "end_ts"}``
    arrays, one entry per surviving run, chronological (``ts`` must already
    be sorted ascending).
    """
    held_mask = held_player_id != NO_PLAYER
    if not held_mask.any():
        return {"player_id": np.array([], dtype=np.int64), "start_ts": np.array([]), "end_ts": np.array([])}

    run_ts = ts[held_mask]
    run_pid = held_player_id[held_mask]

    gap_ms = np.empty_like(run_ts)
    gap_ms[0] = 0.0
    gap_ms[1:] = np.diff(run_ts)
    pid_changed = run_pid != np.concatenate(([run_pid[0] - 1], run_pid[:-1]))
    new_run = pid_changed | (gap_ms > dribble_gap_s * 1000.0)
    new_run[0] = True
    run_id = np.cumsum(new_run) - 1

    n_runs = int(run_id[-1]) + 1
    boundaries = np.arange(n_runs)
    start_idx = np.searchsorted(run_id, boundaries, side="left")
    end_idx = np.searchsorted(run_id, boundaries, side="right") - 1

    start_ts = run_ts[start_idx]
    end_ts = run_ts[end_idx]
    player_id = run_pid[start_idx]
    duration_s = (end_ts - start_ts) / 1000.0

    keep = duration_s >= min_run_s
    return {
        "player_id": player_id[keep],
        "start_ts": start_ts[keep].astype(float),
        "end_ts": end_ts[keep].astype(float),
    }


def backward_trace(runs: dict, shooter_id: int, window_start_ts: float) -> dict:
    """Find the passer, anchored on the play-by-play's own ``shooter_id``
    rather than on whatever the last run's player_id happens to be.

    ``runs`` is ``possession_runs()``'s output, already restricted by the
    caller to frames up to and including the rim crossing (so its last
    entry is the most recent one before the shot). ``window_start_ts`` is
    the make's own kept window's first frame
    (``data/windows/index.parquet``'s ``start_ts_ms``).

    If the last run really is the shooter's, step back one run: a
    different offensive player there is the passer; no earlier run, or the
    shooter's own run reaching back to the window's start, mean no passer
    can be traced (NONE, ``None``).

    If it *isn't* the shooter's -- the shooter's own catch/hold never
    registered as a run at all (e.g. the ball was caught above hold_height
    on an alley-oop) -- that last run is, by elimination, the most recent
    passer: whoever held the ball right before the shooter's (undetected)
    touch. ``trace_problem`` flags this mismatch either way, so it can be
    reviewed separately from a normal trace.

    Returns ``{"traced_passer_id", "trace_problem", "shooter_run_player_id"}``.
    ``traced_passer_id``/``shooter_run_player_id`` are ``None`` for "none
    identified"/"no run matched the shooter" respectively.
    """
    n_runs = len(runs["player_id"])
    if n_runs == 0:
        return {"traced_passer_id": None, "trace_problem": True, "shooter_run_player_id": None}

    last_player_id = int(runs["player_id"][-1])
    last_start = runs["start_ts"][-1]

    if last_player_id != int(shooter_id):
        return {"traced_passer_id": last_player_id, "trace_problem": True, "shooter_run_player_id": None}

    if n_runs < 2 or last_start <= window_start_ts:
        return {"traced_passer_id": None, "trace_problem": False, "shooter_run_player_id": last_player_id}

    prev_player_id = int(runs["player_id"][-2])
    if prev_player_id == last_player_id:
        return {"traced_passer_id": None, "trace_problem": False, "shooter_run_player_id": last_player_id}

    return {"traced_passer_id": prev_player_id, "trace_problem": False, "shooter_run_player_id": last_player_id}


def attribute_passer(assister_id: float | int | None, assisted: bool, traced_passer_id: int | None) -> int | None:
    """``assister_id`` when the play was assisted (the ground truth);
    otherwise the traced passer (or ``None`` for NONE)."""
    if assisted and assister_id is not None and not pd.isna(assister_id):
        return int(assister_id)
    return traced_passer_id
