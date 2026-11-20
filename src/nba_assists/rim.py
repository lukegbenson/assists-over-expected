"""Rim-crossing detection: recover a make's true time from ball tracking.

Used by scripts/detect_makes.py to measure play-by-play clock offset: the
ball's actual downward crossing through the rim, found in SportVU frames, is
compared against the recorded play-by-play ``clock_s`` for the same make.
"""

# pandas' stubs are known to be imprecise for this kind of code (indexing a
# Series/DataFrame by label, boolean masks, .sort_values()/.nunique()/
# .idxmin()): they report a union of every dtype pandas could theoretically
# produce, which the type checker then can't narrow back down, rather than
# any error in the code itself. Verified against tests/test_rim.py. Full
# type checking stays on elsewhere in the project -- this is scoped to this
# file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportReturnType=false

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_assists.config import Config

CANDIDATE_COLUMNS = [
    "basket",
    "basket_x",
    "cross_ts_ms",
    "cross_game_clock",
    "offset_s",
    "clock_check_passed",
    "clock_drop_s",
]


def ball_rows(tracking: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Ball-only rows (the ``sentinels:`` team/player id), sorted by ts_ms."""
    ball_team_id = int(cfg.sentinels.ball_team_id)
    ball_player_id = int(cfg.sentinels.ball_player_id)
    out = tracking.loc[
        (tracking["team_id"] == ball_team_id) & (tracking["player_id"] == ball_player_id),
        ["ts_ms", "quarter", "game_clock", "x", "y", "z"],
    ]
    return out.sort_values("ts_ms", kind="mergesort").reset_index(drop=True)


def _clock_running_check(
    all_ts: np.ndarray,
    all_clock: np.ndarray,
    prev_ts: float,
    prev_clock: float,
    clock_check_s: float,
    clock_min_drop_s: float,
    buzzer_window_s: float,
) -> tuple[bool, float]:
    """Was the clock actually running just before this frame?

    Looked up against the whole quarter's ball frames, not the make's own
    search window, because buzzer_window_s can be wider than that window.
    """
    if prev_clock <= 0.0:
        # The clock already reads 0 going into the crossing (ball still in
        # the air when it expired) -- check further back, over
        # buzzer_window_s, for the moment it was still ticking down, rather
        # than the usual short clock_check_s lookback.
        lookback_start = prev_ts - buzzer_window_s * 1000.0
        in_lookback = (all_ts >= lookback_start) & (all_ts <= prev_ts)
        passed = bool(np.any(all_clock[in_lookback] > 0.0))
        return passed, 0.0

    target_ts = prev_ts - clock_check_s * 1000.0
    idx = int(np.searchsorted(all_ts, target_ts))
    nearby = [i for i in (idx - 1, idx) if 0 <= i < len(all_ts)]
    nearest = min(nearby, key=lambda i: abs(all_ts[i] - target_ts))
    drop = float(all_clock[nearest] - prev_clock)
    return drop >= clock_min_drop_s, drop


def _none_result(reason: str, n_rejected_clock: int = 0) -> dict:
    return {
        "detected": False,
        "reason": reason,
        "case": "none",
        "n_candidates": 0,
        "n_rejected_clock": n_rejected_clock,
        "cross_ts_ms": None,
        "cross_game_clock": None,
        "basket_x": None,
        "offset_s": None,
        "gap_s": None,
    }


def _result(
    case: str, row: pd.Series, n_candidates: int, n_rejected_clock: int, gap_s: float | None
) -> dict:
    return {
        "detected": True,
        "reason": "",
        "case": case,
        "n_candidates": n_candidates,
        "n_rejected_clock": n_rejected_clock,
        "cross_ts_ms": float(row["cross_ts_ms"]),
        "cross_game_clock": float(row["cross_game_clock"]),
        "basket_x": float(row["basket_x"]),
        "offset_s": float(row["offset_s"]),
        "gap_s": gap_s,
    }


def _search_window(quarter_frames: pd.DataFrame, clock_s: float, cfg: Config) -> pd.DataFrame:
    """The slice of a quarter's ball frames detect_make()/window_has_gap() search."""
    rd = cfg.rim_detection
    search_before = float(rd.search_before_s)
    search_after = float(rd.search_after_s)
    return quarter_frames[
        (quarter_frames["game_clock"] <= clock_s + search_before)
        & (quarter_frames["game_clock"] >= clock_s - search_after)
    ]


def window_has_gap(quarter_frames: pd.DataFrame, clock_s: float, cfg: Config) -> bool:
    """Was there a tracking gap, while the clock was running, inside the
    search window detect_make() uses for this make?

    Uses the same max_gap_running_s threshold as scripts/tracking_inventory.py
    (``inventory.max_gap_running_s``), applied to the ball frames actually
    searched here rather than every object on the court, since what matters
    is whether the ball itself was tracked throughout the window. Fewer than
    two frames in the window counts as a gap -- there's nothing to confirm
    coverage with.
    """
    gap_limit_ms = float(cfg.inventory.max_gap_running_s) * 1000.0
    window = _search_window(quarter_frames, clock_s, cfg)
    if len(window) < 2:
        return True

    ts = window["ts_ms"].to_numpy()
    clock = window["game_clock"].to_numpy()
    gaps = np.diff(ts)
    running = clock[:-1] > clock[1:]
    return bool(np.any((gaps >= gap_limit_ms) & running))


def detect_make(
    quarter_frames: pd.DataFrame, clock_s: float, cfg: Config, basket_x: float | None = None
) -> tuple[pd.DataFrame, dict]:
    """Rim-crossing candidates and the chosen result for one make.

    ``quarter_frames`` is every ball frame (ts_ms, game_clock, x, y, z) for
    the make's own quarter, sorted by ts_ms with a reset index -- callers
    pass the whole quarter rather than a pre-cut window, because the
    clock-running and buzzer checks look back further than the search
    window around clock_s.

    ``basket_x``, when given, restricts the geometric search to the single
    basket at that x-coordinate -- e.g. ``attack_direction.parquet``'s
    ``basket_x`` for the scoring team's period -- instead of checking both
    baskets, which rules out ever matching a crossing at the wrong end of
    the court. Left as ``None``, both baskets are searched and a crossing at
    either can be returned (see the ``multi_diff`` case below).

    Returns ``(candidates, result)``: ``candidates`` has one row per
    geometric rim crossing found in the search window (``CANDIDATE_COLUMNS``,
    possibly empty), each tagged with whether it passed the clock-running
    check. ``result`` is the single chosen outcome for this make (the
    ``make_times.parquet`` row; see PROJECT.md).
    """
    rd = cfg.rim_detection
    tol = float(rd.rim_radius_tol)
    rim_z = float(cfg.court.rim_height_z)
    clock_check_s = float(rd.clock_check_s)
    clock_min_drop_s = float(rd.clock_min_drop_s)
    buzzer_window_s = float(rd.buzzer_window_s)

    all_baskets = {name: (float(b.x), float(b.y)) for name, b in cfg.court.baskets.items()}
    baskets = (
        {name: xy for name, xy in all_baskets.items() if xy[0] == basket_x}
        if basket_x is not None
        else all_baskets
    )

    window = _search_window(quarter_frames, clock_s, cfg)
    if len(window) < 2:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS), _none_result("no ball data")

    ts = window["ts_ms"].to_numpy()
    clock = window["game_clock"].to_numpy()
    x = window["x"].to_numpy()
    y = window["y"].to_numpy()
    z = window["z"].to_numpy()
    all_ts = quarter_frames["ts_ms"].to_numpy()
    all_clock = quarter_frames["game_clock"].to_numpy()

    near_rim_any = False
    rows: list[dict] = []
    for i in range(len(window) - 1):
        is_downward_crossing = z[i] > rim_z and z[i + 1] <= rim_z
        # Position at the same height-crossing fraction, not either frame's
        # own position, since the two frames straddle the rim rather than
        # sitting exactly at it.
        frac = (z[i] - rim_z) / (z[i] - z[i + 1]) if is_downward_crossing else None
        for name, (bx, by) in baskets.items():
            near_rim_any = near_rim_any or _dist(x[i], y[i], bx, by) <= tol or _dist(
                x[i + 1], y[i + 1], bx, by
            ) <= tol
            if frac is None:
                continue
            ix = x[i] + frac * (x[i + 1] - x[i])
            iy = y[i] + frac * (y[i + 1] - y[i])
            if _dist(ix, iy, bx, by) > tol:
                continue
            cross_ts = ts[i] + frac * (ts[i + 1] - ts[i])
            cross_clock = clock[i] + frac * (clock[i + 1] - clock[i])
            passed, drop = _clock_running_check(
                all_ts, all_clock, ts[i], clock[i],
                clock_check_s, clock_min_drop_s, buzzer_window_s,
            )
            rows.append(
                {
                    "basket": name,
                    "basket_x": bx,
                    "cross_ts_ms": cross_ts,
                    "cross_game_clock": cross_clock,
                    "offset_s": clock_s - cross_clock,
                    "clock_check_passed": passed,
                    "clock_drop_s": drop,
                }
            )

    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    if candidates.empty:
        reason = "no downward crossing" if near_rim_any else "ball never near a rim"
        return candidates, _none_result(reason)

    valid = candidates[candidates["clock_check_passed"]]
    n_rejected_clock = int((~candidates["clock_check_passed"]).sum())
    if valid.empty:
        return candidates, _none_result("all rejected by clock check", n_rejected_clock)

    if len(valid) == 1:
        return candidates, _result("single", valid.iloc[0], 1, n_rejected_clock, None)

    gap_s = float(valid["cross_game_clock"].max() - valid["cross_game_clock"].min())
    if valid["basket"].nunique() == 1:
        row = valid.sort_values("cross_ts_ms", kind="mergesort").iloc[-1]
        return candidates, _result("multi_same", row, len(valid), n_rejected_clock, gap_s)

    row = valid.loc[valid["offset_s"].abs().idxmin()]
    return candidates, _result("multi_diff", row, len(valid), n_rejected_clock, gap_s)


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
