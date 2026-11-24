"""src/nba_assists/windows.py: window-start resolution, rotation, frame
selection, and agent-track extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_assists.windows import (
    build_agent_tracks,
    build_window,
    compute_offset_used,
    resolve_window_start,
    rotate_and_normalize,
    find_live_stretch_start,
)

MADE_SHOT_CODE = 1


def test_compute_offset_used_nudges_toward_zero_but_not_past():
    assert compute_offset_used(-1.5, 0.5) == pytest.approx(-1.0)
    assert compute_offset_used(-0.2, 0.5) == pytest.approx(0.0)  # capped at 0, not +0.3


def test_resolve_window_start_different_period_uses_period_start():
    result = resolve_window_start(
        prev_same_period=False, prev_msgtype=MADE_SHOT_CODE, prev_clock_s=700.0,
        made_shot_code=MADE_SHOT_CODE, prev_make_detected=True, prev_make_cross_game_clock=699.0,
        offset_s=-1.5, period_max_game_clock=720.0, start_margin_s=0.2, after_prev_make_s=0.3,
        offset_nudge_s=0.5,
    )
    assert result["start_clock"] == pytest.approx(720.0)
    assert result["offset_used"] is None
    assert result["previous_event_problem"] is False


def test_resolve_window_start_no_previous_event_treated_like_different_period():
    result = resolve_window_start(
        prev_same_period=False, prev_msgtype=None, prev_clock_s=None,
        made_shot_code=MADE_SHOT_CODE, prev_make_detected=None, prev_make_cross_game_clock=None,
        offset_s=-1.5, period_max_game_clock=720.0, start_margin_s=0.2, after_prev_make_s=0.3,
        offset_nudge_s=0.5,
    )
    assert result["start_clock"] == pytest.approx(720.0)
    assert result["previous_event_problem"] is False


def test_resolve_window_start_anchors_on_a_detected_previous_make():
    result = resolve_window_start(
        prev_same_period=True, prev_msgtype=MADE_SHOT_CODE, prev_clock_s=650.0,
        made_shot_code=MADE_SHOT_CODE, prev_make_detected=True, prev_make_cross_game_clock=651.2,
        offset_s=-1.5, period_max_game_clock=720.0, start_margin_s=0.2, after_prev_make_s=0.3,
        offset_nudge_s=0.5,
    )
    assert result["start_clock"] == pytest.approx(651.2 - 0.3)
    assert result["offset_used"] is None
    assert result["previous_event_problem"] is False


def test_resolve_window_start_falls_back_when_previous_make_not_detected():
    result = resolve_window_start(
        prev_same_period=True, prev_msgtype=MADE_SHOT_CODE, prev_clock_s=650.0,
        made_shot_code=MADE_SHOT_CODE, prev_make_detected=False, prev_make_cross_game_clock=None,
        offset_s=-1.5, period_max_game_clock=720.0, start_margin_s=0.2, after_prev_make_s=0.3,
        offset_nudge_s=0.5,
    )
    # offset_used = min(-1.5+0.5, 0) = -1.0; start_clock = 650 - (-1.0) - 0.2 = 650.8
    assert result["offset_used"] == pytest.approx(-1.0)
    assert result["start_clock"] == pytest.approx(650.8)
    assert result["previous_event_problem"] is True


def test_resolve_window_start_normal_previous_event():
    result = resolve_window_start(
        prev_same_period=True, prev_msgtype=4.0, prev_clock_s=650.0,  # 4 = rebound, not a make
        made_shot_code=MADE_SHOT_CODE, prev_make_detected=None, prev_make_cross_game_clock=None,
        offset_s=-1.5, period_max_game_clock=720.0, start_margin_s=0.2, after_prev_make_s=0.3,
        offset_nudge_s=0.5,
    )
    assert result["offset_used"] == pytest.approx(-1.0)
    assert result["start_clock"] == pytest.approx(650.8)
    assert result["previous_event_problem"] is False


def test_rotate_and_normalize_left_basket_unchanged():
    u, v = rotate_and_normalize(10.0, 30.0, attack_right=False, length_x=94.0, width_y=50.0, basket_left_x=5.25)
    assert u == pytest.approx(10.0 - 5.25)
    assert v == pytest.approx(30.0 - 25.0)


def test_rotate_and_normalize_right_basket_rotates_180():
    # a player at the right rim itself should land at u=0, v=0 after rotation+shift
    u, v = rotate_and_normalize(88.75, 25.0, attack_right=True, length_x=94.0, width_y=50.0, basket_left_x=5.25)
    assert u == pytest.approx(0.0, abs=1e-6)
    assert v == pytest.approx(0.0, abs=1e-6)


def _ftable(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("ts_ms", kind="mergesort").reset_index(drop=True)


def test_build_window_normal_dense_window():
    ts0, clock0 = 100_000.0, 10.0
    rows = [
        {"ts_ms": ts0 + i * 40, "game_clock": clock0 - i * 0.04, "n_player_rows": 10, "n_ball_rows": 1,
         "clock_running": i > 0}
        for i in range(20)
    ]
    ft = _ftable(rows)
    result = build_window(
        ft, start_clock=clock0, end_ts_ms=ts0 + 19 * 40, intended_length_s=0.76,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["length_frames"] == 19
    assert result["length_s"] == pytest.approx(0.72)
    assert result["coverage_gap"] is False
    assert result["many_missing_frames"] is False


def test_build_window_too_few_frames_is_degenerate():
    ft = _ftable([{"ts_ms": 100_000.0, "game_clock": 10.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False}])
    result = build_window(
        ft, start_clock=10.0, end_ts_ms=100_000.0, intended_length_s=0.0,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["length_frames"] <= 1
    assert result["coverage_gap"] is True
    assert result["many_missing_frames"] is True


def test_build_window_clamps_a_start_clock_that_lands_after_the_end():
    # A fast sequence (e.g. a putback within ~1s of game clock of the miss it
    # follows) can resolve a start_clock whose first matching frame is later
    # in real time than end_ts_ms -- the reported bounds must not invert.
    ts0, clock0 = 100_000.0, 10.0
    rows = [
        {"ts_ms": ts0 + i * 40, "game_clock": clock0 - i * 0.04, "n_player_rows": 10, "n_ball_rows": 1,
         "clock_running": i > 0}
        for i in range(20)
    ]
    ft = _ftable(rows)
    end_ts_ms = ts0 + 5 * 40  # well before the frame matching start_clock
    result = build_window(
        ft, start_clock=clock0 - 0.4, end_ts_ms=end_ts_ms, intended_length_s=0.2,  # only reached at i=10
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["start_ts_ms"] <= result["end_ts_ms"]
    assert result["length_s"] == 0.0


def test_build_window_flags_a_genuine_live_play_gap():
    ts0, clock0 = 100_000.0, 10.0
    rows = [
        {"ts_ms": ts0, "game_clock": clock0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": ts0 + 40, "game_clock": clock0 - 0.04, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        # 6s real gap, clock still decreasing across it -> a genuine tracking gap
        {"ts_ms": ts0 + 40 + 6000, "game_clock": clock0 - 6.04, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": ts0 + 40 + 6000 + 40, "game_clock": clock0 - 6.08, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
    ]
    ft = _ftable(rows)
    # the achieved span (6.04s) alone already clears the absolute floor --
    # this must still be flagged, on the *internal* gap inside it, not the
    # start-coverage rule (which this intended_length_s wouldn't trip anyway).
    result = build_window(
        ft, start_clock=clock0, end_ts_ms=rows[-1]["ts_ms"], intended_length_s=6.08,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["coverage_gap"] is True


def test_build_window_flags_tracking_that_starts_partway_through_the_period():
    # The period's tracking coverage begins only at game_clock=12.0 (e.g. the
    # real period start, 20.0, was never captured) -- a start_clock reaching
    # back toward the true period start must not silently fall back to
    # whatever the earliest available frame happens to be.
    ts0 = 100_000.0
    rows = [
        {"ts_ms": ts0, "game_clock": 12.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": ts0 + 40, "game_clock": 11.96, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": ts0 + 80, "game_clock": 11.92, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
    ]
    ft = _ftable(rows)
    # wanted 20.0 - 11.92 = 8.08s; only got 0.04s -- nowhere near either
    # the 5s floor or 80% of what was intended.
    result = build_window(
        ft, start_clock=20.0, end_ts_ms=rows[-1]["ts_ms"], intended_length_s=20.0 - 11.92,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["coverage_gap"] is True


def test_build_window_flags_a_dropout_right_before_the_computed_start():
    # A genuine tracking dropout (e.g. a camera/feed outage) sits just before
    # start_clock's own matching frame: before_start lands exactly on the
    # frame right after the dropout, so the gap itself never appears inside
    # [raw_start_ts, end_ts_ms]. The resulting sliver of coverage (0.04s) is
    # nowhere near enough of what was wanted (6.04s) -- still excluded.
    ts0, clock0 = 100_000.0, 20.0
    rows = [
        {"ts_ms": ts0, "game_clock": clock0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": ts0 + 40, "game_clock": clock0 - 0.04, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        # 6s real gap landing right at game_clock=10.0 -- the chosen anchor frame
        {"ts_ms": ts0 + 40 + 6000, "game_clock": 10.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": ts0 + 40 + 6000 + 40, "game_clock": 9.96, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
    ]
    ft = _ftable(rows)
    # start_clock=16.0 is still swallowed by the same gap (anything between
    # 10.0 and 19.96 lands on the same post-gap frame) but wants a
    # meaningfully longer window: 16.0 - 9.96 = 6.04s, vs. 0.04s achieved.
    result = build_window(
        ft, start_clock=16.0, end_ts_ms=rows[-1]["ts_ms"], intended_length_s=16.0 - 9.96,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["coverage_gap"] is True


def test_build_window_allows_a_short_start_gap_under_the_absolute_floor():
    # Missing the first chunk of the intended window, but what survives
    # (5.16s) already clears min_start_coverage_s on its own, even though
    # it's a small fraction (~11%) of what was originally wanted (45.2s).
    ts0, clock0 = 400_000.0, 10.0
    n = 131  # i=1..130 kept -> (130-1)*40ms = 5.16s
    rows = [
        {"ts_ms": ts0 + i * 40, "game_clock": clock0 - i * 0.04, "n_player_rows": 10, "n_ball_rows": 1,
         "clock_running": i > 0}
        for i in range(n)
    ]
    ft = _ftable(rows)
    final_game_clock = clock0 - (n - 1) * 0.04  # 4.8
    start_clock = 50.0  # far beyond anything tracked -> raw_start_ts falls back to the earliest frame we have
    result = build_window(
        ft, start_clock=start_clock, end_ts_ms=rows[-1]["ts_ms"], intended_length_s=start_clock - final_game_clock,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["length_s"] == pytest.approx(5.16)
    assert result["coverage_gap"] is False


def test_build_window_allows_a_short_start_gap_under_the_fraction_floor():
    # A short intended window (2.0s) where 1.76s survives -- 88% of what was
    # wanted, comfortably over the 80% floor, even though 1.76s alone is
    # nowhere near the 5s absolute floor.
    ts0, clock0 = 500_000.0, 10.0
    n = 46  # i=1..45 kept -> (45-1)*40ms = 1.76s
    rows = [
        {"ts_ms": ts0 + i * 40, "game_clock": clock0 - i * 0.04, "n_player_rows": 10, "n_ball_rows": 1,
         "clock_running": i > 0}
        for i in range(n)
    ]
    ft = _ftable(rows)
    final_game_clock = clock0 - (n - 1) * 0.04  # 8.2
    start_clock = final_game_clock + 2.0  # 10.2 -- also beyond the tracked max (10.0)
    result = build_window(
        ft, start_clock=start_clock, end_ts_ms=rows[-1]["ts_ms"], intended_length_s=2.0,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["length_s"] == pytest.approx(1.76)
    assert result["coverage_gap"] is False


def test_build_window_flags_a_start_gap_that_fails_both_thresholds():
    # Only 0.36s survives out of an intended 20.0s -- far under both the 5s
    # floor and 80% of what was wanted.
    ts0, clock0 = 600_000.0, 10.0
    n = 11  # i=1..10 kept -> (10-1)*40ms = 0.36s
    rows = [
        {"ts_ms": ts0 + i * 40, "game_clock": clock0 - i * 0.04, "n_player_rows": 10, "n_ball_rows": 1,
         "clock_running": i > 0}
        for i in range(n)
    ]
    ft = _ftable(rows)
    final_game_clock = clock0 - (n - 1) * 0.04  # 9.6
    start_clock = final_game_clock + 20.0  # 29.6
    result = build_window(
        ft, start_clock=start_clock, end_ts_ms=rows[-1]["ts_ms"], intended_length_s=20.0,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["length_s"] == pytest.approx(0.36)
    assert result["coverage_gap"] is True


def test_build_window_discards_real_action_before_a_later_stoppage():
    # ~2s of substantial, real live play (i=0..50), then an 8s real-time
    # freeze (the clock is FROZEN, not running: a timeout/foul/turnover),
    # then play resumes. There's no "is the leading run substantial"
    # exception: once a stoppage is found anywhere in the window, the
    # window always re-anchors on the last resumption minus the buffer,
    # discarding everything before it -- a different possession attempt,
    # whether or not it had real action of its own.
    ts0, clock0 = 100_000.0, 10.0
    leading = [
        {"ts_ms": ts0 + i * 40, "game_clock": clock0 - i * 0.04, "n_player_rows": 10, "n_ball_rows": 1,
         "clock_running": i > 0}
        for i in range(51)  # ts0 .. ts0+2000, game_clock 10.0 .. 8.0
    ]
    # an 8s freeze with continuous native frames (game_clock frozen at 8.0),
    # so the buffer's 6s reach-back (landing at ts0+4040) finds a real frame.
    frozen = [
        {"ts_ms": t, "game_clock": 8.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False}
        for t in np.arange(ts0 + 2040, ts0 + 10001, 40.0)
    ]
    trailing = [
        {"ts_ms": ts0 + 10040, "game_clock": 7.96, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": ts0 + 10080, "game_clock": 7.92, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
    ]
    ft = _ftable(leading + frozen + trailing)
    result = build_window(
        ft, start_clock=clock0, end_ts_ms=ts0 + 10080, intended_length_s=10.08,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["dead_ball_buffer_applied"] is True
    assert result["start_ts_ms"] == pytest.approx(ts0 + 4040)  # 6s before the resumption, well into the freeze
    assert result["length_s"] == pytest.approx(6.04)
    assert result["coverage_gap"] is False


def test_build_window_leaves_a_window_with_no_stoppage_untouched():
    ts0, clock0 = 200_000.0, 10.0
    rows = [
        {"ts_ms": ts0 + i * 40, "game_clock": clock0 - i * 0.04, "n_player_rows": 10, "n_ball_rows": 1,
         "clock_running": i > 0}
        for i in range(51)
    ]
    ft = _ftable(rows)
    result = build_window(
        ft, start_clock=clock0, end_ts_ms=rows[-1]["ts_ms"], intended_length_s=2.0,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["dead_ball_buffer_applied"] is False
    assert result["coverage_gap"] is False
    assert result["start_ts_ms"] == pytest.approx(ts0 + 40)  # the one boundary-artifact frame, unaffected


def test_build_window_flags_many_missing_frames():
    ts0, clock0 = 100_000.0, 10.0
    rows = []
    for i in range(20):
        n_player_rows = 8 if i < 5 else 10  # 5/20 = 25% missing a player, well above 2%
        rows.append(
            {"ts_ms": ts0 + i * 40, "game_clock": clock0 - i * 0.04, "n_player_rows": n_player_rows,
             "n_ball_rows": 1, "clock_running": i > 0}
        )
    ft = _ftable(rows)
    result = build_window(
        ft, start_clock=clock0, end_ts_ms=ts0 + 19 * 40, intended_length_s=0.76,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["many_missing_frames"] is True


def test_build_window_truncates_to_max_window_s():
    ts0, clock0 = 100_000.0, 10.0
    rows = [
        {"ts_ms": ts0 + i * 40, "game_clock": clock0 - i * 0.04, "n_player_rows": 10, "n_ball_rows": 1,
         "clock_running": i > 0}
        for i in range(50)  # spans 1.96s of real time
    ]
    ft = _ftable(rows)
    result = build_window(
        ft, start_clock=clock0, end_ts_ms=ts0 + 49 * 40, intended_length_s=0.5,
        max_window_s=0.5, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["length_s"] <= 0.5 + 1e-6
    assert result["start_ts_ms"] > ts0  # the early frames were trimmed off


def _running_ftable(ts_running_pairs: list[tuple[float, bool]]) -> pd.DataFrame:
    return _ftable(
        [
            {"ts_ms": t, "game_clock": 0.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": r}
            for t, r in ts_running_pairs
        ]
    )


def test_find_live_stretch_start_stops_right_after_a_dead_to_live_transition():
    ft = _running_ftable([(0.0, True), (40.0, True), (6000.0, False), (6040.0, True), (6080.0, True)])
    assert find_live_stretch_start(ft, end_ts_ms=6080.0) == pytest.approx(6040.0)


def test_find_live_stretch_start_ignores_an_isolated_blip():
    # 4000 is a lone clock-running frame isolated by dead frames on both
    # sides from the real resumption at 8040 -- the scan must not stop there.
    ft = _running_ftable([(0.0, False), (4000.0, True), (8000.0, False), (8040.0, True), (8080.0, True)])
    assert find_live_stretch_start(ft, end_ts_ms=8080.0) == pytest.approx(8040.0)


def test_find_live_stretch_start_scans_to_the_start_when_nothing_stopped():
    ts = np.arange(0.0, 500.0, 40.0)
    ft = _running_ftable([(t, i > 0) for i, t in enumerate(ts)])
    # stops right after the one boundary-artifact frame (the very first
    # frame of any series has no predecessor to diff against, so it's
    # never "running" -- that's not a real stoppage).
    assert find_live_stretch_start(ft, end_ts_ms=ts[-1]) == pytest.approx(ts[1])


def test_build_window_applies_dead_ball_buffer_around_an_isolated_blip():
    # Mirrors a real inbound-after-timeout case: a lone clock-running frame
    # (101_000) right as the stoppage begins looks "live" in isolation, but
    # is isolated by dead-ball frames on both sides from the real
    # resumption (105_600) -- plenty of earlier data exists (back to
    # 95_000), so the full 6s buffer before the real resumption is
    # achieved exactly, sweeping in the dead frames at 99_600/99_640 too.
    rows = [
        {"ts_ms": 95_000.0, "game_clock": 20.5, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": 99_600.0, "game_clock": 20.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": 99_640.0, "game_clock": 20.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": 101_000.0, "game_clock": 19.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": 101_040.0, "game_clock": 19.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": 105_600.0, "game_clock": 18.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": 105_640.0, "game_clock": 17.96, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": 105_680.0, "game_clock": 17.92, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
    ]
    ft = _ftable(rows)
    result = build_window(
        ft, start_clock=20.5, end_ts_ms=rows[-1]["ts_ms"], intended_length_s=6.08,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["dead_ball_buffer_applied"] is True
    assert result["start_ts_ms"] == pytest.approx(99_600.0)  # exactly 6s before the real resumption
    assert result["length_frames"] == 7  # the 95_000 frame (before the buffer target) stays excluded
    assert result["coverage_gap"] is False


def test_build_window_dead_ball_buffer_clamped_when_not_enough_data():
    # same shape, but tracking only reaches back to 100_000 -- less than a
    # full 6s before the 105_600 resumption -- so the buffer goes as far
    # back as it can rather than past the earliest available frame.
    rows = [
        {"ts_ms": 100_000.0, "game_clock": 20.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": 101_000.0, "game_clock": 19.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": 101_040.0, "game_clock": 19.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": 105_600.0, "game_clock": 18.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": 105_640.0, "game_clock": 17.96, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        {"ts_ms": 105_680.0, "game_clock": 17.92, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
    ]
    ft = _ftable(rows)
    result = build_window(
        ft, start_clock=20.0, end_ts_ms=rows[-1]["ts_ms"], intended_length_s=5.68,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02,
        min_start_coverage_s=5.0, min_start_coverage_fraction=0.8,
        dead_ball_buffer_s=6.0, dead_ball_min_gap_s=1.0, players_on_court=10,
    )
    assert result["dead_ball_buffer_applied"] is True
    assert result["start_ts_ms"] == pytest.approx(100_000.0)  # clamped to the earliest available frame
    assert result["length_frames"] == 6
    assert result["coverage_gap"] is False


def test_build_agent_tracks_masks_absent_frames_and_sets_roles():
    kept_ts = np.array([100_000.0, 100_040.0, 100_080.0])
    ball_xyz = pd.DataFrame(
        {"ts_ms": kept_ts, "x": [10.0, 11.0, 12.0], "y": [25.0, 25.0, 25.0], "z": [8.0, 9.0, 10.0]}
    )
    # player 1 (shooter, offense) present all 3 frames; player 2 (defense) absent at the last frame
    players = pd.DataFrame(
        [
            {"ts_ms": 100_000.0, "team_id": 1, "player_id": 1, "x": 20.0, "y": 20.0},
            {"ts_ms": 100_040.0, "team_id": 1, "player_id": 1, "x": 21.0, "y": 20.0},
            {"ts_ms": 100_080.0, "team_id": 1, "player_id": 1, "x": 22.0, "y": 20.0},
            {"ts_ms": 100_000.0, "team_id": 2, "player_id": 2, "x": 30.0, "y": 30.0},
            {"ts_ms": 100_040.0, "team_id": 2, "player_id": 2, "x": 31.0, "y": 30.0},
        ]
    )
    out = build_agent_tracks(players, ball_xyz, kept_ts, shooter_team_id=1, shooter_id=1)
    assert out["pos"].shape == (3, 3, 3)  # T=3, A=1 ball + 2 players
    assert out["mask"][:, 0].all()  # ball present every frame
    assert out["mask"][:, 1].all()  # player 1 present every frame
    assert list(out["mask"][:, 2]) == [True, True, False]  # player 2 absent at t=2
    assert out["roles"][0].tolist() == [True, False, False]  # ball
    assert out["roles"][1].tolist() == [False, True, True]  # offense, shooter
    assert out["roles"][2].tolist() == [False, False, False]  # defense, not shooter
