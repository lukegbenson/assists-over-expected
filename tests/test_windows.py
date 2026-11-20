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
        ft, start_clock=clock0, end_ts_ms=ts0 + 19 * 40,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02, players_on_court=10,
    )
    assert result["length_frames"] == 19
    assert result["length_s"] == pytest.approx(0.72)
    assert result["coverage_gap"] is False
    assert result["many_missing_frames"] is False


def test_build_window_too_few_frames_is_degenerate():
    ft = _ftable([{"ts_ms": 100_000.0, "game_clock": 10.0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False}])
    result = build_window(
        ft, start_clock=10.0, end_ts_ms=100_000.0,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02, players_on_court=10,
    )
    assert result["length_frames"] <= 1
    assert result["coverage_gap"] is True
    assert result["many_missing_frames"] is True


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
    result = build_window(
        ft, start_clock=clock0, end_ts_ms=rows[-1]["ts_ms"],
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02, players_on_court=10,
    )
    assert result["coverage_gap"] is True


def test_build_window_does_not_flag_a_dead_ball_stretch_as_a_gap():
    ts0, clock0 = 100_000.0, 10.0
    rows = [
        {"ts_ms": ts0, "game_clock": clock0, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": ts0 + 40, "game_clock": clock0 - 0.04, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
        # 6s real gap, but the clock is FROZEN across it (a timeout/foul) -> not running, not a coverage gap
        {"ts_ms": ts0 + 40 + 6000, "game_clock": clock0 - 0.04, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": False},
        {"ts_ms": ts0 + 40 + 6000 + 40, "game_clock": clock0 - 0.08, "n_player_rows": 10, "n_ball_rows": 1, "clock_running": True},
    ]
    ft = _ftable(rows)
    result = build_window(
        ft, start_clock=clock0, end_ts_ms=rows[-1]["ts_ms"],
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02, players_on_court=10,
    )
    assert result["coverage_gap"] is False


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
        ft, start_clock=clock0, end_ts_ms=ts0 + 19 * 40,
        max_window_s=100.0, max_gap_running_s=5.0, max_missing_frame_share=0.02, players_on_court=10,
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
        ft, start_clock=clock0, end_ts_ms=ts0 + 49 * 40,
        max_window_s=0.5, max_gap_running_s=5.0, max_missing_frame_share=0.02, players_on_court=10,
    )
    assert result["length_s"] <= 0.5 + 1e-6
    assert result["start_ts_ms"] > ts0  # the early frames were trimmed off


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
