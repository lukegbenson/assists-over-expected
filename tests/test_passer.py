"""src/nba_assists/passer.py: possession geometry, runs, and backward trace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_assists.passer import (
    NO_PLAYER,
    attribute_passer,
    backward_trace,
    classify_held,
    possession_geometry,
    possession_runs,
)


def test_possession_geometry_nearest_player_and_missing_ball():
    kept_ts = np.array([0.0, 40.0, 80.0])
    ball_xyz_q = pd.DataFrame({"ts_ms": [0.0, 40.0], "x": [10.0, 10.0], "y": [10.0, 10.0], "z": [5.0, 9.0]})
    player_positions_q = pd.DataFrame(
        [
            {"ts_ms": 0.0, "team_id": 1, "player_id": 101, "x": 10.0, "y": 10.0},
            {"ts_ms": 0.0, "team_id": 1, "player_id": 102, "x": 50.0, "y": 50.0},
            {"ts_ms": 40.0, "team_id": 1, "player_id": 101, "x": 10.0, "y": 10.0},
            {"ts_ms": 40.0, "team_id": 1, "player_id": 102, "x": 50.0, "y": 50.0},
            {"ts_ms": 80.0, "team_id": 1, "player_id": 101, "x": 10.0, "y": 10.0},
        ]
    )
    geo = possession_geometry(player_positions_q, ball_xyz_q, offense_team_id=1, kept_ts=kept_ts)

    assert list(geo["off_player_id"][:2]) == [101, 101]
    assert geo["off_dist"][0] == pytest.approx(0.0)
    assert geo["off_dist"][1] == pytest.approx(0.0)
    # ts=80 has no ball row at all -- forced to NO_PLAYER / inf regardless of position data
    assert geo["off_player_id"][2] == NO_PLAYER
    assert geo["off_dist"][2] == np.inf
    assert np.isnan(geo["ball_z"][2])


def test_classify_held_requires_distance_height_and_a_real_ball():
    geometry = {
        "off_player_id": np.array([101, 101, 55]),
        "off_dist": np.array([0.5, 3.0, 0.5]),
        "ball_z": np.array([5.0, 5.0, np.nan]),
    }
    held = classify_held(geometry, hold_dist=2.5, hold_height=8.5)
    assert list(held) == [101, NO_PLAYER, NO_PLAYER]


def test_possession_runs_merges_a_dribble_gap_and_splits_on_player_change():
    ts = np.array([0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0])
    held = np.array([101, 101, NO_PLAYER, 101, 102, 102, 102, 102])
    runs = possession_runs(ts, held, min_run_s=0.1, dribble_gap_s=0.5)
    assert list(runs["player_id"]) == [101, 102]
    assert list(runs["start_ts"]) == [0.0, 160.0]
    assert list(runs["end_ts"]) == [120.0, 280.0]


def test_possession_runs_splits_and_drops_a_gap_wider_than_dribble_gap_s():
    # same held pattern as above, but the gap at ts=80-120 (80ms) now exceeds
    # a tight 50ms dribble_gap_s -- player 101 splits into a 0.04s run and a
    # single-frame (0.0s) run, and that second sliver is dropped by min_run_s.
    ts = np.array([0.0, 40.0, 80.0, 120.0, 160.0, 200.0])
    held = np.array([101, 101, NO_PLAYER, 101, 102, 102])
    runs = possession_runs(ts, held, min_run_s=0.03, dribble_gap_s=0.05)
    assert list(runs["player_id"]) == [101, 102]
    assert list(runs["start_ts"]) == [0.0, 160.0]
    assert list(runs["end_ts"]) == [40.0, 200.0]


def test_possession_runs_empty_when_nothing_held():
    ts = np.array([0.0, 40.0])
    held = np.array([NO_PLAYER, NO_PLAYER])
    runs = possession_runs(ts, held, min_run_s=0.1, dribble_gap_s=0.5)
    assert len(runs["player_id"]) == 0


def _runs(player_id, start_ts, end_ts) -> dict:
    return {
        "player_id": np.array(player_id, dtype=np.int64),
        "start_ts": np.array(start_ts, dtype=float),
        "end_ts": np.array(end_ts, dtype=float),
    }


def test_backward_trace_finds_a_different_offensive_player_as_passer():
    runs = _runs([201, 101], [0.0, 200.0], [180.0, 300.0])
    result = backward_trace(runs, shooter_id=101, window_start_ts=0.0)
    assert result == {"traced_passer_id": 201, "trace_problem": False, "shooter_run_player_id": 101}


def test_backward_trace_none_when_no_earlier_run():
    runs = _runs([101], [50.0], [300.0])
    result = backward_trace(runs, shooter_id=101, window_start_ts=0.0)
    assert result == {"traced_passer_id": None, "trace_problem": False, "shooter_run_player_id": 101}


def test_backward_trace_none_when_shooter_run_starts_at_window_start():
    runs = _runs([201, 101], [0.0, 0.0], [0.0, 300.0])
    result = backward_trace(runs, shooter_id=101, window_start_ts=0.0)
    assert result == {"traced_passer_id": None, "trace_problem": False, "shooter_run_player_id": 101}


def test_backward_trace_none_when_the_run_before_is_the_same_player():
    runs = _runs([101, 101], [0.0, 200.0], [180.0, 300.0])
    result = backward_trace(runs, shooter_id=101, window_start_ts=0.0)
    assert result == {"traced_passer_id": None, "trace_problem": False, "shooter_run_player_id": 101}


def test_backward_trace_reassigns_to_the_last_run_when_the_shooter_was_never_held():
    # the shooter (101) caught the ball above hold_height (e.g. an
    # alley-oop) -- the only run detected belongs to player 201 (the real
    # passer). The recorded shooter is never in doubt, so that last run is
    # attributed directly as the passer rather than trusted as "the
    # shooter's run" and stepped back from.
    runs = _runs([201], [0.0], [180.0])
    result = backward_trace(runs, shooter_id=101, window_start_ts=0.0)
    assert result == {"traced_passer_id": 201, "trace_problem": True, "shooter_run_player_id": None}


def test_backward_trace_reassigns_even_with_an_earlier_unrelated_run():
    # same as above, but an earlier, unrelated run by the shooter himself
    # (e.g. a loose-ball touch near the start of the window) must not be
    # mistaken for "the shooter's run at the moment of the shot".
    runs = _runs([101, 201], [0.0, 50.0], [10.0, 180.0])
    result = backward_trace(runs, shooter_id=101, window_start_ts=0.0)
    assert result == {"traced_passer_id": 201, "trace_problem": True, "shooter_run_player_id": None}


def test_backward_trace_none_when_no_runs_at_all():
    runs = _runs([], [], [])
    result = backward_trace(runs, shooter_id=101, window_start_ts=0.0)
    assert result == {"traced_passer_id": None, "trace_problem": True, "shooter_run_player_id": None}


def test_attribute_passer_prefers_the_known_assister():
    assert attribute_passer(assister_id=55, assisted=True, traced_passer_id=201) == 55


def test_attribute_passer_falls_back_to_traced_when_assister_is_missing():
    assert attribute_passer(assister_id=np.nan, assisted=True, traced_passer_id=201) == 201


def test_attribute_passer_uses_traced_when_unassisted():
    assert attribute_passer(assister_id=None, assisted=False, traced_passer_id=201) == 201
    assert attribute_passer(assister_id=None, assisted=False, traced_passer_id=None) is None
