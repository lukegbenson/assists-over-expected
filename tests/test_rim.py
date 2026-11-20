"""src/nba_assists/rim.py: ball-row filtering and rim-crossing detection."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_assists.config import get_config
from nba_assists.rim import ball_rows, detect_make, window_has_gap


@pytest.fixture(scope="module")
def cfg():
    return get_config()


def _row(ts_ms, game_clock, x, y, z) -> dict:
    return {"ts_ms": float(ts_ms), "game_clock": float(game_clock), "x": float(x), "y": float(y), "z": float(z)}


def _qframes(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("ts_ms", kind="mergesort").reset_index(drop=True)


def test_ball_rows_filters_sentinel_and_sorts(cfg):
    tracking = pd.DataFrame(
        [
            {"ts_ms": 200, "quarter": 1, "game_clock": 700.0, "team_id": 1, "player_id": 10, "x": 1.0, "y": 1.0, "z": 0.0},
            {"ts_ms": 100, "quarter": 1, "game_clock": 701.0, "team_id": -1, "player_id": -1, "x": 5.0, "y": 25.0, "z": 9.0},
            {"ts_ms": 200, "quarter": 1, "game_clock": 700.0, "team_id": -1, "player_id": -1, "x": 5.5, "y": 25.0, "z": 10.5},
        ]
    )
    out = ball_rows(tracking, cfg)
    assert list(out["ts_ms"]) == [100, 200]
    assert list(out.columns) == ["ts_ms", "quarter", "game_clock", "x", "y", "z"]


def test_detect_make_single_candidate(cfg):
    bx, by = float(cfg.court.baskets.left.x), float(cfg.court.baskets.left.y)
    clock0, ts0 = 500.0, 100_000.0
    frames = _qframes(
        [
            _row(ts0 - 200.0, clock0 + 0.2, bx, by, 12.0),  # clock-check reference
            _row(ts0, clock0, bx, by, 10.5),  # first frame of the crossing
            _row(ts0 + 40.0, clock0 - 0.04, bx, by, 9.5),  # second frame of the crossing
        ]
    )
    candidates, result = detect_make(frames, clock_s=clock0, cfg=cfg)
    assert len(candidates) == 1
    assert bool(candidates.iloc[0]["clock_check_passed"]) is True
    assert result["detected"] is True
    assert result["case"] == "single"
    assert result["basket_x"] == pytest.approx(bx)
    assert result["cross_game_clock"] == pytest.approx(clock0 - 0.02, abs=1e-6)
    assert result["offset_s"] == pytest.approx(0.02, abs=1e-6)
    assert result["n_candidates"] == 1
    assert result["n_rejected_clock"] == 0
    assert result["gap_s"] is None


def test_detect_make_rejected_by_clock_check(cfg):
    bx, by = float(cfg.court.baskets.left.x), float(cfg.court.baskets.left.y)
    clock0, ts0 = 500.0, 100_000.0
    frames = _qframes(
        [
            _row(ts0 - 200.0, clock0, bx, by, 12.0),  # clock flat, not running -> a dead-ball shot
            _row(ts0, clock0, bx, by, 10.5),
            _row(ts0 + 40.0, clock0 - 0.04, bx, by, 9.5),
        ]
    )
    candidates, result = detect_make(frames, clock_s=clock0, cfg=cfg)
    assert len(candidates) == 1
    assert bool(candidates.iloc[0]["clock_check_passed"]) is False
    assert result == {
        "detected": False,
        "reason": "all rejected by clock check",
        "case": "none",
        "n_candidates": 0,
        "n_rejected_clock": 1,
        "cross_ts_ms": None,
        "cross_game_clock": None,
        "basket_x": None,
        "offset_s": None,
        "gap_s": None,
    }


def test_detect_make_no_ball_data(cfg):
    empty = pd.DataFrame(columns=["ts_ms", "game_clock", "x", "y", "z"])
    candidates, result = detect_make(empty, clock_s=500.0, cfg=cfg)
    assert candidates.empty
    assert result["detected"] is False
    assert result["reason"] == "no ball data"


def test_detect_make_ball_never_near_rim(cfg):
    clock0, ts0 = 500.0, 100_000.0
    frames = _qframes(
        [
            _row(ts0 - 200.0, clock0 + 0.2, 45.0, 25.0, 12.0),  # center court, far from both rims
            _row(ts0, clock0, 45.0, 25.0, 10.5),
            _row(ts0 + 40.0, clock0 - 0.04, 45.0, 25.0, 9.5),
        ]
    )
    candidates, result = detect_make(frames, clock_s=clock0, cfg=cfg)
    assert candidates.empty
    assert result["reason"] == "ball never near a rim"


def test_detect_make_no_downward_crossing(cfg):
    bx, by = float(cfg.court.baskets.left.x), float(cfg.court.baskets.left.y)
    clock0, ts0 = 500.0, 100_000.0
    frames = _qframes(
        [
            _row(ts0 - 200.0, clock0 + 0.2, bx, by, 12.0),
            _row(ts0, clock0, bx, by, 11.0),
            _row(ts0 + 40.0, clock0 - 0.04, bx, by, 10.5),  # never drops to <= rim height
        ]
    )
    candidates, result = detect_make(frames, clock_s=clock0, cfg=cfg)
    assert candidates.empty
    assert result["reason"] == "no downward crossing"


def test_detect_make_multi_same_basket_takes_the_later_one(cfg):
    bx, by = float(cfg.court.baskets.left.x), float(cfg.court.baskets.left.y)
    frames = _qframes(
        [
            _row(99_800.0, 500.5, bx, by, 12.0),  # crossing 1's clock-check reference
            _row(100_000.0, 500.3, bx, by, 10.5),  # crossing 1 (a putback's miss)
            _row(100_040.0, 500.26, bx, by, 9.5),
            _row(100_600.0, 499.5, bx, by, 12.0),  # crossing 2's clock-check reference
            _row(100_800.0, 499.3, bx, by, 10.5),  # crossing 2 (the putback's make)
            _row(100_840.0, 499.26, bx, by, 9.5),
        ]
    )
    # clock_s matches crossing 1 exactly, so "closest to clock_s" would pick it
    # wrongly -- multi_same must always take the chronologically later one.
    clock_s = 500.28
    candidates, result = detect_make(frames, clock_s=clock_s, cfg=cfg)
    assert len(candidates) == 2
    assert result["case"] == "multi_same"
    assert result["cross_game_clock"] == pytest.approx(499.28, abs=1e-6)
    assert result["offset_s"] == pytest.approx(1.0, abs=1e-6)
    assert result["gap_s"] == pytest.approx(1.0, abs=1e-6)
    assert result["n_candidates"] == 2
    assert result["n_rejected_clock"] == 0


def test_detect_make_multi_different_baskets_picks_closest_and_flags(cfg):
    left_x, left_y = float(cfg.court.baskets.left.x), float(cfg.court.baskets.left.y)
    right_x, right_y = float(cfg.court.baskets.right.x), float(cfg.court.baskets.right.y)
    frames = _qframes(
        [
            _row(99_800.0, 500.5, left_x, left_y, 12.0),
            _row(100_000.0, 500.3, left_x, left_y, 10.5),  # crossing at the left rim
            _row(100_040.0, 500.26, left_x, left_y, 9.5),
            _row(100_600.0, 499.5, right_x, right_y, 12.0),
            _row(100_800.0, 499.3, right_x, right_y, 10.5),  # crossing at the right rim
            _row(100_840.0, 499.26, right_x, right_y, 9.5),
        ]
    )
    clock_s = 499.5  # closer to the right-rim crossing (499.28) than the left one (500.28)
    candidates, result = detect_make(frames, clock_s=clock_s, cfg=cfg)
    assert len(candidates) == 2
    assert result["case"] == "multi_diff"
    assert result["basket_x"] == pytest.approx(right_x)
    assert result["offset_s"] == pytest.approx(0.22, abs=1e-6)
    assert result["gap_s"] == pytest.approx(1.0, abs=1e-6)


def test_detect_make_buzzer_beater_passes_on_wider_lookback(cfg):
    bx, by = float(cfg.court.baskets.left.x), float(cfg.court.baskets.left.y)
    ts0 = 200_000.0
    frames = _qframes(
        [
            _row(ts0 - 1000.0, 0.5, bx, by, 15.0),  # clock still running 1s before the crossing
            _row(ts0, 0.0, bx, by, 10.5),  # clock already reads 0 (ball still in the air)
            _row(ts0 + 40.0, 0.0, bx, by, 9.5),
        ]
    )
    candidates, result = detect_make(frames, clock_s=0.3, cfg=cfg)
    assert len(candidates) == 1
    assert bool(candidates.iloc[0]["clock_check_passed"]) is True
    assert result["detected"] is True
    assert result["cross_game_clock"] == pytest.approx(0.0, abs=1e-6)
    assert result["offset_s"] == pytest.approx(0.3, abs=1e-6)


def test_detect_make_basket_x_restricts_to_one_basket(cfg):
    left_x, left_y = float(cfg.court.baskets.left.x), float(cfg.court.baskets.left.y)
    right_x, right_y = float(cfg.court.baskets.right.x), float(cfg.court.baskets.right.y)
    frames = _qframes(
        [
            _row(99_800.0, 500.5, left_x, left_y, 12.0),
            _row(100_000.0, 500.3, left_x, left_y, 10.5),  # crossing at the left rim
            _row(100_040.0, 500.26, left_x, left_y, 9.5),
            _row(100_600.0, 499.5, right_x, right_y, 12.0),
            _row(100_800.0, 499.3, right_x, right_y, 10.5),  # crossing at the right rim
            _row(100_840.0, 499.26, right_x, right_y, 9.5),
        ]
    )
    # clock_s is closer to the right-rim crossing, but the scoring team is
    # known (e.g. from attack_direction.parquet) to be attacking the left
    # basket -- basket_x should rule the right-rim crossing out entirely,
    # not just lose a tie-break to it.
    candidates, result = detect_make(frames, clock_s=499.5, cfg=cfg, basket_x=left_x)
    assert len(candidates) == 1
    assert result["case"] == "single"
    assert result["basket_x"] == pytest.approx(left_x)


def test_detect_make_basket_x_excludes_a_real_crossing_at_the_other_end(cfg):
    left_x, left_y = float(cfg.court.baskets.left.x), float(cfg.court.baskets.left.y)
    right_x = float(cfg.court.baskets.right.x)
    frames = _qframes(
        [
            _row(99_800.0, 500.2, left_x, left_y, 12.0),
            _row(100_000.0, 500.0, left_x, left_y, 10.5),  # crossing only at the left rim
            _row(100_040.0, 499.96, left_x, left_y, 9.5),
        ]
    )
    candidates, result = detect_make(frames, clock_s=500.0, cfg=cfg, basket_x=right_x)
    assert candidates.empty
    assert result["reason"] == "ball never near a rim"


def test_window_has_gap_false_when_densely_covered(cfg):
    clock0, ts0 = 500.0, 100_000.0
    frames = _qframes(
        [_row(ts0 + i * 40.0, clock0 - i * 0.04, 20.0, 25.0, 5.0) for i in range(-10, 11)]
    )
    assert window_has_gap(frames, clock_s=clock0, cfg=cfg) is False


def test_window_has_gap_true_for_a_long_gap_while_clock_running(cfg):
    gap_limit_s = float(cfg.inventory.max_gap_running_s)
    search_before = float(cfg.rim_detection.search_before_s)
    search_after = float(cfg.rim_detection.search_after_s)
    clock0, ts0 = 500.0, 100_000.0
    frames = _qframes(
        [
            _row(ts0, clock0 + search_before, 20.0, 25.0, 5.0),
            _row(ts0 + (gap_limit_s + 1.0) * 1000.0, clock0 - search_after, 20.0, 25.0, 5.0),
        ]
    )
    assert window_has_gap(frames, clock_s=clock0, cfg=cfg) is True


def test_window_has_gap_false_when_clock_is_flat_across_the_gap(cfg):
    # a long real-time gap, but the clock didn't move -- a dead-ball stretch
    # (timeout, replay), not a live-play tracking defect
    gap_limit_s = float(cfg.inventory.max_gap_running_s)
    clock0, ts0 = 500.0, 100_000.0
    frames = _qframes(
        [
            _row(ts0, clock0, 20.0, 25.0, 5.0),
            _row(ts0 + (gap_limit_s + 1.0) * 1000.0, clock0, 20.0, 25.0, 5.0),
        ]
    )
    assert window_has_gap(frames, clock_s=clock0, cfg=cfg) is False


def test_window_has_gap_true_with_fewer_than_two_frames_in_window(cfg):
    frames = _qframes([_row(100_000.0, 500.0, 20.0, 25.0, 5.0)])
    assert window_has_gap(frames, clock_s=500.0, cfg=cfg) is True


def test_detect_make_buzzer_rejected_if_clock_was_already_at_zero(cfg):
    bx, by = float(cfg.court.baskets.left.x), float(cfg.court.baskets.left.y)
    ts0 = 200_000.0
    frames = _qframes(
        [
            _row(ts0 - 1000.0, 0.0, bx, by, 15.0),  # clock already at 0 well before the crossing too
            _row(ts0, 0.0, bx, by, 10.5),
            _row(ts0 + 40.0, 0.0, bx, by, 9.5),
        ]
    )
    candidates, result = detect_make(frames, clock_s=0.3, cfg=cfg)
    assert len(candidates) == 1
    assert bool(candidates.iloc[0]["clock_check_passed"]) is False
    assert result["detected"] is False
    assert result["reason"] == "all rejected by clock check"
