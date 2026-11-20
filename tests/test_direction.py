"""src/nba_assists/direction.py: ball-holder vote and consistency checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_assists.config import get_config
from nba_assists.direction import consistency_checks, tally_votes, votes


@pytest.fixture(scope="module")
def cfg():
    return get_config()


# Five filler native frames (not sampled: stride is 5, so only indices 0 and 5
# of a 6-frame sequence get voted on) giving the sampled frame a predecessor
# to diff its game_clock against.
FILLER_TS = [0, 40, 80, 120, 160]
FILLER_CLOCK = [700.20, 700.16, 700.12, 700.08, 700.04]


def _filler_rows() -> list[dict]:
    return [
        {
            "ts_ms": ts, "quarter": 1, "game_clock": clock, "shot_clock": 10.0,
            "team_id": -1, "player_id": -1, "x": 20.0, "y": 25.0, "z": 5.0,
        }
        for ts, clock in zip(FILLER_TS, FILLER_CLOCK)
    ]


def _good_frame5_rows(x_shift: float = 0.0) -> list[dict]:
    """A frame that should pass every filter: the ball held in one half, 10 players."""
    rows = [
        {
            "ts_ms": 200, "quarter": 1, "game_clock": 700.00, "shot_clock": 10.0,
            "team_id": -1, "player_id": -1, "x": 20.0 + x_shift, "y": 25.0, "z": 5.0,
        }
    ]
    players = [
        (100, 1, 21.0, 25.0),   # nearest to the ball -> casts the vote
        (100, 2, 15.0, 20.0),
        (100, 3, 12.0, 30.0),
        (100, 4, 10.0, 15.0),
        (100, 5, 8.0, 35.0),
        (200, 6, 25.0, 10.0),
        (200, 7, 30.0, 40.0),
        (200, 8, 35.0, 20.0),
        (200, 9, 18.0, 45.0),
        (200, 10, 40.0, 25.0),  # farthest from the basket in this half
    ]
    for team_id, player_id, x, y in players:
        rows.append(
            {
                "ts_ms": 200, "quarter": 1, "game_clock": 700.00, "shot_clock": 10.0,
                "team_id": team_id, "player_id": player_id, "x": x + x_shift, "y": y, "z": 0.0,
            }
        )
    return rows


def _tracking(frame5_rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(_filler_rows() + frame5_rows)


def test_votes_picks_left_basket(cfg):
    out = votes(_tracking(_good_frame5_rows()), cfg)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["ts_ms"] == 200
    assert row["period"] == 1
    assert row["voter_team_id"] == 100
    assert row["basket_x"] == pytest.approx(float(cfg.court.baskets.left.x))
    assert bool(row["agree"]) is False  # farthest player (player 10) is on team 200


def test_votes_picks_right_basket(cfg):
    # shift the whole frame across halfcourt (94/2 = 47)
    out = votes(_tracking(_good_frame5_rows(x_shift=50.0)), cfg)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["voter_team_id"] == 100
    assert row["basket_x"] == pytest.approx(float(cfg.court.baskets.right.x))


def test_votes_excludes_clock_not_running(cfg):
    rows = _good_frame5_rows()
    for r in rows:
        r["game_clock"] = 700.04  # same as the previous native frame -> not dropping
    assert votes(_tracking(rows), cfg).empty


@pytest.mark.parametrize("shot_clock", [4.0, 19.0])
def test_votes_excludes_shot_clock_out_of_range(cfg, shot_clock):
    rows = _good_frame5_rows()
    for r in rows:
        r["shot_clock"] = shot_clock
    assert votes(_tracking(rows), cfg).empty


def test_votes_excludes_wrong_player_count(cfg):
    rows = _good_frame5_rows()[:-1]  # drop the last player -> 9, not 10
    assert votes(_tracking(rows), cfg).empty


def test_votes_excludes_missing_ball(cfg):
    rows = [r for r in _good_frame5_rows() if r["team_id"] != -1]
    assert votes(_tracking(rows), cfg).empty


def test_votes_excludes_different_halves(cfg):
    rows = _good_frame5_rows()
    rows[-1]["x"] = 60.0  # one player on the other side of halfcourt
    assert votes(_tracking(rows), cfg).empty


def test_votes_excludes_nearest_player_too_far(cfg):
    rows = _good_frame5_rows()
    rows[1]["x"] = 25.0  # nearest player now 5 ft from the ball (hold_dist is 3.0)
    assert votes(_tracking(rows), cfg).empty


def test_votes_excludes_ball_too_high(cfg):
    rows = _good_frame5_rows()
    for r in rows:
        if r["team_id"] == -1:
            r["z"] = 10.0  # hold_height is 9.0
    assert votes(_tracking(rows), cfg).empty


def test_tally_votes_computes_share_and_clear(cfg):
    left_x = float(cfg.court.baskets.left.x)
    votes_df = pd.DataFrame(
        {
            "ts_ms": range(10),
            "period": [1] * 10,
            "voter_team_id": [100] * 9 + [200],
            "basket_x": [left_x] * 10,
            "agree": [True] * 10,
        }
    )
    tally = tally_votes(votes_df, cfg)
    row = tally[tally["team_id"] == 100].iloc[0]
    assert row["n_votes"] == 9
    assert row["vote_share"] == pytest.approx(1.0)
    # min_votes is 200, so 9 votes isn't "clear" even at 100% share
    assert bool(row["clear"]) is False


def test_tally_votes_uses_the_lower_overtime_vote_threshold(cfg):
    left_x = float(cfg.court.baskets.left.x)
    min_votes_ot = int(cfg.attack_direction.min_votes_ot)
    min_votes = int(cfg.attack_direction.min_votes)
    assert min_votes_ot < min_votes  # otherwise this test proves nothing

    n = min_votes_ot + 10
    votes_df = pd.DataFrame(
        {
            "ts_ms": range(n),
            "period": [5] * n,  # period 5 = first overtime
            "voter_team_id": [100] * n,
            "basket_x": [left_x] * n,
            "agree": [True] * n,
        }
    )
    tally = tally_votes(votes_df, cfg)
    row = tally[tally["team_id"] == 100].iloc[0]
    assert row["n_votes"] == n
    # fewer votes than the regulation min_votes, but enough for min_votes_ot
    assert bool(row["clear"]) is True


def test_consistency_checks_flags_same_basket_violation(cfg):
    left_x = float(cfg.court.baskets.left.x)
    direction_rows = pd.DataFrame(
        [
            {"period": 1, "team_id": 100, "basket_x": left_x, "n_votes": 500, "vote_share": 0.99, "clear": True},
            {"period": 1, "team_id": 200, "basket_x": left_x, "n_votes": 480, "vote_share": 0.98, "clear": True},
        ]
    )
    result = consistency_checks(direction_rows)
    assert result["direction_opposite_ok"] is False


def test_consistency_checks_flags_a_padded_zero_vote_period(cfg):
    left_x = float(cfg.court.baskets.left.x)
    direction_rows = pd.DataFrame(
        [
            {"period": 1, "team_id": 100, "basket_x": left_x, "n_votes": 500, "vote_share": 0.99, "clear": True},
            # team 200 cast zero votes this period -- padded in by
            # scripts/attack_direction.py's process_game(), not dropped
            {"period": 1, "team_id": 200, "basket_x": np.nan, "n_votes": 0, "vote_share": np.nan, "clear": False},
            {"period": 2, "team_id": 100, "basket_x": left_x, "n_votes": 500, "vote_share": 0.99, "clear": True},
            {"period": 2, "team_id": 200, "basket_x": left_x, "n_votes": 500, "vote_share": 0.99, "clear": True},
        ]
    )
    result = consistency_checks(direction_rows)
    assert result["direction_opposite_ok"] is False
    assert any("period 1" in v for v in result["violations"])
    # team 200's missing period-1 vote shouldn't be compared as a real
    # mismatch against period 2 and falsely trip the flip check too
    assert result["direction_flips_ok"] is True


def test_consistency_checks_flags_flip_violation(cfg):
    left_x = float(cfg.court.baskets.left.x)
    right_x = float(cfg.court.baskets.right.x)
    direction_rows = pd.DataFrame(
        [
            {"period": 1, "team_id": 100, "basket_x": left_x, "n_votes": 500, "vote_share": 0.99, "clear": True},
            {"period": 1, "team_id": 200, "basket_x": right_x, "n_votes": 500, "vote_share": 0.99, "clear": True},
            {"period": 2, "team_id": 100, "basket_x": right_x, "n_votes": 500, "vote_share": 0.99, "clear": True},
            {"period": 2, "team_id": 200, "basket_x": left_x, "n_votes": 500, "vote_share": 0.99, "clear": True},
        ]
    )
    result = consistency_checks(direction_rows)
    assert result["direction_flips_ok"] is False


def test_consistency_checks_passes_a_well_formed_game(cfg):
    left_x = float(cfg.court.baskets.left.x)
    right_x = float(cfg.court.baskets.right.x)
    baskets_by_period = {
        1: (left_x, right_x),
        2: (left_x, right_x),
        3: (right_x, left_x),
        4: (right_x, left_x),
        5: (right_x, left_x),  # overtime matches period 4
    }
    rows = []
    for period, (team100_basket, team200_basket) in baskets_by_period.items():
        rows.append(
            {"period": period, "team_id": 100, "basket_x": team100_basket, "n_votes": 500, "vote_share": 0.99, "clear": True}
        )
        rows.append(
            {"period": period, "team_id": 200, "basket_x": team200_basket, "n_votes": 500, "vote_share": 0.99, "clear": True}
        )
    result = consistency_checks(pd.DataFrame(rows))
    assert result["direction_opposite_ok"] is True
    assert result["direction_flips_ok"] is True
    assert result["direction_unclear_periods"] == 0
