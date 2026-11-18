"""scripts/build_makes.py's pure logic: roster loading and the makes table."""

from __future__ import annotations

import csv

import pandas as pd
import pytest

# scripts/ and src/ are on the path via [tool.pytest.ini_options] pythonpath
# in pyproject.toml (also declared under [tool.pyright] extraPaths, for the
# type checker, which doesn't execute a sys.path.insert() the way pytest does).
from build_makes import build_makes, event_type_names, load_rosters
from nba_assists.config import get_config


@pytest.fixture(scope="module")
def cfg():
    return get_config()


def test_event_type_names_round_trip(cfg):
    names = event_type_names(cfg)
    assert names[1] == "made_shot"
    assert names[4] == "rebound"
    assert len(names) == len(cfg.pbp.event_types)


def test_load_rosters(tmp_path):
    roster_dir = tmp_path / "_rosters"
    roster_dir.mkdir()
    header = "game_id,side,team_id,team_abbr,player_id,first_name,last_name,jersey,position\n"
    (roster_dir / "some.game.stem.csv").write_text(
        header
        + "0000000001,home,1,TOR,100,A,B,1,G\n"
        + "0000000001,home,1,TOR,200,C,D,2,F\n"
        + "0000000001,visitor,2,BOS,400,E,F,3,C\n"
    )
    rosters = load_rosters(roster_dir)
    assert set(rosters) == {"0000000001"}
    r = rosters["0000000001"]
    assert r["home_team_id"] == 1
    assert r["player_ids"] == {100, 200, 400}


def _mini_pbp() -> pd.DataFrame:
    # Two games. Game 1: rebound -> assisted make (home) -> unassisted make
    # (visitor, shooter not on roster). Game 2: a single make with nothing
    # before it (no previous event).
    rows = [
        dict(GAME_ID="0000000001", EVENTNUM=1, EVENTMSGTYPE=4, PERIOD=1, clock_s=700,
             HOMEDESCRIPTION="Smith REBOUND", VISITORDESCRIPTION=None, NEUTRALDESCRIPTION=None,
             PLAYER1_ID=100, PLAYER1_TEAM_ID=1, PLAYER2_ID=0, EVENTMSGACTIONTYPE=0),
        dict(GAME_ID="0000000001", EVENTNUM=2, EVENTMSGTYPE=1, PERIOD=1, clock_s=695,
             HOMEDESCRIPTION="Smith made shot", VISITORDESCRIPTION=None, NEUTRALDESCRIPTION=None,
             PLAYER1_ID=100, PLAYER1_TEAM_ID=1, PLAYER2_ID=200, EVENTMSGACTIONTYPE=5),
        dict(GAME_ID="0000000001", EVENTNUM=3, EVENTMSGTYPE=1, PERIOD=1, clock_s=680,
             HOMEDESCRIPTION=None, VISITORDESCRIPTION="Jones made shot", NEUTRALDESCRIPTION=None,
             PLAYER1_ID=300, PLAYER1_TEAM_ID=2, PLAYER2_ID=0, EVENTMSGACTIONTYPE=5),
        dict(GAME_ID="0000000002", EVENTNUM=1, EVENTMSGTYPE=1, PERIOD=1, clock_s=719,
             HOMEDESCRIPTION="Early make", VISITORDESCRIPTION=None, NEUTRALDESCRIPTION=None,
             PLAYER1_ID=500, PLAYER1_TEAM_ID=9, PLAYER2_ID=0, EVENTMSGACTIONTYPE=5),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def rosters():
    return {
        "0000000001": {"home_team_id": 1, "player_ids": {100, 200}},
        "0000000002": {"home_team_id": 9, "player_ids": {500}},
    }


def test_build_makes_schema_and_counts(cfg, rosters):
    out = build_makes(_mini_pbp(), rosters, cfg)
    assert len(out) == 3  # three EVENTMSGTYPE == 1 rows
    assert set(out["game_id"]) == {"0000000001", "0000000002"}


def test_build_makes_assist_detection(cfg, rosters):
    out = build_makes(_mini_pbp(), rosters, cfg)
    assisted_row = out[(out["game_id"] == "0000000001") & (out["eventnum"] == 2)].iloc[0]
    unassisted_row = out[(out["game_id"] == "0000000001") & (out["eventnum"] == 3)].iloc[0]

    assert assisted_row["assisted"] is True or assisted_row["assisted"] == True  # noqa: E712
    assert assisted_row["assister_id"] == 200
    assert unassisted_row["assisted"] is False or unassisted_row["assisted"] == False  # noqa: E712
    assert pd.isna(unassisted_row["assister_id"])


def test_build_makes_home_team_flag(cfg, rosters):
    out = build_makes(_mini_pbp(), rosters, cfg)
    home_row = out[(out["game_id"] == "0000000001") & (out["eventnum"] == 2)].iloc[0]
    visitor_row = out[(out["game_id"] == "0000000001") & (out["eventnum"] == 3)].iloc[0]
    assert bool(home_row["is_home_team"]) is True
    assert bool(visitor_row["is_home_team"]) is False


def test_build_makes_previous_event_by_time(cfg, rosters):
    out = build_makes(_mini_pbp(), rosters, cfg)
    row2 = out[(out["game_id"] == "0000000001") & (out["eventnum"] == 2)].iloc[0]
    row3 = out[(out["game_id"] == "0000000001") & (out["eventnum"] == 3)].iloc[0]

    assert row2["prev_eventnum"] == 1
    assert row2["prev_msgtype"] == 4  # rebound
    assert row2["prev_description"] == "Smith REBOUND"
    assert bool(row2["prev_same_period"]) is True

    # the make itself (eventnum 2) is the previous event for eventnum 3
    assert row3["prev_eventnum"] == 2
    assert row3["prev_msgtype"] == 1


def test_build_makes_no_previous_event_for_first_row_of_game(cfg, rosters):
    out = build_makes(_mini_pbp(), rosters, cfg)
    first_row = out[out["game_id"] == "0000000002"].iloc[0]
    assert pd.isna(first_row["prev_eventnum"])
    assert bool(first_row["prev_same_period"]) is False


def test_build_makes_shooter_off_roster_flagged(cfg, rosters):
    out = build_makes(_mini_pbp(), rosters, cfg)
    off_roster_row = out[(out["game_id"] == "0000000001") & (out["eventnum"] == 3)].iloc[0]
    on_roster_row = out[(out["game_id"] == "0000000001") & (out["eventnum"] == 2)].iloc[0]
    assert bool(off_roster_row["_shooter_on_roster"]) is False  # 300 not in roster
    assert bool(on_roster_row["_shooter_on_roster"]) is True
