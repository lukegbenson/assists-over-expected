"""Pure-logic pieces of scripts/tracking_inventory.py that don't need real data."""

from __future__ import annotations

import pytest

# scripts/ is on the path via [tool.pytest.ini_options] pythonpath in
# pyproject.toml (also declared under [tool.pyright] extraPaths, for the type
# checker, which doesn't execute a sys.path.insert() the way pytest does).
from tracking_inventory import parse_date_from_filename, read_roster_teams


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("01.01.2016.CHA.at.TOR", "2016-01-01"),
        ("12.25.2015.LAL.at.GSW", "2015-12-25"),
        ("not-a-date-stem", None),
        ("2016.01.01.CHA.at.TOR", None),  # wrong field order is not matched
        # upstream naming quirk: path prefix baked into some Dec-5-2015 stems
        ("2016.NBA.Raw.SportVU.Game.Logs12.05.2015.BOS.at.SAS", "2015-12-05"),
    ],
)
def test_parse_date_from_filename(stem, expected):
    assert parse_date_from_filename(stem) == expected


def test_read_roster_teams(tmp_path):
    roster = tmp_path / "game.csv"
    roster.write_text(
        "game_id,side,team_id,team_abbr,player_id,first_name,last_name,jersey,position\n"
        "0021500492,home,1610612761,TOR,101,A,B,1,G\n"
        "0021500492,home,1610612761,TOR,102,C,D,2,F\n"
        "0021500492,visitor,1610612766,CHA,201,E,F,3,C\n"
        "0021500492,visitor,1610612766,CHA,202,G,H,4,G\n"
    )
    home, away = read_roster_teams(roster)
    assert (home, away) == ("TOR", "CHA")


def test_read_roster_teams_missing_side(tmp_path):
    roster = tmp_path / "game.csv"
    roster.write_text(
        "game_id,side,team_id,team_abbr,player_id,first_name,last_name,jersey,position\n"
        "0021500492,home,1610612761,TOR,101,A,B,1,G\n"
    )
    home, away = read_roster_teams(roster)
    assert (home, away) == ("TOR", None)
