"""scripts/prepare_pbp.py's pure logic: clock parsing and game-id filtering."""

from __future__ import annotations

import pandas as pd
import pytest

# scripts/ is on the path via [tool.pytest.ini_options] pythonpath in
# pyproject.toml (also declared under [tool.pyright] extraPaths, for the type
# checker, which doesn't execute a sys.path.insert() the way pytest does).
from prepare_pbp import parse_clock_s


@pytest.mark.parametrize(
    ("pctimestring", "expected"),
    [
        ("12:00", 720),   # start of a regulation period
        ("00:00", 0),     # buzzer
        ("5:00", 300),    # start of overtime
        ("0:01", 1),
    ],
)
def test_parse_clock_s(pctimestring, expected):
    result = parse_clock_s(pd.Series([pctimestring]))
    assert result.iloc[0] == expected


def test_parse_clock_s_counts_down_like_game_clock():
    # Later in a period means a smaller value, same convention as
    # tracking's game_clock (see PROJECT.md).
    series = pd.Series(["12:00", "6:30", "0:05"])
    result = parse_clock_s(series)
    assert result.is_monotonic_decreasing


def test_parse_clock_s_unparseable_is_nan():
    # "1:1" is deliberately rejected too: the real feed always zero-pads
    # seconds to two digits, so a bare single digit is a malformed row worth
    # surfacing as a failure rather than guessing its value.
    series = pd.Series(["12:00", "garbage", None, "1:1"])
    result = parse_clock_s(series)
    assert result.iloc[0] == 720
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])
