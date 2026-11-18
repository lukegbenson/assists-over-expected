"""Clean the raw 2015-16 play-by-play CSV into data/pbp/pbp_2015_16.parquet.

    .venv/bin/python scripts/prepare_pbp.py

Source: Historical/PBP Data/2015-16_pbp.csv from
https://github.com/sumitrodatta/nba-alt-awards, saved to data/pbp/ (download
it first if it isn't there yet).

Transformations, applied to every original NBA stats column unchanged except:
  * GAME_ID is zero-padded to tracking.game_id_length characters (the source
    already stores it zero-padded as text; this is a safety net against any
    dtype coercion during the read).
  * clock_s is added: PCTIMESTRING ("mm:ss") converted to seconds remaining
    in the period, same direction as tracking's game_clock (see PROJECT.md).
  * rows are kept only for games with at least one frame in data/tracking/ —
    the distinct game_id values actually present in the tracking Parquet
    data, not just a matching filename. One game currently has a tracking
    Parquet with zero rows (see PROJECT.md, Tracking QA inventory) and so
    contributes no game_id; it is therefore dropped here too.
  * rows are sorted by game_id, then PERIOD, then clock_s descending, then
    EVENTNUM — time is the primary key because EVENTNUM is sometimes out of
    order in the source feed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_assists.config import get_config, parquet_compression  # noqa: E402

SOURCE_FILENAME = "2015-16_pbp.csv"
OUTPUT_FILENAME = "pbp_2015_16.parquet"

# ID-like columns the source stores as floats (because of embedded NaNs for
# "nobody") — kept as pandas nullable Int64 instead of lossy float64.
NULLABLE_INT_COLUMNS = [
    "PLAYER1_ID", "PLAYER1_TEAM_ID",
    "PLAYER2_ID", "PLAYER2_TEAM_ID",
    "PLAYER3_ID", "PLAYER3_TEAM_ID",
    "PERSON1TYPE", "PERSON2TYPE", "PERSON3TYPE",
]


def tracked_game_ids(tracking_dir: Path) -> set[str]:
    """Distinct game_id values actually present in data/tracking/*.parquet."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT game_id FROM read_parquet('{tracking_dir}/*.parquet')"
    ).fetchall()
    con.close()
    return {r[0] for r in rows}


def parse_clock_s(pctimestring: pd.Series) -> pd.Series:
    """'mm:ss' -> seconds remaining in the period. Unparseable -> NaN."""
    parts = pctimestring.str.extract(r"^(\d+):(\d{2})$")
    minutes = pd.to_numeric(parts[0], errors="coerce")
    seconds = pd.to_numeric(parts[1], errors="coerce")
    return minutes * 60 + seconds


def main() -> int:
    cfg = get_config()
    pbp_dir = cfg.get_path("pbp")
    tracking_dir = cfg.get_path("tracking")
    game_id_width = int(cfg.tracking.game_id_length)

    src_path = pbp_dir / SOURCE_FILENAME
    if not src_path.is_file():
        print(f"missing source file: {src_path}")
        print("download it first, e.g.:")
        print(
            "  curl -sL -o "
            f"{src_path} "
            '"https://raw.githubusercontent.com/sumitrodatta/nba-alt-awards/'
            'main/Historical/PBP%20Data/2015-16_pbp.csv"'
        )
        return 1

    df = pd.read_csv(src_path, index_col=0, dtype={"GAME_ID": str})
    n_read = len(df)

    df["GAME_ID"] = df["GAME_ID"].str.zfill(game_id_width)
    bad_game_id = df["GAME_ID"].str.len() != game_id_width
    n_bad_game_id = int(bad_game_id.sum())

    df["clock_s"] = parse_clock_s(df["PCTIMESTRING"])
    n_bad_clock = int(df["clock_s"].isna().sum())

    for col in NULLABLE_INT_COLUMNS:
        df[col] = df[col].astype("Int64")

    keep_ids = tracked_game_ids(tracking_dir)
    pbp_game_ids = set(df["GAME_ID"].unique())
    n_games_in_pbp = len(pbp_game_ids)
    n_games_kept = len(pbp_game_ids & keep_ids)
    n_games_untracked = n_games_in_pbp - n_games_kept

    df = df[df["GAME_ID"].isin(keep_ids)].copy()

    df = df.sort_values(
        by=["GAME_ID", "PERIOD", "clock_s", "EVENTNUM"],
        ascending=[True, True, False, True],
        kind="mergesort",  # stable: ties keep their original file order
    ).reset_index(drop=True)

    out_path = pbp_dir / OUTPUT_FILENAME
    df.to_parquet(out_path, compression=parquet_compression(cfg), index=False)

    print(f"wrote {out_path} ({len(df):,} rows)")
    print(
        f"summary: rows_read={n_read:,} rows_kept={len(df):,} "
        f"games_in_source={n_games_in_pbp} games_kept={n_games_kept} "
        f"games_not_in_tracking={n_games_untracked} "
        f"bad_game_id={n_bad_game_id} unparseable_clock={n_bad_clock}"
    )
    return 1 if (n_bad_game_id or n_bad_clock) else 0


if __name__ == "__main__":
    raise SystemExit(main())
