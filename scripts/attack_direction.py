"""Determine which basket each team attacks, per game and period.

    .venv/bin/python scripts/attack_direction.py

Uses only the tracking Parquet files and roster CSVs (no play-by-play): a
ball-holder vote (see src/nba_assists/direction.py and config.yaml's
attack_direction: section). This will later confirm the scoring team's
basket for each made field goal.

Writes:
  * data/derived/attack_direction.parquet -- one row per (game_id, period,
    team_id): the winning basket, its vote_share, and the secondary check's
    agreement rate.
  * reports/tracking_inventory.csv -- the existing per-game QA inventory
    (scripts/tracking_inventory.py), with four columns appended:
    direction_min_share, direction_unclear_periods, direction_opposite_ok,
    direction_flips_ok.
  * reports/attack_direction.md -- games/team-periods processed, the
    vote_share distribution, unclear team-periods, games with a consistency
    violation (listed by game_id and period), and the secondary check's
    overall agreement rate.
"""

# pandas' stubs are known to be imprecise for this kind of code (see
# src/nba_assists/direction.py for the full explanation); verified against
# tests/test_direction.py and by running this script. Full type checking
# stays on elsewhere in the project -- this is scoped to this file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportReturnType=false

from __future__ import annotations

import csv
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_assists.config import get_config, parquet_compression  # noqa: E402
from nba_assists.direction import consistency_checks, secondary_agreement, tally_votes, votes  # noqa: E402

DIRECTION_COLUMNS = [
    "game_id", "period", "team_id", "team_abbr", "basket_x",
    "n_votes", "vote_share", "secondary_agreement", "clear",
]
INVENTORY_UPDATE_COLUMNS = [
    "direction_min_share", "direction_unclear_periods", "direction_opposite_ok", "direction_flips_ok",
]
PERCENTILES = (1, 5)


def roster_team_abbrs(roster_path: Path) -> dict[int, str]:
    """team_id -> team_abbr for every team in a _rosters/<game>.csv."""
    if not roster_path.is_file():
        return {}
    out: dict[int, str] = {}
    with roster_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[int(row["team_id"])] = row["team_abbr"]
    return out


def game_id_to_path(tracking_dir: Path) -> dict[str, Path]:
    """Canonical game_id -> its tracking Parquet path."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT game_id, filename FROM read_parquet('{tracking_dir}/*.parquet', filename=true)"
    ).fetchall()
    con.close()
    return {game_id: Path(filename) for game_id, filename in rows}


def process_game(path: Path, game_id: str, roster_dir: Path, cfg) -> tuple[pd.DataFrame, dict]:
    """(direction_rows, consistency_dict) for one game's tracking Parquet.

    direction_rows always has one row per (period, team_id) actually played
    in this game, even a team-period with zero votes (n_votes=0, basket_x
    unknown, clear=False) -- a later step (alignment) needs every
    combination to look up, not just the ones that cleared the vote
    thresholds.
    """
    tracking = pd.read_parquet(
        path, columns=["quarter", "ts_ms", "game_clock", "shot_clock", "team_id", "player_id", "x", "y", "z"]
    )
    votes_df = votes(tracking, cfg)
    tally = tally_votes(votes_df, cfg)
    agreement = secondary_agreement(votes_df)

    team_abbrs = roster_team_abbrs(roster_dir / (path.stem + ".csv"))
    periods = sorted(int(q) for q in tracking["quarter"].unique())
    team_ids = sorted(team_abbrs)

    full_index = pd.MultiIndex.from_product([periods, team_ids], names=["period", "team_id"])
    tally = tally.set_index(["period", "team_id"]).reindex(full_index).reset_index()
    tally["n_votes"] = tally["n_votes"].fillna(0).astype(int)
    tally["clear"] = tally["clear"].fillna(False).astype(bool)

    # merged after reindexing, so a team-period padded in above still gets
    # its period's agreement rate if the other team in that period voted
    tally = tally.merge(agreement, on="period", how="left")
    tally["game_id"] = game_id
    tally["team_abbr"] = tally["team_id"].map(team_abbrs).fillna("")
    direction_rows = tally[DIRECTION_COLUMNS]

    consistency = consistency_checks(direction_rows)
    consistency["game_id"] = game_id
    return direction_rows, consistency


def update_tracking_inventory(inventory_csv_path: Path, updates: pd.DataFrame) -> None:
    """Append/replace the direction_* columns on the existing per-game inventory CSV."""
    inventory = pd.read_csv(inventory_csv_path, dtype={"game_id": str}, keep_default_na=False)
    inventory = inventory.drop(columns=INVENTORY_UPDATE_COLUMNS, errors="ignore")
    merged = inventory.merge(updates, on="game_id", how="left")
    for col in ("direction_min_share",):
        merged[col] = merged[col].map(lambda v: round(v, 4) if pd.notna(v) else "")
    for col in ("direction_unclear_periods", "direction_opposite_ok", "direction_flips_ok"):
        merged[col] = merged[col].where(merged[col].notna(), "")
    merged.to_csv(inventory_csv_path, index=False)


def write_report(
    direction_df: pd.DataFrame, consistency_df: pd.DataFrame, report_path: Path
) -> None:
    n_games = consistency_df["game_id"].nunique()
    n_team_periods = len(direction_df)
    n_unclear = int((~direction_df["clear"]).sum())

    share = direction_df["vote_share"]
    share_stats = (
        f"median={share.median():.4f}  "
        + "  ".join(f"p{p}={share.quantile(p / 100):.4f}" for p in PERCENTILES)
        + f"  min={share.min():.4f}"
        if len(share)
        else "(no team-periods)"
    )

    violating = consistency_df[~consistency_df["direction_opposite_ok"] | ~consistency_df["direction_flips_ok"]]

    lines = [
        "# Attack direction, 2015-16",
        "",
        f"- **Games processed:** {n_games:,}",
        f"- **Team-periods processed:** {n_team_periods:,}",
        f"- **Unclear team-periods:** {n_unclear:,} (vote_share/n_votes below the"
        " attack_direction: min_share/min_votes thresholds)",
        f"- **Games with a consistency violation:** {len(violating):,}",
        f"- **vote_share distribution:** {share_stats}",
    ]

    if len(direction_df):
        agree = direction_df.drop_duplicates(["game_id", "period"])["secondary_agreement"]
        lines.append(f"- **Secondary check overall agreement rate:** {agree.mean():.2%}")

    if n_unclear:
        lines += [
            "",
            "## Unclear team-periods",
            "",
            "| game_id | period | team_id | team_abbr | n_votes | vote_share |",
            "|---|---|---|---|---|---|",
        ]
        for _, row in direction_df.loc[~direction_df["clear"]].iterrows():
            lines.append(
                f"| {row['game_id']} | {row['period']} | {row['team_id']} | {row['team_abbr']} | "
                f"{row['n_votes']} | {row['vote_share']:.4f} |"
            )

    if len(violating):
        lines += ["", "## Games with a consistency violation", ""]
        for _, row in violating.iterrows():
            lines.append(f"- **{row['game_id']}**")
            for v in row["violations"]:
                lines.append(f"  - {v}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cfg = get_config()
    tracking_dir = cfg.get_path("tracking")
    roster_dir = cfg.get_path("rosters")
    derived_dir = cfg.get_path("derived")
    reports_dir = cfg.get_path("reports")
    derived_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    inventory_csv_path = reports_dir / "tracking_inventory.csv"
    if not inventory_csv_path.is_file():
        print(f"missing {inventory_csv_path}; run scripts/tracking_inventory.py first")
        return 1

    game_paths = game_id_to_path(tracking_dir)

    direction_rows: list[pd.DataFrame] = []
    consistency_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for game_id, path in sorted(game_paths.items()):
        try:
            rows, consistency = process_game(path, game_id, roster_dir, cfg)
        except Exception as exc:  # keep going on a bad file
            failures.append((path.name, str(exc)))
            continue
        if not rows.empty:  # an all-empty, object-dtype frame would corrupt concat's dtypes below
            direction_rows.append(rows)
        consistency_rows.append(consistency)

    direction_df = (
        pd.concat(direction_rows, ignore_index=True) if direction_rows else pd.DataFrame(columns=DIRECTION_COLUMNS)
    )
    if len(direction_df):
        direction_df["clear"] = direction_df["clear"].astype(bool)
    consistency_df = pd.DataFrame(consistency_rows)
    if len(consistency_df):
        consistency_df["direction_opposite_ok"] = consistency_df["direction_opposite_ok"].astype(bool)
        consistency_df["direction_flips_ok"] = consistency_df["direction_flips_ok"].astype(bool)

    out_path = derived_dir / "attack_direction.parquet"
    direction_df.to_parquet(out_path, compression=parquet_compression(cfg), index=False)

    inventory_updates = consistency_df[["game_id"] + INVENTORY_UPDATE_COLUMNS]
    update_tracking_inventory(inventory_csv_path, inventory_updates)

    report_path = reports_dir / "attack_direction.md"
    write_report(direction_df, consistency_df, report_path)

    n_unclear = int((~direction_df["clear"]).sum()) if len(direction_df) else 0
    print(f"wrote {out_path} ({len(direction_df):,} rows)")
    print(f"updated {inventory_csv_path}")
    print(f"wrote {report_path}")
    print(
        f"summary: games={len(game_paths):,} processed={len(consistency_rows):,} "
        f"team_periods={len(direction_df):,} unclear={n_unclear:,} failed_games={len(failures)}"
    )
    if failures:
        print("failures:")
        for name, err in failures:
            print(f"  {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
