"""Detect each make's true rim-crossing time from ball tracking.

    .venv/bin/python scripts/detect_makes.py

For every row in data/derived/makes.parquet, searches that game's ball
tracking frames (see src/nba_assists/rim.py) for the downward crossing
through a rim around the play-by-play clock_s, to measure how far clock_s is
from the ball's actual recorded time (the "clock offset").

The geometric search is restricted to the single basket the scoring team
(makes.parquet's shooter_team_id) was attacking that period, per
data/derived/attack_direction.parquet (run scripts/attack_direction.py
first) -- this rules out ever matching a crossing at the wrong end of the
court, which the old dual-basket search could otherwise do (its "multi_diff"
case).

Writes:
  * data/derived/rim_candidates.parquet -- every geometric candidate found,
    one row per candidate, tagged with the clock-running check's result.
  * data/derived/make_times.parquet -- one row per make: the chosen result
    (or why none was chosen), plus window_gap (whether the search window had
    a tracking gap while the clock was running). make_id = "<game_id>-<eventnum>".
  * reports/clock_offset_by_game.csv -- detection rate and median offset per
    game, with its home team (the arena proxy: one home arena per team in
    this dataset).
  * reports/clock_offset_by_arena.csv -- each home team's game count and the
    mean of its games' median offsets.
  * reports/clock_offset.md -- the high-level report: detection rate, counts
    by case, offset statistics, the by-arena breakdown, detection vs.
    tracking coverage, and random samples for manual review.
"""

# pandas' stubs are known to be imprecise for this kind of code (see
# src/nba_assists/rim.py for the full explanation); verified against
# tests/test_rim.py and by running this script. Full type checking stays on
# elsewhere in the project -- this is scoped to this file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_assists.config import get_config, parquet_compression, seed_everything  # noqa: E402
from nba_assists.rim import CANDIDATE_COLUMNS, ball_rows, detect_make, window_has_gap  # noqa: E402

EMPTY_QFRAME = pd.DataFrame(columns=["ts_ms", "quarter", "game_clock", "x", "y", "z"])

RESULT_COLUMNS = [
    "make_id", "game_id", "eventnum",
    "detected", "reason", "case", "n_candidates", "n_rejected_clock",
    "cross_ts_ms", "cross_game_clock", "basket_x", "offset_s", "gap_s", "window_gap",
]

CASES = ["none", "single", "multi_same", "multi_diff"]

# Offset-statistic groupings: all detected makes, then split by candidate count.
STAT_GROUPS = [
    ("all detected", lambda r: r["detected"]),
    ("single-candidate", lambda r: r["case"] == "single"),
    ("multi-candidate", lambda r: r["case"].isin(["multi_same", "multi_diff"])),
]

PERCENTILES = (1, 5, 25, 75, 95, 99)
N_SAMPLE_PER_CASE = 20
HISTOGRAM_BINS = 20
HISTOGRAM_WIDTH = 40


def game_id_to_path(tracking_dir: Path) -> dict[str, Path]:
    """Canonical game_id -> its tracking Parquet path."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT game_id, filename FROM read_parquet('{tracking_dir}/*.parquet', filename=true)"
    ).fetchall()
    con.close()
    return {game_id: Path(filename) for game_id, filename in rows}


def attack_direction_lookup(derived_dir: Path) -> dict[tuple[str, int, int], float]:
    """(game_id, period, team_id) -> the basket that team was attacking.

    Omits any row with no determined basket (a team-period that cast zero
    votes -- see src/nba_assists/direction.py) rather than looking up NaN.
    """
    direction_path = derived_dir / "attack_direction.parquet"
    direction = pd.read_parquet(direction_path, columns=["game_id", "period", "team_id", "basket_x"])
    direction = direction.dropna(subset=["basket_x"])
    return {
        (game_id, int(period), int(team_id)): float(basket_x)
        for game_id, period, team_id, basket_x in direction.itertuples(index=False)
    }


def home_team_abbr(roster_path: Path) -> str | None:
    """The home team's abbreviation from a tracking game's roster CSV."""
    if not roster_path.is_file():
        return None
    with roster_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["side"] == "home":
                return row["team_abbr"]
    return None


def _stats_block(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {}
    stats = {
        "n": int(values.count()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    for p in PERCENTILES:
        stats[f"p{p}"] = float(values.quantile(p / 100))
    return stats


def _ascii_histogram(values: pd.Series) -> list[str]:
    if values.empty:
        return ["(no data)"]
    counts, edges = np.histogram(values, bins=HISTOGRAM_BINS)
    peak = int(counts.max()) or 1
    lines = []
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * max(1, round(HISTOGRAM_WIDTH * count / peak)) if count else ""
        lines.append(f"{lo:8.3f} to {hi:8.3f}  {count:6d}  {bar}")
    return lines


def write_by_game_csv(
    results: pd.DataFrame, game_home_team: dict[str, str | None], report_path: Path
) -> pd.DataFrame:
    by_game = results.groupby("game_id").agg(
        n_makes=("make_id", "size"),
        n_detected=("detected", "sum"),
        median_offset_s=("offset_s", "median"),
    )
    by_game["detection_rate"] = by_game["n_detected"] / by_game["n_makes"]
    by_game = by_game.reset_index().sort_values(
        "detection_rate", kind="mergesort"
    ).reset_index(drop=True)
    by_game["home_team"] = by_game["game_id"].map(game_home_team)
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["game_id", "home_team", "n_makes", "n_detected", "detection_rate", "median_offset_s"])
        for _, row in by_game.iterrows():
            median_offset_s = row["median_offset_s"]
            writer.writerow(
                [
                    row["game_id"],
                    row["home_team"] or "",
                    row["n_makes"],
                    row["n_detected"],
                    round(row["detection_rate"], 4),
                    round(median_offset_s, 4) if pd.notna(median_offset_s) else "",
                ]
            )
    return by_game


def write_by_arena_csv(by_game: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    """Each home team's game count and the mean of its games' median offsets."""
    valid = by_game.dropna(subset=["home_team", "median_offset_s"])
    by_arena = (
        valid.groupby("home_team")
        .agg(n_games=("game_id", "size"), mean_median_offset_s=("median_offset_s", "mean"))
        .reset_index()
        .sort_values("mean_median_offset_s", kind="mergesort")
        .reset_index(drop=True)
    )
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["home_team", "n_games", "mean_median_offset_s"])
        for _, row in by_arena.iterrows():
            writer.writerow([row["home_team"], row["n_games"], round(row["mean_median_offset_s"], 4)])
    return by_arena


def _coverage_table(results: pd.DataFrame) -> list[str]:
    covered = ~results["window_gap"]
    detected = results["detected"]
    n_cov_det = int((covered & detected).sum())
    n_cov_not = int((covered & ~detected).sum())
    n_gap_det = int((~covered & detected).sum())
    n_gap_not = int((~covered & ~detected).sum())

    return [
        "## Detection vs. tracking coverage",
        "",
        "Whether the search window (`clock_s` +/- "
        "`rim_detection.search_before_s`/`search_after_s`) had a tracking gap "
        "-- `inventory.max_gap_running_s` or longer while the clock was "
        "running -- crossed with whether a rim crossing was detected.",
        "",
        "| | detected | not detected |",
        "|---|---|---|",
        f"| **covered** | {n_cov_det:,} (most makes) | {n_cov_not:,} (real detector misses) |",
        f"| **gap** | {n_gap_det:,} (possible wrong matches) | {n_gap_not:,} (expected) |",
        "",
    ]


def write_report(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    by_game: pd.DataFrame,
    by_game_csv_path: Path,
    by_arena: pd.DataFrame,
    by_arena_csv_path: Path,
    report_path: Path,
) -> None:
    n_makes = len(results)
    n_detected = int(results["detected"].sum())
    detection_rate = n_detected / n_makes if n_makes else float("nan")
    n_rejected_total = int((~candidates["clock_check_passed"]).sum())

    case_counts = results["case"].value_counts()

    lines = [
        "# Clock offset, 2015-16",
        "",
        f"- **Makes:** {n_makes:,}",
        f"- **Detected:** {n_detected:,} ({detection_rate:.2%})",
        f"- **Candidates rejected by the clock-running check:** {n_rejected_total:,}",
        f"- **Per-game detection rate:** see `{by_game_csv_path.name}`"
        f" (worst games listed below)",
        "",
        "## Makes by case",
        "",
        "| case | makes |",
        "|---|---|",
    ]
    for case in CASES:
        lines.append(f"| {case} | {int(case_counts.get(case, 0)):,} |")

    lines += ["", *_coverage_table(results)]

    lines += [
        "## Game-clock gap, multi-candidate makes",
        "",
    ]
    gap_values = results.loc[results["case"].isin(["multi_same", "multi_diff"]), "gap_s"]
    gap_stats = _stats_block(gap_values)
    if gap_stats:
        lines.append(
            f"n={gap_stats['n']:,}  median={gap_stats['median']:.3f}s  "
            f"mean={gap_stats['mean']:.3f}s  max={gap_stats['max']:.3f}s"
        )
    else:
        lines.append("(no multi-candidate makes)")

    lines += ["", "## Offset statistics (offset_s = clock_s - cross_game_clock)", ""]
    for label, selector in STAT_GROUPS:
        mask = selector(results)
        stats = _stats_block(results.loc[mask, "offset_s"])
        lines.append(f"### {label}")
        lines.append("")
        if not stats:
            lines += ["(no makes in this group)", ""]
            continue
        header = "| n | median | mean | std | min | p1 | p5 | p25 | p75 | p95 | p99 | max |"
        sep = "|---" * 12 + "|"
        row = (
            f"| {stats['n']:,} | {stats['median']:.3f} | {stats['mean']:.3f} | "
            f"{stats['std']:.3f} | {stats['min']:.3f} | {stats['p1']:.3f} | "
            f"{stats['p5']:.3f} | {stats['p25']:.3f} | {stats['p75']:.3f} | "
            f"{stats['p95']:.3f} | {stats['p99']:.3f} | {stats['max']:.3f} |"
        )
        lines += [header, sep, row, ""]

    detected_offsets = results.loc[results["detected"], "offset_s"]
    lines += ["### Histogram, all detected makes", "", "```"]
    lines += _ascii_histogram(detected_offsets)
    lines += ["```", ""]

    # Per-game medians vs. the per-make distribution above: a much tighter
    # spread here would mean the large per-make offsets are mostly within-game
    # noise around a fairly universal lag; a spread nearly as wide as the
    # per-make one means the offset is closer to a fixed, game-specific
    # quantity (consistent with the by-arena trend below).
    median_per_game = by_game["median_offset_s"].dropna()
    median_stats = _stats_block(median_per_game)
    lines += ["### Histogram, median offset_s per game", ""]
    if median_stats:
        lines.append(
            f"n={median_stats['n']:,} games  median={median_stats['median']:.3f}  "
            f"mean={median_stats['mean']:.3f}  std={median_stats['std']:.3f}  "
            f"min={median_stats['min']:.3f}  max={median_stats['max']:.3f}"
        )
        lines.append(f"(for comparison, per-make std above: {_stats_block(detected_offsets)['std']:.3f})")
        lines += ["", "```"]
        lines += _ascii_histogram(median_per_game)
        lines += ["```", ""]
    else:
        lines += ["(no games with a detected make)", ""]

    if len(detected_offsets):
        share_trunc = float(((detected_offsets >= -1) & (detected_offsets <= 0)).mean())
        share_round = float(((detected_offsets >= -0.5) & (detected_offsets <= 0.5)).mean())
        verdict = "rounded" if share_round > share_trunc else "truncated"
        lines += [
            "## Truncated vs. rounded play-by-play times",
            "",
            f"- share of offsets in [-1, 0] (truncated): {share_trunc:.2%}",
            f"- share of offsets in [-0.5, 0.5] (rounded): {share_round:.2%}",
            f"- play-by-play times look **{verdict}**.",
            "",
        ]

    lines += ["## Worst games by detection rate", "", "| game_id | n_makes | n_detected | detection_rate |", "|---|---|---|---|"]
    for _, row in by_game.head(10).iterrows():
        lines.append(
            f"| {row['game_id']} | {row['n_makes']} | {row['n_detected']} | {row['detection_rate']:.2%} |"
        )

    lines += [
        "",
        "## Mean median offset by arena (home team)",
        "",
        f"Per game: median `offset_s` over its detected makes (`{by_game_csv_path.name}`)."
        f" Per arena: mean of its games' medians, unweighted by make count.",
        "",
        "| home team | games | mean median offset_s |",
        "|---|---|---|",
    ]
    for _, row in by_arena.iterrows():
        lines.append(f"| {row['home_team']} | {row['n_games']} | {row['mean_median_offset_s']:.3f} |")
    lines.append(f"\nFull detail: `{by_arena_csv_path.name}`.")

    lines += ["", "## Random make_ids for manual review, by case", ""]
    for case in CASES:
        ids = results.loc[results["case"] == case, "make_id"].tolist()
        sample = random.sample(ids, min(N_SAMPLE_PER_CASE, len(ids)))
        lines.append(f"- **{case}** ({len(ids):,} total): {', '.join(sorted(sample)) or '(none)'}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cfg = get_config()
    seed_everything(cfg)  # for the random make_id samples in the report

    tracking_dir = cfg.get_path("tracking")
    roster_dir = cfg.get_path("rosters")
    derived_dir = cfg.get_path("derived")
    reports_dir = cfg.get_path("reports")
    derived_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    makes = pd.read_parquet(derived_dir / "makes.parquet")
    game_paths = game_id_to_path(tracking_dir)
    basket_lookup = attack_direction_lookup(derived_dir)

    candidate_frames: list[pd.DataFrame] = []
    result_rows: list[dict] = []
    failed_games: list[str] = []
    game_home_team: dict[str, str | None] = {}
    n_missing_basket = 0

    for game_id, group in makes.groupby("game_id", sort=False):
        path = game_paths.get(game_id)
        if path is None:
            by_quarter: dict[int, pd.DataFrame] = {}
            failed_games.append(game_id)
            game_home_team[game_id] = None
        else:
            tracking = pd.read_parquet(
                path, columns=["ts_ms", "quarter", "game_clock", "team_id", "player_id", "x", "y", "z"]
            )
            balls = ball_rows(tracking, cfg)
            by_quarter = {
                int(q): qf.reset_index(drop=True) for q, qf in balls.groupby("quarter", sort=False)
            }
            game_home_team[game_id] = home_team_abbr(roster_dir / (path.stem + ".csv"))

        for _, make in group.iterrows():
            period = int(make["period"])
            quarter_frames = by_quarter.get(period, EMPTY_QFRAME)
            clock_s = float(make["clock_s"])

            basket_x = basket_lookup.get((make["game_id"], period, int(make["shooter_team_id"])))
            if basket_x is None:
                n_missing_basket += 1  # shouldn't happen: makes.parquet is pre-filtered to
                # games whose attack-direction consistency checks passed; fall back to
                # searching both baskets rather than detecting nothing for this make.

            candidates, result = detect_make(quarter_frames, clock_s, cfg, basket_x=basket_x)
            result["window_gap"] = window_has_gap(quarter_frames, clock_s, cfg)
            make_id = f"{make['game_id']}-{int(make['eventnum'])}"

            if len(candidates):
                candidates = candidates.copy()
                candidates["make_id"] = make_id
                candidates["game_id"] = make["game_id"]
                candidates["eventnum"] = int(make["eventnum"])
                candidate_frames.append(candidates)

            result_rows.append(
                {"make_id": make_id, "game_id": make["game_id"], "eventnum": int(make["eventnum"]), **result}
            )

    candidates_out = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame(columns=["make_id", "game_id", "eventnum"] + CANDIDATE_COLUMNS)
    )
    results_out = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)

    candidates_path = derived_dir / "rim_candidates.parquet"
    make_times_path = derived_dir / "make_times.parquet"
    candidates_out.to_parquet(candidates_path, compression=parquet_compression(cfg), index=False)
    results_out.to_parquet(make_times_path, compression=parquet_compression(cfg), index=False)

    by_game_csv_path = reports_dir / "clock_offset_by_game.csv"
    by_game = write_by_game_csv(results_out, game_home_team, by_game_csv_path)

    by_arena_csv_path = reports_dir / "clock_offset_by_arena.csv"
    by_arena = write_by_arena_csv(by_game, by_arena_csv_path)

    report_path = reports_dir / "clock_offset.md"
    write_report(
        results_out, candidates_out, by_game, by_game_csv_path, by_arena, by_arena_csv_path, report_path
    )

    n_detected = int(results_out["detected"].sum())
    detection_rate = n_detected / len(results_out) if len(results_out) else float("nan")
    print(f"wrote {candidates_path} ({len(candidates_out):,} rows)")
    print(f"wrote {make_times_path} ({len(results_out):,} rows)")
    print(f"wrote {by_game_csv_path}")
    print(f"wrote {report_path}")
    print(
        f"summary: makes={len(results_out):,} detected={n_detected:,} "
        f"detection_rate={detection_rate:.4f} failed_games={len(failed_games)} "
        f"missing_basket_lookup={n_missing_basket}"
    )
    return 1 if failed_games else 0


if __name__ == "__main__":
    raise SystemExit(main())
