"""Build a per-game QA inventory of the tracking Parquet files.

Reads every Parquet in data/tracking (one per game, per PROJECT.md), computes
frame-level coverage and integrity stats per game, and writes one row per
game to reports/tracking_inventory.csv, plus a high-level summary to
reports/tracking_inventory_summary.md.

    .venv/bin/python scripts/tracking_inventory.py

A "frame" is a distinct ts_ms within a game (rows are already de-duplicated on
ts_ms by the build step). "minutes covered" is frame_count / frame_rate / 60:
the amount of sampled game time actually captured, not the wall-clock span
between the first and last frame (which also spans quarter breaks, timeouts,
and halftime).

Games are flagged (see inventory: in config.yaml) when any of:
  * minutes covered is below inventory.min_minutes_required,
  * the share of frames missing a player or the ball exceeds
    inventory.max_missing_frame_share,
  * any gap between consecutive frames in the same quarter, while game_clock
    was actually decreasing, exceeds inventory.max_gap_running_s. A gap while
    game_clock is flat (timeout, dead ball) or across a quarter boundary is
    not a tracking defect and is not flagged by this rule.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_assists.config import format_game_id, get_config  # noqa: E402

# MM.DD.YYYY.AWAY.at.HOME, e.g. "01.01.2016.CHA.at.TOR" -> date only; team
# abbreviations are read from the roster CSV instead, which is authoritative.
# Searched rather than anchored at the start: ten of the Dec-5-2015 archives in
# the upstream repo carry a "2016.NBA.Raw.SportVU.Game.Logs" prefix baked into
# the filename (an upstream naming quirk, confirmed to have no normally-named
# duplicate), so the date does not always start the stem.
DATE_RE = re.compile(r"(?P<mm>\d{2})\.(?P<dd>\d{2})\.(?P<yyyy>\d{4})")

FRAME_STATS_SQL = """
WITH frames AS (
    SELECT
        ts_ms,
        ANY_VALUE(game_id)    AS game_id,
        ANY_VALUE(quarter)    AS quarter,
        ANY_VALUE(game_clock) AS game_clock,
        COUNT(*)                                            AS n_rows,
        SUM(CASE WHEN team_id <> {ball_team_id} THEN 1 ELSE 0 END) AS n_player_rows,
        SUM(CASE WHEN team_id = {ball_team_id} THEN 1 ELSE 0 END)  AS n_ball_rows
    FROM read_parquet(?)
    GROUP BY ts_ms
),
ordered AS (
    SELECT
        *,
        LAG(ts_ms)        OVER (ORDER BY ts_ms) AS prev_ts_ms,
        LAG(quarter)      OVER (ORDER BY ts_ms) AS prev_quarter,
        LAG(game_clock)   OVER (ORDER BY ts_ms) AS prev_game_clock
    FROM frames
)
SELECT
    ANY_VALUE(game_id)                                       AS game_id,
    COUNT(*)                                                 AS frame_count,
    AVG(CASE WHEN n_rows = {players_on_court} + 1
             THEN 1.0 ELSE 0.0 END)                           AS share_frames_11_rows,
    AVG(CASE WHEN n_ball_rows = 0 THEN 1.0 ELSE 0.0 END)      AS share_frames_ball_missing,
    AVG(CASE WHEN n_player_rows < {players_on_court} OR n_ball_rows = 0
             THEN 1.0 ELSE 0.0 END)                           AS share_frames_missing_object,
    MAX(ts_ms - prev_ts_ms)                                   AS max_gap_ms,
    MAX(CASE WHEN quarter = prev_quarter
                  AND game_clock < prev_game_clock
             THEN ts_ms - prev_ts_ms END)                     AS max_gap_running_ms,
    (SELECT STRING_AGG(CAST(q AS VARCHAR), ',' ORDER BY q)
     FROM (SELECT DISTINCT quarter AS q FROM ordered) sub)   AS quarters_present
FROM ordered
"""

INVENTORY_FIELDS = [
    "game_id",
    "date",
    "home_team",
    "away_team",
    "frame_count",
    "minutes_covered",
    "share_frames_11_rows",
    "share_frames_ball_missing",
    "quarters_present",
    "max_gap_ms",
    "max_gap_running_ms",
    "flagged",
    "flag_reasons",
]


def read_roster_teams(roster_path: Path) -> tuple[str | None, str | None]:
    """(home_abbr, away_abbr) from a _rosters/<game>.csv, or (None, None)."""
    home_abbr = away_abbr = None
    with roster_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["side"] == "home" and home_abbr is None:
                home_abbr = row["team_abbr"]
            elif row["side"] == "visitor" and away_abbr is None:
                away_abbr = row["team_abbr"]
    return home_abbr, away_abbr


def _round_or_blank(value: float | None, ndigits: int = 4) -> float | str:
    return round(value, ndigits) if value is not None else ""


def parse_date_from_filename(stem: str) -> str | None:
    m = DATE_RE.search(stem)
    if not m:
        return None
    return f"{m['yyyy']}-{m['mm']}-{m['dd']}"


def inventory_row(parquet_path: Path, roster_dir: Path, cfg, con: duckdb.DuckDBPyConnection) -> dict:
    ball_team_id = int(cfg.sentinels.ball_team_id)
    players_on_court = int(cfg.tracking.players_on_court)
    sql = FRAME_STATS_SQL.format(ball_team_id=ball_team_id, players_on_court=players_on_court)
    result = con.execute(sql, [str(parquet_path)]).fetchone()
    if result is None:
        # Can't happen for this query in practice: it has no GROUP BY, so the
        # aggregates always produce exactly one row, even over zero input
        # rows (an empty tracking Parquet, e.g.). Guarded anyway rather than
        # asserted away, since fetchone()'s declared return type is
        # Optional and a future edit to the query could make it a real path.
        raise RuntimeError(f"aggregate query returned no row for {parquet_path}")
    cols: list[str] = [d[0] for d in con.description]
    stats = {col: val for col, val in zip(cols, result)}

    frame_rate = float(cfg.tracking.source_frame_rate_hz)
    minutes_covered = stats["frame_count"] / frame_rate / 60.0

    raw_game_id = stats["game_id"]
    game_id = format_game_id(raw_game_id, cfg) if raw_game_id is not None else parquet_path.stem

    roster_path = roster_dir / (parquet_path.stem + ".csv")
    home_abbr, away_abbr = (None, None)
    if roster_path.is_file():
        home_abbr, away_abbr = read_roster_teams(roster_path)

    date = parse_date_from_filename(parquet_path.stem)

    thresholds = cfg.inventory
    reasons = []
    if minutes_covered < float(thresholds.min_minutes_required):
        reasons.append(
            f"minutes_covered<{thresholds.min_minutes_required} ({minutes_covered:.1f})"
        )

    # A frame_count of zero means the source archive had no tracking moments
    # at all (metadata/roster present, but every event's moments list was
    # empty) — every per-frame aggregate below is NULL for an empty group, so
    # short-circuit instead of comparing None to a float.
    max_gap_running_ms = stats["max_gap_running_ms"]
    if stats["frame_count"] == 0:
        reasons.append("frame_count=0 (no tracking moments in source archive)")
    else:
        if stats["share_frames_missing_object"] > float(thresholds.max_missing_frame_share):
            reasons.append(
                "share_frames_missing_object>"
                f"{thresholds.max_missing_frame_share} ({stats['share_frames_missing_object']:.4f})"
            )
        gap_limit_ms = float(thresholds.max_gap_running_s) * 1000.0
        if max_gap_running_ms is not None and max_gap_running_ms > gap_limit_ms:
            reasons.append(f"max_gap_running_ms>{gap_limit_ms:.0f} ({max_gap_running_ms:.0f})")

    return {
        "game_id": game_id,
        "date": date or "",
        "home_team": home_abbr or "",
        "away_team": away_abbr or "",
        "frame_count": stats["frame_count"],
        "minutes_covered": round(minutes_covered, 2),
        "share_frames_11_rows": _round_or_blank(stats["share_frames_11_rows"]),
        "share_frames_ball_missing": _round_or_blank(stats["share_frames_ball_missing"]),
        "quarters_present": stats["quarters_present"] or "",
        "max_gap_ms": stats["max_gap_ms"] if stats["max_gap_ms"] is not None else "",
        "max_gap_running_ms": max_gap_running_ms if max_gap_running_ms is not None else "",
        "flagged": bool(reasons),
        "flag_reasons": "; ".join(reasons),
    }


# max_gap_running_ms trips on nearly every game (see PROJECT.md: clip
# boundaries between events routinely land mid-possession), so it's noise in
# a summary meant to surface real defects. A game whose only flag reason is
# this one is excluded from the report below; games also flagged for another
# reason still show it alongside that reason.
GAP_REASON_LABEL = "max_gap_running_ms"


def _flag_labels(flag_reasons: str) -> list[str]:
    """flag_reasons -> the condition name that tripped each reason."""
    if not flag_reasons:
        return []
    return [
        re.split(r"[<>=]", reason, maxsplit=1)[0] for reason in flag_reasons.split("; ")
    ]


def write_summary(
    rows: list[dict], failures: list[tuple[str, str]], report_path: Path
) -> None:
    n_games = len(rows)
    notable = [
        r for r in rows if any(l != GAP_REASON_LABEL for l in _flag_labels(r["flag_reasons"]))
    ]
    n_notable = len(notable)
    n_gap_only = sum(1 for r in rows if r["flagged"]) - n_notable

    minutes = [r["minutes_covered"] for r in rows]
    avg_minutes = sum(minutes) / n_games if n_games else float("nan")

    reason_counts: Counter[str] = Counter()
    for r in notable:
        for label in _flag_labels(r["flag_reasons"]):
            if label != GAP_REASON_LABEL:
                reason_counts[label] += 1

    lines = [
        "# Tracking inventory, 2015-16",
        "",
        f"- **Games:** {n_games:,}",
        f"- **Flagged for a reason other than {GAP_REASON_LABEL}:** {n_notable:,}",
        f"- **Flagged only for {GAP_REASON_LABEL} (not shown below, see "
        "PROJECT.md):** "
        f"{n_gap_only:,}",
        f"- **Failed to process:** {len(failures):,}",
        f"- **Average minutes covered per game:** {avg_minutes:.1f}",
        "",
        "## Flag reasons (excluding the near-universal max_gap_running_ms)",
        "",
        "| reason | games |",
        "|---|---|",
    ]
    for label, count in reason_counts.most_common():
        lines.append(f"| {label} | {count:,} |")

    if notable:
        lines += [
            "",
            "## Flagged games",
            "",
            "| game_id | date | minutes_covered | flag_reasons |",
            "|---|---|---|---|",
        ]
        for r in notable:
            lines.append(
                f"| {r['game_id']} | {r['date']} | {r['minutes_covered']} | "
                f"{r['flag_reasons']} |"
            )

    if failures:
        lines += [
            "",
            "## Files that failed to process",
            "",
            "| file | error |",
            "|---|---|",
        ]
        for name, err in failures:
            lines.append(f"| {name} | {err} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cfg = get_config()
    tracking_dir = cfg.get_path("tracking")
    roster_dir = cfg.get_path("rosters")
    reports_dir = cfg.get_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "tracking_inventory.csv"

    parquet_files = sorted(tracking_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"no Parquet files found in {tracking_dir}")
        return 1

    con = duckdb.connect()
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for path in parquet_files:
        try:
            rows.append(inventory_row(path, roster_dir, cfg, con))
        except Exception as exc:  # keep going on a bad file
            failures.append((path.name, str(exc)))

    con.close()

    rows.sort(key=lambda r: (r["date"], r["game_id"]))

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = reports_dir / "tracking_inventory_summary.md"
    write_summary(rows, failures, summary_path)

    n_flagged = sum(1 for r in rows if r["flagged"])
    print(f"wrote {out_path} ({len(rows)} rows)")
    print(f"wrote {summary_path}")
    print(
        f"summary: games={len(parquet_files)} written={len(rows)} "
        f"flagged={n_flagged} failed={len(failures)}"
    )
    if failures:
        print("failures:")
        for name, err in failures:
            print(f"  {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
