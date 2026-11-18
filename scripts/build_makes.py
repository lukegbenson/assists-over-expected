"""Build data/derived/makes.parquet and reports/makes_summary.md.

    .venv/bin/python scripts/build_makes.py

Reads data/pbp/pbp_2015_16.parquet (already sorted per game by PERIOD, then
clock_s descending, then EVENTNUM — see scripts/prepare_pbp.py), computes the
previous play-by-play event *by that time order* for every row before
filtering, then keeps EVENTMSGTYPE == made_shot (config: pbp.event_types).

makes.parquet columns:
    game_id, eventnum, period, clock_s,
    shooter_id, shooter_team_id, assister_id, assisted, action_type,
    description, is_home_team,
    prev_eventnum, prev_msgtype, prev_clock_s, prev_description, prev_same_period

home_team_id and each game's roster (for the "shooter not on roster" check)
come from data/tracking/_rosters/*.csv, keyed by their own game_id column —
roster filenames are the source archive's stem, not the game id.
"""

# pandas' stubs are known to be imprecise for exactly this kind of multi-step
# Series/DataFrame code (groupby().shift(), .where(), .map(), boolean compares
# against a nullable Series, int(...) on a cell value, etc.): they report a
# union of every dtype pandas could theoretically produce, which a type
# checker then can't narrow back down, rather than any error in the code
# itself. Verified by running this script and diffing its output
# byte-for-byte before and after. Full type checking stays on elsewhere in
# the project — this is scoped to this file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportReturnType=false

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_assists.config import get_config, parquet_compression  # noqa: E402


def load_rosters(roster_dir: Path) -> dict[str, dict]:
    """game_id -> {"home_team_id": int, "player_ids": set[int]}."""
    rosters: dict[str, dict] = {}
    for path in sorted(roster_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        game_id = rows[0]["game_id"]
        home_team_id = next(
            (int(r["team_id"]) for r in rows if r["side"] == "home"), None
        )
        player_ids = {int(r["player_id"]) for r in rows}
        rosters[game_id] = {"home_team_id": home_team_id, "player_ids": player_ids}
    return rosters


def build_makes(df: pd.DataFrame, rosters: dict[str, dict], cfg) -> pd.DataFrame:
    no_player = int(cfg.pbp.no_player_sentinel)
    made_shot_type = int(cfg.pbp.event_types.made_shot)

    # Whichever of home/visitor/neutral description is populated for that row.
    # bfill (not chained fillna) avoids a dtype clash: NEUTRALDESCRIPTION is
    # entirely empty in this dataset, so pandas infers it as float64 rather
    # than string.
    desc_cols = ["HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]
    df["description"] = df[desc_cols].bfill(axis=1).iloc[:, 0]

    # Previous event by time, computed on the full per-game event stream
    # (every type, not just makes) before any row filtering.
    by_game = df.groupby("GAME_ID", sort=False)
    df["prev_eventnum"] = by_game["EVENTNUM"].shift(1)
    df["prev_msgtype"] = by_game["EVENTMSGTYPE"].shift(1)
    df["prev_clock_s"] = by_game["clock_s"].shift(1)
    df["prev_description"] = by_game["description"].shift(1)
    prev_period = by_game["PERIOD"].shift(1)
    df["prev_same_period"] = prev_period == df["PERIOD"]

    makes = df[df["EVENTMSGTYPE"] == made_shot_type].copy()

    assister_id = makes["PLAYER2_ID"]
    assister_id = assister_id.where(assister_id != no_player)

    home_team_id = makes["GAME_ID"].map(
        lambda gid: rosters.get(gid, {}).get("home_team_id")
    )

    out = pd.DataFrame(
        {
            "game_id": makes["GAME_ID"],
            "eventnum": makes["EVENTNUM"],
            "period": makes["PERIOD"],
            "clock_s": makes["clock_s"],
            "shooter_id": makes["PLAYER1_ID"],
            "shooter_team_id": makes["PLAYER1_TEAM_ID"],
            "assister_id": assister_id,
            "assisted": assister_id.notna(),
            "action_type": makes["EVENTMSGACTIONTYPE"],
            "description": makes["description"],
            "is_home_team": makes["PLAYER1_TEAM_ID"] == home_team_id,
            "prev_eventnum": makes["prev_eventnum"],
            "prev_msgtype": makes["prev_msgtype"],
            "prev_clock_s": makes["prev_clock_s"],
            "prev_description": makes["prev_description"],
            "prev_same_period": makes["prev_same_period"],
        }
    ).reset_index(drop=True)

    def on_roster(row) -> bool:
        roster = rosters.get(row["game_id"])
        if roster is None or pd.isna(row["shooter_id"]):
            return False
        return int(row["shooter_id"]) in roster["player_ids"]

    out["_shooter_on_roster"] = out.apply(on_roster, axis=1)
    return out


def event_type_names(cfg) -> dict[int, str]:
    return {int(v): k for k, v in cfg.pbp.event_types.items()}


def write_summary(out: pd.DataFrame, cfg, report_path: Path) -> None:
    names = event_type_names(cfg)
    n_makes = len(out)
    n_assisted = int(out["assisted"].sum())
    share_assisted = n_assisted / n_makes if n_makes else float("nan")

    qa = cfg.makes_qa
    lo, hi = float(qa.expected_assisted_share_min), float(qa.expected_assisted_share_max)
    in_range = lo <= share_assisted <= hi

    prev_counts = (
        out["prev_msgtype"]
        .map(lambda v: names.get(int(v), f"unknown ({int(v)})") if pd.notna(v) else "none (first event of game)")
        .value_counts()
    )

    off_roster = out[~out["_shooter_on_roster"]]

    lines = [
        "# Made field goals, 2015-16 (tracked games only)",
        "",
        f"- **Makes:** {n_makes:,}",
        f"- **Assisted:** {n_assisted:,} ({share_assisted:.2%})",
        f"  - expected range: {lo:.0%}-{hi:.0%} — "
        + ("within range" if in_range else "**OUT OF RANGE, investigate**"),
        f"- **Shooter not found in that game's roster:** {len(off_roster):,}",
        "",
        "## Previous event, by type",
        "",
        "| previous event type | count | share |",
        "|---|---|---|",
    ]
    for label, count in prev_counts.items():
        lines.append(f"| {label} | {count:,} | {count / n_makes:.2%} |")

    if len(off_roster):
        lines += [
            "",
            "## Makes with shooter not on that game's roster",
            "",
            "| game_id | eventnum | shooter_id | description |",
            "|---|---|---|---|",
        ]
        for _, row in off_roster.head(50).iterrows():
            lines.append(
                f"| {row['game_id']} | {row['eventnum']} | {row['shooter_id']} | "
                f"{row['description']} |"
            )
        if len(off_roster) > 50:
            lines.append(f"\n_... and {len(off_roster) - 50} more._")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cfg = get_config()
    pbp_dir = cfg.get_path("pbp")
    roster_dir = cfg.get_path("rosters")
    derived_dir = cfg.get_path("derived")
    reports_dir = cfg.get_path("reports")
    derived_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(pbp_dir / "pbp_2015_16.parquet")
    rosters = load_rosters(roster_dir)

    out = build_makes(df, rosters, cfg)
    n_off_roster = int((~out["_shooter_on_roster"]).sum())

    report_path = reports_dir / "makes_summary.md"
    write_summary(out, cfg, report_path)

    out_path = derived_dir / "makes.parquet"
    out.drop(columns=["_shooter_on_roster"]).to_parquet(
        out_path, compression=parquet_compression(cfg), index=False
    )

    n_assisted = int(out["assisted"].sum())
    share_assisted = n_assisted / len(out) if len(out) else float("nan")
    print(f"wrote {out_path} ({len(out):,} rows)")
    print(f"wrote {report_path}")
    print(
        f"summary: makes={len(out):,} assisted={n_assisted:,} "
        f"share_assisted={share_assisted:.4f} shooter_off_roster={n_off_roster}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
