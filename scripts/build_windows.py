"""Cut a fixed-end tracking window leading up to each detected make.

    .venv/bin/python scripts/build_windows.py

For every row in data/derived/makes.parquet, builds a 25 Hz position track
(ball + both teams' players) ending shortly after the make's detected rim
crossing (data/derived/make_times.parquet) and starting shortly after the
previous play-by-play event, corrected for clock offset. See
src/nba_assists/windows.py and config.yaml's windows: section.

Writes:
  * data/windows/<game_id>.npz, one per game with at least one included
    make: pos [N, T_max, A, 3] (float16; u, v, z -- every play attacks the
    left basket), mask [N, T_max, A], roles [N, A, 3] (is_ball, is_offense,
    is_shooter), length [N] (frames actually present, not padding),
    make_id [N] (string, for joining back to the index), and shot_clock
    [N, T_max] if windows.include_shot_clock.
  * data/windows/index.parquet -- one row per make in makes.parquet
    (included or not): make_id, game_id, eventnum, label (assisted),
    included, reason, start_ts_ms, end_ts_ms, offset_used, length_s,
    length_frames, prev_event_type, many_missing_frames,
    previous_event_problem.
  * reports/windows.md -- counts by exclusion reason, and the assisted
    share among included vs. excluded makes, overall and by team.
"""

# pandas' stubs are known to be imprecise for this kind of code (see
# src/nba_assists/rim.py for the full explanation); verified against
# tests/test_windows.py and by running this script. Full type checking
# stays on elsewhere in the project -- this is scoped to this file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportReturnType=false

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_assists.config import get_config, parquet_compression  # noqa: E402
from nba_assists.direction import frame_table, player_positions  # noqa: E402
from nba_assists.windows import build_agent_tracks, build_window, resolve_window_start, rotate_and_normalize  # noqa: E402

INDEX_COLUMNS = [
    "make_id", "game_id", "eventnum", "label", "included", "reason",
    "start_ts_ms", "end_ts_ms", "offset_used", "length_s", "length_frames",
    "prev_event_type", "many_missing_frames", "previous_event_problem",
]
EXCLUSION_REASONS = ["not detected", "basket mismatch", "unclear direction", "too short", "coverage gap"]


def game_id_to_path(tracking_dir: Path) -> dict[str, Path]:
    """Canonical game_id -> its tracking Parquet path."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT game_id, filename FROM read_parquet('{tracking_dir}/*.parquet', filename=true)"
    ).fetchall()
    con.close()
    return {game_id: Path(filename) for game_id, filename in rows}


def event_type_names(cfg) -> dict[int, str]:
    return {int(v): k for k, v in cfg.pbp.event_types.items()}


def write_report(index: pd.DataFrame, direction_team_abbr: dict, report_path: Path) -> None:
    reason_counts = index.loc[~index["included"], "reason"].value_counts()
    n_included = int(index["included"].sum())

    lines = [
        "# Windows, 2015-16",
        "",
        "## Counts by exclusion reason",
        "",
        "| reason | makes |",
        "|---|---|",
        f"| included | {n_included:,} |",
    ]
    for reason in EXCLUSION_REASONS:
        lines.append(f"| {reason} | {int(reason_counts.get(reason, 0)):,} |")

    included = index.loc[index["included"], "label"]
    excluded = index.loc[~index["included"], "label"]
    lines += [
        "",
        "## Assisted share, included vs. excluded",
        "",
        f"- included: n={len(included):,}, assisted_share={included.mean():.2%}" if len(included) else "- included: (none)",
        f"- excluded: n={len(excluded):,}, assisted_share={excluded.mean():.2%}" if len(excluded) else "- excluded: (none)",
    ]
    if len(included) and len(excluded):
        lines.append(f"- difference (included - excluded): {included.mean() - excluded.mean():+.2%}")

    lines += [
        "",
        "## Assisted share by team (shooter's team), included vs. excluded",
        "",
        "| team | included n | included assisted_share | excluded n | excluded assisted_share | difference |",
        "|---|---|---|---|---|---|",
    ]
    index = index.copy()
    index["team_abbr"] = [
        direction_team_abbr.get((gid, tid), "") for gid, tid in zip(index["game_id"], index["shooter_team_id"])
    ]
    for team, group in index.groupby("team_abbr", sort=True):
        if not team:
            continue
        inc = group.loc[group["included"], "label"]
        exc = group.loc[~group["included"], "label"]
        inc_share = f"{inc.mean():.2%}" if len(inc) else "n/a"
        exc_share = f"{exc.mean():.2%}" if len(exc) else "n/a"
        diff = f"{(inc.mean() - exc.mean()):+.2%}" if len(inc) and len(exc) else "n/a"
        lines.append(f"| {team} | {len(inc):,} | {inc_share} | {len(exc):,} | {exc_share} | {diff} |")

    lines += [
        "",
        "A large included-vs-excluded difference (overall or for one team) means "
        "the exclusions could bias later results, not just shrink the sample.",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cfg = get_config()

    derived_dir = cfg.get_path("derived")
    tracking_dir = cfg.get_path("tracking")
    windows_dir = cfg.get_path("windows")
    reports_dir = cfg.get_path("reports")
    windows_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    w = cfg.windows
    end_buffer_s = float(w.end_buffer_s)
    start_margin_s = float(w.start_margin_s)
    after_prev_make_s = float(w.after_prev_make_s)
    offset_source = str(w.offset_source)
    offset_nudge_s = float(w.offset_nudge_s)
    max_window_s = float(w.max_window_s)
    min_window_s = float(w.min_window_s)
    min_start_coverage_s = float(w.min_start_coverage_s)
    min_start_coverage_fraction = float(w.min_start_coverage_fraction)
    include_shot_clock = bool(w.include_shot_clock)

    max_gap_running_s = float(cfg.inventory.max_gap_running_s)
    max_missing_frame_share = float(cfg.inventory.max_missing_frame_share)
    made_shot_code = int(cfg.pbp.event_types.made_shot)
    length_x = float(cfg.court.length_x)
    width_y = float(cfg.court.width_y)
    basket_left_x = float(cfg.court.baskets.left.x)
    basket_right_x = float(cfg.court.baskets.right.x)
    event_names = event_type_names(cfg)

    makes = pd.read_parquet(derived_dir / "makes.parquet")
    make_times = pd.read_parquet(derived_dir / "make_times.parquet")
    direction = pd.read_parquet(
        derived_dir / "attack_direction.parquet",
        columns=["game_id", "period", "team_id", "team_abbr", "basket_x", "clear"],
    )

    mt_lookup = {
        (r.game_id, int(r.eventnum)): r
        for r in make_times.itertuples(index=False)
    }
    direction_lookup = {
        (r.game_id, int(r.period), int(r.team_id)): (r.basket_x, bool(r.clear))
        for r in direction.itertuples(index=False)
    }
    team_abbr_lookup = {
        (r.game_id, int(r.team_id)): r.team_abbr for r in direction.itertuples(index=False)
    }

    game_median_offset: dict[str, float] = {}
    if offset_source == "game_median":
        by_game_csv = pd.read_csv(reports_dir / "clock_offset_by_game.csv", dtype={"game_id": str})
        by_game_csv["median_offset_s"] = pd.to_numeric(by_game_csv["median_offset_s"], errors="coerce")
        game_median_offset = dict(zip(by_game_csv["game_id"], by_game_csv["median_offset_s"]))

    merged = makes.merge(make_times, on=["game_id", "eventnum"], how="left")
    merged["detected"] = merged["detected"].fillna(False)

    def _basket_info(row) -> tuple[float | None, bool]:
        return direction_lookup.get((row["game_id"], int(row["period"]), int(row["shooter_team_id"])), (None, False))

    basket_info = merged.apply(_basket_info, axis=1)
    merged["attack_basket_x"] = [b[0] for b in basket_info]
    merged["attack_clear"] = [b[1] for b in basket_info]
    merged["prev_event_type"] = merged["prev_msgtype"].map(
        lambda v: event_names.get(int(v), f"unknown ({int(v)})") if pd.notna(v) else ""
    )

    reason = pd.Series("", index=merged.index, dtype=object)
    not_detected = ~merged["detected"]
    reason = reason.mask(not_detected, "not detected")

    basket_mismatch = (
        (~not_detected) & merged["attack_basket_x"].notna() & (merged["basket_x"] != merged["attack_basket_x"])
    )
    reason = reason.mask((reason == "") & basket_mismatch, "basket mismatch")

    unclear = (reason == "") & (~merged["attack_clear"])
    reason = reason.mask(unclear, "unclear direction")
    merged["_early_reason"] = reason

    game_paths = game_id_to_path(tracking_dir)
    index_rows: list[dict] = []
    failed_games: list[str] = []

    for _, make in merged.loc[merged["_early_reason"] != ""].iterrows():
        index_rows.append(_index_row(make, make["_early_reason"]))

    candidates = merged[merged["_early_reason"] == ""]
    for game_id, game_candidates in candidates.groupby("game_id", sort=False):
        path = game_paths.get(game_id)
        if path is None:
            for _, make in game_candidates.iterrows():
                index_rows.append(_index_row(make, "coverage gap"))
            continue
        try:
            needed_periods = sorted(int(p) for p in game_candidates["period"].unique())
            tracking = pd.read_parquet(
                path, columns=["ts_ms", "quarter", "game_clock", "shot_clock", "team_id", "player_id", "x", "y", "z"]
            )
            tracking = tracking[tracking["quarter"].isin(needed_periods)]
            quarter_data: dict[int, dict] = {}
            for period, qdf in tracking.groupby("quarter", sort=False):
                ftab = frame_table(qdf, cfg)
                ftab["clock_running"] = ftab.groupby("period")["game_clock"].diff() < 0
                pp = player_positions(qdf, cfg)
                ball_xyz = ftab[["ts_ms", "ball_x", "ball_y", "ball_z"]].rename(
                    columns={"ball_x": "x", "ball_y": "y", "ball_z": "z"}
                )
                quarter_data[int(period)] = {
                    "frame_table": ftab,
                    "player_positions": pp,
                    "ball_xyz": ball_xyz,
                    "period_max_game_clock": float(ftab["game_clock"].max()),
                }
        except Exception as exc:  # keep going on a bad file
            failed_games.append(game_id)
            print(f"  {path}: {exc}")
            for _, make in game_candidates.iterrows():
                index_rows.append(_index_row(make, "coverage gap"))
            continue

        game_windows: list[dict] = []
        for _, make in game_candidates.iterrows():
            period = int(make["period"])
            qd = quarter_data.get(period)
            if qd is None:
                index_rows.append(_index_row(make, "coverage gap"))
                continue

            prev_eventnum = make["prev_eventnum"]
            prev_make = mt_lookup.get((make["game_id"], int(prev_eventnum))) if pd.notna(prev_eventnum) else None
            offset_for_prev = (
                game_median_offset.get(make["game_id"]) if offset_source == "game_median" else float(make["offset_s"])
            )
            if offset_for_prev is None or pd.isna(offset_for_prev):
                offset_for_prev = 0.0

            start_info = resolve_window_start(
                prev_same_period=bool(make["prev_same_period"]),
                prev_msgtype=make["prev_msgtype"],
                prev_clock_s=make["prev_clock_s"],
                made_shot_code=made_shot_code,
                prev_make_detected=bool(prev_make.detected) if prev_make is not None else None,
                prev_make_cross_game_clock=float(prev_make.cross_game_clock)
                if prev_make is not None and pd.notna(prev_make.cross_game_clock)
                else None,
                offset_s=offset_for_prev,
                period_max_game_clock=qd["period_max_game_clock"],
                start_margin_s=start_margin_s,
                after_prev_make_s=after_prev_make_s,
                offset_nudge_s=offset_nudge_s,
            )

            end_ts_ms = float(make["cross_ts_ms"]) + end_buffer_s * 1000.0
            intended_length_s = start_info["start_clock"] - float(make["cross_game_clock"])
            window = build_window(
                qd["frame_table"], start_info["start_clock"], end_ts_ms, intended_length_s,
                max_window_s, max_gap_running_s, max_missing_frame_share,
                min_start_coverage_s, min_start_coverage_fraction, int(cfg.tracking.players_on_court),
            )

            if window["length_s"] < min_window_s:
                index_rows.append(_index_row(make, "too short", start_info, window))
                continue
            if window["coverage_gap"]:
                index_rows.append(_index_row(make, "coverage gap", start_info, window))
                continue

            kept_ts = window["kept_ts"]
            tracks = build_agent_tracks(
                qd["player_positions"], qd["ball_xyz"], kept_ts,
                int(make["shooter_team_id"]), int(make["shooter_id"]),
            )
            attack_right = make["attack_basket_x"] == basket_right_x
            u, v = rotate_and_normalize(
                tracks["pos"][:, :, 0], tracks["pos"][:, :, 1], attack_right, length_x, width_y, basket_left_x
            )
            tracks["pos"][:, :, 0] = u
            tracks["pos"][:, :, 1] = v

            window_entry = {
                "make_id": make["make_id"],
                "pos": tracks["pos"].astype(np.float16),
                "mask": tracks["mask"],
                "roles": tracks["roles"],
            }
            if include_shot_clock:
                window_entry["shot_clock"] = (
                    qd["frame_table"].set_index("ts_ms")["shot_clock"].reindex(kept_ts).to_numpy(dtype=np.float32)
                )
            game_windows.append(window_entry)
            index_rows.append(_index_row(make, "", start_info, window, included=True))

        if game_windows:
            _write_game_npz(windows_dir / f"{game_id}.npz", game_windows, include_shot_clock)

    index = pd.DataFrame(index_rows, columns=INDEX_COLUMNS + ["shooter_team_id"])
    index_path = windows_dir / "index.parquet"
    index[INDEX_COLUMNS].to_parquet(index_path, compression=parquet_compression(cfg), index=False)

    report_path = reports_dir / "windows.md"
    write_report(index, team_abbr_lookup, report_path)

    n_included = int(index["included"].sum())
    print(f"wrote {index_path} ({len(index):,} rows)")
    print(f"wrote {report_path}")
    print(f"summary: makes={len(index):,} included={n_included:,} failed_games={len(failed_games)}")
    return 1 if failed_games else 0


def _index_row(make, reason: str, start_info: dict | None = None, window: dict | None = None, included: bool = False) -> dict:
    return {
        "make_id": make["make_id"],
        "game_id": make["game_id"],
        "eventnum": int(make["eventnum"]),
        "label": bool(make["assisted"]),
        "included": included,
        "reason": reason,
        "start_ts_ms": window["start_ts_ms"] if window else None,
        "end_ts_ms": window["end_ts_ms"] if window else None,
        "offset_used": start_info["offset_used"] if start_info else None,
        "length_s": window["length_s"] if window else None,
        "length_frames": window["length_frames"] if window else None,
        "prev_event_type": make["prev_event_type"],
        "many_missing_frames": window["many_missing_frames"] if window else None,
        "previous_event_problem": start_info["previous_event_problem"] if start_info else None,
        "shooter_team_id": int(make["shooter_team_id"]),
    }


def _write_game_npz(path: Path, game_windows: list[dict], include_shot_clock: bool) -> None:
    n = len(game_windows)
    t_max = max(w["pos"].shape[0] for w in game_windows)
    a_max = max(w["pos"].shape[1] for w in game_windows)

    pos = np.zeros((n, t_max, a_max, 3), dtype=np.float16)
    mask = np.zeros((n, t_max, a_max), dtype=bool)
    roles = np.zeros((n, a_max, 3), dtype=bool)
    length = np.zeros((n,), dtype=np.int32)
    make_ids = np.array([w["make_id"] for w in game_windows])
    shot_clock = np.zeros((n, t_max), dtype=np.float32) if include_shot_clock else None

    for i, w in enumerate(game_windows):
        t, a = w["pos"].shape[0], w["pos"].shape[1]
        pos[i, :t, :a] = w["pos"]
        mask[i, :t, :a] = w["mask"]
        roles[i, :a] = w["roles"]
        length[i] = t
        if include_shot_clock:
            assert shot_clock is not None
            shot_clock[i, :t] = w["shot_clock"]

    extra = {"shot_clock": shot_clock} if include_shot_clock else {}
    np.savez_compressed(path, pos=pos, mask=mask, roles=roles, length=length, make_id=make_ids, **extra)


if __name__ == "__main__":
    raise SystemExit(main())
