"""Trace who passed the shooter the ball, for every included window.

    .venv/bin/python scripts/attribute_passers.py

For every included row of data/windows/index.parquet, works on that make's
real tracking frames (ts_ms, team_id, player_id, x, y, z -- not the
anonymized model-input tensors in data/windows/*.npz) from the window's own
start through the make's detected rim crossing. See src/nba_assists/passer.py
and config.yaml's passer: section for the method.

Tunes hold_dist, hold_height, min_run_s and dribble_gap_s by grid search,
maximizing agreement between the traced passer and the known assister on a
sample of assisted makes from a seeded split of training games, then
reports agreement and the false-NONE rate on the held-out games. The chosen
combination is then applied to every included make (assisted or not) for
the final attribution.

Writes:
  * data/derived/passers.parquet -- one row per included make: make_id,
    game_id, eventnum, shooter_id, shooter_team_id, assister_id, assisted,
    traced_passer_id, trace_problem, passer_id, is_train_game.
  * reports/passer_trace.md -- the chosen hyperparameters, held-out
    agreement and false-NONE rate, overall trace_problem/NONE rates, and 20
    example disagreements for manual review.
"""

# pandas' stubs are known to be imprecise for this kind of code (see
# src/nba_assists/rim.py for the full explanation); verified against
# tests/test_passer.py and by running this script. Full type checking
# stays on elsewhere in the project -- this is scoped to this file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportReturnType=false

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_assists.config import get_config, parquet_compression  # noqa: E402
from nba_assists.direction import frame_table, player_positions  # noqa: E402
from nba_assists.passer import attribute_passer, backward_trace, classify_held, possession_geometry, possession_runs  # noqa: E402

PASSER_COLUMNS = [
    "make_id", "game_id", "eventnum", "shooter_id", "shooter_team_id",
    "assister_id", "assisted", "traced_passer_id", "trace_problem",
    "passer_id", "is_train_game",
]


def game_id_to_path(tracking_dir: Path) -> dict[str, Path]:
    """Canonical game_id -> its tracking Parquet path."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT game_id, filename FROM read_parquet('{tracking_dir}/*.parquet', filename=true)"
    ).fetchall()
    con.close()
    return {game_id: Path(filename) for game_id, filename in rows}


def build_geometry(cfg, candidates: pd.DataFrame) -> dict[str, dict]:
    """make_id -> {"geometry", "shooter_id", "window_start_ts"} for every
    row of ``candidates`` (one per included make: make_id, game_id,
    eventnum, period, shooter_id, shooter_team_id, start_ts_ms,
    cross_ts_ms). The expensive, hold_dist/hold_height-independent part --
    computed once and reused across the whole grid search.
    """
    tracking_dir = cfg.get_path("tracking")
    game_paths = game_id_to_path(tracking_dir)

    out: dict[str, dict] = {}
    n_no_frames = 0
    n_no_path = 0

    for game_id, game_candidates in candidates.groupby("game_id", sort=False):
        path = game_paths.get(game_id)
        if path is None:
            n_no_path += len(game_candidates)
            continue

        tracking = pd.read_parquet(
            path, columns=["ts_ms", "quarter", "game_clock", "shot_clock", "team_id", "player_id", "x", "y", "z"]
        )
        needed_periods = sorted(int(p) for p in game_candidates["period"].unique())
        tracking = tracking[tracking["quarter"].isin(needed_periods)]

        quarter_data: dict[int, dict] = {}
        for period, qdf in tracking.groupby("quarter", sort=False):
            ftab = frame_table(qdf, cfg)
            pp = player_positions(qdf, cfg)
            ball_xyz = ftab[["ts_ms", "ball_x", "ball_y", "ball_z"]].rename(
                columns={"ball_x": "x", "ball_y": "y", "ball_z": "z"}
            )
            quarter_data[int(period)] = {"frame_table": ftab, "player_positions": pp, "ball_xyz": ball_xyz}

        for _, make in game_candidates.iterrows():
            period = int(make["period"])
            qd = quarter_data.get(period)
            if qd is None:
                n_no_frames += 1
                continue

            # every raw frame in this range, not just clock-running ones --
            # Step 4 already validated this is a continuous, gap-free span
            # for this make (see windows.py's own internal_gap check, which
            # looks at raw frame-to-frame timing regardless of
            # clock_running); clock_running itself can lag real action by
            # several seconds right after a stoppage (the game clock's
            # reading, not a tracking gap), so filtering by it here would
            # throw away real possession/pass frames.
            ftab = qd["frame_table"]
            in_range = ftab.loc[
                (ftab["ts_ms"] >= float(make["start_ts_ms"])) & (ftab["ts_ms"] <= float(make["cross_ts_ms"]))
            ]
            kept_ts = np.sort(in_range["ts_ms"].unique())
            if len(kept_ts) == 0:
                n_no_frames += 1
                continue

            geometry = possession_geometry(qd["player_positions"], qd["ball_xyz"], int(make["shooter_team_id"]), kept_ts)
            out[make["make_id"]] = {
                "geometry": geometry,
                "shooter_id": int(make["shooter_id"]),
                "window_start_ts": float(kept_ts[0]),
            }

    print(
        f"  geometry built for {len(out):,} makes "
        f"({n_no_frames:,} with no live frames before the crossing, {n_no_path:,} with no tracking file for the game)"
    )
    return out


def _trace_one(bundle: dict, hold_dist: float, hold_height: float, min_run_s: float, dribble_gap_s: float) -> dict:
    """classify_held -> possession_runs -> backward_trace for one make, one
    hyperparameter combination. ``bundle["geometry"]`` is already scoped to
    this make's own window (its start through the rim crossing)."""
    geometry = bundle["geometry"]
    held = classify_held(geometry, hold_dist, hold_height)
    runs = possession_runs(geometry["ts"], held, min_run_s, dribble_gap_s)
    return backward_trace(runs, bundle["shooter_id"], bundle["window_start_ts"])


def _agreement_stats(
    make_ids: list[str], geometry_by_make: dict[str, dict], assister_by_make: dict[str, float],
    hold_dist: float, hold_height: float, min_run_s: float, dribble_gap_s: float,
) -> dict:
    """Agreement with the known assister and the false-NONE rate, over the
    given (assisted) make_ids, at one hyperparameter combination."""
    n = len(make_ids)
    n_agree = 0
    n_none = 0
    for make_id in make_ids:
        traced = _trace_one(geometry_by_make[make_id], hold_dist, hold_height, min_run_s, dribble_gap_s)["traced_passer_id"]
        if traced is None:
            n_none += 1
        elif int(traced) == int(assister_by_make[make_id]):
            n_agree += 1
    return {
        "n": n,
        "agreement": n_agree / n if n else float("nan"),
        "false_none_rate": n_none / n if n else float("nan"),
    }


def write_report(
    report_path: Path, best: dict, train_sample_n: int, holdout_stats: dict,
    overall_n: int, overall_none_rate: float, overall_trace_problem_rate: float,
    n_train_games: int, n_holdout_games: int, examples: pd.DataFrame,
) -> None:
    lines = [
        "# Pass attribution, 2015-16",
        "",
        "## Chosen parameters (grid search, maximizing agreement with the known",
        "assister on a training sample of assisted makes; ties broken toward the",
        "lower false-NONE rate -- see config.yaml's passer: section)",
        "",
        f"- hold_dist: {best['hold_dist']:.2f} ft",
        f"- hold_height: {best['hold_height']:.2f} ft",
        f"- min_run_s: {best['min_run_s']:.2f} s",
        f"- dribble_gap_s: {best['dribble_gap_s']:.2f} s",
        "",
        "## Tuning",
        "",
        f"- training games: {n_train_games:,}; held-out games: {n_holdout_games:,}",
        f"- training sample (assisted makes): {train_sample_n:,}",
        f"- training agreement at the chosen combination: {best['train_stats']['agreement']:.2%}"
        f" (false-NONE rate {best['train_stats']['false_none_rate']:.2%})",
        "",
        "## Held-out accuracy",
        "",
        f"- assisted makes evaluated: {holdout_stats['n']:,}",
        f"- **agreement with the known assister: {holdout_stats['agreement']:.2%}**",
        f"- **false-NONE rate (assisted makes traced as NONE): {holdout_stats['false_none_rate']:.2%}**",
        "",
        "## Overall (every included make, assisted or not)",
        "",
        f"- makes with a traced passer attempt: {overall_n:,}",
        f"- share traced as NONE: {overall_none_rate:.2%}",
        f"- share where the run active at the shot wasn't the recorded shooter's"
        f" (trace_problem): {overall_trace_problem_rate:.2%}",
        "",
        "## Example disagreements (assisted makes; traced passer vs. the known assister)",
        "",
        "| game_id | eventnum | description | assister_id | traced_passer_id | trace_problem | set |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in examples.iterrows():
        traced = "" if pd.isna(row["traced_passer_id"]) else int(row["traced_passer_id"])
        lines.append(
            f"| {row['game_id']} | {row['eventnum']} | {row['description']} | {int(row['assister_id'])} | "
            f"{traced} | {row['trace_problem']} | {'train' if row['is_train_game'] else 'holdout'} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cfg = get_config()
    rng = np.random.default_rng(int(cfg.seeds.numpy))

    derived_dir = cfg.get_path("derived")
    windows_dir = cfg.get_path("windows")
    reports_dir = cfg.get_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    p = cfg.passer
    hold_dist_grid = [float(v) for v in p.hold_dist_grid]
    hold_height_grid = [float(v) for v in p.hold_height_grid]
    min_run_s_grid = [float(v) for v in p.min_run_s_grid]
    dribble_gap_s_grid = [float(v) for v in p.dribble_gap_s_grid]
    train_game_share = float(p.train_game_share)
    max_tuning_makes = int(p.max_tuning_makes)
    n_examples = int(p.n_examples)

    index = pd.read_parquet(windows_dir / "index.parquet")
    included = index.loc[index["included"], ["make_id", "game_id", "eventnum", "start_ts_ms"]]

    makes = pd.read_parquet(
        derived_dir / "makes.parquet",
        columns=["game_id", "eventnum", "period", "shooter_id", "shooter_team_id", "assister_id", "assisted", "description"],
    )
    make_times = pd.read_parquet(derived_dir / "make_times.parquet", columns=["game_id", "eventnum", "cross_ts_ms"])

    candidates = included.merge(makes, on=["game_id", "eventnum"], how="left").merge(
        make_times, on=["game_id", "eventnum"], how="left"
    )

    print(f"building possession geometry for {len(candidates):,} included makes...")
    geometry_by_make = build_geometry(cfg, candidates)

    meta = candidates.set_index("make_id").to_dict("index")

    game_ids = np.array(sorted(candidates["game_id"].unique()))
    rng.shuffle(game_ids)
    n_train_games = int(round(len(game_ids) * train_game_share))
    train_games = set(game_ids[:n_train_games])
    n_holdout_games = len(game_ids) - n_train_games

    assisted_make_ids = [
        mid for mid in geometry_by_make if bool(meta[mid]["assisted"]) and pd.notna(meta[mid]["assister_id"])
    ]
    train_assisted = [mid for mid in assisted_make_ids if meta[mid]["game_id"] in train_games]
    holdout_assisted = [mid for mid in assisted_make_ids if meta[mid]["game_id"] not in train_games]

    tuning_sample = train_assisted
    if len(tuning_sample) > max_tuning_makes:
        tuning_sample = list(rng.choice(np.array(tuning_sample, dtype=object), size=max_tuning_makes, replace=False))
    assister_by_make = {mid: float(meta[mid]["assister_id"]) for mid in geometry_by_make if pd.notna(meta[mid]["assister_id"])}

    combos = list(itertools.product(hold_dist_grid, hold_height_grid, min_run_s_grid, dribble_gap_s_grid))
    print(f"grid search: {len(combos):,} combinations over {len(tuning_sample):,} training makes...")

    best: dict | None = None
    for hold_dist, hold_height, min_run_s, dribble_gap_s in combos:
        stats = _agreement_stats(
            tuning_sample, geometry_by_make, assister_by_make, hold_dist, hold_height, min_run_s, dribble_gap_s
        )
        key = (stats["agreement"], -stats["false_none_rate"])
        if best is None or key > best["key"]:
            best = {
                "key": key, "hold_dist": hold_dist, "hold_height": hold_height,
                "min_run_s": min_run_s, "dribble_gap_s": dribble_gap_s, "train_stats": stats,
            }
    assert best is not None
    print(
        f"best: hold_dist={best['hold_dist']} hold_height={best['hold_height']} "
        f"min_run_s={best['min_run_s']} dribble_gap_s={best['dribble_gap_s']} "
        f"(training agreement {best['train_stats']['agreement']:.2%})"
    )

    holdout_stats = _agreement_stats(
        holdout_assisted, geometry_by_make, assister_by_make,
        best["hold_dist"], best["hold_height"], best["min_run_s"], best["dribble_gap_s"],
    )
    print(f"held-out agreement: {holdout_stats['agreement']:.2%} (false-NONE rate {holdout_stats['false_none_rate']:.2%})")

    rows: list[dict] = []
    disagreement_rows: list[dict] = []
    for make_id, bundle in geometry_by_make.items():
        row_meta = meta[make_id]
        result = _trace_one(bundle, best["hold_dist"], best["hold_height"], best["min_run_s"], best["dribble_gap_s"])
        assisted = bool(row_meta["assisted"])
        assister_id = row_meta["assister_id"]
        is_train = row_meta["game_id"] in train_games
        passer_id = attribute_passer(assister_id, assisted, result["traced_passer_id"])

        rows.append(
            {
                "make_id": make_id,
                "game_id": row_meta["game_id"],
                "eventnum": int(row_meta["eventnum"]),
                "shooter_id": int(row_meta["shooter_id"]),
                "shooter_team_id": int(row_meta["shooter_team_id"]),
                "assister_id": int(assister_id) if pd.notna(assister_id) else None,
                "assisted": assisted,
                "traced_passer_id": result["traced_passer_id"],
                "trace_problem": result["trace_problem"],
                "passer_id": passer_id,
                "is_train_game": is_train,
            }
        )

        if assisted and pd.notna(assister_id) and (
            result["traced_passer_id"] is None or int(result["traced_passer_id"]) != int(assister_id)
        ):
            disagreement_rows.append(
                {
                    "game_id": row_meta["game_id"],
                    "eventnum": int(row_meta["eventnum"]),
                    "description": row_meta["description"],
                    "assister_id": int(assister_id),
                    "traced_passer_id": result["traced_passer_id"],
                    "trace_problem": result["trace_problem"],
                    "is_train_game": is_train,
                }
            )

    passers = pd.DataFrame(rows, columns=PASSER_COLUMNS)
    passers_path = derived_dir / "passers.parquet"
    passers.to_parquet(passers_path, compression=parquet_compression(cfg), index=False)

    overall_n = len(passers)
    overall_none_rate = float((passers["traced_passer_id"].isna()).mean()) if overall_n else float("nan")
    overall_trace_problem_rate = float(passers["trace_problem"].mean()) if overall_n else float("nan")

    disagreements = pd.DataFrame(disagreement_rows)
    examples = (
        disagreements.sample(n=min(n_examples, len(disagreements)), random_state=int(cfg.seeds.numpy))
        if len(disagreements)
        else disagreements
    )

    report_path = reports_dir / "passer_trace.md"
    write_report(
        report_path, best, len(tuning_sample), holdout_stats,
        overall_n, overall_none_rate, overall_trace_problem_rate,
        n_train_games, n_holdout_games, examples,
    )

    print(f"wrote {passers_path} ({len(passers):,} rows)")
    print(f"wrote {report_path}")
    print(
        f"summary: makes={overall_n:,} holdout_agreement={holdout_stats['agreement']:.4f} "
        f"holdout_false_none_rate={holdout_stats['false_none_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
