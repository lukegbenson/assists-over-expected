"""Attack-direction detection: which basket each team shoots at, per period.

Uses only tracking Parquet rows (no play-by-play): a ball-holder vote. See
PROJECT.md, "Attack direction", and config.yaml's attack_direction: section.
"""

# pandas' stubs are known to be imprecise for this kind of code (groupby/merge
# chains, boolean masks, idxmin/idxmax, .between()): they report a union of
# every dtype pandas could theoretically produce, which the type checker then
# can't narrow back down, rather than any error in the code itself. Verified
# against tests/test_direction.py and by running scripts/attack_direction.py.
# Full type checking stays on elsewhere in the project -- this is scoped to
# this file.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_assists.config import Config

VOTE_COLUMNS = ["ts_ms", "period", "voter_team_id", "basket_x", "agree"]
TALLY_COLUMNS = ["period", "team_id", "basket_x", "n_votes", "vote_share", "clear"]


def frame_table(tracking: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per distinct ts_ms: period, clocks, object counts, ball position."""
    ball_team_id = int(cfg.sentinels.ball_team_id)
    is_ball = tracking["team_id"] == ball_team_id

    base = tracking.groupby("ts_ms", sort=False).agg(
        period=("quarter", "first"),
        game_clock=("game_clock", "first"),
        shot_clock=("shot_clock", "first"),
        n_rows=("team_id", "size"),
    )
    base["n_ball_rows"] = is_ball.groupby(tracking["ts_ms"]).sum()
    base["n_player_rows"] = base["n_rows"] - base["n_ball_rows"]

    ball = (
        tracking.loc[is_ball, ["ts_ms", "x", "y", "z"]]
        .drop_duplicates("ts_ms", keep="first")
        .set_index("ts_ms")
        .rename(columns={"x": "ball_x", "y": "ball_y", "z": "ball_z"})
    )
    frames = base.join(ball, how="left").reset_index()
    return frames.sort_values("ts_ms", kind="mergesort").reset_index(drop=True)


def player_positions(tracking: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per (ts_ms, team_id, player_id), the ball excluded."""
    ball_team_id = int(cfg.sentinels.ball_team_id)
    out = tracking.loc[
        tracking["team_id"] != ball_team_id, ["ts_ms", "team_id", "player_id", "x", "y"]
    ]
    return out.reset_index(drop=True)


def votes(tracking: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Ball-holder votes from the kept, sampled frames of one game.

    One row per kept frame: ``period``, ``voter_team_id`` (the nearest
    player's team -- the vote), ``basket_x`` (the basket in the ball's half),
    and ``agree`` (whether the player farthest from that basket, the
    secondary check, belongs to the same team as the voter).
    """
    ad = cfg.attack_direction
    halfcourt_x = float(cfg.court.length_x) / 2.0
    hold_dist = float(ad.hold_dist)
    hold_height = float(ad.hold_height)
    shot_clock_min = float(ad.shot_clock_min)
    shot_clock_max = float(ad.shot_clock_max)
    players_on_court = int(cfg.tracking.players_on_court)
    stride = max(1, round(float(cfg.tracking.source_frame_rate_hz) / float(ad.sample_hz)))

    basket_xy = sorted((float(b.x), float(b.y)) for b in cfg.court.baskets.values())
    left_x, basket_y = basket_xy[0]
    right_x, _ = basket_xy[1]

    frames = frame_table(tracking, cfg)
    frames["clock_running"] = frames.groupby("period")["game_clock"].diff() < 0

    sampled = frames.iloc[np.arange(0, len(frames), stride)]
    kept_frames = sampled.loc[
        sampled["clock_running"].fillna(False)
        & sampled["shot_clock"].between(shot_clock_min, shot_clock_max)
        & (sampled["n_ball_rows"] == 1)
        & (sampled["n_player_rows"] == players_on_court),
        ["ts_ms", "period", "ball_x", "ball_y", "ball_z"],
    ]
    if kept_frames.empty:
        return pd.DataFrame(columns=VOTE_COLUMNS)

    players = player_positions(tracking, cfg)
    merged = players.merge(kept_frames, on="ts_ms", how="inner")

    merged["ball_side"] = merged["ball_x"] >= halfcourt_x
    merged["player_side"] = merged["x"] >= halfcourt_x
    bad_ts = merged.loc[merged["player_side"] != merged["ball_side"], "ts_ms"].unique()
    merged = merged[~merged["ts_ms"].isin(bad_ts)]
    if merged.empty:
        return pd.DataFrame(columns=VOTE_COLUMNS)

    merged["basket_x"] = np.where(merged["ball_side"], right_x, left_x)
    merged["dist_to_ball"] = np.hypot(merged["x"] - merged["ball_x"], merged["y"] - merged["ball_y"])
    merged["dist_to_basket"] = np.hypot(merged["x"] - merged["basket_x"], merged["y"] - basket_y)

    nearest = merged.loc[merged.groupby("ts_ms")["dist_to_ball"].idxmin()].set_index("ts_ms")
    farthest = merged.loc[merged.groupby("ts_ms")["dist_to_basket"].idxmax()].set_index("ts_ms")

    hold_ok = (nearest["dist_to_ball"] <= hold_dist) & (nearest["ball_z"] <= hold_height)
    kept_ts = nearest.index[hold_ok]
    if len(kept_ts) == 0:
        return pd.DataFrame(columns=VOTE_COLUMNS)

    out = pd.DataFrame(
        {
            "ts_ms": kept_ts,
            "period": nearest.loc[kept_ts, "period"].to_numpy(),
            "voter_team_id": nearest.loc[kept_ts, "team_id"].to_numpy(),
            "basket_x": nearest.loc[kept_ts, "basket_x"].to_numpy(),
            "agree": (
                farthest.loc[kept_ts, "team_id"].to_numpy()
                == nearest.loc[kept_ts, "team_id"].to_numpy()
            ),
        }
    )
    return out.reset_index(drop=True)


def tally_votes(votes_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per (period, team_id): the winning basket and its vote_share."""
    if votes_df.empty:
        return pd.DataFrame(columns=TALLY_COLUMNS)

    min_share = float(cfg.attack_direction.min_share)
    min_votes = int(cfg.attack_direction.min_votes)
    min_votes_ot = int(cfg.attack_direction.min_votes_ot)

    counts = (
        votes_df.groupby(["period", "voter_team_id", "basket_x"])
        .size()
        .rename("votes")
        .reset_index()
    )
    totals = counts.groupby(["period", "voter_team_id"])["votes"].sum().rename("n_votes")
    counts = counts.merge(totals, on=["period", "voter_team_id"])
    winners = (
        counts.sort_values("votes", ascending=False, kind="mergesort")
        .drop_duplicates(["period", "voter_team_id"])
        .copy()
    )
    winners["vote_share"] = winners["votes"] / winners["n_votes"]
    required_votes = np.where(winners["period"] > 4, min_votes_ot, min_votes)
    winners["clear"] = (winners["vote_share"] >= min_share) & (winners["n_votes"] >= required_votes)
    out = winners.rename(columns={"voter_team_id": "team_id"})[TALLY_COLUMNS]
    return out.sort_values(["period", "team_id"], kind="mergesort").reset_index(drop=True)


def secondary_agreement(votes_df: pd.DataFrame) -> pd.DataFrame:
    """One row per period: the share of kept frames where the secondary
    check (farthest-from-basket player's team) agrees with the vote."""
    if votes_df.empty:
        return pd.DataFrame(columns=["period", "secondary_agreement"])
    return (
        votes_df.groupby("period")["agree"]
        .mean()
        .rename("secondary_agreement")
        .reset_index()
    )


def consistency_checks(direction_rows: pd.DataFrame) -> dict:
    """Per-game consistency summary from that game's tally_votes() rows.

    ``direction_opposite_ok``: both teams hold opposite baskets in every
    period. ``direction_flips_ok``: each team's basket matches between
    periods 1-2, flips for 3-4, and matches period 4 in any overtime.
    ``violations`` (not written to tracking_inventory.csv; for the
    human-readable report) describes each specific check that failed.
    """
    if direction_rows.empty:
        return {
            "direction_min_share": None,
            "direction_unclear_periods": 0,
            "direction_opposite_ok": False,
            "direction_flips_ok": False,
            "violations": ["no attack-direction rows for this game"],
        }

    direction_min_share = float(direction_rows["vote_share"].min())
    direction_unclear_periods = int((~direction_rows["clear"]).sum())

    opposite_ok = True
    violations: list[str] = []
    for period, group in direction_rows.groupby("period"):
        baskets = group["basket_x"].to_numpy()
        if len(group) != 2:
            opposite_ok = False
            violations.append(f"period {period}: {len(group)} teams recorded, expected 2")
        elif pd.isna(baskets).any():
            opposite_ok = False
            violations.append(f"period {period}: at least one team cast no attack-direction vote")
        elif baskets[0] == baskets[1]:
            opposite_ok = False
            violations.append(f"period {period}: both teams assigned basket_x={baskets[0]}")

    flips_ok = True
    for team_id, group in direction_rows.groupby("team_id"):
        # dropna so a padded, voteless period (NaN basket_x -- see
        # scripts/attack_direction.py's process_game()) reads as "unknown"
        # (by_period.get() then returns None) rather than a real mismatch;
        # the opposite_ok check above already flags the missing vote itself.
        by_period = group.dropna(subset=["basket_x"]).set_index("period")["basket_x"]
        p1, p2 = by_period.get(1), by_period.get(2)
        p3, p4 = by_period.get(3), by_period.get(4)
        if p1 is not None and p2 is not None and p1 != p2:
            flips_ok = False
            violations.append(f"team {team_id}: basket differs between periods 1 and 2")
        if p3 is not None and p4 is not None and p3 != p4:
            flips_ok = False
            violations.append(f"team {team_id}: basket differs between periods 3 and 4")
        if p1 is not None and p3 is not None and p1 == p3:
            flips_ok = False
            violations.append(f"team {team_id}: basket did not switch between periods 1 and 3")
        for period in by_period.index:
            if period > 4 and (p4 is None or by_period[period] != p4):
                flips_ok = False
                violations.append(f"team {team_id}: period {period} (overtime) does not match period 4")

    return {
        "direction_min_share": direction_min_share,
        "direction_unclear_periods": direction_unclear_periods,
        "direction_opposite_ok": opposite_ok,
        "direction_flips_ok": flips_ok,
        "violations": violations,
    }
