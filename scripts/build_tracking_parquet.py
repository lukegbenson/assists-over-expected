"""Convert the 2015-16 SportVU game logs (.7z -> JSON) into one Parquet file per game.

Usage:
    python build_tracking_parquet.py <dir_with_7z_files> <output_dir>

Each archive is extracted to a temporary folder, converted, and the ~115 MB JSON is
deleted straight away, so peak disk use stays around one game's JSON. Already-converted
games are skipped, so the script can be stopped and restarted.

Output (one row per object per frame, ball has team_id = -1 and player_id = -1):
    game_id, event_id, quarter, ts_ms, game_clock, shot_clock,
    team_id, player_id, x, y, z
Raw moments are repeated across overlapping events; frames are de-duplicated on ts_ms
and keep the first event_id they appear under, so event_id is only a rough pointer.
To cut the window around a play, use quarter + game_clock (or ts_ms) instead.

Also writes <output_dir>/_rosters/<game>.csv with player names, jerseys and positions.

Requires: pip install py7zr pyarrow
"""
import json
import pathlib
import sys
import tempfile

import py7zr
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("game_id", pa.string()), ("event_id", pa.int16()), ("quarter", pa.int8()),
    ("ts_ms", pa.int64()), ("game_clock", pa.float32()), ("shot_clock", pa.float32()),
    ("team_id", pa.int32()), ("player_id", pa.int32()),
    ("x", pa.float32()), ("y", pa.float32()), ("z", pa.float32()),
])


def convert(json_path: pathlib.Path, dst: pathlib.Path) -> int:
    with open(json_path) as f:
        game = json.load(f)
    cols = {name: [] for name in SCHEMA.names}
    seen = set()
    for ev in game["events"]:
        eid = int(ev["eventId"])
        for m in ev["moments"]:
            ts = m[1]
            if ts in seen:
                continue
            seen.add(ts)
            for team_id, player_id, x, y, z in m[5]:
                cols["game_id"].append(game["gameid"])
                cols["event_id"].append(eid)
                cols["quarter"].append(m[0])
                cols["ts_ms"].append(ts)
                cols["game_clock"].append(m[2])
                cols["shot_clock"].append(m[3])
                cols["team_id"].append(team_id)
                cols["player_id"].append(player_id)
                cols["x"].append(x)
                cols["y"].append(y)
                cols["z"].append(z)
    # Roster (names, jerseys, positions) from the first event's home/visitor blocks.
    roster_dir = dst.parent / "_rosters"
    roster_dir.mkdir(exist_ok=True)
    first = game["events"][0]
    with open(roster_dir / (dst.stem + ".csv"), "w") as f:
        f.write("game_id,side,team_id,team_abbr,player_id,first_name,last_name,jersey,position\n")
        for side in ("home", "visitor"):
            team = first[side]
            for p in team["players"]:
                f.write(",".join(str(v) for v in (
                    game["gameid"], side, team["teamid"], team["abbreviation"], p["playerid"],
                    p["firstname"].replace(",", " "), p["lastname"].replace(",", " "),
                    p["jersey"], p["position"])) + "\n")

    table = pa.table(cols, schema=SCHEMA)
    tmp = dst.with_suffix(".parquet.part")
    pq.write_table(table, tmp, compression="zstd")
    tmp.rename(dst)
    return table.num_rows


def main(src_dir: str, out_dir: str) -> None:
    src, out = pathlib.Path(src_dir), pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archives = sorted(src.glob("*.7z"))
    for i, arc in enumerate(archives, 1):
        dst = out / (arc.stem + ".parquet")
        if dst.exists():
            continue
        if arc.stat().st_size < 1000:
            print(f"[{i}/{len(archives)}] {arc.name}: empty archive, skipped")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            with py7zr.SevenZipFile(arc) as z:
                z.extractall(tmp)
            json_files = list(pathlib.Path(tmp).glob("*.json"))
            if not json_files:
                print(f"[{i}/{len(archives)}] {arc.name}: no JSON inside, skipped")
                continue
            try:
                rows = convert(json_files[0], dst)
            except Exception as e:  # keep going on a bad file
                print(f"[{i}/{len(archives)}] {arc.name}: FAILED ({e})")
                continue
        print(f"[{i}/{len(archives)}] {arc.name}: {rows:,} rows")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])