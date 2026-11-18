# Vendored play-animation script

Copied from https://github.com/linouk23/NBA-Player-Movements (not a dependency
of the `nba_assists` package — only used interactively for reviewing a single
play). `README_upstream.md` is the original repo's README.

## Usage

```
cd scripts/vendor
python main.py --path <path/to/a/game.json> --event <event_index>
```

It expects the raw per-game JSON (not our Parquet output), and reads
`court.png` from the current directory. To animate a game from our pipeline,
re-extract its JSON from the matching archive in `data/raw/`:

```
python -c "import py7zr; py7zr.SevenZipFile('data/raw/<game_id>.7z').extractall('/tmp/game_json')"
```

then point `--path` at the extracted `.json` file.
