# Vendored play-animation script

Copied from https://github.com/linouk23/NBA-Player-Movements (not a dependency
of the `nba_assists` package — only used interactively for reviewing a single
play). `README_upstream.md` is the original repo's README.

## Patched from upstream

`Event.py`'s `show()` used `table.properties()['child_artists']`, a private
matplotlib API removed in recent versions. It now uses the public
`table.get_celld()` instead — same effect (colors the table cell text white),
just compatible with the matplotlib installed here (3.11.2). Everything else
is unmodified.

## Usage

1. `--path` takes the raw per-game JSON, not our Parquet output — re-extract
   it from the matching archive in `data/raw/` (filenames there are
   `<date>.<away>.at.<home>.7z`, not the canonical game id; see PROJECT.md):

   ```
   .venv/bin/python -c "
   import py7zr
   py7zr.SevenZipFile('data/raw/01.01.2016.CHA.at.TOR.7z').extractall('/tmp/game_json')
   "
   ```

   This writes `/tmp/game_json/<game_id>.json`.

2. Run the animation from this directory, so the relative `court.png` load
   resolves, pointing `--path` at that JSON. `--event` is the **positional
   index** into the game's `events` array (0-based), not the `event_id`
   column in our Parquet — print the roster or play-by-play separately if you
   need to map a specific play to its index:

   ```
   cd scripts/vendor
   ../../.venv/bin/python main.py --path /tmp/game_json/0021500492.json --event 5
   ```

   Omitting `--event` defaults to 0; passing an index past the last event
   clamps to the last one. The script prints the valid range (`You can rerun
   the script and choose any event from 0 to N`) on every run.

3. This opens an interactive matplotlib window and blocks until you close it
   — it needs a GUI-capable backend (the default on a normal desktop Python;
   not inside a plain SSH session with no display). To save it to a file
   instead of displaying it, edit the last line of `Event.show()`
   (`plt.show()`) to `anim.save("play.mp4")` or `anim.save("play.gif")`.
