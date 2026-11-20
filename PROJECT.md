# nba-assists — project conventions

This file is the contract for every later step. Read it before writing code that
touches tracking data, play-by-play, windows, or models.

## Repository layout

```
data/raw/              7z archives, one per game (inputs, never modified)
data/tracking/         one Parquet per game, named <date>.<away>.at.<home>.parquet
data/tracking/_rosters/ per-game roster CSV, same stem as its Parquet
data/pbp/              play-by-play, one Parquet per game
data/windows/          extracted event windows
data/derived/          features, labels, splits, intermediate tables
src/nba_assists/       the package (importable library code)
scripts/               runnable entry points, one step per script
scripts/vendor/        vendored third-party code, not a package dependency
notebooks/             exploration only; nothing here is a dependency
models/                trained weights and fitted estimators
reports/               figures, tables, metrics, run summaries
tests/                 pytest suite
```

A tracking Parquet's filename is the source archive's stem (`MM.DD.YYYY.AWAY.at.HOME`),
**not** the canonical game id — that filename order (away team first) is read from
the archive name only to recover the date; home/away team abbreviations are read
from the roster CSV instead, which is authoritative. The canonical 10-character
`game_id` is a column inside the file itself (see Identifiers below); always read
it from there, never parse it from the filename.

Library code goes in `src/nba_assists/`. Scripts in `scripts/` parse arguments,
call into the package, and print a summary. No analysis logic lives in a script
that another step would want to reuse.

## Coordinates

* SportVU feet. `x` runs 0–94 along the court, `y` runs 0–50 across it.
* `z` is ball height in feet. Player rows carry `z = 0`.
* Baskets are at `(5.25, 25)` and `(88.75, 25)`.
* Rim height is 10 ft; rim radius is 0.75 ft.
* Geometry constants are read from `config.yaml` under `court:`, not retyped.

## Clock

* `game_clock` is **seconds remaining in the quarter** and **counts down**.
* "Later" in a period means a **smaller** `game_clock`.
* Sorting a period chronologically means sorting `game_clock` **descending**.
* A regulation period is 720 s; an overtime period is 300 s.
* `shot_clock` may be null (e.g. late-clock situations); handle it explicitly.

## Identifiers

* Game IDs are 10-character zero-padded **strings**: `"0021500431"`.
  Never store or compare them as integers. Use
  `nba_assists.config.format_game_id` to normalize.
* `player_id` and `team_id` are integers.
* Ball rows are marked by `team_id = -1` **and** `player_id = -1`
  (`sentinels:` in `config.yaml`). Filter the ball out explicitly before any
  per-player aggregation.

## Tracking frames

* Native sampling is 25 Hz (`tracking.source_frame_rate_hz`).
* Each frame holds 10 player rows plus 1 ball row (`tracking.players_on_court`
  + the ball); a frame with a different count is a data defect, tracked by
  `scripts/tracking_inventory.py` as a share of the game's frames, not
  silently dropped.
* **The source data is not a continuous 25 Hz stream for the whole quarter.**
  Each play-by-play event in the raw JSON carries its own short `moments`
  window; the union of those windows leaves real, multi-second — sometimes
  multi-minute — stretches of wall-clock time with zero frames, even while
  `game_clock` was ticking down the whole time. This is a property of every
  game in this dataset, not a defect specific to a few of them. See "Tracking
  QA inventory" below.

## Tracking data source

* Raw archives come from https://github.com/linouk23/NBA-Player-Movements,
  `data/2016.NBA.Raw.SportVU.Game.Logs/*.7z` (2015-16 season). Fetch with a
  shallow clone (`git clone --depth 1`, still ~7 GB since the files are
  committed to the tip, not history) into a temp dir, move the archives into
  `data/raw/`, then delete the clone.
* `scripts/vendor/` holds that repo's play-animation tool (`main.py` +
  `Game.py`/`Event.py`/`Moment.py`/`Player.py`/`Team.py`/`Ball.py`/
  `Constant.py`/`court.png`), copied for reviewing a single play. It is not a
  dependency of `nba_assists` and takes the raw per-game JSON as input, not
  our Parquet — see `scripts/vendor/README.md`.
* `scripts/build_tracking_parquet.py <raw_dir> <tracking_dir>` converts one
  archive at a time (extract → Parquet → delete the JSON immediately, so peak
  extra disk use is about one game's JSON), skips archives under 1 KB (empty)
  and already-converted games, so it is safe to stop and resume.

## Tracking QA inventory

* `scripts/tracking_inventory.py` reads every tracking Parquet and writes
  `reports/tracking_inventory.csv`, one row per game: `game_id` (from the data
  itself, not the filename — see Repository layout), `date`, `home_team`,
  `away_team`, `frame_count`, `minutes_covered`, `share_frames_11_rows`,
  `share_frames_ball_missing`, `quarters_present`, `max_gap_ms`,
  `max_gap_running_ms`, `flagged`, `flag_reasons`; and a high-level summary to
  `reports/tracking_inventory_summary.md` (games flagged, by what reason, and
  the list of flagged games).
* `minutes_covered` is `frame_count / tracking.source_frame_rate_hz / 60`: the
  amount of sampled game time actually captured, not the wall-clock span
  between the first and last frame (which also counts quarter breaks and
  halftime).
* A game is flagged (`inventory:` in `config.yaml`) when: minutes covered is
  below `min_minutes_required`; the share of frames missing a player or the
  ball exceeds `max_missing_frame_share`; or any gap between consecutive
  frames in the same quarter, while `game_clock` was strictly decreasing,
  exceeds `max_gap_running_s`.
* **That last rule trips on nearly every game**, because of the non-continuous
  source data described above — a clip boundary between two events can land
  in the middle of a live possession. It still ranks games usefully by
  severity (`max_gap_running_ms` in the tens of seconds is the baseline for
  essentially all games; values in the hundreds of seconds mark real outliers
  worth a closer look), but on its own it does not separate "normal" games
  from damaged ones the way the other two rules do.

## Play-by-play

* Source: `Historical/PBP Data/2015-16_pbp.csv` from
  https://github.com/sumitrodatta/nba-alt-awards (full 2015-16 season, 1,230
  games — about twice what we have tracking for), fetched via its raw file
  URL and saved to `data/pbp/2015-16_pbp.csv`.
* `scripts/prepare_pbp.py` reads it, zero-pads `GAME_ID`, adds `clock_s`
  (`PCTIMESTRING` "mm:ss" → seconds remaining in the period, same direction
  as tracking's `game_clock`), keeps only rows for games that actually have
  tracking frames — the distinct `game_id` values present in the tracking
  Parquet data itself, not merely a matching file in `data/tracking/` (the
  one zero-frame game from the QA inventory contributes no `game_id` and so
  drops out here too) — **and whose attack-direction consistency checks both
  passed** (`direction_opposite_ok`/`direction_flips_ok` in
  `reports/tracking_inventory.csv`; run `scripts/tracking_inventory.py` then
  `scripts/attack_direction.py` first). A game failing either check means its
  basket labeling, and so which team a play's direction belongs to, can't be
  trusted. Rows are then sorted by `PERIOD`, then `clock_s` descending, then
  `EVENTNUM` (time is the primary key: `EVENTNUM` is sometimes out of order
  in the source feed). Writes `data/pbp/pbp_2015_16.parquet`, otherwise
  unchanged from the source columns.
* `scripts/build_makes.py` reads that Parquet — already sorted by time per
  game — and, before any filtering, computes each row's previous event by
  shifting within each `game_id` group (`prev_eventnum`, `prev_msgtype`,
  `prev_clock_s`, `prev_description`, `prev_same_period`). Only after that
  does it filter to `EVENTMSGTYPE == pbp.event_types.made_shot`, so a make's
  "previous event" is the true previous play of any type, not the previous
  make. `PLAYER2_ID == pbp.no_player_sentinel` (0 in this feed — unrelated to
  our own `-1` ball sentinel) means no assist. `home_team_id` and each game's
  roster (for the shooter-on-roster check) come from
  `data/tracking/_rosters/*.csv`, keyed by that file's own `game_id` column —
  roster filenames are the source archive's stem, not the game id. Writes
  `data/derived/makes.parquet` and `reports/makes_summary.md`; the expected
  assisted-share range it checks against lives in `makes_qa:` in
  `config.yaml`.

## Rim-crossing detection / clock offset

* `src/nba_assists/rim.py` finds, for one make, the ball's actual downward
  crossing through a rim in the tracking frames: a pair of consecutive ball
  frames within `rim_detection.search_before_s`/`search_after_s` of the
  make's `clock_s`, straddling `court.rim_height_z` in height and within
  `rim_detection.rim_radius_tol` feet of a basket at the interpolated
  crossing point. Each candidate is checked against whether the game clock
  was actually running shortly before it (`rim_detection.clock_check_s`/
  `clock_min_drop_s`), with a wider `buzzer_window_s` lookback used instead
  when the clock already reads 0 going into the crossing — this is what
  rejects free throws and other dead-ball shots through the same rim.
  `detect_make()` is the single entry point; see its docstring for the
  "choosing among candidates" rules (one candidate: take it; several at the
  same basket: take the latest; several at different baskets: take whichever
  is closest to `clock_s` — a likely tracking glitch).
* `scripts/detect_makes.py` runs this for every row of `makes.parquet`
  against that game's ball frames (`rim.ball_rows()`, the inverse of the
  per-player ball filter: `team_id`/`player_id` equal to the ball
  sentinels). Writes `data/derived/rim_candidates.parquet` (every geometric
  candidate, clock-check result included) and `data/derived/make_times.parquet`
  (one row per make: `make_id = "<game_id>-<eventnum>"`, the chosen result or
  why none was chosen), plus `reports/clock_offset_by_game.csv` and
  `reports/clock_offset.md`.
* **The detection rate is well under 100%** (57% on the full 2015-16 set),
  for two different reasons, both real properties of the data rather than a
  detection bug: the same non-continuous tracking coverage described above
  (a make's rim crossing can simply fall in a stretch with zero frames), and
  — confirmed by hand on several of the worst-detection games — a handful of
  games carry a clock-offset between the pbp and tracking feeds larger than
  the fixed ±2 s search window, so every make in that game searches the
  wrong slice of frames and detects nothing. `clock_offset_by_game.csv` has
  the full per-game breakdown; a widened or per-game-calibrated search
  window is a candidate follow-up, not yet implemented.
* **The offsets found are not simple truncation or rounding.** The spec
  question was whether `clock_s` is a truncated or rounded reading of the
  true time; in practice detected offsets center around -1.2s (the
  recorded pbp time trailing the actual rim crossing by over a second, not a
  sub-second rounding artifact), consistent with known lag in the NBA's
  play-by-play event logging relative to real game time. See
  `reports/clock_offset.md` for the full distribution.

## Attack direction

* `src/nba_assists/direction.py` determines, per game/period/team, which
  basket a team is attacking, from tracking data alone (no play-by-play): a
  ball-holder vote. On frames sampled at `attack_direction.sample_hz`, kept
  only when the game clock is running (native-frame-to-native-frame, before
  sampling), `shot_clock` is in `[shot_clock_min, shot_clock_max]`, the frame
  has exactly one ball row and `tracking.players_on_court` player rows, and
  the ball and every player are in the same half (`court.length_x / 2`), the
  player nearest the ball (within `hold_dist` feet, ball height at or below
  `hold_height`) casts one vote for the basket in that half. `votes()` is the
  per-game entry point; `tally_votes()` picks each (period, team_id)'s
  winning basket and `vote_share`; `secondary_agreement()` reports, per
  period, how often the player farthest from that basket (a secondary,
  report-only check) belongs to the same team as the vote; `consistency_checks()`
  flags a game where the two teams don't hold opposite baskets in every
  period, or a team's basket doesn't follow the expected periods-1-2/3-4
  flip-and-overtime-matches-period-4 pattern.
* `scripts/attack_direction.py` runs this for every game in `data/tracking/`.
  Writes `data/derived/attack_direction.parquet` (one row per game/period/team:
  `game_id`, `period`, `team_id`, `team_abbr`, `basket_x`, `n_votes`,
  `vote_share`, `secondary_agreement`, `clear` — `clear` is
  `vote_share >= attack_direction.min_share` and `n_votes` at or above
  `attack_direction.min_votes` (`min_votes_ot` instead for a period past 4,
  since overtime is shorter); appends `direction_min_share`,
  `direction_unclear_periods`, `direction_opposite_ok`, `direction_flips_ok`
  to `reports/tracking_inventory.csv` (run `scripts/tracking_inventory.py`
  first); and writes `reports/attack_direction.md`.
* **`attack_direction.parquet` always has one row per (period, team_id)
  actually played in a game**, even a team-period that cast zero votes
  (`n_votes=0`, `basket_x` null, `clear=False`) — a later step (alignment)
  needs every combination to look up, not just the ones with enough votes to
  clear the thresholds. `scripts/attack_direction.py`'s `process_game()`
  reindexes `tally_votes()`'s output onto the full period × team_id grid
  (periods from the tracking data itself, teams from the roster) before
  running `consistency_checks()`, so a period where one or both teams cast no
  vote is itself a flagged violation (`direction_opposite_ok=False`), not
  silently absent the way it was before this reindex existed.
* **Some games have `shot_clock` null for part or all of a game** (a
  source-data defect, confirmed by hand — frame counts and coverage are
  otherwise normal): 12 games have it null throughout (every period of those
  games casts zero votes), and a handful more have it null for just one or
  two periods. Either way the shot-clock-range filter correctly excludes a
  null reading rather than guessing, and the resulting zero-vote periods show
  up as consistency violations in `attack_direction.md` and so are excluded
  by `scripts/prepare_pbp.py` (see "Play-by-play" above).

## Configuration

* Every tunable parameter a script actually reads lives in `config.yaml`.
  **No magic numbers in code, and no speculative config either**: a new
  constant means a new config key read by real code in the same change, not
  a guess at a value some future step might want.
* Code reads it through `nba_assists.config.get_config()`.
* `cfg.seeds.global_` is the attribute spelling of the YAML key `seeds.global`,
  since `global` is a Python keyword.
* Override the config path for experiments with the `NBA_ASSISTS_CONFIG`
  environment variable.

## Randomness

* All randomness is seeded from `config.yaml` under `seeds:`.
* Call `nba_assists.config.seed_everything()` once at the top of any script that
  samples, shuffles, splits, or trains. It seeds Python, NumPy and Torch.

## Splits

* **All train/validation/test splits will be by game, never by play.** No game
  may contribute rows to two splits; plays from one game are highly
  correlated. Nothing builds a split yet — no `splits:` config section exists
  until the step that reads it does.

## Scripts

Each script:

1. loads the config, and seeds randomness if it uses any,
2. writes its outputs under `data/` or `reports/`,
3. prints a short summary of **counts and failures** on exit, e.g.
   `games=612 frames=11,204,338 passes=84,221 failed_games=3`.

Re-running a script currently just overwrites its outputs, except
`scripts/build_tracking_parquet.py`, which skips a game whose Parquet already
exists (so it can be stopped and resumed on a 636-archive run without
redoing finished work).

Check new scripts with `.venv/bin/pyright scripts/ src/` (installed via the
`dev` extra). Two kinds of finding show up there:

* A real gap — e.g. a call whose declared return type includes `None` but the
  code doesn't check for it. Fix the code, don't suppress the warning.
* Noise from a library's own stubs rather than a real issue — duckdb ships no
  stubs for `Connection.description`/`fetchone()`, and pandas' stubs report a
  union of every dtype an operation could theoretically produce and then
  can't narrow it back down, for code that is otherwise correct and tested.
  Prefer restructuring to dodge the false positive (a dict comprehension
  instead of `dict(zip(...))`, say) where that's easy; otherwise scope a
  `# pyright: reportXxx=false` comment to the top of that one file rather
  than disabling a rule project-wide, and say why in a comment next to it.
  Confirm the fix changed no output before moving on (diff the script's
  Parquet/CSV against a copy from before the change).

## Data hygiene

* `data/` and `models/` are git-ignored. Nothing large is committed.
* Parquet is the interchange format, read and written through
  `nba_assists.config.parquet_compression()` rather than passing a codec
  string directly (pandas types that parameter as a `Literal` of five known
  codecs, not a plain `str`).
* Raw archives in `data/raw/` are treated as read-only.

## Environment

* Python 3.12 in `.venv/`.
* Install with `.venv/bin/pip install -e ".[dev]"`.
* Run tests with `.venv/bin/pytest`.
