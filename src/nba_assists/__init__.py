"""Passing and assist modeling from SportVU tracking data.

Conventions (coordinates, clock direction, id formats) are recorded in
PROJECT.md at the repository root.  Tunable parameters live in config.yaml.
"""

from nba_assists.config import (
    Config,
    ensure_dirs,
    format_game_id,
    get_config,
    load_config,
    parquet_compression,
    project_paths,
    repo_root,
    resolve_device,
    seed_everything,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ensure_dirs",
    "format_game_id",
    "get_config",
    "load_config",
    "parquet_compression",
    "project_paths",
    "repo_root",
    "resolve_device",
    "seed_everything",
    "__version__",
]
