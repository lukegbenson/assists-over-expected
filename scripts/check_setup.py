#!/usr/bin/env python3
"""Verify the project skeleton: directories, config, seeds, environment.

Run this first after cloning. It creates any missing directory, reports which
data stages are populated, and exits non-zero if something is wrong.

    .venv/bin/python scripts/check_setup.py
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Allow running without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_assists.config import (  # noqa: E402
    config_path,
    ensure_dirs,
    format_game_id,
    get_config,
    repo_root,
    resolve_device,
    seed_everything,
)

REQUIRED_PACKAGES = [
    "py7zr",
    "pyarrow",
    "polars",
    "duckdb",
    "pandas",
    "numpy",
    "sklearn",
    "torch",
    "matplotlib",
    "statsmodels",
    "tqdm",
    "yaml",
    "pytest",
]

# Data stages whose contents are worth counting in the summary.
STAGE_GLOBS = {
    "raw": "*.7z",
    "tracking": "*.parquet",
    "rosters": "*.parquet",
    "pbp": "*.parquet",
    "windows": "*.parquet",
    "derived": "*",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="report missing directories instead of creating them",
    )
    args = parser.parse_args()

    failures: list[str] = []

    cfg = get_config()
    seed = seed_everything(cfg)

    print(f"repo root   : {repo_root()}")
    print(f"config      : {config_path()}")
    print(f"python      : {sys.version.split()[0]}")
    print(f"global seed : {seed}")
    print(f"device      : {resolve_device(cfg)}")

    # --- directories -------------------------------------------------------
    paths = {k: (repo_root() / str(v)).resolve() for k, v in cfg["paths"].items()}
    if args.no_create:
        missing = [k for k, p in paths.items() if not p.is_dir()]
        failures += [f"missing directory: {paths[k]}" for k in missing]
    else:
        paths = ensure_dirs(cfg)

    # --- dependencies ------------------------------------------------------
    missing_pkgs = []
    for name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing_pkgs.append(name)
    failures += [f"missing package: {name}" for name in missing_pkgs]

    # --- config sanity -----------------------------------------------------
    fractions = cfg["splits"]
    total = sum(
        float(fractions[k]) for k in ("train_fraction", "val_fraction", "test_fraction")
    )
    if abs(total - 1.0) > 1e-9:
        failures.append(f"split fractions sum to {total:.4f}, expected 1.0")
    if str(cfg["splits"]["split_by"]) != "game":
        failures.append("splits.split_by must be 'game'")
    if format_game_id(21500431) != "0021500431":
        failures.append("game id padding is wrong")

    # --- data stage counts -------------------------------------------------
    print("\ndata stages:")
    for key, pattern in STAGE_GLOBS.items():
        path = paths.get(key)
        count = len([p for p in path.glob(pattern) if p.is_file()]) if path and path.is_dir() else 0
        state = "ok" if path and path.is_dir() else "MISSING"
        print(f"  {key:9s} {count:6d} file(s)  [{state}]  {path}")

    # --- summary -----------------------------------------------------------
    print(
        f"\nsummary: directories={len(paths)} "
        f"packages_ok={len(REQUIRED_PACKAGES) - len(missing_pkgs)}/{len(REQUIRED_PACKAGES)} "
        f"failures={len(failures)}"
    )
    for problem in failures:
        print(f"  FAIL {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
