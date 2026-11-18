"""Configuration loading.

Every tunable parameter lives in ``config.yaml`` at the repository root.  Code
reads it through :func:`load_config` and never inlines a constant.

    from nba_assists.config import get_config
    cfg = get_config()
    rim_z = cfg.court.rim_height_z
    seed = cfg.seeds.global_
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

CONFIG_FILENAME = "config.yaml"
CONFIG_ENV_VAR = "NBA_ASSISTS_CONFIG"

# ``global`` is a Python keyword, so ``cfg.seeds.global_`` is the attribute form.
_KEYWORD_SUFFIX = "_"
_PYTHON_KEYWORDS = {"global", "class", "lambda", "from", "import", "None", "True", "False"}


class Config(Mapping):
    """Read-only nested mapping with attribute access.

    Behaves like a dict (``cfg["court"]["rim_height_z"]``) and like an object
    (``cfg.court.rim_height_z``).  Nested dicts are wrapped lazily on access.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    # -- mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attribute access ---------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        key = name
        if name.endswith(_KEYWORD_SUFFIX) and name[:-1] in _PYTHON_KEYWORDS:
            key = name[:-1]
        try:
            return _wrap(self._data[key])
        except KeyError:
            raise AttributeError(
                f"no config key {key!r}; available: {sorted(self._data)}"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is read-only; edit config.yaml instead")

    def __dir__(self):
        return list(self._data)

    def __repr__(self) -> str:
        return f"Config({sorted(self._data)!r})"

    # -- helpers ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict copy, safe to mutate."""
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.items()}

    def get_path(self, key: str) -> Path:
        """Resolve a key under ``paths:`` against the repository root."""
        return (repo_root() / str(self._data["paths"][key])).resolve()


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def repo_root() -> Path:
    """Repository root: the nearest ancestor directory holding ``config.yaml``."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).resolve().parent
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / CONFIG_FILENAME).is_file():
            return parent
    return here.parents[2]  # src/nba_assists/config.py -> repo root


def config_path() -> Path:
    """Location of the active config file."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).resolve()
    return repo_root() / CONFIG_FILENAME


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load a config file from disk, uncached."""
    target = Path(path).resolve() if path is not None else config_path()
    if not target.is_file():
        raise FileNotFoundError(f"config file not found: {target}")
    with target.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{target} must contain a YAML mapping at the top level")
    return Config(data)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached config for the default path.  Use this from scripts."""
    return load_config()


def project_paths(cfg: Config | None = None) -> dict[str, Path]:
    """Every entry under ``paths:`` resolved to an absolute Path."""
    cfg = cfg or get_config()
    root = repo_root()
    return {key: (root / str(val)).resolve() for key, val in cfg["paths"].items()}


def ensure_dirs(cfg: Config | None = None) -> dict[str, Path]:
    """Create every configured directory.  Returns the resolved paths."""
    paths = project_paths(cfg)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def seed_everything(cfg: Config | None = None) -> int:
    """Seed Python, NumPy and (if installed) Torch from ``seeds:`` in the config.

    Returns the global seed actually applied.
    """
    cfg = cfg or get_config()
    seeds = cfg["seeds"]
    base = int(seeds["global"])

    random.seed(base)
    os.environ["PYTHONHASHSEED"] = str(base)

    try:
        import numpy as np

        np.random.seed(int(seeds.get("numpy", base)))
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(int(seeds.get("torch", base)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seeds.get("torch", base)))
    except ImportError:
        pass

    return base


def resolve_device(cfg: Config | None = None) -> str:
    """Turn ``training.device: auto`` into a concrete torch device string."""
    cfg = cfg or get_config()
    requested = str(cfg["training"]["device"]).lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_game_id(game_id: str | int, cfg: Config | None = None) -> str:
    """Zero-pad a game id to the configured width, e.g. ``"0021500431"``."""
    cfg = cfg or get_config()
    width = int(cfg["tracking"]["game_id_length"])
    text = str(game_id).strip()
    if len(text) > width:
        raise ValueError(f"game id {text!r} is longer than {width} characters")
    return text.zfill(width)


__all__ = [
    "Config",
    "config_path",
    "ensure_dirs",
    "format_game_id",
    "get_config",
    "load_config",
    "project_paths",
    "repo_root",
    "resolve_device",
    "seed_everything",
]
