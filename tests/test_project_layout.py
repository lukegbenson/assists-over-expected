"""The directory skeleton and the documents that describe it."""

from __future__ import annotations

import pytest

from nba_assists.config import repo_root

EXPECTED_DIRS = [
    "data/raw",
    "data/tracking",
    "data/tracking/_rosters",
    "data/pbp",
    "data/windows",
    "data/derived",
    "src/nba_assists",
    "scripts",
    "notebooks",
    "models",
    "reports",
    "tests",
]

EXPECTED_FILES = ["pyproject.toml", "config.yaml", "PROJECT.md", ".gitignore"]


@pytest.mark.parametrize("rel", EXPECTED_DIRS)
def test_directory_exists(rel):
    assert (repo_root() / rel).is_dir()


@pytest.mark.parametrize("rel", EXPECTED_FILES)
def test_file_exists(rel):
    assert (repo_root() / rel).is_file()


def test_gitignore_excludes_data_and_models():
    lines = {
        line.strip()
        for line in (repo_root() / ".gitignore").read_text().splitlines()
    }
    assert "data/" in lines
    assert "models/" in lines


def test_project_md_records_the_conventions():
    text = (repo_root() / "PROJECT.md").read_text()
    for token in ["0021500431", "5.25", "88.75", "0.75", "game_clock", "team_id = -1"]:
        assert token in text, f"PROJECT.md does not mention {token!r}"
