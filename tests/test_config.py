"""The config contract: keys the rest of the pipeline relies on."""

from __future__ import annotations

import math

import pytest

from nba_assists.config import (
    Config,
    ensure_dirs,
    format_game_id,
    get_config,
    load_config,
    project_paths,
    repo_root,
    seed_everything,
)


@pytest.fixture(scope="module")
def cfg() -> Config:
    return get_config()


def test_config_file_loads(cfg):
    assert isinstance(cfg, Config)
    assert len(cfg) > 0


def test_attribute_and_item_access_agree(cfg):
    assert cfg.court.rim_height_z == cfg["court"]["rim_height_z"]


def test_config_is_read_only(cfg):
    with pytest.raises(AttributeError):
        cfg.court = None


def test_missing_key_is_attribute_error(cfg):
    with pytest.raises(AttributeError):
        _ = cfg.not_a_real_section


def test_court_geometry_matches_conventions(cfg):
    court = cfg.court
    assert court.length_x == 94.0
    assert court.width_y == 50.0
    assert court.rim_height_z == 10.0
    assert court.rim_radius == 0.75
    assert (court.baskets.left.x, court.baskets.left.y) == (5.25, 25.0)
    assert (court.baskets.right.x, court.baskets.right.y) == (88.75, 25.0)
    # Baskets are mirror images about mid-court.
    assert math.isclose(
        court.baskets.left.x + court.baskets.right.x, court.length_x, abs_tol=1e-9
    )


def test_ball_sentinels(cfg):
    assert cfg.sentinels.ball_team_id == -1
    assert cfg.sentinels.ball_player_id == -1


def test_frame_rate_and_period_length(cfg):
    assert cfg.tracking.source_frame_rate_hz == 25.0
    assert cfg.tracking.period_length_s == 720.0
    assert cfg.tracking.players_on_court == 10


def test_window_margins_present_and_ordered(cfg):
    w = cfg.windows
    assert w.pre_margin_s > 0
    assert w.post_margin_s > 0
    assert w.min_length_s < w.max_length_s


def test_thresholds_are_physically_sane(cfg):
    assert 0 < cfg.possession.max_dist_ft < cfg.possession.release_dist_ft
    assert cfg.pass_detection.min_duration_s < cfg.pass_detection.max_duration_s
    assert cfg.pass_detection.max_apex_height_ft > cfg.court.rim_height_z


def test_model_sizes_present(cfg):
    assert len(cfg.model.mlp.hidden_sizes) >= 1
    assert all(h > 0 for h in cfg.model.mlp.hidden_sizes)
    assert cfg.model.lstm.hidden_size > 0


def test_splits_are_by_game_and_sum_to_one(cfg):
    s = cfg.splits
    assert s.split_by == "game"
    assert math.isclose(
        s.train_fraction + s.val_fraction + s.test_fraction, 1.0, abs_tol=1e-9
    )


def test_seeds_present(cfg):
    seeds = cfg.seeds
    assert isinstance(seeds["global"], int)
    assert seeds.global_ == seeds["global"]
    assert isinstance(seeds.split, int)


def test_seed_everything_is_reproducible(cfg):
    import random

    seed_everything(cfg)
    first = [random.random() for _ in range(5)]
    seed_everything(cfg)
    assert [random.random() for _ in range(5)] == first


def test_paths_resolve_under_repo_root(cfg):
    root = repo_root()
    for path in project_paths(cfg).values():
        assert str(path).startswith(str(root))


def test_ensure_dirs_creates_every_path(cfg):
    for path in ensure_dirs(cfg).values():
        assert path.is_dir()


def test_to_dict_round_trips(cfg):
    plain = cfg.to_dict()
    assert plain["court"]["rim_radius"] == 0.75
    plain["court"]["rim_radius"] = 99.0  # mutating the copy is safe
    assert cfg.court.rim_radius == 0.75


def test_load_config_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0021500431", "0021500431"),
        (21500431, "0021500431"),
        ("21500431", "0021500431"),
        (" 21500431 ", "0021500431"),
    ],
)
def test_format_game_id(raw, expected):
    assert format_game_id(raw) == expected
    assert len(format_game_id(raw)) == 10


def test_format_game_id_rejects_overlong():
    with pytest.raises(ValueError):
        format_game_id("123456789012")
