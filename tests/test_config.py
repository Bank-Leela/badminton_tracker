import pytest

from config import REPO_ROOT, load_config


def test_defaults_load():
    cfg = load_config()
    assert cfg.shuttle.eval_mode in {"weight", "average", "nonoverlap"}
    assert cfg.overlay.traj_len > 0


def test_relative_paths_resolve_against_repo_root():
    cfg = load_config()
    assert cfg.paths.cache_dir == str(REPO_ROOT / "data" / "cache")


def test_override_is_typed_not_string():
    cfg = load_config(overrides=["shuttle.batch_size=4", "device=cpu"])
    assert cfg.shuttle.batch_size == 4
    assert cfg.device == "cpu"


def test_override_can_create_a_key():
    cfg = load_config(overrides=["shuttle.new_key=true"])
    assert cfg.shuttle.new_key is True


def test_malformed_override_rejected():
    with pytest.raises(ValueError):
        load_config(overrides=["shuttle.batch_size"])


def test_dotted_get_with_default():
    cfg = load_config()
    assert cfg.get("shuttle.nope", 7) == 7
    assert cfg.get("nope.nope", None) is None
