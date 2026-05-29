"""Regression tests for config_loader.py — M2 path containment + core/overlay merge."""
from __future__ import annotations

from pathlib import Path

import pytest

import src.config_loader as config_loader
from src.config_loader import _deep_merge, _load_config, reload_config


@pytest.fixture(autouse=True)
def _restore_config_cache_after_test():
    """Prevent this module from leaking a core-only/polluted config cache into
    later test modules.

    These tests exercise `_load_config()` and the cache with the overlay env
    var deleted/redirected via monkeypatch. monkeypatch restores the env at
    teardown, but the module-global `src.config_loader._config` is NOT reverted
    — a core-only cache can persist. Several production modules read
    `get_config()["llm"]` at IMPORT time, so a later test module that first
    imports them would crash with `KeyError: 'llm'`. After each test here, with
    env already restored by monkeypatch, force a cache reload so the global
    cache reflects the (restored) environment.
    """
    yield
    reload_config()


class TestPathContainment:
    """M2: SIMULATION_CONFIG_PATH must stay within config/."""

    def test_rejects_path_outside_config_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        escape_path = tmp_path / "evil.toml"
        escape_path.write_text("")
        monkeypatch.setenv("SIMULATION_CONFIG_PATH", str(escape_path))
        with pytest.raises(ValueError, match="must be within"):
            _load_config()

    def test_rejects_traversal_attack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "SIMULATION_CONFIG_PATH",
            str(config_loader._CONFIG_DIR / ".." / ".." / "etc" / "passwd"),
        )
        with pytest.raises(ValueError, match="must be within"):
            _load_config()

    def test_accepts_override_inside_config_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "SIMULATION_CONFIG_PATH", str(config_loader._CONFIG_DIR / "core.toml")
        )
        cfg = _load_config()
        assert isinstance(cfg, dict)

    def test_default_path_loads_core_toml(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SIMULATION_CONFIG_PATH", raising=False)
        monkeypatch.delenv("SIMULATION_CONFIG_OVERLAY_PATH", raising=False)
        cfg = _load_config()
        assert isinstance(cfg, dict)
        assert "power" in cfg

    def test_overlay_config_merges(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = (tmp_path / "config").resolve()
        config_dir.mkdir()
        core_path = config_dir / "core.toml"
        overlay_path = config_dir / "overlay.toml"
        core_path.write_text("[power]\ndefault_alpha = 0.05\n", encoding="utf-8")
        overlay_path.write_text(
            "[power]\ntarget_power = 0.8\n[experiment]\ndefault_pairs = 12\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_CORE_CONFIG_PATH", core_path)
        monkeypatch.delenv("SIMULATION_CONFIG_PATH", raising=False)
        monkeypatch.setenv("SIMULATION_CONFIG_OVERLAY_PATH", str(overlay_path))

        cfg = _load_config()
        assert cfg["power"]["default_alpha"] == 0.05
        assert cfg["power"]["target_power"] == 0.8
        assert cfg["experiment"]["default_pairs"] == 12

    def test_missing_file_in_config_dir_raises_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "SIMULATION_CONFIG_PATH", str(config_loader._CONFIG_DIR / "nonexistent.toml")
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_config()


class TestDeepMerge:
    def test_flat_merge(self) -> None:
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}
        assert _deep_merge(base, overlay) == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        overlay = {"x": {"b": 3, "c": 4}}
        assert _deep_merge(base, overlay) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_base_unchanged(self) -> None:
        base = {"a": {"nested": 1}}
        _deep_merge(base, {"a": {"nested": 2}})
        assert base["a"]["nested"] == 1


class TestReloadConfig:
    def test_reload_returns_fresh_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SIMULATION_CONFIG_PATH", raising=False)
        monkeypatch.delenv("SIMULATION_CONFIG_OVERLAY_PATH", raising=False)
        cfg = reload_config()
        assert isinstance(cfg, dict)
