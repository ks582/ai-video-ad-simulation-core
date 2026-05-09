"""Singleton TOML config loader.

Loads ``config/core.toml`` unconditionally. Optionally merges a
deployment-specific overlay when ``SIMULATION_CONFIG_OVERLAY_PATH`` is set.
The merged dict is exposed via :func:`get_config`.

The config path is resolved relative to the project root (two levels
above this file: ``src/config_loader.py`` → project root).
``SIMULATION_CONFIG_PATH`` can point at a single combined file instead.
Both env-configured paths must stay under ``config/``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = (_PROJECT_ROOT / "config").resolve()
_CORE_CONFIG_PATH = _CONFIG_DIR / "core.toml"
_CONFIG_OVERRIDE_ENV = "SIMULATION_CONFIG_PATH"
_CONFIG_OVERLAY_ENV = "SIMULATION_CONFIG_OVERLAY_PATH"

_config: dict[str, Any] | None = None


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into a shallow copy of *base*."""
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_config() -> dict[str, Any]:
    """Return the parsed TOML config dict (cached after first load)."""
    global _config  # noqa: PLW0603
    if _config is None:
        _config = _load_config()
    return _config


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a single TOML file with path-containment check."""
    path = path.resolve()
    if not path.is_relative_to(_CONFIG_DIR):
        raise ValueError(f"Config path must be within {_CONFIG_DIR}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def _resolve_env_config_path(env_var: str) -> Path | None:
    """Resolve an optional env-configured TOML path under config/."""
    raw_path = os.environ.get(env_var, "")
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    if not path.is_relative_to(_CONFIG_DIR):
        raise ValueError(f"Config path must be within {_CONFIG_DIR}: {path}")
    return path


def _load_config() -> dict[str, Any]:
    """Load core.toml plus an optional deployment-specific overlay.

    ``SIMULATION_CONFIG_PATH`` overrides the default core-plus-overlay merge and
    loads a single file instead (for developer overrides / tests).
    The resolved path must stay under ``config/`` (M2 — path containment).
    """
    override = _resolve_env_config_path(_CONFIG_OVERRIDE_ENV)
    if override is not None:
        return _load_toml(override)

    config = _load_toml(_CORE_CONFIG_PATH)

    overlay = _resolve_env_config_path(_CONFIG_OVERLAY_ENV)
    if overlay is not None:
        config = _deep_merge(config, _load_toml(overlay))

    return config


def reload_config() -> dict[str, Any]:
    """Force-reload the config from disk. Useful for testing."""
    global _config  # noqa: PLW0603
    _config = None
    return get_config()
