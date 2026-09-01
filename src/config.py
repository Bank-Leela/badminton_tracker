"""Configuration loading.

One YAML file, loaded once, passed down explicitly. No globals, no implicit
lookup of the active config from inside library code.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

# Keys under `paths:` are resolved against the repo root when relative.
_PATH_SECTION = "paths"


class Config:
    """Dict wrapper with attribute access and dotted-key lookup."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(f"no config key {name!r}") from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _resolve_paths(data: dict[str, Any]) -> dict[str, Any]:
    paths = data.get(_PATH_SECTION)
    if not isinstance(paths, dict):
        return data
    for key, value in paths.items():
        if isinstance(value, str):
            paths[key] = str(Path(value) if os.path.isabs(value) else REPO_ROOT / value)
    return data


def _apply_override(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override {dotted!r}: {part!r} is not a section")
    node[parts[-1]] = value


def _coerce(text: str) -> Any:
    """Parse an override value with YAML rules, so `8` is an int and `true` a bool."""
    return yaml.safe_load(text)


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> Config:
    """Load a config file, then apply `key.subkey=value` override strings."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        key, _, raw = override.partition("=")
        _apply_override(data, key.strip(), _coerce(raw.strip()))

    return Config(_resolve_paths(data))


def cache_dir(cfg: Config, match_id: str) -> Path:
    """Per-match cache directory, created on demand."""
    path = Path(cfg.paths.cache_dir) / match_id
    path.mkdir(parents=True, exist_ok=True)
    return path
