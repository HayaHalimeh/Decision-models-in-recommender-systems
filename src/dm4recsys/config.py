"""Configuration loading.

A run config is built from:
  1. configs/base.yaml
  2. configs/datasets/<name>.yaml   -> cfg["datasets"][<name>]
  3. configs/models/*.yaml          -> merged into cfg["models"]
  4. the run file passed on the CLI (default configs/server.yaml), deep-merged on top
  5. optional `--set a.b.c=value` overrides
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_value(s: str) -> Any:
    return yaml.safe_load(s)


def load_config(mode_file: str | os.PathLike | None = None, overrides: list[str] | None = None) -> dict:
    cfg = _read(CONFIG_DIR / "base.yaml")
    cfg.setdefault("datasets", {})
    for p in sorted((CONFIG_DIR / "datasets").glob("*.yaml")):
        cfg["datasets"][p.stem] = deep_merge(cfg["datasets"].get(p.stem, {}), _read(p))
    cfg.setdefault("models", {})
    for p in sorted((CONFIG_DIR / "models").glob("*.yaml")):
        cfg["models"] = deep_merge(cfg["models"], _read(p))
    if mode_file:
        mp = Path(mode_file)
        if not mp.is_absolute() and not mp.exists():
            mp = REPO_ROOT / mp
        cfg = deep_merge(cfg, _read(mp))
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    _resolve_paths(cfg)
    return cfg


def _resolve_paths(cfg: dict) -> None:
    """Make every entry in cfg['paths'] absolute (relative to the repo root)."""
    paths = cfg.setdefault("paths", {})
    for k, v in list(paths.items()):
        p = Path(str(v).format(mode=cfg.get("mode", "full")))
        paths[k] = str(p if p.is_absolute() else REPO_ROOT / p)


def path(cfg: dict, key: str, *parts: str) -> Path:
    p = Path(cfg["paths"][key]).joinpath(*parts)
    return p


def dataset_names(cfg: dict) -> list[str]:
    return list(cfg["run_datasets"])


def model_cfg(cfg: dict, name: str) -> dict:
    if name not in cfg["models"]:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(cfg['models'])}")
    m = copy.deepcopy(cfg["models"][name])
    m["name"] = name
    return m


def models_in_group(cfg: dict, group: str) -> list[str]:
    return [n for n, m in cfg["models"].items() if m.get("group") == group and m.get("enabled", True)]


def zero_shot_models(cfg: dict) -> list[str]:
    """Every model that is run by the generic runner (everything except trained sequential models)."""
    return [n for n, m in cfg["models"].items() if m.get("group") != "sequential" and m.get("enabled", True)]
