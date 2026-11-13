"""配置加载：Hydra 风格三组配置 + CLI 覆盖，不依赖 hydra（负责人 C）。"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _read(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _read_group(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    """分组配置可 extends 同组配置；别名不复制数值，循环继承须报错。"""
    if path in ancestors:
        raise ValueError(f"cyclic config inheritance: {path.name}")
    cfg = _read(path)
    parent = cfg.pop("extends", None)
    if parent is not None:
        base = _read_group(path.with_name(f"{parent}.yaml"), ancestors + (path,))
        return _deep_merge(base, cfg)
    return cfg


def load_config(
    cli_tokens: list[str] | None = None, config_dir: Path | None = None
) -> dict:
    """``env=formation algo=bac`` 选择分组，``key=value`` 覆盖任意字段。"""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg = _read(cdir / "config.yaml")
    groups = dict(cfg.get("defaults", {}))
    overrides: dict = {}

    for tok in cli_tokens or []:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k in ("env", "algo", "experiment"):
            groups[k] = v
        else:
            overrides[k] = v

    for group, name in groups.items():
        p = cdir / group / f"{name}.yaml"
        if p.exists():
            _deep_merge(cfg, _read_group(p))

    for k, v in overrides.items():
        _set_path(cfg, k, _parse_scalar(v))
    return cfg


def _set_path(d: dict, path: str, value) -> None:
    parts = path.split(".") if "." in path else [path]
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _parse_scalar(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def apply_overrides(cfg: dict, tokens: list[str]) -> dict:
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k in ("env", "algo", "experiment"):
            continue
        _set_path(cfg, k, _parse_scalar(v))
    return cfg
