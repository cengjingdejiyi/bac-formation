"""checkpoint 读写（负责人 C）。"""
from __future__ import annotations

from pathlib import Path


def save_checkpoint(path: str | Path, actor, critic, meta: dict | None = None) -> Path:
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(), 'meta': meta or {}}, p)
    return p


def load_checkpoint(path: str | Path, actor=None, critic=None, map_location='cpu'):
    import torch

    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if actor is not None and 'actor' in ckpt:
        actor.load_state_dict(ckpt['actor'])
    if critic is not None and 'critic' in ckpt:
        critic.load_state_dict(ckpt['critic'])
    return ckpt
