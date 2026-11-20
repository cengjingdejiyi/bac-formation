"""checkpoint 读写（负责人 C；续训所需的对偶/优化器/缩放字段由 A 补齐）。"""

from __future__ import annotations

from pathlib import Path


def save_checkpoint(
    path: str | Path,
    actor,
    critic,
    meta: dict | None = None,
    optimizer=None,
    extra: dict | None = None,
) -> Path:
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, p)
    return p


def load_checkpoint(
    path: str | Path,
    actor=None,
    critic=None,
    map_location: str = "cpu",
    optimizer=None,
):
    import torch

    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if actor is not None and "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    if critic is not None and "critic" in ckpt:
        critic.load_state_dict(ckpt["critic"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
