"""演示：跑一轮编队并渲染发言门控与注意力权重（负责人 F）。

两种跑法：

    # 1) 用训练好的模型（读运行目录里的 config.yaml 与 ckpt/last.ckpt）
    python render_demo.py --run results/main/formation_6n_bac_L4_K3_B2.0_s0 --out results/demo

    # 2) 不加载模型，用脚本控制器跑一轮（mock 后端，无需 torch/mpe2）
    python render_demo.py --out results/demo_scripted --backend mock --max-steps 80

输出（`<out>` 目录下）：

    demo.gif                 每帧叠加门控与注意力的动画（10 fps 默认）
    frames/frame_XXX.png     逐帧图片
    trajectory.npy           位置序列 [T, N, 2]（兼容旧调用）
    demo_stats.json          每步 gate / attn / bits / formation_err

缺 matplotlib 时只落 npy 与 json 并打印提示，不报错。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT, ROOT / "src"):
    if (candidate / "bacmarl").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402


def _flatten_attn(attn: np.ndarray, comm_mask: np.ndarray) -> np.ndarray:
    """把模块给出的注意力对齐成 [N, N]（i 收 j），并强制尊重通信掩码。"""
    a = np.asarray(attn, dtype=np.float32)
    if a.ndim == 3:  # 逐环境的 batch 维
        a = a[0]
    a = np.where(comm_mask, a, 0.0)
    return a


def _uniform_attn(comm_mask: np.ndarray) -> np.ndarray:
    """无模型时：可见邻居均分权重。"""
    w = comm_mask.astype(np.float32)
    denom = w.sum(axis=1, keepdims=True)
    return w / np.clip(denom, 1e-6, None)


def _rollout_scripted(env, max_steps: int, seed: int):
    """脚本控制器：门控全开、注意力可见邻居均分。"""
    from bacmarl.envs.scripted import ScriptedController

    cfg = env.cfg
    ctrl = ScriptedController(cfg.n_agents, cfg.formation_radius)
    step = env.reset(seed=seed)
    frames = []
    for _ in range(max_steps):
        gate = np.ones(cfg.n_agents, dtype=np.float32)
        frames.append(
            {
                "pos": np.asarray(step.info["pos"], dtype=np.float32).copy(),
                "center": np.asarray(step.info["center"], dtype=np.float32).copy(),
                "mask": np.asarray(step.comm_mask, dtype=bool).copy(),
                "gate": gate,
                "attn": _uniform_attn(step.comm_mask),
                "bits": float("nan"),
                "speak_rate": 1.0,
                "formation_err": float(step.info["formation_err"]),
                "success": bool(step.info["success"]),
                "t": int(step.info["t"]),
            }
        )
        if step.done:
            break
        step = env.step(ctrl.act(step.obs, step.obs_mask))
    return frames


def _rollout_actor(env, actor, torch, max_steps: int, seed: int):
    """确定性策略：记录门控与注意力（注意力用无噪信道重算，只看"想发给谁"）。"""
    cfg = env.cfg
    step = env.reset(seed=seed)
    torch.manual_seed(seed)
    hidden = actor.init_hidden(1, cfg.n_agents)
    frames = []
    for _ in range(max_steps):
        obs = torch.as_tensor(step.obs)[None]
        obs_mask = torch.as_tensor(step.obs_mask)[None]
        comm_mask = torch.as_tensor(step.comm_mask)[None]
        with torch.no_grad():
            out = actor(
                obs,
                obs_mask,
                comm_mask,
                hidden,
                {"drop_p": 0.0, "flip_p": 0.0},
                deterministic=True,
            )
            clean = actor.comm(
                out.h,
                comm_mask,
                out.gate,
                {"drop_p": 0.0, "flip_p": 0.0},
            )
        hidden = out.h
        frame = {
            "pos": np.asarray(step.info["pos"], dtype=np.float32).copy(),
            "center": np.asarray(step.info["center"], dtype=np.float32).copy(),
            "mask": np.asarray(step.comm_mask, dtype=bool).copy(),
            "gate": out.gate[0].numpy().astype(np.float32),
            "attn": _flatten_attn(clean.attn[0].numpy(), step.comm_mask),
            "bits": float(out.bits.mean()),
            "speak_rate": float(out.gate.mean()),
            "formation_err": float(step.info["formation_err"]),
            "success": bool(step.info["success"]),
            "t": int(step.info["t"]),
        }
        frames.append(frame)
        if step.done:
            break
        action = out.action_logits.argmax(-1)[0].numpy()
        step = env.step(action)
    return frames


def _render(
    frames, out_dir: Path, fps: int, title: str, radius: float = 0.5
) -> Path | None:
    """逐帧作图，返回 GIF 路径；缺依赖时返回 None。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 未装 matplotlib，跳过出图（{exc}）；pip install matplotlib")
        return None

    # 图例与坐标轴有中文：优先用系统里的中文字体，缺了就退回默认（会报 glyph warning）
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    images = []
    want_gif = True
    pos_all = np.stack([f["pos"] for f in frames]).reshape(-1, 2)
    lo = pos_all.min(axis=0) - 1.0
    hi = pos_all.max(axis=0) + 1.0
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=140)

    for idx, frame in enumerate(frames):
        pos, gate, attn = frame["pos"], frame["gate"], frame["attn"]
        n = pos.shape[0]
        ax.clear()
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_aspect("equal")
        ax.grid(alpha=0.15, linestyle=":")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # 编队目标：中心点 + 参考半径环（只作任务示意，不参与计算）
        center = np.asarray(frame.get("center", np.zeros(2)), dtype=np.float32)
        ax.add_patch(
            Circle(
                center,
                radius=radius,
                fill=False,
                linestyle="--",
                linewidth=0.8,
                edgecolor="#888888",
                alpha=0.6,
                zorder=1,
            )
        )
        ax.plot(center[0], center[1], marker="+", color="#888888", ms=8, zorder=2)

        # 通信边：透明度与线宽映射注意力权重（i 收 j 的 j→i 有向边）；
        # 允许通信但权重≈0 的边画成浅灰虚线，便于看出拓扑与注意力的区别
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w = float(attn[i, j])
                if w <= 1e-4:
                    if frame.get("mask") is not None and bool(frame["mask"][i, j]):
                        ax.plot(
                            [pos[j, 0], pos[i, 0]],
                            [pos[j, 1], pos[i, 1]],
                            color="#999999",
                            alpha=0.35,
                            linewidth=0.6,
                            linestyle=(0, (2, 2)),
                            zorder=1,
                        )
                    continue
                ax.annotate(
                    "",
                    xy=pos[i],
                    xytext=pos[j],
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#1f77b4",
                        alpha=min(0.9, 0.15 + 0.75 * w),
                        linewidth=0.6 + 2.4 * w,
                        shrinkA=6,
                        shrinkB=6,
                    ),
                )

        # 无人机：发言者实心、静默者空心；大小随门控值
        speaking = gate > 0.5
        ax.scatter(
            pos[speaking, 0],
            pos[speaking, 1],
            s=60 + 140 * gate[speaking],
            c="#d62728",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label="发言",
        )
        ax.scatter(
            pos[~speaking, 0],
            pos[~speaking, 1],
            s=60 + 140 * gate[~speaking],
            facecolors="none",
            edgecolors="#555555",
            linewidths=1.2,
            zorder=4,
            label="静默",
        )
        for i in range(n):
            ax.text(pos[i, 0] + 0.06, pos[i, 1] + 0.06, f"{gate[i]:.2f}", fontsize=6)

        bits = frame["bits"]
        bits_txt = "n/a" if not np.isfinite(bits) else f"{bits:.2f}"
        ax.set_title(title, fontsize=9)
        hud = (
            f"step {frame['t']:>3}  ferr {frame['formation_err']:.3f}\n"
            f"bits {bits_txt}  speak {frame['speak_rate']:.2f}  "
            f"{'success' if frame['success'] else 'running'}"
        )
        ax.text(
            0.02,
            0.98,
            hud,
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, linewidth=0.5),
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

        png = frames_dir / f"frame_{idx:03d}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=140)

        if want_gif:
            try:
                import io

                from PIL import Image

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100)
                buf.seek(0)
                images.append(Image.open(buf).convert("RGB").copy())
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] 无法收集 GIF 帧（{exc}）")
                want_gif = False

    plt.close(fig)

    if not images:
        return None
    gif_path = out_dir / "demo.gif"
    try:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=max(1, int(1000 / max(1, fps))),
            loop=0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] GIF 写入失败（{exc}）")
        return None
    return gif_path


def _load_actor(cfg: dict, run_dir: Path, torch):
    """按运行目录的 config.yaml 复建 Actor 并加载 ckpt。"""
    from bacmarl.comm.base import build_comm
    from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv
    from bacmarl.models.actor import Actor
    from bacmarl.utils.checkpoint import load_checkpoint

    hidden = int((cfg.get("train") or {}).get("hidden", 64))
    env_cfg = EnvConfig.from_dict(cfg.get("env", {}))
    comm = build_comm(cfg["algo"]["comm"], hidden, cfg["algo"])
    prot = FormationCommEnv(env_cfg)
    try:
        actor = Actor(prot.obs_dim, env_cfg.n_obs, 5, hidden, comm)
    finally:
        prot.close()
    ckpt = run_dir / "ckpt" / "last.ckpt"
    if not ckpt.exists():
        print(f"[WARN] 未找到 {ckpt}，退回脚本控制器")
        return None
    load_checkpoint(ckpt, actor=actor)
    actor.eval()
    print(f"[INFO] 已加载 {ckpt}")
    return actor


def _read_cfg(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    if not p.exists():
        raise SystemExit(f"[ERR] {run_dir} 里没有 config.yaml，别用 --run 或换个目录")
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="编队通信 Demo 渲染（F）")
    ap.add_argument("--run", default=None, help="运行目录（含 config.yaml 与 ckpt/last.ckpt）")
    ap.add_argument("--out", default=None, help="输出目录，默认 results/demo")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--backend", default="mock", choices=["mock", "mpe2"], help="无 --run 时的后端"
    )
    ap.add_argument("--n-agents", type=int, default=6)
    ap.add_argument("--no-scripted", action="store_true", help="有 --run 但不加载模型")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    from bacmarl.envs.formation_comm import EnvConfig, FormationCommEnv

    out_dir = Path(args.out) if args.out else ROOT / "results" / "demo"
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    actor = None
    cfg: dict = {}
    if args.run:
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = Path.cwd() / run_dir
        cfg = _read_cfg(run_dir)
        if not args.no_scripted:
            try:
                import torch

                actor = _load_actor(cfg, run_dir, torch)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] 加载模型失败（{exc}），退回脚本控制器")

    if actor is not None:
        env_cfg = EnvConfig.from_dict(cfg.get("env", {}))
        title = f"learned policy · {Path(args.run).name}"
        torch_obj = __import__("torch")
    else:
        env_cfg = EnvConfig.from_dict(
            {
                "backend": args.backend,
                "n_agents": args.n_agents,
                "max_steps": args.max_steps or 80,
            }
        )
        title = "scripted controller"
        torch_obj = None

    if args.max_steps:
        env_cfg.max_steps = int(args.max_steps)

    all_frames, traj, stats = [], [], []
    for ep in range(args.episodes):
        env = FormationCommEnv(env_cfg)
        try:
            if actor is not None:
                frames = _rollout_actor(
                    env, actor, torch_obj, env_cfg.max_steps, args.seed + ep
                )
            else:
                frames = _rollout_scripted(env, env_cfg.max_steps, args.seed + ep)
        finally:
            env.close()

        all_frames.extend(frames)
        traj.append(np.stack([f["pos"] for f in frames]))
        for f in frames:
            stats.append(
                {
                    "episode": ep,
                    "t": f["t"],
                    "formation_err": f["formation_err"],
                    "bits": None if not np.isfinite(f["bits"]) else f["bits"],
                    "speak_rate": f["speak_rate"],
                    "gate": [float(v) for v in f["gate"]],
                    "attn": [[float(v) for v in row] for row in f["attn"]],
                }
            )

    np.save(out_dir / "trajectory.npy", traj[0] if len(traj) == 1 else np.stack(traj))
    (out_dir / "demo_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    gif = _render(
        all_frames, out_dir, args.fps, title, float(env_cfg.formation_radius)
    )
    last = all_frames[-1] if all_frames else None
    if last is not None:
        print(
            f"[DONE] {len(all_frames)} 帧，末步编队误差 {last['formation_err']:.3f}"
            f"，speak_rate {last['speak_rate']:.2f}"
            + (f" → {gif}" if gif else f" → {out_dir}")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

