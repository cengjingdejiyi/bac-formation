"""Demo 渲染冒烟测试（负责人 F）。

缺 matplotlib 时只验证数据落盘；装了才断言 GIF 帧数。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import render_demo  # noqa: E402


def _has_mpl() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def test_scripted_demo_writes_data(tmp_path):
    out = tmp_path / "demo"
    assert (
        render_demo.main(["--out", str(out), "--backend", "mock", "--max-steps", "4"])
        == 0
    )
    assert (out / "trajectory.npy").exists()
    stats = (out / "demo_stats.json").read_text(encoding="utf-8")
    assert "speak_rate" in stats and "attn" in stats


@pytest.mark.skipif(not _has_mpl(), reason="未装 matplotlib")
def test_scripted_demo_writes_gif(tmp_path):
    out = tmp_path / "demo"
    assert (
        render_demo.main(["--out", str(out), "--backend", "mock", "--max-steps", "4"])
        == 0
    )
    gif = out / "demo.gif"
    assert gif.exists() and gif.stat().st_size > 0
    frames = sorted((out / "frames").glob("frame_*.png"))
    assert len(frames) >= 1
