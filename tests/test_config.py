"""配置加载与覆盖测试（contracts 第 5 节）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from bacmarl.utils.config import load_config  # noqa: E402


def test_default_groups_merge():
    cfg = load_config([])
    assert cfg['env']['n_agents'] >= 1
    assert cfg['algo']['comm'] in ('none', 'ic3net', 'tarmac', 'bac')
    assert cfg['train']['hidden'] >= 8


def test_cli_override():
    cfg = load_config(['env=formation', 'algo=ippo', 'env.n_agents=9', 'algo.msg_bits=8', 'seed=3'])
    assert cfg['algo']['comm'] == 'none'
    assert cfg['env']['n_agents'] == 9
    assert cfg['algo']['msg_bits'] == 8
    assert cfg['seed'] == 3
