# MPE2 simple_formation 实测备忘（B 接手必读）

本文件记录 2026-11-13 在 Python 3.13 / numpy 2.5.4 环境下对 `mpe2==1.1.1` 的实测结果，
用于在写 `backend="mpe2"` 适配前对齐真实形状，避免按文档猜。

## 版本与依赖（已核实，不需要降 numpy）

| 包 | 版本 | 关键约束 |
|---|---|---|
| mpe2 | 1.1.1 | `Requires-Python>=3.10`；依赖 `pettingzoo>=1.24.3`、`pygame-ce>=2.3.0`、`scipy>=1.7.0` |
| pettingzoo | 1.27.0 | `Requires-Python<3.15,>=3.10`；`numpy>=1.21.0`（**无上限**） |

结论：numpy 2.x 与 mpe2 可以共存，本机 `numpy==2.5.4` 实装后 `simple_formation` 正常实例化并 step 通。
六个人统一用 `requirements.txt` 即可，不必为 mpe2 降到 1.26 以下。

## 入口（注意是 v1，不是 v3）

`mpe2` 顶层**没有** `simple_formation` 导出，必须按子模块导入：

```python
from mpe2 import simple_formation_v1 as sf   # 只有 v1，没有 simple_formation_v3
env = sf.parallel_env(N=6, max_cycles=25, continuous_actions=False)
```

可用构造签名（`raw_env`）：

```
raw_env(N: int = 4, max_cycles: int = 25, continuous_actions: bool = False,
        render_mode: str | None = None, dynamic_rescaling: bool = True,
        benchmark_data: bool = False, terminate_on_success: bool = False)
```

`terminate_on_success=False` 是默认值——这正好对上报告里「保持 H 步才算完成」的判定需求：
**不要**依赖环境自带的 terminate，完成判定在我们自己的 `envs/metrics.py` 里算。

## 实测形状（N=6）

| 项 | 实测值 | 与本骨架契约的差距（B 要补的映射） |
|---|---|---|
| `env.agents` | `['agent_0' … 'agent_5']` | 按序压成数组即可 |
| `obs` 单智能体 | `(6,)` float32 | 契约要求 `[N, 4 + 2*n_obs] = (16,)`：MPE2 原版**看不到队友**，邻居槽位必须自己从 `env.unwrapped.world.agents` 的位置算 |
| obs 前 6 维 | `[vx, vy, ?, ?, ?, ?]` | 需逐维核对语义（速度 / 自身位置 / 地标相对位置），前两维是速度 |
| `action_space` | `Discrete(5)` | 与契约一致（0 悬停 / 1 左 / 2 右 / 3 下 / 4 上），无需映射 |
| `reward` | 各智能体相同（共享团队奖励，实测 `-0.7082`） | 契约要求 `[N]` 逐机奖励，需广播后再叠加安全距离惩罚 |
| `step()` 返回 | 5 元组 `(obs, rew, term, trunc, info)` | PettingZoo 并行接口，照搬 |
| `info` | 空 dict（实测 `[]`） | 编队误差、碰撞数、完成标记必须我们自己算并塞进 `info` |
| 地标位置 | 一个中心 landmark（实测 `[0.357, -0.466]` 附近） | 编队中心 `p^c` 用它；报告里 p^c 来源就是这里，不需要另设 |
| 初始位置 | N 架散布在半径约 1 的圆内 | 与 mock 后端量级相近，参数可迁移 |

## 具体建议

1. `axis 顺序`：先按上面 6 维语义对齐，再补 `2*n_obs` 维邻居相对位置（用 `world.agents[i].state.p_pos` 算，注意 MPE2 的 `p_pos` 是绝对坐标，要转成 `pos_j - pos_i`）。
2. `comm_mask`：MPE2 没有通信概念，按 `comm_radius` + `recv_k` 自己算，和 mock 后端用同一段代码即可（建议抽成 `envs/metrics.py` 或独立函数共享）。
3. `以 MPE2 为唯一物理后端`，mock 只用来跑冒烟：两套后端共用同一份 `EnvStep` 组装代码，避免形状漂移。
4. 记得 `env.close()`；并行环境中每个 agent 的 `observation_space` 都要断言 shape，早失败比后面对不上强。
5. 若要做 `simple_spread` 通用性验证，模块名同样是 `simple_spread_v3`（这个有 v3）。
