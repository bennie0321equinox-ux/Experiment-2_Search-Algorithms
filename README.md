# 实验二 搜索算法

10243330403 戴贝妮

## 运行环境

- Python 3.10 及以上
- 题目一至题目四及所有对照实验仅使用 Python 标准库
- 重新生成报告图表时，额外安装 `numpy` 与 `matplotlib`

```bash
python3 -m pip install numpy matplotlib
```

在本目录中运行以下命令。代码不依赖外部数据文件，随机迷宫实验使用固定随机种子。

## 基础题

```bash
python3 question1_maze_astar.py
python3 question2_weighted_terrain_astar.py
python3 question3_minimax_alpha_beta.py
python3 question4_treasure_game.py
```

预期核心结果：题目一输出讲义给定的最短路径；题目二输出 `7`；题目三输出 `5`；题目四输出 6×6 距离矩阵，并在 H=2、H=4 时分别得到 `T2, -1` 和 `T3, 3`。

## 对照与拓展实验

```bash
python3 question1_heuristic_experiment.py
python3 question2_weighted_astar_experiment.py
python3 question3_child_order_experiment.py
python3 question4_action_order_experiment.py
python3 question4_threat_aware_ordering.py
```

| 文件 | 内容 |
| --- | --- |
| `question1_heuristic_experiment.py` | 比较 h=0 与曼哈顿距离在不同障碍密度下的展开节点数 |
| `question2_weighted_astar_experiment.py` | 比较标准 A* 与 Weighted A* 的节点数和路径代价 |
| `question3_child_order_experiment.py` | 比较不同子节点访问顺序的 Alpha-Beta 剪枝效果 |
| `question4_action_order_experiment.py` | 比较原始顺序、即时收益排序和反向排序 |
| `question4_threat_aware_ordering.py` | 比较威胁感知排序与两种基线排序 |

## 图表

```bash
python3 generate_report_figures.py
```

该命令用于生成 `figures/` 下，攥写实验报告时所用的 5 张图和 2 张表。图表中的数据与上述固定实验设置一致。
