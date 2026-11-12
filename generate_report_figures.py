from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from question4_treasure_game import LOCATION_ORDER, build_distance_matrix

OUTPUT = Path(__file__).with_name("figures")
OUTPUT.mkdir(exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 150


def save(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(OUTPUT / filename, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_grid(
    axis: plt.Axes,
    grid: list[list[str | int]],
    path: list[tuple[int, int]],
    title: str,
    cumulative_costs: list[int] | None = None,
) -> None:
    rows, cols = len(grid), len(grid[0])
    terrain_colors = {"X": "#3c4655", "S": "#81c995", "E": "#f28b82", "A": "#81c995", "B": "#f28b82"}
    for row in range(rows):
        for col in range(cols):
            cell = grid[row][col]
            if cell == "X":
                color = terrain_colors["X"]
            elif isinstance(cell, int):
                color = {1: "#dfefff", 3: "#c8ddff", 8: "#8db6f2", 9: "#789fe0"}.get(cell, "#dfefff")
            else:
                color = terrain_colors.get(cell, "#fff3cd")
            axis.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="#ffffff", linewidth=1.3))
            text_color = "white" if cell == "X" else "#1f2937"
            axis.text(col + 0.5, row + 0.45, str(cell), ha="center", va="center", color=text_color, fontsize=10, weight="medium")

    xs = [col + 0.5 for row, col in path]
    ys = [row + 0.5 for row, col in path]
    axis.plot(xs, ys, color="#e65100", linewidth=2.5, marker="o", markersize=4, zorder=3)
    if cumulative_costs:
        for (row, col), cost in zip(path, cumulative_costs):
            axis.text(col + 0.77, row + 0.77, f"g={cost}", ha="center", va="center", color="#7c2d12", fontsize=7, zorder=4)

    axis.set_xlim(0, cols)
    axis.set_ylim(rows, 0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=12, weight="medium")


def figure_paths() -> None:
    maze = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    maze_path = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    terrain = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    terrain_path = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (4, 3), (4, 4)]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    draw_grid(axes[0], maze, maze_path, "题一：最短路径（步数 7）")
    draw_grid(axes[1], terrain, terrain_path, "题二：最低代价路径（总代价 7）", [0, 1, 2, 3, 4, 5, 6, 7, 7])
    save(figure, "figure-01-astar-paths.png")


def figure_maze_density() -> None:
    density = np.array([10, 20, 30])
    zero = np.array([360.6, 318.7, 253.9])
    manhattan = np.array([59.5, 68.0, 88.3])
    figure, axis = plt.subplots(figsize=(6.6, 4.3))
    axis.plot(density, zero, marker="o", linewidth=2.2, label="h=0（UCS / Dijkstra）", color="#4e79a7")
    axis.plot(density, manhattan, marker="o", linewidth=2.2, label="曼哈顿启发函数", color="#e15759")
    for x, y in zip(density, zero):
        axis.annotate(f"{y:.1f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    for x, y in zip(density, manhattan):
        axis.annotate(f"{y:.1f}", (x, y), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9)
    axis.set_xlabel("障碍物密度（%）")
    axis.set_ylabel("可解迷宫中的平均展开节点数")
    axis.set_xticks(density)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("题一：障碍密度下的启发函数搜索开销")
    save(figure, "figure-02-maze-density.png")


def figure_weighted_tradeoff() -> None:
    labels = ["h=0", "w=1.0", "w=1.2", "w=1.5", "w=2.0"]
    nodes = [57, 47, 45, 42, 25]
    costs = [23, 23, 23, 23, 27]
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    colors = ["#9aa5b1", "#4e79a7", "#4e79a7", "#4e79a7", "#e15759"]
    axes[0].bar(labels, nodes, color=colors)
    axes[0].set_title("搜索开销")
    axes[0].set_ylabel("展开节点数")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, costs, color=colors)
    axes[1].set_title("路径总代价")
    axes[1].set_ylabel("总代价")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("题二：Weighted A* 的效率—最优性取舍", fontsize=13, weight="medium")
    save(figure, "figure-03-weighted-astar-tradeoff.png")


def figure_alpha_beta_order() -> None:
    labels = ["Minimax", "Alpha-Beta\n左到右", "Alpha-Beta\n右到左"]
    nodes = [40, 20, 40]
    leaves = [27, 11, 27]
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    colors = ["#9aa5b1", "#59a14f", "#e15759"]
    axes[0].bar(labels, nodes, color=colors)
    axes[0].set_title("访问节点数")
    axes[0].set_ylabel("节点数")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, leaves, color=colors)
    axes[1].set_title("评估叶子数")
    axes[1].set_ylabel("叶子数")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("题三：访问顺序决定 Alpha-Beta 的剪枝效果", fontsize=13, weight="medium")
    save(figure, "figure-04-alpha-beta-order.png")


def figure_game_ordering() -> None:
    labels = ["原始顺序", "即时收益", "威胁感知"]
    h2 = [13, 11, 11]
    h4 = [47, 37, 39]
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.1, 4.3))
    bars_h2 = axis.bar(x - width / 2, h2, width, label="H=2", color="#4e79a7")
    bars_h4 = axis.bar(x + width / 2, h4, width, label="H=4", color="#f28e2b")
    for bars in (bars_h2, bars_h4):
        axis.bar_label(bars, padding=3, fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Alpha-Beta 访问节点数")
    axis.set_title("题四：行动排序的剪枝效率")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save(figure, "figure-05-game-action-ordering.png")


def table_distance_matrix() -> None:
    distances = build_distance_matrix()
    values = [[distances[source][target] for target in LOCATION_ORDER] for source in LOCATION_ORDER]
    figure, axis = plt.subplots(figsize=(6.3, 3.5))
    axis.axis("off")
    table = axis.table(
        cellText=values,
        rowLabels=LOCATION_ORDER,
        colLabels=LOCATION_ORDER,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.45)
    axis.set_title("题四：A、B 与四个宝藏的最短移动代价矩阵", pad=14, fontsize=12, weight="medium")
    save(figure, "table-01-distance-matrix.png")


def table_depth_decisions() -> None:
    figure, axis = plt.subplots(figsize=(6.3, 2.5))
    axis.axis("off")
    table = axis.table(
        cellText=[
            ["H=2", "T2", "-1", "T2，-1"],
            ["H=4", "T3", "3", "T3，3"],
        ],
        colLabels=["搜索深度", "Minimax 首步", "最优收益差", "Alpha-Beta 结果"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.55)
    axis.set_title("题四：搜索深度改变最优首步", pad=14, fontsize=12, weight="medium")
    save(figure, "table-02-depth-decisions.png")


if __name__ == "__main__":
    figure_paths()
    figure_maze_density()
    figure_weighted_tradeoff()
    figure_alpha_beta_order()
    figure_game_ordering()
    table_distance_matrix()
    table_depth_decisions()
    print(f"已生成 7 个图表：{OUTPUT}")
