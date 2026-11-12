from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Literal

Coordinate = tuple[int, int]
Cell = str | int
HeuristicMode = Literal["scaled_manhattan", "zero"]


@dataclass(frozen=True)
class TerrainSearchResult:
    cost: int
    path: list[Coordinate]
    expanded_nodes: int


def solve_terrain(
    start: Coordinate,
    goal: Coordinate,
    grid: Sequence[Sequence[Cell]],
    *,
    heuristic_mode: HeuristicMode = "scaled_manhattan",
) -> TerrainSearchResult:
    if heuristic_mode not in ("scaled_manhattan", "zero"):
        raise ValueError("heuristic_mode 只能是 'scaled_manhattan' 或 'zero'")
    if not grid or not grid[0]:
        return TerrainSearchResult(-1, [], 0)

    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("地图必须是矩形二维网格")

    terrain_costs: list[int] = []
    for row in grid:
        for cell in row:
            if cell in ("X", "S", "E"):
                continue
            if isinstance(cell, bool):
                raise ValueError(f"无效的地形代价: {cell!r}")
            try:
                cost = int(cell)
            except (TypeError, ValueError) as error:
                raise ValueError(f"无效的地形代价: {cell!r}") from error
            if cost < 0:
                raise ValueError("地形代价不能为负数")
            terrain_costs.append(cost)

    def is_walkable(position: Coordinate) -> bool:
        row, col = position
        return 0 <= row < rows and 0 <= col < cols and grid[row][col] != "X"

    if not is_walkable(start) or not is_walkable(goal):
        return TerrainSearchResult(-1, [], 0)

    minimum_entry_cost = min(terrain_costs, default=0)

    def entry_cost(position: Coordinate) -> int:
        cell = grid[position[0]][position[1]]
        if position == goal or cell in ("S", "E"):
            return 0
        return int(cell)

    def heuristic(position: Coordinate) -> int:
        if heuristic_mode == "zero":
            return 0
        distance = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
        return max(0, distance - 1) * minimum_entry_cost

    sequence = count()
    frontier: list[tuple[int, int, int, Coordinate, int]] = []
    start_h = heuristic(start)
    heappush(frontier, (start_h, start_h, next(sequence), start, 0))

    came_from: dict[Coordinate, Coordinate] = {}
    best_cost = {start: 0}
    expanded: set[Coordinate] = set()
    expanded_nodes = 0
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))

    while frontier:
        _, _, _, current, queued_cost = heappop(frontier)
        if queued_cost != best_cost.get(current) or current in expanded:
            continue

        expanded.add(current)
        expanded_nodes += 1
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return TerrainSearchResult(queued_cost, path, expanded_nodes)

        for row_delta, col_delta in directions:
            neighbor = (current[0] + row_delta, current[1] + col_delta)
            if not is_walkable(neighbor) or neighbor in expanded:
                continue

            new_cost = queued_cost + entry_cost(neighbor)
            if new_cost >= best_cost.get(neighbor, float("inf")):
                continue

            best_cost[neighbor] = new_cost
            came_from[neighbor] = current
            neighbor_h = heuristic(neighbor)
            heappush(
                frontier,
                (new_cost + neighbor_h, neighbor_h, next(sequence), neighbor, new_cost),
            )

    return TerrainSearchResult(-1, [], expanded_nodes)


def a_star_terrain_cost(
    start: Coordinate,
    goal: Coordinate,
    grid: Sequence[Sequence[Cell]],
) -> int:
    return solve_terrain(start, goal, grid).cost


def compare_heuristics(
    start: Coordinate,
    goal: Coordinate,
    grid: Sequence[Sequence[Cell]],
) -> dict[str, int]:
    dijkstra = solve_terrain(start, goal, grid, heuristic_mode="zero")
    astar = solve_terrain(start, goal, grid, heuristic_mode="scaled_manhattan")
    if dijkstra.cost != astar.cost:
        raise AssertionError("可采纳启发函数不应改变最优总代价")
    return {
        "最优总代价": astar.cost,
        "h=0 展开节点数": dijkstra.expanded_nodes,
        "缩放曼哈顿距离 展开节点数": astar.expanded_nodes,
    }


if __name__ == "__main__":
    weighted_map = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    print(a_star_terrain_cost((0, 0), (4, 4), weighted_map))
