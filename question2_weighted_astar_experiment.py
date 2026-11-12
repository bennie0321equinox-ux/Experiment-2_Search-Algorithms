from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

from question2_weighted_terrain_astar import Cell, Coordinate, solve_terrain


@dataclass(frozen=True)
class WeightedAStarResult:
    weight: float
    cost: int
    path: list[Coordinate]
    expanded_nodes: int


def weighted_a_star(
    start: Coordinate,
    goal: Coordinate,
    grid: Sequence[Sequence[Cell]],
    weight: float,
) -> WeightedAStarResult:
    if weight < 1:
        raise ValueError("本实验只比较 weight >= 1 的情形")
    if not grid or not grid[0]:
        return WeightedAStarResult(weight, -1, [], 0)

    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("地图必须是矩形二维网格")

    def is_walkable(position: Coordinate) -> bool:
        row, col = position
        return 0 <= row < rows and 0 <= col < cols and grid[row][col] != "X"

    if not is_walkable(start) or not is_walkable(goal):
        return WeightedAStarResult(weight, -1, [], 0)

    terrain_costs = [
        int(cell)
        for row in grid
        for cell in row
        if cell not in ("X", "S", "E")
    ]
    if any(cost < 0 for cost in terrain_costs):
        raise ValueError("地形代价不能为负数")
    minimum_entry_cost = min(terrain_costs, default=0)

    def entry_cost(position: Coordinate) -> int:
        cell = grid[position[0]][position[1]]
        return 0 if position == goal or cell in ("S", "E") else int(cell)

    def heuristic(position: Coordinate) -> int:
        distance = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
        return max(0, distance - 1) * minimum_entry_cost

    sequence = count()
    frontier: list[tuple[float, int, int, Coordinate, int]] = []
    start_h = heuristic(start)
    heappush(frontier, (weight * start_h, start_h, next(sequence), start, 0))
    best_cost = {start: 0}
    came_from: dict[Coordinate, Coordinate] = {}
    expanded: set[Coordinate] = set()
    expanded_nodes = 0

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
            return WeightedAStarResult(weight, queued_cost, list(reversed(path)), expanded_nodes)

        for row_delta, col_delta in ((0, 1), (1, 0), (0, -1), (-1, 0)):
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
                (
                    new_cost + weight * neighbor_h,
                    neighbor_h,
                    next(sequence),
                    neighbor,
                    new_cost,
                ),
            )

    return WeightedAStarResult(weight, -1, [], expanded_nodes)


def run_experiment() -> list[WeightedAStarResult]:
    designed_map: list[list[Cell]] = [
        ["S", "X", 1, 9, 1, 1, 5, 1],
        [1, "X", 1, 1, 1, 1, 2, 5],
        [2, 1, 2, 2, 5, 2, 5, 2],
        [2, 2, 5, 5, 5, 2, 2, 2],
        [1, 2, 5, 5, 2, 2, 2, 1],
        [1, "X", 1, 2, 1, 9, 2, 1],
        [1, "X", 1, 2, 2, 1, 2, 5],
        [1, 9, 1, 5, 5, 1, 2, "E"],
    ]
    start, goal = (0, 0), (7, 7)

    dijkstra = solve_terrain(start, goal, designed_map, heuristic_mode="zero")
    standard = solve_terrain(start, goal, designed_map)
    assert dijkstra.cost == standard.cost == 23

    results = [weighted_a_star(start, goal, designed_map, weight) for weight in (1.0, 1.2, 1.5, 2.0)]
    assert results[0].cost == standard.cost

    print("创新实验：Weighted A* 的效率—最优性取舍")
    print(f"h=0（Dijkstra）：总代价={dijkstra.cost}，展开节点={dijkstra.expanded_nodes}")
    print(f"标准 A*：总代价={standard.cost}，展开节点={standard.expanded_nodes}")
    print("weight | 总代价 | 相对最优代价增加 | 展开节点数 | 是否保持最优")
    for result in results:
        extra_cost = result.cost - standard.cost
        optimal = "是" if extra_cost == 0 else "否"
        print(
            f"{result.weight:>6.1f} | {result.cost:>6} | {extra_cost:>14} |"
            f" {result.expanded_nodes:>10} | {optimal}"
        )
    print("\nw=1.0 的最优路径：", results[0].path)
    print("w=2.0 的更快但非最优路径：", results[-1].path)
    return results


if __name__ == "__main__":
    run_experiment()
