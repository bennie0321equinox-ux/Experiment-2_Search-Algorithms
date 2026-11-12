from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from typing import Sequence

Coordinate = tuple[int, int]


def _search(
    start: Coordinate,
    goal: Coordinate,
    grid: Sequence[Sequence[str]],
    use_heuristic: bool,
) -> tuple[list[Coordinate], int]:
    if not grid or not grid[0]:
        return [], 0

    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("地图必须是矩形二维网格")

    def is_walkable(position: Coordinate) -> bool:
        row, col = position
        return (
            0 <= row < rows
            and 0 <= col < cols
            and grid[row][col] != "X"
        )

    if not is_walkable(start) or not is_walkable(goal):
        return [], 0

    def heuristic(position: Coordinate) -> int:
        if not use_heuristic:
            return 0
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    sequence = count()
    frontier: list[tuple[int, int, int, Coordinate, int]] = []
    start_h = heuristic(start)
    heappush(frontier, (start_h, start_h, next(sequence), start, 0))

    came_from: dict[Coordinate, Coordinate] = {}
    best_cost = {start: 0}
    expanded: set[Coordinate] = set()
    expanded_count = 0
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))

    while frontier:
        _, _, _, current, queued_cost = heappop(frontier)
        if queued_cost != best_cost.get(current) or current in expanded:
            continue

        expanded.add(current)
        expanded_count += 1
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, expanded_count

        for row_delta, col_delta in directions:
            neighbor = (current[0] + row_delta, current[1] + col_delta)
            if not is_walkable(neighbor) or neighbor in expanded:
                continue

            new_cost = queued_cost + 1
            if new_cost >= best_cost.get(neighbor, float("inf")):
                continue

            best_cost[neighbor] = new_cost
            came_from[neighbor] = current
            neighbor_h = heuristic(neighbor)
            heappush(
                frontier,
                (
                    new_cost + neighbor_h,
                    neighbor_h,
                    next(sequence),
                    neighbor,
                    new_cost,
                ),
            )

    return [], expanded_count


def a_star_maze(
    start: Coordinate,
    goal: Coordinate,
    grid: Sequence[Sequence[str]],
) -> list[Coordinate]:
    path, _ = _search(start, goal, grid, use_heuristic=True)
    return path


def compare_heuristics(
    start: Coordinate,
    goal: Coordinate,
    grid: Sequence[Sequence[str]],
) -> dict[str, int]:
    dijkstra_expanded = _search(start, goal, grid, use_heuristic=False)[1]
    astar_expanded = _search(start, goal, grid, use_heuristic=True)[1]
    return {
        "h=0": dijkstra_expanded,
        "曼哈顿距离": astar_expanded,
    }


if __name__ == "__main__":
    maze = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    start, goal = (0, 0), (3, 4)

    print(a_star_maze(start, goal, maze))
