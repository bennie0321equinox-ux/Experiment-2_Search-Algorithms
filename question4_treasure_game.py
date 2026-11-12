from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf

Coordinate = tuple[int, int]
Location = str

MAP: list[list[str | int]] = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]

LOCATIONS: dict[Location, Coordinate] = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}
TREASURE_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
LOCATION_ORDER = ("A", "B", "T1", "T2", "T3", "T4")
TREASURE_ORDER = ("T1", "T2", "T3", "T4")


def a_star_cost(start: Coordinate, goal: Coordinate, grid: list[list[str | int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    minimum_cost = min(
        cell for row in grid for cell in row if isinstance(cell, int) and not isinstance(cell, bool)
    )

    def walkable(position: Coordinate) -> bool:
        row, col = position
        return 0 <= row < rows and 0 <= col < cols and grid[row][col] != "X"

    def entry_cost(position: Coordinate) -> int:
        cell = grid[position[0]][position[1]]
        return 1 if isinstance(cell, str) else cell

    def heuristic(position: Coordinate) -> int:
        return (
            abs(position[0] - goal[0]) + abs(position[1] - goal[1])
        ) * minimum_cost

    if not walkable(start) or not walkable(goal):
        return -1

    sequence = count()
    frontier: list[tuple[int, int, Coordinate, int]] = []
    heappush(frontier, (heuristic(start), next(sequence), start, 0))
    best_cost = {start: 0}

    while frontier:
        _, _, current, current_cost = heappop(frontier)
        if current_cost != best_cost.get(current):
            continue
        if current == goal:
            return current_cost

        for row_delta, col_delta in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            neighbor = (current[0] + row_delta, current[1] + col_delta)
            if not walkable(neighbor):
                continue
            new_cost = current_cost + entry_cost(neighbor)
            if new_cost >= best_cost.get(neighbor, inf):
                continue
            best_cost[neighbor] = new_cost
            heappush(
                frontier,
                (new_cost + heuristic(neighbor), next(sequence), neighbor, new_cost),
            )
    return -1


def build_distance_matrix() -> dict[Location, dict[Location, int]]:
    return {
        source: {
            destination: a_star_cost(LOCATIONS[source], LOCATIONS[destination], MAP)
            for destination in LOCATION_ORDER
        }
        for source in LOCATION_ORDER
    }


@dataclass(frozen=True)
class GameState:
    a_position: Location
    b_position: Location
    remaining: tuple[Location, ...]
    score_a: int
    score_b: int
    current_player: str

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


@dataclass
class SearchStats:
    visited_nodes: int = 0
    evaluated_states: int = 0
    pruned_branches: int = 0


def initial_state() -> GameState:
    return GameState("A", "B", TREASURE_ORDER, 0, 0, "A")


def next_state(
    state: GameState,
    target: Location,
    distances: dict[Location, dict[Location, int]],
) -> GameState:
    if target not in state.remaining:
        raise ValueError(f"宝藏 {target} 已被获取或不存在")

    remaining = tuple(item for item in state.remaining if item != target)
    if state.current_player == "A":
        gain = TREASURE_VALUES[target] - distances[state.a_position][target]
        return GameState(
            target,
            state.b_position,
            remaining,
            state.score_a + gain,
            state.score_b,
            "B",
        )

    gain = TREASURE_VALUES[target] - distances[state.b_position][target]
    return GameState(
        state.a_position,
        target,
        remaining,
        state.score_a,
        state.score_b + gain,
        "A",
    )


def _minimax(
    state: GameState,
    distances: dict[Location, dict[Location, int]],
    depth: int,
    stats: SearchStats,
) -> int:
    stats.visited_nodes += 1
    if depth == 0 or not state.remaining:
        stats.evaluated_states += 1
        return state.utility

    child_values = [
        _minimax(next_state(state, target, distances), distances, depth - 1, stats)
        for target in state.remaining
    ]
    return max(child_values) if state.current_player == "A" else min(child_values)


def minimax_decision(
    distances: dict[Location, dict[Location, int]], depth: int
) -> tuple[dict[str, int | str], SearchStats]:
    state = initial_state()
    stats = SearchStats()
    best_target = ""
    best_value = -inf
    for target in state.remaining:
        value = _minimax(next_state(state, target, distances), distances, depth - 1, stats)
        if value > best_value:
            best_target, best_value = target, value
    return {"best_target": best_target, "best_value": int(best_value)}, stats


def _alpha_beta(
    state: GameState,
    distances: dict[Location, dict[Location, int]],
    depth: int,
    alpha: float,
    beta: float,
    stats: SearchStats,
) -> int:
    stats.visited_nodes += 1
    if depth == 0 or not state.remaining:
        stats.evaluated_states += 1
        return state.utility

    if state.current_player == "A":
        value = -inf
        for index, target in enumerate(state.remaining):
            value = max(
                value,
                _alpha_beta(
                    next_state(state, target, distances), distances, depth - 1, alpha, beta, stats
                ),
            )
            alpha = max(alpha, value)
            if alpha >= beta:
                stats.pruned_branches += len(state.remaining) - index - 1
                break
        return int(value)

    value = inf
    for index, target in enumerate(state.remaining):
        value = min(
            value,
            _alpha_beta(
                next_state(state, target, distances), distances, depth - 1, alpha, beta, stats
            ),
        )
        beta = min(beta, value)
        if alpha >= beta:
            stats.pruned_branches += len(state.remaining) - index - 1
            break
    return int(value)


def alpha_beta_decision(
    distances: dict[Location, dict[Location, int]], depth: int
) -> tuple[dict[str, int | str], SearchStats]:
    state = initial_state()
    stats = SearchStats()
    best_target = ""
    best_value = -inf
    alpha, beta = -inf, inf
    for target in state.remaining:
        value = _alpha_beta(
            next_state(state, target, distances), distances, depth - 1, alpha, beta, stats
        )
        if value > best_value:
            best_target, best_value = target, value
        alpha = max(alpha, best_value)
    return {"best_target": best_target, "best_value": int(best_value)}, stats


def print_distance_matrix(distances: dict[Location, dict[Location, int]]) -> None:
    print("      " + "  ".join(f"{name:>2}" for name in LOCATION_ORDER))
    for source in LOCATION_ORDER:
        values = "  ".join(f"{distances[source][target]:>2}" for target in LOCATION_ORDER)
        print(f"{source:>2}  {values}")


if __name__ == "__main__":
    distances = build_distance_matrix()
    print_distance_matrix(distances)
    for depth in (2, 4):
        minimax_result, _ = minimax_decision(distances, depth)
        alpha_beta_result, _ = alpha_beta_decision(distances, depth)
        assert minimax_result == alpha_beta_result
        print(f"H={depth} Minimax    => {minimax_result}")
        print(f"H={depth} Alpha-Beta => {alpha_beta_result}")
