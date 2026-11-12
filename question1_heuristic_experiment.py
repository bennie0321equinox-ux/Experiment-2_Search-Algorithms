from __future__ import annotations

from random import Random

from question1_maze_astar import _search

Coordinate = tuple[int, int]


def generate_maze(
    rows: int, cols: int, obstacle_rate: float, rng: Random
) -> tuple[list[list[str]], Coordinate, Coordinate]:
    start, goal = (0, 0), (rows - 1, cols - 1)
    maze = [
        ["X" if rng.random() < obstacle_rate else "." for _ in range(cols)]
        for _ in range(rows)
    ]
    maze[start[0]][start[1]] = "S"
    maze[goal[0]][goal[1]] = "E"
    return maze, start, goal


def run_experiment() -> None:
    rows = cols = 20
    samples_per_rate = 50
    obstacle_rates = (0.10, 0.20, 0.30)

    print("障碍率 | 可解比例 | h=0平均展开节点 | 曼哈顿平均展开节点 | 平均减少")
    for obstacle_rate in obstacle_rates:
        rng = Random(20261105 + int(obstacle_rate * 100))
        solved = 0
        zero_expanded: list[int] = []
        manhattan_expanded: list[int] = []

        for _ in range(samples_per_rate):
            maze, start, goal = generate_maze(rows, cols, obstacle_rate, rng)
            zero_path, zero_nodes = _search(start, goal, maze, use_heuristic=False)
            manhattan_path, manhattan_nodes = _search(
                start, goal, maze, use_heuristic=True
            )
            if not zero_path:
                assert not manhattan_path
                continue
            assert len(zero_path) == len(manhattan_path)
            solved += 1
            zero_expanded.append(zero_nodes)
            manhattan_expanded.append(manhattan_nodes)

        average_zero = sum(zero_expanded) / solved
        average_manhattan = sum(manhattan_expanded) / solved
        reduction = (1 - average_manhattan / average_zero) * 100
        print(
            f"{obstacle_rate:>5.0%} | {solved:>2}/{samples_per_rate:<2}   |"
            f" {average_zero:>15.1f} | {average_manhattan:>20.1f} |"
            f" {reduction:>6.1f}%"
        )


if __name__ == "__main__":
    run_experiment()
