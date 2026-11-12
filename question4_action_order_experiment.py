from __future__ import annotations

from math import inf
from typing import Literal

from question4_treasure_game import (
    GameState,
    Location,
    SearchStats,
    TREASURE_VALUES,
    alpha_beta_decision,
    build_distance_matrix,
    initial_state,
    minimax_decision,
    next_state,
)

OrderMode = Literal["original", "gain_desc", "gain_asc"]


def ordered_targets(
    state: GameState,
    distances: dict[Location, dict[Location, int]],
    mode: OrderMode,
) -> list[Location]:
    targets = list(state.remaining)
    if mode == "original":
        return targets

    position = state.a_position if state.current_player == "A" else state.b_position
    targets.sort(
        key=lambda target: TREASURE_VALUES[target] - distances[position][target],
        reverse=mode == "gain_desc",
    )
    return targets


def _alpha_beta_with_order(
    state: GameState,
    distances: dict[Location, dict[Location, int]],
    depth: int,
    alpha: float,
    beta: float,
    mode: OrderMode,
    stats: SearchStats,
) -> int:
    stats.visited_nodes += 1
    if depth == 0 or not state.remaining:
        stats.evaluated_states += 1
        return state.utility

    targets = ordered_targets(state, distances, mode)
    if state.current_player == "A":
        value = -inf
        for index, target in enumerate(targets):
            value = max(
                value,
                _alpha_beta_with_order(
                    next_state(state, target, distances),
                    distances,
                    depth - 1,
                    alpha,
                    beta,
                    mode,
                    stats,
                ),
            )
            alpha = max(alpha, value)
            if alpha >= beta:
                stats.pruned_branches += len(targets) - index - 1
                break
        return int(value)

    value = inf
    for index, target in enumerate(targets):
        value = min(
            value,
            _alpha_beta_with_order(
                next_state(state, target, distances),
                distances,
                depth - 1,
                alpha,
                beta,
                mode,
                stats,
            ),
        )
        beta = min(beta, value)
        if alpha >= beta:
            stats.pruned_branches += len(targets) - index - 1
            break
    return int(value)


def alpha_beta_ordered_decision(
    distances: dict[Location, dict[Location, int]], depth: int, mode: OrderMode
) -> tuple[dict[str, int | str], SearchStats]:
    root = initial_state()
    stats = SearchStats(visited_nodes=1)
    best_target = ""
    best_value = -inf
    alpha, beta = -inf, inf

    for target in ordered_targets(root, distances, mode):
        value = _alpha_beta_with_order(
            next_state(root, target, distances),
            distances,
            depth - 1,
            alpha,
            beta,
            mode,
            stats,
        )
        if value > best_value:
            best_target, best_value = target, value
        alpha = max(alpha, best_value)

    return {"best_target": best_target, "best_value": int(best_value)}, stats


def run_experiment() -> None:
    distances = build_distance_matrix()
    modes: tuple[tuple[str, OrderMode], ...] = (
        ("原始顺序", "original"),
        ("收益优先", "gain_desc"),
        ("收益反向", "gain_asc"),
    )

    for depth in (2, 4):
        minimax_result, _ = minimax_decision(distances, depth)
        baseline_result, _ = alpha_beta_decision(distances, depth)
        assert baseline_result == minimax_result

        print(f"H={depth}（Minimax 基准：{minimax_result}）")
        print("顺序     | 最优首步 | 最优收益 | 访问节点 | 评估叶子 | 剪枝分支")
        for label, mode in modes:
            result, stats = alpha_beta_ordered_decision(distances, depth, mode)
            if result != minimax_result:
                raise AssertionError("行动排序不应改变 Alpha-Beta 的最优决策")
            print(
                f"{label:<6} | {result['best_target']:^8} | {result['best_value']:^8} |"
                f" {stats.visited_nodes:^8} | {stats.evaluated_states:^8} |"
                f" {stats.pruned_branches:^8}"
            )


if __name__ == "__main__":
    run_experiment()
