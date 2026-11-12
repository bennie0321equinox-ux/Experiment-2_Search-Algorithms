from __future__ import annotations

from math import inf

from question4_action_order_experiment import alpha_beta_ordered_decision
from question4_treasure_game import (
    GameState,
    Location,
    SearchStats,
    TREASURE_VALUES,
    build_distance_matrix,
    initial_state,
    minimax_decision,
    next_state,
)


def immediate_gain(
    state: GameState,
    target: Location,
    distances: dict[Location, dict[Location, int]],
) -> int:
    position = state.a_position if state.current_player == "A" else state.b_position
    return TREASURE_VALUES[target] - distances[position][target]


def threat_aware_targets(
    state: GameState, distances: dict[Location, dict[Location, int]]
) -> list[Location]:
    def priority(target: Location) -> int:
        after_move = next_state(state, target, distances)
        opponent_best_gain = max(
            (immediate_gain(after_move, item, distances) for item in after_move.remaining),
            default=0,
        )
        return immediate_gain(state, target, distances) - opponent_best_gain

    return sorted(state.remaining, key=priority, reverse=True)


def _alpha_beta_threat_aware(
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

    targets = threat_aware_targets(state, distances)
    if state.current_player == "A":
        value = -inf
        for index, target in enumerate(targets):
            value = max(
                value,
                _alpha_beta_threat_aware(
                    next_state(state, target, distances),
                    distances,
                    depth - 1,
                    alpha,
                    beta,
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
            _alpha_beta_threat_aware(
                next_state(state, target, distances),
                distances,
                depth - 1,
                alpha,
                beta,
                stats,
            ),
        )
        beta = min(beta, value)
        if alpha >= beta:
            stats.pruned_branches += len(targets) - index - 1
            break
    return int(value)


def alpha_beta_threat_aware_decision(
    distances: dict[Location, dict[Location, int]], depth: int
) -> tuple[dict[str, int | str], SearchStats]:
    root = initial_state()
    stats = SearchStats(visited_nodes=1)
    best_target = ""
    best_value = -inf
    alpha, beta = -inf, inf

    for target in threat_aware_targets(root, distances):
        value = _alpha_beta_threat_aware(
            next_state(root, target, distances),
            distances,
            depth - 1,
            alpha,
            beta,
            stats,
        )
        if value > best_value:
            best_target, best_value = target, value
        alpha = max(alpha, best_value)
    return {"best_target": best_target, "best_value": int(best_value)}, stats


def run_experiment() -> None:
    distances = build_distance_matrix()
    for depth in (2, 4):
        exact_result, _ = minimax_decision(distances, depth)
        original_result, original_stats = alpha_beta_ordered_decision(
            distances, depth, "original"
        )
        immediate_result, immediate_stats = alpha_beta_ordered_decision(
            distances, depth, "gain_desc"
        )
        threat_result, threat_stats = alpha_beta_threat_aware_decision(distances, depth)

        assert original_result == immediate_result == threat_result == exact_result
        print(f"H={depth}（精确结果：{exact_result}）")
        print("策略         | 访问节点 | 评估叶子 | 剪枝分支")
        for label, stats in (
            ("原始顺序", original_stats),
            ("即时收益排序", immediate_stats),
            ("威胁感知排序", threat_stats),
        ):
            print(
                f"{label:<10} | {stats.visited_nodes:^8} |"
                f" {stats.evaluated_states:^8} | {stats.pruned_branches:^8}"
            )


if __name__ == "__main__":
    run_experiment()
