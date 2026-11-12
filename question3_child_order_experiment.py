from __future__ import annotations

from question3_minimax_alpha_beta import TreeNode, alpha_beta, minimax


def build_experiment_tree() -> TreeNode:
    def max_node(value: int) -> TreeNode:
        return TreeNode(children=[TreeNode(value=value) for _ in range(3)])

    def min_node(values: list[int]) -> TreeNode:
        return TreeNode(children=[max_node(value) for value in values])

    return TreeNode(
        children=[
            min_node([8, 9, 10]),
            min_node([1, 9, 10]),
            min_node([2, 9, 10]),
        ]
    )


def run_experiment() -> None:
    root = build_experiment_tree()
    minimax_score, minimax_stats = minimax(root, return_stats=True)

    rows: list[tuple[str, int, int, int, int]] = []
    for label, order in (("左到右", "left_to_right"), ("右到左", "right_to_left")):
        score, stats = alpha_beta(root, return_stats=True, child_order=order)
        if score != minimax_score:
            raise AssertionError("改变访问顺序不应改变 Alpha-Beta 的最优分数")
        rows.append(
            (
                label,
                score,
                stats.visited_nodes,
                stats.evaluated_leaves,
                stats.pruned_nodes,
            )
        )

    print(f"Minimax 基线：最优分数={minimax_score}，访问节点={minimax_stats.visited_nodes}")
    print("顺序   | 最优分数 | Alpha-Beta访问节点 | 评估叶子 | 剪掉节点")
    for label, score, visited, leaves, pruned in rows:
        print(f"{label:<4} | {score:^8} | {visited:^20} | {leaves:^8} | {pruned:^8}")


if __name__ == "__main__":
    run_experiment()
