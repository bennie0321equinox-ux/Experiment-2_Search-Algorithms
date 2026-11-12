from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Any, Literal

ChildOrder = Literal["left_to_right", "right_to_left"]


@dataclass
class TreeNode:
    value: int | None = None
    children: list[TreeNode] = field(default_factory=list)
    is_maximizing_player: bool | None = None


@dataclass
class SearchStats:
    visited_nodes: int = 0
    evaluated_leaves: int = 0
    pruned_children: int = 0
    pruned_nodes: int = 0


def _ordered_children(node: TreeNode, child_order: ChildOrder) -> list[TreeNode]:
    if child_order == "left_to_right":
        return node.children
    if child_order == "right_to_left":
        return list(reversed(node.children))
    raise ValueError("child_order 只能是 'left_to_right' 或 'right_to_left'")


def _subtree_size(node: TreeNode) -> int:
    return 1 + sum(_subtree_size(child) for child in node.children)


def tree_from_dict(data: dict[str, Any]) -> TreeNode:
    if "value" in data:
        return TreeNode(value=int(data["value"]))

    children_data = data.get("children")
    if not isinstance(children_data, list) or not children_data:
        raise ValueError("非叶节点必须包含非空 children 列表")

    player = data.get("is_maximizing_player")
    if player is not None and not isinstance(player, bool):
        raise ValueError("is_maximizing_player 必须是布尔值")

    return TreeNode(
        children=[tree_from_dict(child) for child in children_data],
        is_maximizing_player=player,
    )


def _leaf_value(node: TreeNode) -> int:
    if node.children:
        if node.value is not None:
            raise ValueError("节点不能同时包含 value 和 children")
        raise ValueError("内部节点需要递归搜索，不能直接读取 value")
    if node.value is None:
        raise ValueError("叶节点必须包含 value")
    return node.value


def _minimax(
    node: TreeNode,
    maximizing_player: bool,
    stats: SearchStats,
    child_order: ChildOrder,
) -> int:
    stats.visited_nodes += 1
    is_maximizing = (
        node.is_maximizing_player
        if node.is_maximizing_player is not None
        else maximizing_player
    )

    if not node.children:
        stats.evaluated_leaves += 1
        return _leaf_value(node)

    child_values = [
        _minimax(child, not is_maximizing, stats, child_order)
        for child in _ordered_children(node, child_order)
    ]
    return max(child_values) if is_maximizing else min(child_values)


def minimax(
    root: TreeNode,
    *,
    return_stats: bool = False,
    child_order: ChildOrder = "left_to_right",
) -> int | tuple[int, SearchStats]:
    stats = SearchStats()
    score = _minimax(root, True, stats, child_order)
    if return_stats:
        return score, stats
    return score


def _alpha_beta(
    node: TreeNode,
    maximizing_player: bool,
    alpha: float,
    beta: float,
    stats: SearchStats,
    child_order: ChildOrder,
) -> int:
    stats.visited_nodes += 1
    is_maximizing = (
        node.is_maximizing_player
        if node.is_maximizing_player is not None
        else maximizing_player
    )

    if not node.children:
        stats.evaluated_leaves += 1
        return _leaf_value(node)

    if is_maximizing:
        best_value = -inf
        children = _ordered_children(node, child_order)
        for index, child in enumerate(children):
            best_value = max(
                best_value,
                _alpha_beta(child, False, alpha, beta, stats, child_order),
            )
            alpha = max(alpha, best_value)
            if alpha >= beta:
                skipped = children[index + 1 :]
                stats.pruned_children += len(skipped)
                stats.pruned_nodes += sum(_subtree_size(item) for item in skipped)
                break
        return int(best_value)

    best_value = inf
    children = _ordered_children(node, child_order)
    for index, child in enumerate(children):
        best_value = min(
            best_value,
            _alpha_beta(child, True, alpha, beta, stats, child_order),
        )
        beta = min(beta, best_value)
        if alpha >= beta:
            skipped = children[index + 1 :]
            stats.pruned_children += len(skipped)
            stats.pruned_nodes += sum(_subtree_size(item) for item in skipped)
            break
    return int(best_value)


def alpha_beta(
    root: TreeNode,
    *,
    return_stats: bool = False,
    child_order: ChildOrder = "left_to_right",
) -> int | tuple[int, SearchStats]:
    stats = SearchStats()
    score = _alpha_beta(root, True, -inf, inf, stats, child_order)
    if return_stats:
        return score, stats
    return score


def example_tree() -> TreeNode:
    return TreeNode(
        children=[
            TreeNode(
                children=[
                    TreeNode(children=[TreeNode(value=3), TreeNode(value=5)]),
                    TreeNode(children=[TreeNode(value=6), TreeNode(value=9)]),
                ]
            ),
            TreeNode(
                children=[
                    TreeNode(children=[TreeNode(value=6), TreeNode(value=9)]),
                    TreeNode(children=[TreeNode(value=1), TreeNode(value=2)]),
                ]
            ),
        ]
    )


def ordering_experiment_tree() -> TreeNode:
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


def compare_child_orders(root: TreeNode) -> dict[str, dict[str, int]]:
    results: dict[str, dict[str, int]] = {}
    for order in ("left_to_right", "right_to_left"):
        minimax_score, minimax_stats = minimax(root, return_stats=True, child_order=order)
        alpha_beta_score, alpha_beta_stats = alpha_beta(
            root, return_stats=True, child_order=order
        )
        if minimax_score != alpha_beta_score:
            raise AssertionError("Alpha-Beta 必须与 Minimax 返回相同最优分数")
        results[order] = {
            "最优分数": alpha_beta_score,
            "Minimax访问节点": minimax_stats.visited_nodes,
            "Alpha-Beta访问节点": alpha_beta_stats.visited_nodes,
            "Alpha-Beta评估叶子": alpha_beta_stats.evaluated_leaves,
            "被剪掉的子树根数": alpha_beta_stats.pruned_children,
            "被剪掉的节点数": alpha_beta_stats.pruned_nodes,
        }
    return results


if __name__ == "__main__":
    print(alpha_beta(example_tree()))
