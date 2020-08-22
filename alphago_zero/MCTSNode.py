# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Tuple, Dict, ClassVar
import numpy as np
from ConnectNGame import Pos


class TreeNode:
    """
    MCTS Tree Node
    """

    c_puct: ClassVar[int] = 5  # class-wise global param c_puct, exploration weight factor.

    _parent: TreeNode
    _children: Dict[int, TreeNode]  # map from action to TreeNode
    _visit_num: int
    _Q: float   # Q value of the node, which is the mean action value.
    _prior: float

    def __init__(self, parent_node: TreeNode, prior: float):
        self._parent = parent_node
        self._children = {}
        self._visit_num = 0
        self._Q = 0.0
        self._prior = prior

    def get_puct(self) -> float:
        """
        Computes AlphaGo Zero PUCT (polynomial upper confidence trees) of the node.

        :return: Node PUCT value.
        """
        U = (TreeNode.c_puct * self._prior * np.sqrt(self._parent._visit_num) / (1 + self._visit_num))
        return self._Q + U

    def expand(self, action: int, prob: np.float) -> TreeNode:
        """
        Expands the node by adding one child

        :param action: which action leads from current node to the child
        :param prob: initial prob of choosing the child
        :return: the newly created child
        """
        child_node = TreeNode(self, prob)
        self._children[action] = child_node
        return child_node

    def select(self) -> Tuple[Pos, TreeNode]:
        """
        Selects an action(Pos) having max UCB value.

        :return: Action and corresponding node
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_puct())

    def propagate_to_root(self, leaf_value: float):
        """
        Updates current node with observed leaf_value and propagates to root node.

        :param leaf_value:
        :return:
        """
        self._update(leaf_value)
        if self._parent:
            self._parent.propagate_to_root(-leaf_value)

    def _update(self, leaf_value: float):
        """
        Updates the node by newly observed leaf_value.

        :param leaf_value:
        :return:
        """
        self._visit_num += 1
        # new Q is updated towards deviation from existing Q
        self._Q += 0.2 * (leaf_value - self._Q)

    def is_leaf(self) -> bool:
        """
        Checks if the node is not expanded yet. It may be a terminal node as well.

        :return:
        """
        return len(self._children) == 0





