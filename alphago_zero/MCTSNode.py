# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple, Dict

import numpy as np


class TreeNode:
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent_node: TreeNode, prior_p: float):
        self._parent = parent_node
        self._children: Dict[int, TreeNode] = {}  # a map from action to TreeNode
        self._visit_num = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action: int, prob: np.ndarray) -> TreeNode:
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        childNode = TreeNode(self, prob)
        self._children[action] = childNode
        return childNode

    def select(self, c_puct) -> Tuple[int, TreeNode]:
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_node_ucb(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._visit_num += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._visit_num

    def update_til_root(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_til_root(-leaf_value)
        self.update(leaf_value)

    def get_node_ucb(self, c_puct) -> float:
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._visit_num) / (1 + self._visit_num))
        return self._Q + self._u

    def is_leaf(self) -> bool:
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}





