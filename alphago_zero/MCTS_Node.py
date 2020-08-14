# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple, Dict

import numpy as np


class TreeNode:
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parentNode: TreeNode, priorProb: float):
        self._parent = parentNode
        self._children: Dict[int, TreeNode] = {}  # a map from action to TreeNode
        self._visitsNum = 0
        self._Q = 0
        self._u = 0
        self._P = priorProb

    def expand(self, action: int, prob: np.ndarray) -> TreeNode:
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        childNode = TreeNode(self, prob)
        self._children[action] = childNode
        return childNode

    def select(self, cPuct) -> Tuple[int, TreeNode]:
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].getNodeUCB(cPuct))

    def update(self, leafValue):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._visitsNum += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leafValue - self._Q) / self._visitsNum

    def updateToRoot(self, leafValue):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.updateToRoot(-leafValue)
        self.update(leafValue)

    def getNodeUCB(self, cPuct) -> float:
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (cPuct * self._P * np.sqrt(self._parent._visitsNum) / (1 + self._visitsNum))
        return self._Q + self._u

    def isLeaf(self) -> bool:
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}





