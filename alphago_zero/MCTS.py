# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

"""
from __future__ import annotations
from typing import List, Tuple, Dict, Iterator, ClassVar
import numpy as np
import copy
from scipy.special import softmax

from PyGameConnectN import PyGameBoard
from alphago_zero.PolicyValueNetwork import PolicyValueNet
from connect_n import ConnectNGame, GameStatus, Move1D


class TreeNode(object):
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


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""
    statusToNodeMap: ClassVar[Dict[GameStatus, TreeNode]] = {}  # gameStatus => TreeNode

    def __init__(self, policyValueNet: PolicyValueNet, cPuct=5, playoutNum=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policyValueNet = policyValueNet
        self._cPuct = cPuct
        self._playoutNum = playoutNum

    def _playout(self, game: ConnectNGame):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = MCTS.statusToNodeMap[game.getStatus()]
        while True:
            if node.isLeaf():
                break
            # Greedily select next move.
            action, node = node.select(self._cPuct)
            game.move1D(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        actionWithProbs, leafValue = self._policyValueNet.policy_value_fn(game)
        # Check for end of game.
        end, winner = game.gameOver, game.gameResult
        if not end:
            for action, prob in actionWithProbs:
                game.move1D(action)
                childNode = node.expand(action, prob)
                MCTS.statusToNodeMap[game.getStatus()] = childNode
                # print(f'nodes {len(MCTS.statusToNodeMap)}')
                game.undo()

        else:
            # for end state，return the "true" leaf_value
            return float(winner)

        # Update value and visit count of nodes in this traversal.
        node.updateToRoot(-leafValue)

    def predictOneStepByPlayouts(self, game: ConnectNGame, temp=1e-3) -> Tuple[List[Move1D], np.ndarray]:
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._playoutNum):
            state_copy = copy.deepcopy(game)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        currentNode = MCTS.statusToNodeMap[game.getStatus()]
        act_visits = [(act, node._visitsNum) for act, node in currentNode._children.items()]
        acts, visits = zip(*act_visits)
        actProbs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, actProbs

    def reset(self, initialGame: ConnectNGame):
        MCTS.statusToNodeMap = {}
        self._root = TreeNode(None, 1.0)
        MCTS.statusToNodeMap[initialGame.getStatus()] = self._root

    def trainGetNextAction(self, board: PyGameBoard, selfPlay=True, temperature=1e-3) -> Tuple[Move1D, np.ndarray]:
        availableActions = board.getAvailablePositions1D()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        moveProbs = np.zeros(board.board_size * board.board_size)
        if len(availableActions) > 0:
            acts, probs = self.predictOneStepByPlayouts(board.connectNGame, temperature)
            moveProbs[list(acts)] = probs
            if selfPlay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                assert move in board.connectNGame.getAvailablePositions1D()
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)

            return move, moveProbs
        else:
            raise Exception('No actions')




