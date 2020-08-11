# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

"""
from typing import List, Tuple, Dict
from __future__ import annotations


import numpy as np
import copy
from scipy.special import softmax

from PyGameConnectN import PyGameBoard
from connect_n import ConnectNGame, GameStatus


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    statusToNodeMap: Dict[GameStatus, TreeNode] = {}   # gameStatus => TreeNode

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children: Dict[int, TreeNode] = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action: int, prob: np.ndarray, status: GameStatus):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        childNode = TreeNode(self, prob)
        self._children[action] = childNode
        TreeNode.statusToNodeMap[status] = childNode

    def select(self, c_puct) -> Tuple[int, TreeNode]:
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].nodeUCB(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def updateToRoot(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.updateToRoot(-leaf_value)
        self.update(leaf_value)

    def nodeUCB(self, c_puct) -> float:
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def isLeaf(self) -> bool:
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    @classmethod
    def reset(cls):
        TreeNode.statusToNodeMap = {}



class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policyValueNet, c_puct=5, n_playout=10000):
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
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, game: ConnectNGame):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = TreeNode.statusToNodeMap[game.getStatus()]
        while True:
            if node.isLeaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            game.move1D(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        actionWithProbs, leafValue = self._policyValueNet.policy_value_fn(game)
        # Check for end of game.
        end, winner = game.gameOver, game.gameResult
        if not end:
            for action, prob in actionWithProbs:
                game.move()
                node.expand(action, prob, game.getStatus())
                game.undo()

        else:
            # for end state，return the "true" leaf_value
            return float(winner)

        # Update value and visit count of nodes in this traversal.
        node.updateToRoot(-leafValue)

    def get_move_probs(self, state: ConnectNGame, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def reset(self):
        TreeNode.reset()


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policyValueNet, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policyValueNet, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.reset()

    def get_action(self, board: PyGameBoard, temp=1e-3, return_prob=0):
        sensible_moves = board.getAvailablePositions1D()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.board_size * board.board_size)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board.connectNGame, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.reset()
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
