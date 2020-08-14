# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Dict, Iterator, ClassVar
import numpy as np
import copy
from scipy.special import softmax

from PyGameConnectN import PyGameBoard
from alphago_zero.MCTSNode import TreeNode
from alphago_zero.PolicyValueNetwork import PolicyValueNet
from base_agent import BaseAgent
from connect_n import ConnectNGame, GameStatus, Pos
from operator import itemgetter


class MCTSRolloutPlayer(BaseAgent):
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

    def get_action(self, game: PyGameBoard) -> Pos:
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
            game_copy = copy.deepcopy(game)
            self._playout(game_copy)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def _playout(self, game: ConnectNGame):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = MCTSRolloutPlayer.statusToNodeMap[game.get_status()]
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._cPuct)
            game.move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        actionWithProbs, leaf_value = self._policyValueNet.policy_value_fn(game)
        # Check for end of game.
        end, winner = game.gameOver, game.gameResult
        if not end:
            childNode = node.expand(actionWithProbs)
            MCTSRolloutPlayer.statusToNodeMap[game.get_status()] = childNode
        else:
            # for end state，return the "true" leaf_value
            return float(winner)

        leaf_value = self._evaluate_rollout(game)
        # Update value and visit count of nodes in this traversal.
        node.update_til_root(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = self.rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def reset(self, initialGame: ConnectNGame):
        MCTSRolloutPlayer.statusToNodeMap = {}
        self._root = TreeNode(None, 1.0)

    def rollout_policy_fn(self, board):
        """a coarse, fast version of policy_fn used in the rollout phase."""
        # rollout randomly
        action_probs = np.random.rand(len(board.availables))
        return zip(board.availables, action_probs)


    def policy_value_fn(self, board):
        """a function that takes in a state and outputs a list of (action, probability)
        tuples and a score for the state"""
        # return uniform probabilities and 0 score for pure MCTS
        action_probs = np.ones(len(board.availables))/len(board.availables)
        return zip(board.availables, action_probs), 0