# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Dict, Iterator, ClassVar
import numpy as np
import copy

from PyGameConnectN import PyGameBoard
from alphago_zero.MCTSNode import TreeNode
from agent import BaseAgent
from ConnectNGame import ConnectNGame, GameStatus, Pos, GameResult
from operator import itemgetter

from alphago_zero.PolicyValueNetwork import MoveWithProb


class MCTSRolloutPlayer(BaseAgent):
    """An implementation of Monte Carlo Tree Search."""
    # status_2_node_map: ClassVar[Dict[GameStatus, TreeNode]] = {}  # gameStatus => TreeNode

    def __init__(self, playout_num=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._playout_num = playout_num

    def get_action(self, board: PyGameBoard) -> Pos:
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        game = copy.deepcopy(board.connect_n_game)
        node = TreeNode(None, 1.0)

        for n in range(self._playout_num):
            game_copy = copy.deepcopy(game)
            self._playout(game_copy, node)
        return max(node._children.items(), key=lambda act_node: act_node[1]._visit_num)[0]

    def _playout(self, game: ConnectNGame, node: TreeNode):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select()
            game.move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_and_probs, leaf_value = self.rollout_policy_value_fn(game)
        # Check for end of game.
        end, winner = game.game_over, game.game_result
        if not end:
            for action, prob in action_and_probs:
                child_node = node.expand(action, prob)

        player = game.current_player
        result = self._rollout_simulate_to_end(game)
        if result == ConnectNGame.RESULT_TIE:
            leaf_value = float(ConnectNGame.RESULT_TIE)
        else:
            leaf_value = 1.0 if result == player else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_til_root(leaf_value)

    def _rollout_simulate_to_end(self, game: ConnectNGame) -> GameResult:
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        while True:
            end, result = game.game_over, game.game_result
            if end:
                break
            action_probs = self.rollout_policy_fn(game)
            max_action = max(action_probs, key=itemgetter(1))[0]
            game.move(max_action)
        return result

    def rollout_policy_fn(self, game: ConnectNGame) -> Iterator[MoveWithProb]:
        """a coarse, fast version of policy_fn used in the rollout phase."""
        # rollout randomly
        action_probs = np.random.rand(len(game.get_avail_pos()))
        return zip(game.get_avail_pos(), action_probs)


    def rollout_policy_value_fn(self, game: ConnectNGame) -> Tuple[Iterator[MoveWithProb], float]:
        """a function that takes in a state and outputs a list of (action, probability)
        tuples and a score for the state"""
        # return uniform probabilities and 0 score for pure MCTS
        move_list = game.get_avail_pos()
        action_probs = np.ones(len(move_list)) / len(move_list)
        return zip(move_list, action_probs), 0
