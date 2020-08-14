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
from alphago_zero.MCTSNode import TreeNode
from alphago_zero.PolicyValueNetwork import PolicyValueNet
from agent import BaseAgent
from ConnectNGame import ConnectNGame, GameStatus, Pos

class MCTSAlphaGoZeroPlayer(BaseAgent):
    """An implementation of Monte Carlo Tree Search."""
    status_2_node_map: ClassVar[Dict[GameStatus, TreeNode]] = {}  # gameStatus => TreeNode

    def __init__(self, policy_value_net: PolicyValueNet, c_puct=5, playout_num=10000):
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
        self._policy_value_net = policy_value_net
        self._c_puct = c_puct
        self._playout_num = playout_num

    def _playout(self, game: ConnectNGame):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = MCTSAlphaGoZeroPlayer.status_2_node_map[game.get_status()]
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            game.move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_and_probs, leaf_value = self._policy_value_net.policy_value_fn(game)
        # Check for end of game.
        end, winner = game.gameOver, game.gameResult
        if not end:
            for action, prob in action_and_probs:
                game.move(action)
                child_node = node.expand(action, prob)
                MCTSAlphaGoZeroPlayer.status_2_node_map[game.get_status()] = child_node
                # print(f'nodes {len(MCTS.statusToNodeMap)}')
                game.undo()

        else:
            # for end state，return the "true" leaf_value
            return float(winner)

        # Update value and visit count of nodes in this traversal.
        node.update_til_root(-leaf_value)

    def predict_one_step(self, game: ConnectNGame, temp=1e-3) -> Tuple[List[Pos], np.ndarray]:
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._playout_num):
            state_copy = copy.deepcopy(game)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        current_node = MCTSAlphaGoZeroPlayer.status_2_node_map[game.get_status()]
        act_visits = [(act, node._visit_num) for act, node in current_node._children.items()]
        acts, visits = zip(*act_visits)
        actProbs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, actProbs

    def reset(self, initial_state: ConnectNGame):
        MCTSAlphaGoZeroPlayer.status_2_node_map = {}
        self._root = TreeNode(None, 1.0)
        MCTSAlphaGoZeroPlayer.status_2_node_map[initial_state.get_status()] = self._root

    def get_action(self, game: PyGameBoard) -> Pos:
        return self.train_get_next_action(game)[0]

    def train_get_next_action(self, board: PyGameBoard, self_play=True, temperature=1e-3) -> Tuple[Pos, np.ndarray]:
        avail_pos = board.get_avail_pos()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.board_size * board.board_size)
        if len(avail_pos) > 0:
            acts, probs = self.predict_one_step(board.connect_n_game, temperature)
            move_probs[list(acts)] = probs
            if self_play:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                assert move in board.connect_n_game.get_avail_pos()
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)

            return move, move_probs
        else:
            raise Exception('No actions')