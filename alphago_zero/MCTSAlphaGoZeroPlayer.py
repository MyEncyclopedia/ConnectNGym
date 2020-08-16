# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

"""

from __future__ import annotations
from typing import List, Tuple, Dict, Iterator, ClassVar, Any
import numpy as np
import copy

from nptyping import NDArray
from scipy.special import softmax

from PyGameConnectN import PyGameBoard
from alphago_zero.MCTSNode import TreeNode
from alphago_zero.PolicyValueNetwork import PolicyValueNet
from agent import BaseAgent
from ConnectNGame import ConnectNGame, GameStatus, Pos

ActionProbs = NDArray[(Any), np.float]
MoveWithProb = Tuple[Pos, ActionProbs]

class MCTSAlphaGoZeroPlayer(BaseAgent):
    """An implementation of Monte Carlo Tree Search."""
    status_2_node_map: ClassVar[Dict[GameStatus, TreeNode]] = {}  # gameStatus => TreeNode

    def __init__(self, policy_value_net: PolicyValueNet, initial_state: ConnectNGame, playout_num=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._policy_value_net = policy_value_net
        self._playout_num = playout_num
        self._initial_state = initial_state
        self.reset()

    def _playout(self, game: ConnectNGame):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        player = game.current_player

        node = MCTSAlphaGoZeroPlayer.status_2_node_map[game.get_status()]
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select()
            game.move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_and_probs, leaf_value = self._policy_value_net.policy_value_fn(game)
        # Check for end of game.
        end, winner = game.game_over, game.game_result
        if not end:
            for action, prob in action_and_probs:
                game.move(action)
                child_node = node.expand(action, prob)
                MCTSAlphaGoZeroPlayer.status_2_node_map[game.get_status()] = child_node
                # print(f'nodes {len(MCTS.statusToNodeMap)}')
                game.undo()
        else:
            if winner == ConnectNGame.RESULT_TIE:
                leaf_value = ConnectNGame.RESULT_TIE
            else:
                leaf_value = 1 if winner == player else -1
            leaf_value = float(leaf_value)

        # Update value and visit count of nodes in this traversal.
        node.update_til_root(leaf_value)

    def predict_one_step(self, game: ConnectNGame, temperature=1e-3) -> Tuple[List[Pos], ActionProbs]:
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
        act_probs = softmax(1.0 / temperature * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def reset(self):
        MCTSAlphaGoZeroPlayer.status_2_node_map = {}
        self._root = TreeNode(None, 1.0)
        MCTSAlphaGoZeroPlayer.status_2_node_map[self._initial_state.get_status()] = self._root

    def get_action(self, board: PyGameBoard) -> Pos:
        return self.train_get_next_action(copy.deepcopy(board.connect_n_game))[0]

    def train_get_next_action(self, game: ConnectNGame, self_play=True, temperature=1e-3) -> Tuple[MoveWithProb]:
        avail_pos = game.get_avail_pos()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs: ActionProbs = np.zeros(game.board_size * game.board_size)
        if len(avail_pos) > 0:
            acts, probs = self.predict_one_step(game, temperature)
            move_probs[list(acts)] = probs
            if self_play:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                assert move in game.get_avail_pos()
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)

            return move, move_probs
        else:
            raise Exception('No actions')