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
from alphago_zero.PolicyValueNetwork import PolicyValueNet, ActionProbs, MoveWithProb, NetGameState, convert_game_state
from agent import BaseAgent
from ConnectNGame import ConnectNGame, GameStatus, Pos, GameResult


class MCTSAlphaGoZeroPlayer(BaseAgent):
    """
    AlphaGo Zero MCTS player.
    """

    # Keeping track of all nodes in MCTS tree currently constructed.
    # It is emptied after reset() is called.
    status_2_node_map: ClassVar[Dict[GameStatus, TreeNode]] = {}  # GameStatus => TreeNode
    # temperature param during training
    temperature: float

    _policy_value_net: PolicyValueNet
    _playout_num: int
    _root: TreeNode
    _initial_state: ConnectNGame  # used in reset() to construct root node.

    def __init__(self, policy_value_net: PolicyValueNet, initial_state: ConnectNGame, playout_num=1000):
        self._policy_value_net = policy_value_net
        self._playout_num = playout_num
        self._initial_state = initial_state
        self._root = None
        self.reset()

    def self_play_one_game(self, game: ConnectNGame) \
            -> Tuple[GameResult, List[Tuple[NetGameState, ActionProbs, NDArray[(Any), np.float]]]]:
        """

        :return:
            winner: int
            List[]
        """
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """

        states: List[NetGameState] = []
        mcts_probs: List[ActionProbs] = []
        current_players: List[float] = []
        while True:
            move, move_probs = self.train_get_next_action(game)
            # store the data
            states.append(convert_game_state(game))
            mcts_probs.append(move_probs)
            current_players.append(game.current_player)
            # perform a move
            game.move(move)

            end, result = game.game_over, game.game_result
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if result != ConnectNGame.RESULT_TIE:
                    winners_z[np.array(current_players) == result] = 1.0
                    winners_z[np.array(current_players) != result] = -1.0

                self.reset()
                return result, list(zip(states, mcts_probs, winners_z))

    def get_action(self, board: PyGameBoard) -> Pos:
        """
        Method defined in BaseAgent.

        :param board:
        :return:
        """
        return self.train_get_next_action(copy.deepcopy(board.connect_n_game))[0]

    def train_get_next_action(self, game: ConnectNGame, self_play=True) -> Tuple[MoveWithProb]:
        avail_pos = game.get_avail_pos()
        move_probs: ActionProbs = np.zeros(game.board_size * game.board_size)
        if len(avail_pos) > 0:
            # the pi defined in AlphaGo Zero paper
            acts, probs = self._next_step(game)
            move_probs[list(acts)] = probs
            if self_play:
                # add Dirichlet Noise when training to favour exploration
                move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                assert move in game.get_avail_pos()
            else:
                move = np.random.choice(acts, p=probs)

            return move, move_probs
        else:
            raise Exception('No actions')

    def reset(self):
        """
        Releases all nodes in MCTS tree and resets root node.
        """
        MCTSAlphaGoZeroPlayer.status_2_node_map = {}
        self._root = TreeNode(None, 1.0)
        MCTSAlphaGoZeroPlayer.status_2_node_map[self._initial_state.get_status()] = self._root

    def _next_step(self, game: ConnectNGame) -> Tuple[List[Pos], ActionProbs]:
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
        act_probs = softmax(1.0 / MCTSAlphaGoZeroPlayer.temperature * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

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
        node.propagate_to_root(leaf_value)

