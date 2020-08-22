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
    _is_training: bool

    def __init__(self, policy_value_net: PolicyValueNet, initial_state: ConnectNGame, playout_num=1000, is_training=True):
        self._policy_value_net = policy_value_net
        self._playout_num = playout_num
        self._initial_state = initial_state
        self._root = None
        self._is_training = is_training
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
            move, move_probs = self._get_action(game)
            # store the data
            states.append(convert_game_state(game))
            mcts_probs.append(move_probs)
            current_players.append(game.current_player)
            # perform a move
            game.move(move)

            if game.game_over:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if game.game_result != ConnectNGame.RESULT_TIE:
                    winners_z[np.array(current_players) == game.game_result] = 1.0
                    winners_z[np.array(current_players) != game.game_result] = -1.0

                self.reset()
                return game.game_result, list(zip(states, mcts_probs, winners_z))

    def get_action(self, board: PyGameBoard) -> Pos:
        """
        Method defined in BaseAgent.

        :param board:
        :return:
        """
        return self._get_action(copy.deepcopy(board.connect_n_game))[0]

    def _get_action(self, game: ConnectNGame) -> Tuple[MoveWithProb]:
        avail_pos = game.get_avail_pos()
        move_probs: ActionProbs = np.zeros(game.board_size * game.board_size)
        if len(avail_pos) > 0:
            # the pi defined in AlphaGo Zero paper
            acts, act_probs = self._next_step_play_act_probs(game)
            move_probs[list(acts)] = act_probs
            if self._is_training:
                # add Dirichlet Noise when training in favour of exploration
                move = np.random.choice(acts, p=0.75 * act_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(act_probs))))
                assert move in game.get_avail_pos()
            else:
                move = np.random.choice(acts, p=act_probs)

            return move, move_probs
        else:
            raise Exception('No available actions')

    def reset(self):
        """
        Releases all nodes in MCTS tree and resets root node.
        """
        MCTSAlphaGoZeroPlayer.status_2_node_map = {}
        self._root = TreeNode(None, 1.0)
        MCTSAlphaGoZeroPlayer.status_2_node_map[self._initial_state.get_status()] = self._root

    def _next_step_play_act_probs(self, game: ConnectNGame) -> Tuple[List[Pos], ActionProbs]:
        """
        For the given game status, run playouts number of times specified by self._playout_num.
        Returns the action distribution according to AlphaGo Zero MCTS play formula.

        :param game:
        :return:
        """

        for n in range(self._playout_num):
            self._playout(copy.deepcopy(game))

        current_node = MCTSAlphaGoZeroPlayer.status_2_node_map[game.get_status()]
        act_visits = [(act, node._visit_num) for act, node in current_node._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / MCTSAlphaGoZeroPlayer.temperature * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def _playout(self, game: ConnectNGame):
        """
        From current game status, run a sequence down to a leaf node, either because game ends or unexplored node.
        Get the leaf value of the leaf node, either the actual reward of game or action value returned by policy net.
        And propagate upwards to root node.

        :param game:
        """
        player_id = game.current_player

        node = MCTSAlphaGoZeroPlayer.status_2_node_map[game.get_status()]
        while True:
            if node.is_leaf():
                break
            act, node = node.select()
            game.move(act)

        act_and_probs: Iterator[MoveWithProb]
        act_and_probs, leaf_value = self._policy_value_net.policy_value_fn(game)

        if not game.game_over:
            # case where encountering an unexplored leaf node, update leaf_value estimated by policy net to root
            for act, prob in act_and_probs:
                game.move(act)
                child_node = node.expand(act, prob)
                MCTSAlphaGoZeroPlayer.status_2_node_map[game.get_status()] = child_node
                game.undo()
        else:
            # case where game ends, update actual leaf_value to root
            if game.game_result == ConnectNGame.RESULT_TIE:
                leaf_value = ConnectNGame.RESULT_TIE
            else:
                leaf_value = 1 if game.game_result == player_id else -1
            leaf_value = float(leaf_value)

        # Update leaf_value until root node
        node.propagate_to_root(leaf_value)

