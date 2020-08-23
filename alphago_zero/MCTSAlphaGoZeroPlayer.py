# -*- coding: utf-8 -*-
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

    # temperature param during training
    temperature: float = 1.0

    _policy_value_net: PolicyValueNet
    _playout_num: int
    _current_root: TreeNode
    _is_training: bool

    def __init__(self, policy_value_net: PolicyValueNet, playout_num=1000, is_training=True):
        self._policy_value_net = policy_value_net
        self._playout_num = playout_num
        self._current_root = None
        self._is_training = is_training
        self.reset()

    def self_play_one_game(self, game: ConnectNGame) \
            -> List[Tuple[NetGameState, ActionProbs, NDArray[(Any), np.float]]]:
        """

        :param game:
        :return:
        """

        states: List[NetGameState] = []
        probs: List[ActionProbs] = []
        current_players: List[np.float] = []
        while True:
            move, move_probs = self._get_action(game)
            states.append(convert_game_state(game))
            probs.append(move_probs)
            current_players.append(game.current_player)
            game.move(move)

            if game.game_over:
                current_player_z = np.zeros(len(current_players))
                if game.game_result != ConnectNGame.RESULT_TIE:
                    current_player_z[np.array(current_players) == game.game_result] = 1.0
                    current_player_z[np.array(current_players) != game.game_result] = -1.0

                self.reset()
                return list(zip(states, probs, current_player_z))

    def get_action(self, board: PyGameBoard) -> Pos:
        """
        Method defined in BaseAgent.

        :param board:
        :return: next move for the given game board.
        """
        return self._get_action(copy.deepcopy(board.connect_n_game))[0]

    def _get_action(self, game: ConnectNGame) -> Tuple[MoveWithProb]:
        epsilon = 0.25
        avail_pos = game.get_avail_pos()
        move_probs: ActionProbs = np.zeros(game.board_size * game.board_size)
        assert len(avail_pos) > 0

        # the pi defined in AlphaGo Zero paper
        acts, act_probs = self._next_step_play_act_probs(game)
        move_probs[list(acts)] = act_probs
        if self._is_training:
            # add Dirichlet Noise when training in favour of exploration
            p_ = (1-epsilon) * act_probs + epsilon * np.random.dirichlet(0.3 * np.ones(len(act_probs)))
            move = np.random.choice(acts, p=p_)
            assert move in game.get_avail_pos()
        else:
            move = np.random.choice(acts, p=act_probs)

        self.reset()
        return move, move_probs

    def reset(self):
        """
        Releases all nodes in MCTS tree and resets root node.
        """
        # MCTSAlphaGoZeroPlayer.status_2_node_map = {}
        self._current_root = TreeNode(None, 1.0)
        # MCTSAlphaGoZeroPlayer.status_2_node_map[self._initial_state.get_status()] = self._current_root

    def _next_step_play_act_probs(self, game: ConnectNGame) -> Tuple[List[Pos], ActionProbs]:
        """
        For the given game status, run playouts number of times specified by self._playout_num.
        Returns the action distribution according to AlphaGo Zero MCTS play formula.

        :param game:
        :return: actions and their probability
        """

        for n in range(self._playout_num):
            self._playout(copy.deepcopy(game))

        act_visits = [(act, node._visit_num) for act, node in self._current_root._children.items()]
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

        node = self._current_root
        while True:
            if node.is_leaf():
                break
            act, node = node.select()
            game.move(act)

        # now game state is a leaf node in the tree, either a terminal node or an unexplored node
        act_and_probs: Iterator[MoveWithProb]
        act_and_probs, leaf_value = self._policy_value_net.policy_value_fn(game)

        if not game.game_over:
            # case where encountering an unexplored leaf node, update leaf_value estimated by policy net to root
            for act, prob in act_and_probs:
                game.move(act)
                child_node = node.expand(act, prob)
                # MCTSAlphaGoZeroPlayer.status_2_node_map[game.get_status()] = child_node
                game.undo()
        else:
            # case where game ends, update actual leaf_value to root
            if game.game_result == ConnectNGame.RESULT_TIE:
                leaf_value = ConnectNGame.RESULT_TIE
            else:
                leaf_value = 1 if game.game_result == player_id else -1
            leaf_value = float(leaf_value)

        # Update leaf_value until root node
        node.propagate_to_root(-leaf_value)

