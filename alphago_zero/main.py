import argparse
import copy
import random
from collections import deque
from typing import Tuple, List

import torch

from PyGameConnectN import PyGameBoard
from alphago_zero.MCTSAlphaGoZeroPlayer import MCTSAlphaGoZeroPlayer
from alphago_zero.PolicyValueNetwork import PolicyValueNet, convertGameState
from ConnectNGame import ConnectNGame
import numpy as np


# def getRotatedStatus(play_data: list):
#     """augment the data set by rotation and flipping
#     play_data: [(state, mcts_prob, winner_z), ..., ...]
#     """
#     extend_data = []
#     for state, mcts_porb, winner in play_data:
#         for i in [1, 2, 3, 4]:
#             # rotate counterclockwise
#             equi_state = np.array([np.rot90(s, i) for s in state])
#             equi_mcts_prob = np.rot90(np.flipud(
#                 mcts_porb.reshape((state.shape[1], state.shape[1]))), i)
#             extend_data.append((equi_state,
#                                 np.flipud(equi_mcts_prob).flatten(),
#                                 winner))
#             # flip horizontally
#             equi_state = np.array([np.fliplr(s) for s in equi_state])
#             equi_mcts_prob = np.fliplr(equi_mcts_prob)
#             extend_data.append((equi_state,
#                                 np.flipud(equi_mcts_prob).flatten(),
#                                 winner))
#     return extend_data

def start_play(self, player1, player2, start_player=0, is_shown=1):
    """start a game between two players"""
    if start_player not in (0, 1):
        raise Exception('start_player should be either 0 (player1 first) '
                        'or 1 (player2 first)')
    self.board.init_board(start_player)
    p1, p2 = self.board.players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    if is_shown:
        self.graphic(self.board, player1.player, player2.player)
    while True:
        current_player = self.board.get_current_player()
        player_in_turn = players[current_player]
        move = player_in_turn.get_action(self.board)
        self.board.do_move(move)
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        end, winner = self.board.game_end()
        if end:
            if is_shown:
                if winner != -1:
                    print("Game end. Winner is", players[winner])
                else:
                    print("Game end. Tie")
            return winner

def self_play_one_game(player: MCTSAlphaGoZeroPlayer, pygame_board: PyGameBoard, temperature, show_gui=False) \
        -> Tuple[int, List[Tuple[np.ndarray, np.ndarray, np.float64]]]:
    """

    :param player:
    :param args:
    :param show_gui:
    :return:
        winner: int
        List[]
    """
    """ start a self-play game using a MCTS player, reuse the search tree,
    and store the self-play data: (state, mcts_probs, z) for training
    """

    states: list[np.ndarray] = []
    mcts_probs: list[np.ndarray] = []
    current_players: list[int] = []
    while True:
        move, move_probs = player.train_get_next_action(pygame_board, temperature=temperature)
        # store the data
        states.append(convertGameState(pygame_board.connectNGame))
        mcts_probs.append(move_probs)
        current_players.append(pygame_board.get_current_player())
        # perform a move
        pygame_board.move(move)
        if show_gui:
            pygame_board.display()
        end, winner = pygame_board.connectNGame.gameOver, pygame_board.connectNGame.gameResult
        if end:
            # winner from the perspective of the current player of each state
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            # reset MCTS root node
            # player.resetPlayer()
            if show_gui:
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
            return winner, list(zip(states, mcts_probs, winners_z))


def update_policy(mini_batch, policy_value_net, args):
    """update the policy-value net"""
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)
    for i in range(args.epochs):
        loss, entropy = policy_value_net.train_step(
            state_batch,
            mcts_probs_batch,
            winner_batch,
            args.learning_rate * args.lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                            axis=1)
                     )
        if kl > args.kl_targ * 4:  # early stopping if D_KL diverges badly
            break
    # adaptively adjust the learning rate
    if kl > args.kl_targ * 2 and args.lr_multiplier > 0.1:
        args.lr_multiplier /= 1.5
    elif kl < args.kl_targ / 2 and args.lr_multiplier < 10:
        args.lr_multiplier *= 1.5

    explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
    explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
    print(("kl:{:.5f},"
           "lr_multiplier:{:.3f},"
           "loss:{},"
           "entropy:{},"
           "explained_var_old:{:.3f},"
           "explained_var_new:{:.3f}"
           ).format(kl,
                    args.lr_multiplier,
                    loss,
                    entropy,
                    explained_var_old,
                    explained_var_new))
    return loss, entropy


# def policy_evaluate(self, n_games=10):
#     """
#     Evaluate the trained policy by playing against the pure MCTS player
#     Note: this is only for monitoring the progress of training
#     """
#     current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
#                                      c_puct=self.c_puct,
#                                      n_playout=self.n_playout)
#     pure_mcts_player = MCTS_Pure(c_puct=5,
#                                  n_playout=self.pure_mcts_playout_num)
#     win_cnt = defaultdict(int)
#     for i in range(n_games):
#         winner = self.game.start_play(current_mcts_player,
#                                       pure_mcts_player,
#                                       start_player=i % 2,
#                                       is_shown=0)
#         win_cnt[winner] += 1
#     win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
#     print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
#             self.pure_mcts_playout_num,
#             win_cnt[1], win_cnt[2], win_cnt[-1]))
#     return win_ratio

def train(args):
    initial_game = ConnectNGame(board_size=args.board_size, N=args.n_in_row)
    # pygame_board = PyGameBoard(connectNGame=game)

    data_buffer = deque(maxlen=args.buffer_size)

    policy_value_net = PolicyValueNet(args.board_size, args.board_size)
    mctsPlayer = MCTSAlphaGoZeroPlayer(policy_value_net, c_puct=args.c_puct, playout_num=args.n_playout)
    mctsPlayer.reset(initial_game)

    try:
        for i in range(args.game_batch_num):
            for b in range(args.play_batch_size):
                game = copy.deepcopy(initial_game)
                pygame_board = PyGameBoard(connectNGame=game)
                winner, play_data = self_play_one_game(mctsPlayer, pygame_board, args.temperature)
                mctsPlayer.reset(initial_game)
                play_data = list(play_data)[:]
                # augment the data
                # play_data = getRotatedStatus(play_data)
                # data_buffer.extend(play_data)
                episode_len = len(play_data)
                print(f'batch i:{i + 1}, episode_len:{episode_len}')
                if len(data_buffer) > args.batch_size:
                    mini_batch = random.sample(data_buffer, args.batch_size)
                    loss, entropy = update_policy(mini_batch, policy_value_net, args)
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % args.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()
                # self.policy_value_net.save_model('./current_policy.model')
                # if win_ratio > self.best_win_ratio:
                #     print("New best policy!!!!!!!!")
                #     self.best_win_ratio = win_ratio
                #     # update the best_policy
                #     self.policy_value_net.save_model('./best_policy.model')
                #     if (self.best_win_ratio == 1.0 and
                #             self.pure_mcts_playout_num < 5000):
                #         self.pure_mcts_playout_num += 1000
                #         self.best_win_ratio = 0.0
    except KeyboardInterrupt:
        print('\nquit')


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser("ConnectN_AlphaGo_Zero")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--board_size", type=int, default=4)
    parser.add_argument("--n_in_row", type=int, default=3)
    # training params
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--lr_multiplier", type=float, default=1.0)  # adaptively adjust the learning rate based on KL
    parser.add_argument("--temperature", type=float, default=1.0)  # the temperature param
    parser.add_argument("--n_playout", type=int, default=400)  # num of simulations for each move
    parser.add_argument("--c_puct", type=int, default=5)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)  # mini-batch size for training
    parser.add_argument("--play_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)  # num of train_steps for each update
    parser.add_argument("--kl_targ", type=float, default=0.02)
    parser.add_argument("--check_freq", type=int, default=50)
    parser.add_argument("--game_batch_num", type=int, default=1500)
    parser.add_argument("--best_win_ratio", type=float, default=0.0)
    parser.add_argument("--pure_mcts_playout_num", type=int, default=1000)  # num of simulations used for the pure mcts,
    # which is used as the opponent to evaluate the trained policy

    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        args.use_cuda = True
        torch.cuda.device(args.gpu)
    else:
        args.use_cuda = False

    train(args)
