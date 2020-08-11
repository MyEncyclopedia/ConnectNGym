import argparse
import random
from collections import deque
from typing import Tuple, List

import torch

from PyGameConnectN import PyGameBoard
from alphago_zero.MCTS import MCTSPlayer
from alphago_zero.PolicyValueNetwork import PolicyValueNet, convertGameState
from connect_n import ConnectNGame
import numpy as np


def get_equi_data(play_data: list):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
                mcts_porb.reshape((state.shape[1], state.shape[1]))), i)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
    return extend_data

def selfPlayRollout(player: MCTSPlayer, args, is_shown=0) -> Tuple[int, List[Tuple[np.ndarray, np.ndarray, np.float64]]]:
    """

    :param player:
    :param args:
    :param is_shown:
    :return:
        winner: int
        List[]
    """
    """ start a self-play game using a MCTS player, reuse the search tree,
    and store the self-play data: (state, mcts_probs, z) for training
    """
    game = ConnectNGame(board_size=args.board_size, N=args.n_in_row)
    pygameBoard = PyGameBoard(connectNGame=game)

    states: list[np.ndarray] = []
    mcts_probs:list[np.ndarray] = []
    current_players: list[int] = []
    while True:
        move, move_probs = player.get_action(pygameBoard, temp=args.temp, return_prob=1)
        # store the data
        states.append(convertGameState(pygameBoard.connectNGame))
        mcts_probs.append(move_probs)
        current_players.append(pygameBoard.getCurrentPlayer())
        # perform a move
        pygameBoard.move1D(move)
        if is_shown:
            pygameBoard.display()
        end, winner = pygameBoard.connectNGame.gameOver, pygameBoard.connectNGame.gameResult
        if end:
            # winner from the perspective of the current player of each state
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            # reset MCTS root node
            player.reset_player()
            if is_shown:
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
            return winner, list(zip(states, mcts_probs, winners_z))


def train(args):
    game = ConnectNGame(board_size=args.board_size, N=args.n_in_row)
    pygameBoard = PyGameBoard(connectNGame=game)

    data_buffer = deque(maxlen=args.buffer_size)

    policy_value_net = PolicyValueNet(args.board_size, args.board_size)
    mcts_player = MCTSPlayer(policy_value_net,
                             c_puct=args.c_puct,
                             n_playout=args.n_playout,
                             is_selfplay=1)

    def collect_selfplay_data(n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = selfPlayRollout(mcts_player, args)
            play_data = list(play_data)[:]
            # augment the data
            play_data = get_equi_data(play_data)
            data_buffer.extend(play_data)
            episode_len = len(play_data)
            return episode_len

    def policy_update():
        """update the policy-value net"""
        mini_batch = random.sample(data_buffer, args.batch_size)
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

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
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

    try:
        for i in range(args.game_batch_num):
            episode_len = collect_selfplay_data(args.play_batch_size)
            print(f'batch i:{i + 1}, episode_len:{episode_len}')
            if len(data_buffer) > args.batch_size:
                loss, entropy = policy_update()
            # check the performance of the current model,
            # and save the model params
            if (i + 1) % args.check_freq == 0:
                pass
                # print("current self-play batch: {}".format(i + 1))
                # win_ratio = self.policy_evaluate()
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
    parser.add_argument("--temp", type=float, default=1.0)  # the temperature param
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
