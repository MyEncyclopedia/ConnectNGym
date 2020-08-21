import argparse
import copy
import logging
import random
from collections import deque
from typing import Tuple, List, Any

import numpy as np
import torch
from nptyping import NDArray

from ConnectNGame import ConnectNGame, GameResult
from alphago_zero import MCTSNode
from alphago_zero.MCTSAlphaGoZeroPlayer import MCTSAlphaGoZeroPlayer
from alphago_zero.MCTSRolloutPlayer import MCTSRolloutPlayer
from alphago_zero.PolicyValueNetwork import PolicyValueNet, convert_game_state, NetGameState, ActionProbs
from alphago_zero.temp_play import battle


def self_play_one_game(player: MCTSAlphaGoZeroPlayer, game: ConnectNGame, temperature: float) \
        -> Tuple[GameResult, List[Tuple[NetGameState, ActionProbs, NDArray[(Any), np.float]]]]:
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

    states: List[NetGameState] = []
    mcts_probs: List[ActionProbs] = []
    current_players: List[float] = []
    while True:
        move, move_probs = player.train_get_next_action(game, temperature=temperature)
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
            return result, list(zip(states, mcts_probs, winners_z))


def update_policy(mini_batch: List[Tuple[NetGameState, ActionProbs, NDArray[(Any), np.float]]], policy_value_net: PolicyValueNet, args):
    """update the policy-value net"""
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)
    for i in range(args.epochs):
        loss, entropy = policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, args.learning_rate)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
        if kl > args.kl_targ * 4:  # early stopping if D_KL diverges badly
            break

    logging.warning(f'kl:{kl:.5f}, loss:{loss}, entropy:{entropy}')
    return loss, entropy


def train(args):
    initial_game = ConnectNGame(board_size=args.board_size, n=args.n_in_row)
    game_records: deque[Tuple[NetGameState, ActionProbs, NDArray[(Any), np.float]]]
    game_records = deque(maxlen=args.buffer_size)
    loss_q = deque(maxlen=20)

    policy_value_net = PolicyValueNet(args.board_size, args.board_size, use_gpu=args.use_cuda)
    alphago_zero_player = MCTSAlphaGoZeroPlayer(policy_value_net, playout_num=args.playout_num, initial_state=initial_game)

    best_win_ratio = 0.0

    for i in range(args.game_batch_num):
        game = copy.deepcopy(initial_game)
        winner, one_game_records = self_play_one_game(alphago_zero_player, game, temperature=args.temperature)
        alphago_zero_player.reset()
        game_records.extend(one_game_records)
        episode_len = len(one_game_records)
        logging.warning(f'batch i:{i + 1}, episode_len:{episode_len}, {len(game_records)}')
        if len(game_records) > args.batch_size:
            mini_batch = random.sample(game_records, args.batch_size)
            loss, entropy = update_policy(mini_batch, policy_value_net, args)
            loss_q.append(loss)
            if len(loss_q) == loss_q.maxlen:
                if all(loss_q[last_idx] < loss_q[0] for last_idx in range(-1, -6, -1)):
                    args.learning_rate = 1e-2

        # check the performance of the current model,and save the model params
        if i % args.check_freq == 0:
            initial_game = ConnectNGame(board_size=args.board_size, n=args.n_in_row)
            alphago_zero_player = MCTSAlphaGoZeroPlayer(policy_value_net, playout_num=args.playout_num, initial_state=initial_game)
            mcts_rollout_player = MCTSRolloutPlayer(playout_num=args.rollout_playout_num)
            win_ratio = battle(initial_game, alphago_zero_player, mcts_rollout_player, n_games=3)
            logging.warning(f'current self-play batch: {i+1}, win_ratio:{win_ratio:.3f}')
            policy_value_net.save_model('./current_policy.model')
            if win_ratio > best_win_ratio:
                logging.warning(f'best policy {win_ratio:.3f}')
                best_win_ratio = win_ratio
                # update the best_policy
                policy_value_net.save_model('./best_policy.model')
                # if best_win_ratio == 1.0 and args.rollout_playout_num < 5000:
                #     args.rollout_playout_num += 1000
                #     best_win_ratio = 0.0

def logging_config():
    import logging.config
    import yaml
    with open('../logging_config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def parse_args():
    parser = argparse.ArgumentParser("ConnectN_AlphaGo_Zero")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--board_size", type=int, default=4)
    parser.add_argument("--n_in_row", type=int, default=3)
    # training params
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=1.0)  # the temperature param
    parser.add_argument("--playout_num", type=int, default=1000)  # num of simulations for each move
    parser.add_argument("--rollout_playout_num", type=int, default=900)  # num of simulations for each move
    parser.add_argument("--c_puct", type=int, default=5)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)  # mini-batch size for training
    parser.add_argument("--epochs", type=int, default=8)  # num of train_steps for each update
    parser.add_argument("--kl_targ", type=float, default=0.02)
    parser.add_argument("--check_freq", type=int, default=50)
    parser.add_argument("--game_batch_num", type=int, default=3000)
    parser.add_argument("--best_win_ratio", type=float, default=0.0)

    args = parser.parse_args()
    if args.gpu >= 0 and torch.cuda.is_available():
        args.use_cuda = True
        torch.cuda.device(args.gpu)
    else:
        args.use_cuda = False
    return args

if __name__ == "__main__":
    logging_config()
    args = parse_args()
    MCTSNode.c_puct = args.c_puct
    train(args)
