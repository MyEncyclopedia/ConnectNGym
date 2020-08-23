import copy
import logging
from collections import defaultdict

from ConnectNGame import ConnectNGame
from ConnectNGym import ConnectNGym
from PyGameConnectN import PyGameBoard
from agent import play, AIAgent, BaseAgent
from alphago_zero.MCTSRolloutPlayer import MCTSRolloutPlayer
from minimax.PlannedStrategy import PlannedMinimaxStrategy


def battle(initial_game: ConnectNGame, player1: BaseAgent, player2: BaseAgent, n_games=10) -> float:
    win_counts = defaultdict(int)
    board = PyGameBoard(connect_n_game=copy.deepcopy(initial_game))
    env = ConnectNGym(board, display_milli_sec=100)
    for i in range(n_games):
        winner = play(env, player1, player2, render=False)
        win_counts[winner] += 1
    logging.warning(f'first: win: {win_counts[1]}, lose: {win_counts[-1]}, tie:{win_counts[0]}')
    # for i in range(n_games):
    #     winner = play(env, player_second, player_first, render=False)
    #     win_counts[-winner] += 1
    win_ratio = 1.0*(win_counts[1] + 0.5*win_counts[0]) / n_games
    logging.warning(f'total {win_counts}')
    # logging.warning(f'second first: win: {win_counts[1]}, lose: {win_counts[-1]}, tie:{win_counts[0]}')

    return win_ratio

def minimax_mcts():
    initial_game = ConnectNGame(board_size=4, n=3)
    strategy = PlannedMinimaxStrategy(initial_game)
    strategy.load_state()

    planned_minimax_agent = AIAgent(strategy)
    mcts_rollout_player = MCTSRolloutPlayer(playout_num=1000)
    battle(initial_game, planned_minimax_agent, mcts_rollout_player, n_games=20)
    # battle(initial_game, planned_minimax_agent, planned_minimax_agent, n_games=20)


def get_equi_data(play_data, size):
    import numpy as np

    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(size, size)), i)
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


if __name__ == '__main__':
    # mcts_play()
    minimax_mcts()
