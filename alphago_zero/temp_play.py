import copy
import logging
from collections import defaultdict

from ConnectNGame import ConnectNGame
from ConnectNGym import ConnectNGym
from PyGameConnectN import PyGameBoard
from agent import play, AIAgent
from alphago_zero.MCTSRolloutPlayer import MCTSRolloutPlayer
from alphago_zero.mcts_pure import MCTSPlayer
from minimax.PlannedStrategy import PlannedMinimaxStrategy


def battle(initial_game: ConnectNGame, player_first, player_second, n_games=10):
    win_counts = defaultdict(int)
    board = PyGameBoard(connect_n_game=copy.deepcopy(initial_game))
    env = ConnectNGym(board, display_milli_sec=100)
    for i in range(n_games):
        winner = play(env, player_first, player_second, render=False)
        win_counts[winner] += 1
    logging.warning(f'first: win: {win_counts[1]}, lose: {win_counts[-1]}, tie:{win_counts[0]}')
    # for i in range(n_games):
    #     winner = play(env, player_second, player_first, render=False)
    #     win_counts[-winner] += 1
    win_ratio = 1.0*(win_counts[1] + 0.5*win_counts[0]) / n_games
    logging.warning(f'total {win_counts}')
    # logging.warning(f'second first: win: {win_counts[1]}, lose: {win_counts[-1]}, tie:{win_counts[0]}')

    return win_ratio

def mcts_play():
    initial_game = ConnectNGame(board_size=4, n=3)
    mcts_rollout_player = MCTSRolloutPlayer(playout_num=1000)
    mcts_pure_player = MCTSPlayer(c_puct=5, n_playout=1000)
    battle(initial_game, mcts_rollout_player, mcts_pure_player, n_games=20)

def minimax_mcts():
    initial_game = ConnectNGame(board_size=4, n=3)
    strategy = PlannedMinimaxStrategy(initial_game)
    strategy.load_state()

    planned_minimax_agent = AIAgent(strategy)
    mcts_rollout_player = MCTSRolloutPlayer(playout_num=1000)
    battle(initial_game, planned_minimax_agent, mcts_rollout_player, n_games=20)
    # battle(initial_game, planned_minimax_agent, planned_minimax_agent, n_games=20)



if __name__ == '__main__':
    # mcts_play()
    minimax_mcts()
