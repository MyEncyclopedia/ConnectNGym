import copy
from collections import defaultdict

from ConnectNGame import ConnectNGame
from ConnectNGym import ConnectNGym
from PyGameConnectN import PyGameBoard
from agent import play
from alphago_zero.MCTSRolloutPlayer import MCTSRolloutPlayer
from alphago_zero.mcts_pure import MCTSPlayer


def battle(n_games=10):
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    initial_game = ConnectNGame(board_size=4, n=3)
    mcts_rollout_player = MCTSRolloutPlayer(playout_num=1000)
    mcts_pure_player = MCTSPlayer(c_puct=5, n_playout=1000)
    win_counts = defaultdict(int)
    board = PyGameBoard(connect_n_game=copy.deepcopy(initial_game))
    env = ConnectNGym(board, display_milli_sec=100)
    for i in range(n_games):
        winner = play(env, mcts_rollout_player, mcts_pure_player, render=False)
        win_counts[winner] += 1
    print(f'rollout first: win: {win_counts[1]}, lose: {win_counts[2]}, tie:{win_counts[-1]}')
    for i in range(n_games):
        winner = play(env, mcts_pure_player, mcts_rollout_player, render=False)
        win_counts[winner] += 1
    win_ratio = 1.0*(win_counts[1] + 0.5*win_counts[-1]) / n_games
    print(f'pure first: win: {win_counts[1]}, lose: {win_counts[2]}, tie:{win_counts[-1]}')
    return win_ratio

if __name__ == '__main__':
    battle(n_games=20)