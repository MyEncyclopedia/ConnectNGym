import copy
import math
from typing import Tuple

from ConnectNGame import ConnectNGame, GameResult, GameStatus, Pos
from minimax.strategy import Strategy


class CountingMinimaxStrategy(Strategy):
    def action(self, game: ConnectNGame) -> Tuple[GameResult, Pos]:
        self.game = copy.deepcopy(game)
        self.dp_map = {}
        result, move = self.minimax(game.get_status())
        return result, move

    def minimax(self, game_status: GameStatus) -> Tuple[GameResult, Pos]:
        # print(f'Current {len(strategy.dpMap)}')

        if game_status in self.dp_map:
            return self.dp_map[game_status]

        game = self.game
        best_move = None
        assert not game.game_over
        if game.current_player == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.get_avail_pos():
                move = pos
                result = game.move(pos)
                if result is None:
                    assert not game.game_over
                    result, opp_move = self.minimax(game.get_status())
                    self.dp_map[game.get_status()] = result, opp_move
                else:
                    self.dp_map[game.get_status()] = result, move
                game.undo()
                ret = max(ret, result)
                best_move = move if ret == result else best_move
            self.dp_map[game_status] = ret, best_move
            return ret, best_move
        else:
            ret = math.inf
            for pos in game.get_avail_pos():
                move = pos
                result = game.move(pos)

                if result is None:
                    assert not game.game_over
                    result, opp_move = self.minimax(game.get_status())
                    self.dp_map[game.get_status()] = result, opp_move
                else:
                    self.dp_map[game.get_status()] = result, move
                game.undo()
                ret = min(ret, result)
                best_move = move if ret == result else best_move
            self.dp_map[game_status] = ret, best_move
            return ret, best_move


if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(n=3, board_size=3)
    strategy = CountingMinimaxStrategy()
    strategy.action(tic_tac_toe)
    print(f'Game States Number {len(strategy.dp_map)}')
