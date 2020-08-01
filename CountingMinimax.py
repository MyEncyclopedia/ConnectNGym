import copy
import math
from typing import Tuple

from connect_n import ConnectNGame
from strategy import Strategy


class CountingMinimaxStrategy(Strategy):
    def action(self, game: ConnectNGame) -> Tuple[int, Tuple[int, int]]:
        self.game = copy.deepcopy(game)
        self.dpMap = {}
        result, move = self.minimax(game.getStatus())
        return result, move

    def minimax(self, gameStatus: Tuple[Tuple[int, ...]]) -> Tuple[int, Tuple[int, int]]:
        # print(f'Current {len(strategy.dpMap)}')

        if gameStatus in self.dpMap:
            return self.dpMap[gameStatus]

        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.minimax(game.getStatus())
                    self.dpMap[game.getStatus()] = result, oppMove
                else:
                    self.dpMap[game.getStatus()] = result, move
                game.undo()
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
            self.dpMap[gameStatus] = ret, bestMove
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.move(*pos)

                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.minimax(game.getStatus())
                    self.dpMap[game.getStatus()] = result, oppMove
                else:
                    self.dpMap[game.getStatus()] = result, move
                game.undo()
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
            self.dpMap[gameStatus] = ret, bestMove
            return ret, bestMove


if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(N=3, board_size=3)
    strategy = CountingMinimaxStrategy()
    strategy.action(tic_tac_toe)
    print(f'Game States Number {len(strategy.dpMap)}')
