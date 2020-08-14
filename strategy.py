import copy
import math
import os
import sys
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple

from connect_n import ConnectNGame, Move2D


class Strategy(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def action(self, game: ConnectNGame) -> Tuple[int, Move2D]:
        pass


class MinimaxStrategy(Strategy):
    def action(self, game: ConnectNGame) -> Tuple[int, Move2D]:
        self.game = copy.deepcopy(game)
        result, move = self.minimax()
        return result, move

    def minimax(self) -> Tuple[int, Tuple[int, int]]:
        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions2D():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.minimax()
                game.undo()
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == 1:
                    return 1, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions2D():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.minimax()
                game.undo()
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == -1:
                    return -1, move
            return ret, bestMove


class MinimaxDPStrategy(Strategy):
    def action(self, game) -> Tuple[int, Move2D]:
        self.game = game
        result, move = self.minimax_dp(self.game.getStatus())
        return result, move

    @lru_cache(maxsize=None)
    def minimax_dp(self, gameState: Tuple[Tuple[int, ...]]) -> Tuple[int, Move2D]:
        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions2D():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.minimax_dp(game.getStatus())
                game.undo()
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == 1:
                    return 1, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions2D():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.minimax_dp(game.getStatus())
                game.undo()
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == -1:
                    return -1, move
            return ret, bestMove


class AlphaBetaStrategy(Strategy):
    def action(self, game: ConnectNGame) -> Tuple[int, Move2D]:
        self.game = game
        result, move = self.alpha_beta(self.game.getStatus(), -math.inf, math.inf)
        return result, move

    def alpha_beta(self, gameStatus: Tuple[Tuple[int, ...]], alpha: int = None, beta: int = None) \
            -> Tuple[int, Move2D]:
        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions2D():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.alpha_beta(game.getStatus(), alpha, beta)
                game.undo()
                alpha = max(alpha, result)
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == 1:
                    return ret, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions2D():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.alpha_beta(game.getStatus(), alpha, beta)
                game.undo()
                beta = min(beta, result)
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == -1:
                    return ret, move
            return ret, bestMove


class AlphaBetaDPStrategy(Strategy):
    def action(self, game: ConnectNGame) -> Tuple[int, Move2D]:
        self.game = game
        self.alphaBetaStack = [(-math.inf, math.inf)]
        result, move = self.alpha_beta_dp(self.game.getStatus())
        return result, move

    @lru_cache(maxsize=None)
    def alpha_beta_dp(self, gameStatus: Tuple[Tuple[int, ...]]) -> Tuple[int, Move2D]:
        alpha, beta = self.alphaBetaStack[-1]
        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions2D():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    self.alphaBetaStack.append((alpha, beta))
                    result, oppMove = self.alpha_beta_dp(game.getStatus())
                    self.alphaBetaStack.pop()
                game.undo()
                alpha = max(alpha, result)
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == 1:
                    return ret, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions2D():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    self.alphaBetaStack.append((alpha, beta))
                    result, oppMove = self.alpha_beta_dp(game.getStatus())
                    self.alphaBetaStack.pop()
                game.undo()
                beta = min(beta, result)
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == -1:
                    return ret, move
            return ret, bestMove


if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(N=5, board_size=7)
    # strategy = MinimaxDPStrategy(tic_tac_toe)
    strategy = AlphaBetaDPStrategy(tic_tac_toe)
    print(strategy.action())
    sys.exit(1)

    tic_tac_toe = ConnectNGame(N=5, board_size=5)
    # tic_tac_toe.move(0, 0)
    # tic_tac_toe.move(1, 1)
    # tic_tac_toe.move(1, 2)
    # tic_tac_toe.move(1, 0)
    # tic_tac_toe.move(0, 1)
    # tic_tac_toe.drawText()
    # strategy1 = MinimaxDPStrategy(tic_tac_toe)
    # strategy2 = AlphaBetaStrategy(tic_tac_toe)
    # strategy3 = AlphaBetaDPStrategy(tic_tac_toe)

    while not tic_tac_toe.gameOver:
        strategy = MinimaxStrategy(tic_tac_toe)
        r, action = strategy.action()
        # print(f'{tic_tac_toe.getStatus()} [{r}] : {action}')
        tic_tac_toe.move2D(action[0], action[1])
        tic_tac_toe.drawText()
        # print('------------------------------------------------------------')
