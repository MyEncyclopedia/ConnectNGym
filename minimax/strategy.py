import copy
import math
import os
import sys
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple

from ConnectNGame import ConnectNGame, Move2D, GameStatus, GameAbsoluteResult


class Strategy(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def action(self, game: ConnectNGame) -> Tuple[GameAbsoluteResult, Move2D]:
        pass


class MinimaxStrategy(Strategy):
    def action(self, game: ConnectNGame) -> Tuple[GameAbsoluteResult, Move2D]:
        self.game = copy.deepcopy(game)
        result, move = self.minimax()
        return result, move

    def minimax(self) -> Tuple[GameAbsoluteResult, Move2D]:
        game = self.game
        bestMove = None
        assert not game.game_over
        if game.current_player == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.get_avail_pos_2d():
                move = pos
                result = game.move_2d(*pos)
                if result is None:
                    assert not game.game_over
                    result, oppMove = self.minimax()
                game.undo()
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == 1:
                    return 1, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.get_avail_pos_2d():
                move = pos
                result = game.move_2d(*pos)
                if result is None:
                    assert not game.game_over
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
        result, move = self.minimax_dp(self.game.get_status())
        return result, move

    @lru_cache(maxsize=None)
    def minimax_dp(self, game_state: GameStatus) -> Tuple[GameAbsoluteResult, Move2D]:
        game = self.game
        bestMove = None
        assert not game.game_over
        if game.current_player == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.get_avail_pos_2d():
                move = pos
                result = game.move_2d(*pos)
                if result is None:
                    assert not game.game_over
                    result, oppMove = self.minimax_dp(game.get_status())
                game.undo()
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == 1:
                    return 1, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.get_avail_pos_2d():
                move = pos
                result = game.move_2d(*pos)
                if result is None:
                    assert not game.game_over
                    result, oppMove = self.minimax_dp(game.get_status())
                game.undo()
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == -1:
                    return -1, move
            return ret, bestMove


class AlphaBetaStrategy(Strategy):
    def action(self, game: ConnectNGame) -> Tuple[GameAbsoluteResult, Move2D]:
        self.game = game
        result, move = self.alpha_beta(self.game.get_status(), -math.inf, math.inf)
        return result, move

    def alpha_beta(self, game_status: GameStatus, alpha: int=None, beta:int=None) \
            -> Tuple[int, Move2D]:
        game = self.game
        bestMove = None
        assert not game.game_over
        if game.current_player == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.get_avail_pos_2d():
                move = pos
                result = game.move_2d(*pos)
                if result is None:
                    assert not game.game_over
                    result, oppMove = self.alpha_beta(game.get_status(), alpha, beta)
                game.undo()
                alpha = max(alpha, result)
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == 1:
                    return ret, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.get_avail_pos_2d():
                move = pos
                result = game.move_2d(*pos)
                if result is None:
                    assert not game.game_over
                    result, oppMove = self.alpha_beta(game.get_status(), alpha, beta)
                game.undo()
                beta = min(beta, result)
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == -1:
                    return ret, move
            return ret, bestMove


class AlphaBetaDPStrategy(Strategy):
    def action(self, game: ConnectNGame) -> Tuple[GameAbsoluteResult, Move2D]:
        self.game = game
        self.alpha_beta_stack = [(-math.inf, math.inf)]
        result, move = self.alpha_beta_dp(self.game.get_status())
        return result, move

    @lru_cache(maxsize=None)
    def alpha_beta_dp(self, game_status: GameStatus) -> Tuple[GameAbsoluteResult, Move2D]:
        alpha, beta = self.alpha_beta_stack[-1]
        game = self.game
        bestMove = None
        assert not game.game_over
        if game.current_player == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.get_avail_pos_2d():
                move = pos
                result = game.move_2d(*pos)
                if result is None:
                    assert not game.game_over
                    self.alpha_beta_stack.append((alpha, beta))
                    result, oppMove = self.alpha_beta_dp(game.get_status())
                    self.alpha_beta_stack.pop()
                game.undo()
                alpha = max(alpha, result)
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == 1:
                    return ret, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.get_avail_pos_2d():
                move = pos
                result = game.move_2d(*pos)
                if result is None:
                    assert not game.game_over
                    self.alpha_beta_stack.append((alpha, beta))
                    result, oppMove = self.alpha_beta_dp(game.get_status())
                    self.alpha_beta_stack.pop()
                game.undo()
                beta = min(beta, result)
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == -1:
                    return ret, move
            return ret, bestMove


if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(n=5, board_size=7)
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
        tic_tac_toe.move_2d(action[0], action[1])
        tic_tac_toe.draw_text()
        # print('------------------------------------------------------------')
