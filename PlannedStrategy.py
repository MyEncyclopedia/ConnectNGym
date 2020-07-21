import copy
import math
from typing import Tuple

from connect_n import ConnectNGame
from strategy import Strategy


class PlannedMinimaxStrategy(Strategy):
    def __init__(self, game):
        super().__init__()
        self.game = copy.deepcopy(game)
        self.dpMap = {}
        self.result = self.minimax()
        print(self.result)

    def action(self, game) -> Tuple[int, Tuple[int, int]]:
        # self.game = copy.deepcopy(game)

        self.dpMap = {}

        return result

    def updateDP(self, status, result):
        similarStates = self.similarStatus(status)
        for s in similarStates:
            if not s in self.dpMap:
                self.dpMap[s] = result

    def minimax(self) -> int:
        similarStates = self.similarStatus(self.game.getStatus())
        for s in similarStates:
            if s in self.dpMap:
                return self.dpMap[s]
        print(f'{len(self.game.actionStack)}: {len(self.dpMap)}')

        game = self.game
        bestMove = None
        assert not game.gameOver
        thisState = game.getStatus()

        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result = self.minimax()
                game.undo()
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == 1:
                    self.updateDP(thisState, ret)
                    return 1
            self.updateDP(thisState, ret)
            return ret
        else:
            ret = math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result = self.minimax()
                game.undo()
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == -1:
                    self.updateDP(thisState, ret)
                    return -1
            self.updateDP(thisState, ret)
            return ret


    def similarStatus(self, status):
        ret = []
        rotatedS = status
        for _ in range(4):
            rotatedS = self.rotate(rotatedS)
            ret.append(rotatedS)

        return ret


    def rotate(self, s):
        N = len(s)
        board = [[ConnectNGame.AVAILABLE] * N for _ in range(N)]

        for r in range(N):
            for c in range(N):
                board[c][N-1-r] = s[r][c]

        return tuple([tuple(board[i]) for i in range(N)])


if __name__ == '__main__':
    connectNGame = ConnectNGame(N=3, board_size=3)

    strategy = PlannedMinimaxStrategy(connectNGame)
