import copy
import math
from typing import Tuple, List

from ConnectNGame import ConnectNGame, GameStatus, Move2D, GameResult, Pos
from minimax.strategy import Strategy


class PlannedMinimaxStrategy(Strategy):
    def __init__(self, game: ConnectNGame):
        super().__init__()
        self.game = copy.deepcopy(game)
        self.dp_map = {}  # game_status => result, move
        self.result = self.minimax(game.get_status())
        print(f'best result: {self.result}')

    def action(self, game: ConnectNGame) -> Tuple[GameResult, Pos]:
        game = copy.deepcopy(game)

        player = game.current_player
        best_result = player * -1  # assume opponent win as worst result
        best_move = None
        for move in game.get_avail_pos():
            game.move(move)
            status = game.get_status()
            game.undo()

            result = self.dp_map[status]

            if player == ConnectNGame.PLAYER_A:
                best_result = max(best_result, result)
            else:
                best_result = min(best_result, result)
            # update best_move if any improvement
            best_move = move if best_result == result else best_move
            print(f'move {move} => {result}')

        # if best_result == game.currentPlayer:
        #     return best_result, move

        return best_result, best_move

    def minimax(self, game_status: GameStatus) -> Tuple[GameResult, Pos]:
        similar_states = self.get_similar_status(self.game.get_status())
        for s in similar_states:
            if s in self.dp_map:
                return self.dp_map[s]
        print(f'{len(self.game.action_stack)}: {len(self.dp_map)}')

        game = self.game
        best_move = None
        assert not game.game_over
        this_state = game.get_status()

        if game.current_player == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.get_avail_pos():
                move = pos
                result = game.move(pos)
                if result is None:
                    assert not game.game_over
                    result = self.minimax(game.get_status())
                    self._update_dp(game.get_status(), result)
                else:
                    self._update_dp(game.get_status(), result)
                game.undo()
                ret = max(ret, result)
                best_move = move if ret == result else best_move
            self._update_dp(this_state, ret)
            return ret
        else:
            ret = math.inf
            for pos in game.get_avail_pos():
                move = pos
                result = game.move(pos)
                if result is None:
                    assert not game.game_over
                    result = self.minimax(game.get_status())
                    self._update_dp(game.get_status(), result)
                else:
                    self._update_dp(game.get_status(), result)
                game.undo()
                ret = min(ret, result)
                best_move = move if ret == result else best_move
            self._update_dp(this_state, ret)
            return ret

    def _update_dp(self, status: GameStatus, result: GameResult):
        similarStates = self.get_similar_status(status)
        for s in similarStates:
            if not s in self.dp_map:
                self.dp_map[s] = result

    def get_similar_status(self, status: GameStatus) -> List[GameStatus]:
        ret = []
        rotatedS = status
        for _ in range(4):
            rotatedS = self.rotate(rotatedS)
            ret.append(rotatedS)
        return ret

    def rotate(self, status: GameStatus) -> GameStatus:
        N = len(status)
        board = [[ConnectNGame.AVAILABLE] * N for _ in range(N)]

        for r in range(N):
            for c in range(N):
                board[c][N - 1 - r] = status[r][c]

        return tuple([tuple(board[i]) for i in range(N)])


if __name__ == '__main__':
    connectNGame = ConnectNGame(n=3, board_size=3)

    strategy = PlannedMinimaxStrategy(connectNGame)
