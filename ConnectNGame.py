from typing import List, Tuple

from typing import NewType

GameStatus = Tuple[Tuple[int, ...]]
Move2D = Tuple[int, int]
Pos = int

class ConnectNGame:
    PLAYER_A = 1
    PLAYER_B = -1
    AVAILABLE = 0
    RESULT_TIE = 0
    RESULT_A_WIN = 1
    RESULT_B_WIN = -1

    def __init__(self, N: int = 3, board_size: int = 3):
        assert N <= board_size
        self.N = N
        self.board_size = board_size
        self.board = [[ConnectNGame.AVAILABLE] * self.board_size for _ in range(self.board_size)]
        self.gameOver = False
        self.gameResult = None
        self.currentPlayer = ConnectNGame.PLAYER_A
        self.remainingPosNum = self.board_size * self.board_size
        self.actionStack = []

    def move(self, pos: Pos) -> int:
        r, c = pos // self.board_size, pos % self.board_size
        return self.move_2d(r, c)

    def move_2d(self, r: int, c: int) -> int:
        """

        :param r:
        :param c:
        :return: None: game ongoing
        """
        assert self.board[r][c] == ConnectNGame.AVAILABLE
        self.board[r][c] = self.currentPlayer
        self.actionStack.append((r, c))
        self.remainingPosNum -= 1
        if self.check_win(r, c):
            self.gameOver = True
            self.gameResult = self.currentPlayer
            return self.currentPlayer
        if self.remainingPosNum == 0:
            self.gameOver = True
            self.gameResult = ConnectNGame.RESULT_TIE
            return ConnectNGame.RESULT_TIE
        self.currentPlayer *= -1

    def undo(self):
        if len(self.actionStack) > 0:
            lastAction = self.actionStack.pop()
            r, c = lastAction
            self.board[r][c] = ConnectNGame.AVAILABLE
            self.currentPlayer = ConnectNGame.PLAYER_A if len(self.actionStack) % 2 == 0 else ConnectNGame.PLAYER_B
            self.remainingPosNum += 1
            self.gameOver = False
            self.gameResult = None
        else:
            raise Exception('No lastAction')

    def check_win(self, r: int, c: int) -> bool:
        north = self.get_connected_num(r, c, -1, 0)
        south = self.get_connected_num(r, c, 1, 0)

        east = self.get_connected_num(r, c, 0, 1)
        west = self.get_connected_num(r, c, 0, -1)

        south_east = self.get_connected_num(r, c, 1, 1)
        north_west = self.get_connected_num(r, c, -1, -1)

        north_east = self.get_connected_num(r, c, -1, 1)
        south_west = self.get_connected_num(r, c, 1, -1)

        if (north + south + 1 >= self.N) or (east + west + 1 >= self.N) or \
                (south_east + north_west + 1 >= self.N) or (north_east + south_west + 1 >= self.N):
            return True
        return False

    def get_connected_num(self, r: int, c: int, dr: int, dc: int) -> int:
        player = self.board[r][c]
        result = 0
        i = 1
        while True:
            new_r = r + dr * i
            new_c = c + dc * i
            if 0 <= new_r < self.board_size and 0 <= new_c < self.board_size:
                if self.board[new_r][new_c] == player:
                    result += 1
                else:
                    break
            else:
                break
            i += 1
        return result

    def get_avail_pos_2d(self) -> List[Move2D]:
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if
                self.board[i][j] == ConnectNGame.AVAILABLE]

    def get_avail_pos(self) -> List[Pos]:
        return [i * self.board_size + j
                for i in range(self.board_size)
                for j in range(self.board_size)
                if self.board[i][j] == ConnectNGame.AVAILABLE]

    def get_status(self) -> GameStatus:
        return tuple([tuple(self.board[i]) for i in range(self.board_size)])

    def check_action(self, r: int, c: int) -> bool:
        return self.board[r][c] == ConnectNGame.AVAILABLE

    def reset(self):
        self.board = [[ConnectNGame.AVAILABLE] * self.board_size for _ in range(self.board_size)]
        self.gameOver = False
        self.gameResult = None
        self.currentPlayer = ConnectNGame.PLAYER_A
        self.remainingPosNum = self.board_size * self.board_size
        self.actionStack = []

    def draw_text(self):
        print('')
        print('------')
        for r in range(self.N):
            row = ''
            for c in range(self.N):
                row += 'O' if self.board[r][c] == ConnectNGame.PLAYER_A else 'X' if self.board[r][
                                                                                        c] == ConnectNGame.PLAYER_B else '.'
            print(row)
        print('------')


if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(N=3, board_size=3)
    tic_tac_toe.move_2d(0, 0)
    tic_tac_toe.move_2d(1, 1)

    # print(minimax(tic_tac_toe, True))
    # print(minimax_dp(tic_tac_toe, tic_tac_toe.getStatus()))
    # print(alpha_beta(tic_tac_toe, tic_tac_toe.getStatus(), -math.inf, math.inf))
