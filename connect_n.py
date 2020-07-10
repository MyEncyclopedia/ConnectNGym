from typing import List, Tuple

class ConnectNGame:

    PLAYER_A = 1
    PLAYER_B = -1
    AVAILABLE = 0
    RESULT_TIE = 0
    RESULT_A_WIN = 1
    RESULT_B_WIN = -1

    def __init__(self, N:int = 3, board_size:int = 3):
        assert N <= board_size
        self.N = N
        self.board_size = board_size
        self.board = [[ConnectNGame.AVAILABLE] * board_size for _ in range(board_size)]
        self.gameOver = False
        self.gameResult = None
        self.currentPlayer = ConnectNGame.PLAYER_A
        self.remainingPosNum = board_size * board_size
        self.actionStack = []

    def move(self, r: int, c: int):
        """

        :param r:
        :param c:
        :return: None: game ongoing
        """
        assert self.board[r][c] == ConnectNGame.AVAILABLE
        self.board[r][c] = self.currentPlayer
        self.actionStack.append((r, c))
        self.remainingPosNum -= 1
        if self.checkWin(r, c):
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

    def checkWin(self, r: int, c: int) -> bool:
        north = self.getConnectedNum(r, c, -1, 0)
        south = self.getConnectedNum(r, c, 1, 0)

        east = self.getConnectedNum(r, c, 0, 1)
        west = self.getConnectedNum(r, c, 0, -1)

        south_east = self.getConnectedNum(r, c, 1, 1)
        north_west = self.getConnectedNum(r, c, -1, -1)

        north_east = self.getConnectedNum(r, c, -1, 1)
        south_west = self.getConnectedNum(r, c, 1, -1)

        if (north + south + 1 >= self.N) or (east + west + 1 >= self.N) or \
                (south_east + north_west + 1 >= self.N) or (north_east + south_west + 1 >= self.N):
            return True
        return False

    def getConnectedNum(self, r: int, c: int, dr: int, dc: int) -> int:
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

    def getAvailablePositions(self) -> List[Tuple[int, int]]:
        return [(i,j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i][j] == ConnectNGame.AVAILABLE]

    def getStatus(self):
        return tuple([tuple(self.board[i]) for i in range(self.board_size)])


    def checkAction(self, r, c):
        return self.board[r][c] == ConnectNGame.AVAILABLE

    def drawText(self):
        print('')
        print('------')
        for r in range(self.N):
            row = ''
            for c in range(self.N):
                row += 'O' if self.board[r][c] == ConnectNGame.PLAYER_A else 'X' if self.board[r][c] == ConnectNGame.PLAYER_B else '.'
            print(row)
        print('------')


if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(N=3, board_size=3)
    tic_tac_toe.move(0, 0)
    tic_tac_toe.move(1, 1)

    # print(minimax(tic_tac_toe, True))
    # print(minimax_dp(tic_tac_toe, tic_tac_toe.getStatus()))
    # print(alpha_beta(tic_tac_toe, tic_tac_toe.getStatus(), -math.inf, math.inf))

