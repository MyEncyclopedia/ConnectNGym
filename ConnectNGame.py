from typing import List, Tuple

from typing import NewType

GameStatus = Tuple[Tuple[int, ...]]
Move2D = Tuple[int, int]
Pos = int
GameAbsoluteResult = int # 1: A wins; 0: Tie; -1: B wins
GameRelativeResult = int # 1: current player wins; 0: Tie; -1: opponent wins


class ConnectNGame:
    PLAYER_A = 1
    PLAYER_B = -1
    AVAILABLE = 0
    RESULT_TIE = 0
    # RESULT_A_WIN = 1
    # RESULT_B_WIN = -1

    def __init__(self, n: int = 3, board_size: int = 3):
        assert n <= board_size
        self.n = n
        self.board_size = board_size
        self.board = [[ConnectNGame.AVAILABLE] * self.board_size for _ in range(self.board_size)]
        self.game_over = False
        self.game_result = None
        self.current_player = ConnectNGame.PLAYER_A
        self.remaining_pos_num = self.board_size * self.board_size
        self.action_stack = []

    def move(self, pos: Pos) -> GameAbsoluteResult:
        r, c = pos // self.board_size, pos % self.board_size
        return self._move_2d(r, c)

    def _move_2d(self, r: int, c: int) -> GameAbsoluteResult:
        """

        :param r:
        :param c:
        :return: None: game ongoing
        """
        assert self.board[r][c] == ConnectNGame.AVAILABLE
        self.board[r][c] = self.current_player
        self.action_stack.append((r, c))
        self.remaining_pos_num -= 1
        if self.check_win(r, c):
            self.game_over = True
            self.game_result = self.current_player
            return self.current_player
        if self.remaining_pos_num == 0:
            self.game_over = True
            self.game_result = ConnectNGame.RESULT_TIE
            return ConnectNGame.RESULT_TIE
        self.current_player *= -1

    def undo(self):
        if len(self.action_stack) > 0:
            last_action = self.action_stack.pop()
            r, c = last_action
            self.board[r][c] = ConnectNGame.AVAILABLE
            self.current_player = ConnectNGame.PLAYER_A if len(self.action_stack) % 2 == 0 else ConnectNGame.PLAYER_B
            self.remaining_pos_num += 1
            self.game_over = False
            self.game_result = None
        else:
            raise Exception('No last_action')

    def check_win(self, r: int, c: int) -> bool:
        north = self.get_connected_num(r, c, -1, 0)
        south = self.get_connected_num(r, c, 1, 0)

        east = self.get_connected_num(r, c, 0, 1)
        west = self.get_connected_num(r, c, 0, -1)

        south_east = self.get_connected_num(r, c, 1, 1)
        north_west = self.get_connected_num(r, c, -1, -1)

        north_east = self.get_connected_num(r, c, -1, 1)
        south_west = self.get_connected_num(r, c, 1, -1)

        if (north + south + 1 >= self.n) or (east + west + 1 >= self.n) or \
                (south_east + north_west + 1 >= self.n) or (north_east + south_west + 1 >= self.n):
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

    # def get_avail_pos_2d(self) -> List[Move2D]:
    #     return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if
    #             self.board[i][j] == ConnectNGame.AVAILABLE]

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
        self.game_over = False
        self.game_result = None
        self.current_player = ConnectNGame.PLAYER_A
        self.remaining_pos_num = self.board_size * self.board_size
        self.action_stack = []

    def draw_text(self):
        for r in range(self.n):
            row = ''
            for c in range(self.n):
                row += 'O' if self.board[r][c] == ConnectNGame.PLAYER_A else 'X' \
                    if self.board[r][c] == ConnectNGame.PLAYER_B else '.'
            print(f'{row}\n')


if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(n=3, board_size=3)
    tic_tac_toe._move_2d(0, 0)
    tic_tac_toe._move_2d(1, 1)

    # print(minimax(tic_tac_toe, True))
    # print(minimax_dp(tic_tac_toe, tic_tac_toe.getStatus()))
    # print(alpha_beta(tic_tac_toe, tic_tac_toe.getStatus(), -math.inf, math.inf))
