import sys
from typing import List, Tuple
import pygame
from pygame.event import Event
from ConnectNGame import ConnectNGame, Move2D, Pos, GameStatus


class PyGameBoard:

    def __init__(self, connect_n_game: ConnectNGame):
        self.connect_n_game = connect_n_game
        self.board_size = connect_n_game.board_size
        self.connect_num = connect_n_game.n
        self.grid_size = 40
        self.start_x, self.start_y = 30, 50
        self.edge_size = self.grid_size / 2
        self.action = None

        pygame.init()

        window_size = max(300, self.grid_size * self.board_size + 80)
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption(f"Connect-{self.connect_num}, {self.board_size}x{self.board_size}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)

    def next_user_input(self) -> Pos:
        self.action = None
        while not self.action:
            self.check_event()
            self._render()
            self.clock.tick(60)
        return self.action[0] * self.board_size + self.action[1]

    def check_event(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self._handle_user_input(e)

    def display(self, milli_sec=2000):
        # tick_num = sec * 1000
        while milli_sec >= 0:
            pygame.event.get()
            self._render()
            passed = self.clock.tick(1)
            # print(tick_num)
            milli_sec -= passed

    # proxy methods
    def move(self, pos: Pos) -> int:
        # r, c = pos // self.board_size, pos % self.board_size
        return self.connect_n_game.move(pos)

    def is_game_over(self) -> bool:
        return self.connect_n_game.game_over

    # def get_avail_positions_2d(self) -> List[Move2D]:
    #     return self.connect_n_game.get_avail_pos_2d()

    def get_avail_pos(self) -> List[Pos]:
        return self.connect_n_game.get_avail_pos()

    def get_current_player(self) -> int:
        return self.connect_n_game.current_player

    def get_status(self) -> GameStatus:
        return self.connect_n_game.get_status()

    def _render(self):
        self.screen.fill((255, 255, 255))
        # print(self.clock.get_fps())
        msg = None
        if self.connect_n_game.game_over:
            title = f"Game Over"
            pygame.display.set_caption(title)
            if self.connect_n_game.game_result == ConnectNGame.RESULT_TIE:
                msg = 'Draw'
            else:
                msg = "{0} Win".format("Black" if self.connect_n_game.game_result == ConnectNGame.PLAYER_A else "White")
        else:
            pygame.display.set_caption(f"Connect-{self.connect_num}, {self.board_size}x{self.board_size}")
            msg = "{0} Turn".format("Black" if self.connect_n_game.current_player == ConnectNGame.PLAYER_A else "White")
        self.screen.blit(self.font.render(msg, True, (0, 122, 255)),
                         (self.grid_size * self.board_size + 30, 50))

        self._draw()
        # if self.connectNGame.gameOver:
        #     winner_msg = "{0} Win".format("Black" if self.connectNGame.gameResult == ConnectNGame.PLAYER_A else "White")
        #     title = "Game Over " + winner_msg + title

        pygame.display.update()

    def _handle_user_input(self, e: Event) -> Move2D:
        origin_x = self.start_x - self.edge_size
        origin_y = self.start_y - self.edge_size
        size = (self.board_size - 1) * self.grid_size + self.edge_size * 2
        pos = e.pos
        if origin_x <= pos[0] <= origin_x + size and origin_y <= pos[1] <= origin_y + size:
            if not self.connect_n_game.game_over:
                x = pos[0] - origin_x
                y = pos[1] - origin_y
                r = int(y // self.grid_size)
                c = int(x // self.grid_size)
                valid = self.connect_n_game.check_action(r, c)
                if valid:
                    self.action = (r, c)
                    return self.action

    def _draw(self):
        screen = self.screen
        pygame.draw.rect(screen, (192, 192, 192),
                         [self.start_x - self.edge_size, self.start_y - self.edge_size,
                          (self.board_size - 1) * self.grid_size + self.edge_size * 2,
                          (self.board_size - 1) * self.grid_size + self.edge_size * 2], 0)

        for r in range(self.board_size):
            y = self.start_y + r * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [self.start_x, y],
                             [self.start_x + self.grid_size * (self.board_size - 1), y], 2)

        for c in range(self.board_size):
            x = self.start_x + c * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [x, self.start_y],
                             [x, self.start_y + self.grid_size * (self.board_size - 1)], 2)

        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.connect_n_game.board[r][c]
                if piece != ConnectNGame.AVAILABLE:
                    if piece == ConnectNGame.PLAYER_A:
                        color = (0, 0, 0)
                    else:
                        color = (255, 255, 255)

                    x = self.start_x + c * self.grid_size
                    y = self.start_y + r * self.grid_size
                    pygame.draw.circle(screen, color, [x, y], self.grid_size // 2)


if __name__ == '__main__':
    connectNGame = ConnectNGame()
    board = PyGameBoard(connectNGame)
    while not board.is_game_over():
        pos = board.next_user_input()
        board.move(pos)


    pygame.quit()
