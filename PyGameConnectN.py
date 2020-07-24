from typing import List, Tuple

import pygame
from pygame.event import Event

from connect_n import ConnectNGame


class PyGameBoard:

    def __init__(self, board_size=3, connect_num=3):
        self.grid_size = 30
        self.start_x, self.start_y = 30, 50
        self.edge_size = self.grid_size / 2
        self.board_size = board_size
        self.connectNGame = ConnectNGame(N=connect_num, board_size=board_size)
        self.action = None

        pygame.init()

        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption(f"Connect-{connect_num}, {board_size}x{board_size}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 24)
        self.going = True


    def check_event(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.going = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self.handle_user_input(e)

    def next_user_input(self) -> Tuple[int, int]:
        self.action = None
        while not self.action:
            self.check_event()
            self.render()
            self.clock.tick(60)
        return self.action


    def display(self, sec=2):
        tick_num = sec * 1000
        while tick_num >= 0:
            pygame.event.get()
            self.render()
            passed = self.clock.tick(1)
            print(tick_num)
            tick_num -= passed

    # proxy methods
    def move(self, r: int, c: int) -> int:
        return self.connectNGame.move(r, c)

    def isGameOver(self) -> bool:
        return self.connectNGame.gameOver

    def getAvailablePositions(self) -> List[Tuple[int, int]]:
        return self.connectNGame.getAvailablePositions()

    def getCurrentPlayer(self) -> int:
        return self.connectNGame.currentPlayer

    def getStatus(self) -> Tuple[Tuple[int, ...]]:
        return self.connectNGame.getStatus()

    def render(self):
        self.screen.fill((255, 255, 255))
        # self.screen.blit(self.font.render("FPS: {0:.2F}".format(self.clock.get_fps()), True, (0, 0, 0)), (10, 10))

        self.draw()
        if self.connectNGame.gameOver:
            title = f"Connect-{self.connectNGame.N}, {self.connectNGame.board_size}x{self.connectNGame.board_size}"
            winner_msg = "{0} Win".format("Black" if self.connectNGame.gameResult == ConnectNGame.PLAYER_A else "White")
            title = "Game Over " + winner_msg + title
            pygame.display.set_caption(title)

        pygame.display.update()

    def handle_user_input(self, e: Event) -> Tuple[int, int]:
        origin_x = self.start_x - self.edge_size
        origin_y = self.start_y - self.edge_size
        size = (self.board_size - 1) * self.grid_size + self.edge_size * 2
        pos = e.pos
        if origin_x <= pos[0] <= origin_x + size and origin_y <= pos[1] <= origin_y + size:
            if not self.connectNGame.gameOver:
                x = pos[0] - origin_x
                y = pos[1] - origin_y
                r = int(y // self.grid_size)
                c = int(x // self.grid_size)
                valid = self.connectNGame.checkAction(r, c)
                if valid:
                    self.action = (r, c)
                    return self.action

    def draw(self):
        screen = self.screen
        pygame.draw.rect(screen, (185, 122, 87),
                         [self.start_x - self.edge_size, self.start_y - self.edge_size,
                          (self.board_size - 1) * self.grid_size + self.edge_size * 2, (self.board_size - 1) * self.grid_size + self.edge_size * 2], 0)

        for r in range(self.board_size):
            y = self.start_y + r * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [self.start_x, y], [self.start_x + self.grid_size * (self.board_size - 1), y], 2)

        for c in range(self.board_size):
            x = self.start_x + c * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [x, self.start_y], [x, self.start_y + self.grid_size * (self.board_size - 1)], 2)

        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.connectNGame.board[r][c]
                if piece != ConnectNGame.AVAILABLE:
                    if piece == ConnectNGame.PLAYER_A:
                        color = (0, 0, 0)
                    else:
                        color = (255, 255, 255)

                    x = self.start_x + c * self.grid_size
                    y = self.start_y + r * self.grid_size
                    pygame.draw.circle(screen, color, [x, y], self.grid_size // 2)


if __name__ == '__main__':
    # game = PyGameBoard(board_size=15, connect_num=5)
    game = PyGameBoard()
    while game.going:
        pos = game.next_user_input()
        game.move(*pos)

    pygame.quit()

