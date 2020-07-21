import copy
import time

import gym
from gym import spaces

from PyGameConnectN import PyGameBoard

REWARD_A = 1
REWARD_B = -1
REWARD_TIE = 0
REWARD_NONE = None

class ConnectNGym(gym.Env):

    def __init__(self, grid_num=3, connect_num=3):
        self.grid_num = grid_num
        self.connect_num = connect_num
        self.action_space = spaces.Discrete(grid_num* grid_num)
        self.observation_space = spaces.Discrete(grid_num * grid_num)

        # self.boardGame = PyGameBoard(board_size=grid_num, connect_num=connect_num)

        self.seed()
        # self.reset()

    def reset(self) -> PyGameBoard:
        self.boardGame = PyGameBoard(board_size=self.grid_num, connect_num=self.connect_num)
        # return copy.deepcopy(self.boardGame.connectNGame)
        return self.boardGame

    def step(self, action):
        """Step environment by action.

        Args:
            action (int): Location

        Returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """
        # assert self.action_space.contains(action)

        r, c = action
        # loc = action
        # if self.done:
        #     return self._get_obs(), 0, True, None

        reward = REWARD_NONE
        # place
        result = self.boardGame.move(r, c)
        if self.boardGame.isGameOver():
            reward = result

        return copy.deepcopy(self.boardGame.connectNGame), reward, not result is None, None

    def render(self, mode='human', close=False):
        self.action = self.boardGame.next_user_input()
        return self.action

        # if close:
        #     return
        # if mode == 'human':
        #     # self._show_board(print)  # NOQA
        #     print('')
        # else:
        #     pass
        #     # self._show_board(logging.info)
        #     # logging.info('')

    def get_available_actions(self):
        return self.boardGame.getAvailablePositions()

    def show_board(self, gui=False, sec=2):
        if not gui:
            self.boardGame.connectNGame.drawText()
            time.sleep(sec)
        else:
            self.boardGame.display(sec=sec)
