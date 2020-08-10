import copy
import time
from typing import Tuple, List

import gym
from gym import spaces

from PyGameConnectN import PyGameBoard
from connect_n import ConnectNGame, Move1D

REWARD_A = 1
REWARD_B = -1
REWARD_TIE = 0
REWARD_NONE = None


class ConnectNGym(gym.Env):

    def __init__(self, pygameBoard: PyGameBoard, isGUI=True, displaySec=2):
        self.pygameBoard = pygameBoard
        self.isGUI = isGUI
        self.displaySec = displaySec
        self.action_space = spaces.Discrete(pygameBoard.board_size * pygameBoard.board_size)
        self.observation_space = spaces.Discrete(pygameBoard.board_size * pygameBoard.board_size)
        self.seed()
        self.reset()

    def reset(self) -> ConnectNGame:
        """Resets the state of the environment and returns an initial observation.

        Returns:
          observation (object): the initial observation.
        """
        self.pygameBoard.connectNGame.reset()
        return copy.deepcopy(self.pygameBoard.connectNGame)

    def step(self, action: Move1D) -> Tuple[ConnectNGame, int, bool, None]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
          action (object): an action provided by the agent

        Returns:
          observation (object): agent's observation of the current environment
          reward (float) : amount of reward returned after previous action
          done (bool): whether the episode has ended, in which case further step() calls will return undefined results
          info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # assert self.action_space.contains(action)

        r, c = action
        reward = REWARD_NONE
        result = self.pygameBoard.move(r, c)
        if self.pygameBoard.isGameOver():
            reward = result

        return copy.deepcopy(self.pygameBoard.connectNGame), reward, not result is None, None

    def render(self, mode='human'):
        """
        Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
        Make sure that your class's metadata 'render.modes' key includes
        the list of supported modes. It's recommended to call super()
        in implementations to use the functionality of this method.

        Args:
          mode (str): the mode to render with
        """
        if not self.isGUI:
            self.pygameBoard.connectNGame.drawText()
            time.sleep(self.displaySec)
        else:
            self.pygameBoard.display(sec=self.displaySec)

    def get_available_actions(self) -> List[Tuple[int, int]]:
        return self.pygameBoard.getAvailablePositions()
