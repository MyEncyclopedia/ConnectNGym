import copy
import time
from typing import Tuple, List

import gym
from gym import spaces

from PyGameConnectN import PyGameBoard
from connect_n import ConnectNGame

REWARD_A = 1
REWARD_B = -1
REWARD_TIE = 0
REWARD_NONE = None

class ConnectNGym(gym.Env):

	def __init__(self, pygameBoard: PyGameBoard):
		self.boardGame = pygameBoard
		self.board_size = pygameBoard.board_size
		self.action_space = spaces.Discrete(self.board_size * self.board_size)
		self.observation_space = spaces.Discrete(self.board_size * self.board_size)
		self.seed()
		self.reset()

	def reset(self) -> ConnectNGame:
		"""Resets the state of the environment and returns an initial observation.

		Returns:
			observation (object): the initial observation.
		"""
		connectNGame = ConnectNGame(board_size=self.board_size, N=self.connect_num)
		self.boardGame = PyGameBoard(connectNGame)
		return self.boardGame

	def step(self, action: Tuple[int, int]) -> Tuple[ConnectNGame, int, bool, None]:
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
		# loc = action
		# if self.done:
		#     return self._get_obs(), 0, True, None

		reward = REWARD_NONE
		# place
		result = self.boardGame.move(r, c)
		if self.boardGame.isGameOver():
			reward = result

		return copy.deepcopy(self.boardGame.connectNGame), reward, not result is None, None

	def render(self, mode='human', close=False) -> Tuple[int, int]:
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

	def get_available_actions(self) -> List[Tuple[int, int]]:
		return self.boardGame.getAvailablePositions()

	def show_board(self, gui=False, sec=2):
		if not gui:
			self.boardGame.connectNGame.drawText()
			time.sleep(sec)
		else:
			self.boardGame.display(sec=sec)
