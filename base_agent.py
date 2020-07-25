#!/usr/bin/env python
import random
import time

from ConnectNGym import ConnectNGym
from PlannedStrategy import PlannedMinimaxStrategy
from PyGameConnectN import PyGameBoard
from connect_n import ConnectNGame
from strategy import MinimaxStrategy, Strategy


class BaseAgent(object):
	def __init__(self):
		pass

	def act(self, game: PyGameBoard, available_actions):
		return random.choice(available_actions)


class AIAgent(BaseAgent):
	def __init__(self, strategy: Strategy):
		self.strategy = strategy

	def act(self, game: PyGameBoard, available_actions):
		result, move = self.strategy.action(game.connectNGame)
		assert move in available_actions
		return move


class HumanAgent(BaseAgent):
	def __init__(self):
		pass

	def act(self, game: PyGameBoard, available_actions):
		return game.next_user_input()


def play_human_vs_human(env: ConnectNGym):
	play(env, HumanAgent(), AIAgent(MinimaxStrategy()))


def play_human_vs_ai(env: ConnectNGym):
	pygameBoard: PyGameBoard = env.reset()
	plannedMinimaxAgent = AIAgent(PlannedMinimaxStrategy(pygameBoard.connectNGame))
	play(env, HumanAgent(), plannedMinimaxAgent)


def play_ai_vs_ai(env: ConnectNGym):
	pygameBoard: PyGameBoard = env.reset()
	plannedMinimaxAgent = AIAgent(PlannedMinimaxStrategy(pygameBoard.connectNGame))
	play(env, plannedMinimaxAgent, plannedMinimaxAgent)


def play(env: ConnectNGym, agent1: BaseAgent, agent2: BaseAgent):
	agents = [agent1, agent2]

	while True:
		env.reset()
		done = False
		env.show_board(True)
		agent_id = -1
		while not done:
			agent_id = (agent_id + 1) % 2
			available_actions = env.get_available_actions()
			agent = agents[agent_id]
			action = agent.act(pygameBoard, available_actions)
			_, reward, done, info = env.step(action)
			env.show_board(True)

			if done:
				print(f'result={reward}')
				time.sleep(3)
				break


if __name__ == '__main__':
	pygameBoard = PyGameBoard(connectNGame=ConnectNGame(board_size=3, N=3))
	env = ConnectNGym(pygameBoard)

	# play_ai_vs_ai(env)
	play_human_vs_ai(env)
