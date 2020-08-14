#!/usr/bin/env python
import random
import time

from ConnectNGym import ConnectNGym
from PlannedStrategy import PlannedMinimaxStrategy
from PyGameConnectN import PyGameBoard
from connect_n import ConnectNGame, Pos
from strategy import MinimaxStrategy, Strategy


class BaseAgent(object):
    def __init__(self):
        pass

    def get_action(self, game: PyGameBoard) -> Pos:
        available_actions = game.get_avail_pos()
        return random.choice(available_actions)


class AIAgent(BaseAgent):
    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def get_action(self, game: PyGameBoard) -> Pos:
        available_actions = game.get_avail_pos()
        result, move = self.strategy.action(game.connectNGame)
        assert move in available_actions
        return move


class HumanAgent(BaseAgent):
    def __init__(self):
        pass

    def get_action(self, game: PyGameBoard) -> Pos:
        return game.next_user_input()


def play_human_vs_human(env: ConnectNGym):
    play(env, HumanAgent(), HumanAgent())


def play_human_vs_ai(env: ConnectNGym):
    plannedMinimaxAgent = AIAgent(PlannedMinimaxStrategy(env.pygameBoard.connectNGame))
    play(env, HumanAgent(), plannedMinimaxAgent)


def play_ai_vs_ai(env: ConnectNGym):
    plannedMinimaxAgent = AIAgent(PlannedMinimaxStrategy(env.pygameBoard.connectNGame))
    play(env, plannedMinimaxAgent, plannedMinimaxAgent)


def play(env: ConnectNGym, agent1: BaseAgent, agent2: BaseAgent):
    agents = [agent1, agent2]

    while True:
        env.reset()
        done = False
        agent_id = -1
        while not done:
            agent_id = (agent_id + 1) % 2
            agent = agents[agent_id]
            action = agent.get_action(pygameBoard)
            _, reward, done, info = env.step(action)
            env.render(True)

            if done:
                print(f'result={reward}')
                time.sleep(3)
                break


if __name__ == '__main__':
    pygameBoard = PyGameBoard(connectNGame=ConnectNGame(board_size=3, N=3))
    env = ConnectNGym(pygameBoard)
    env.render(True)

    play_ai_vs_ai(env)
# play_human_vs_ai(env)
