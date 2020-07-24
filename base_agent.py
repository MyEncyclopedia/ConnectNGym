#!/usr/bin/env python
import random

from ConnectNGym import ConnectNGym
from PlannedStrategy import PlannedMinimaxStrategy
from PyGameConnectN import PyGameBoard
from strategy import MinimaxStrategy, Strategy


class BaseAgent(object):
    def __init__(self):
        pass

    def act(self, game:PyGameBoard, available_actions):
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

def play():
    env = ConnectNGym()

    pygameBoard: PyGameBoard = env.reset()
    # agents = [HumanAgent(), AIAgent(MinimaxStrategy())]
    agents = [HumanAgent(), AIAgent(PlannedMinimaxStrategy(pygameBoard.connectNGame))]

    done = False
    env.show_board(True)
    agent_id = 0
    while not done:
        available_actions = env.get_available_actions()
        agent = agents[agent_id]
        action = agent.act(pygameBoard, available_actions)
        _, reward, done, info = env.step(action)
        env.show_board(True)

        if done:
            print(reward)
            break

        agent_id = (agent_id + 1) % 2

if __name__ == '__main__':
    play()
