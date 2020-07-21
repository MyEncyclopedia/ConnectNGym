#!/usr/bin/env python
import random

from ConnectNGym import ConnectNGym
from PyGameConnectN import PyGameBoard
from strategy import MinimaxStrategy


class BaseAgent(object):
    def __init__(self):
        pass

    def act(self, game:PyGameBoard, available_actions):
        return random.choice(available_actions)

class StrategyAgent(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def act(self, game: PyGameBoard, available_actions):
        s = MinimaxStrategy()
        result, move = s.action(game.connectNGame)
        assert move in available_actions
        return move

class HumanAgent(object):
    def __init__(self):
        pass

    def act(self, game: PyGameBoard, available_actions):
        return game.next_user_input()

def play():
    env = ConnectNGym()
    # agents = [StrategyAgent(None), StrategyAgent(None)]
    agents = [HumanAgent(), StrategyAgent(None)]

    game: PyGameBoard = env.reset()
    done = False
    env.show_board(True)
    agent_id = 0
    while not done:
        available_actions = env.get_available_actions()
        agent = agents[agent_id]
        action = agent.act(game, available_actions)
        _, reward, done, info = env.step(action)
        env.show_board(True)

        if done:
            print(reward)
            break

        agent_id = (agent_id + 1) % 2

if __name__ == '__main__':
    play()
