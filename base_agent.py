#!/usr/bin/env python
import random

from ConnectNGym import ConnectNGym
from strategy import MinimaxStrategy


class BaseAgent(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def act(self, game, available_actions):
        self.game = game
        return random.choice(available_actions)

class StrategyAgent(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def act(self, game, available_actions):
        s = MinimaxStrategy()
        result, move = s.action(game)
        assert move in available_actions
        return move

def play():
    env = ConnectNGym()
    agents = [StrategyAgent(None), StrategyAgent(None)]

    game = env.reset()
    done = False
    env.show_board(False)
    agent_id = 0
    while not done:
        available_actions = env.get_available_actions()
        agent = agents[agent_id]
        action = agent.act(game, available_actions)
        game, reward, done, info = env.step(action)
        env.show_board(False)

        if done:
            print(reward)
            break

        agent_id = (agent_id + 1) % 2

if __name__ == '__main__':
    play()
