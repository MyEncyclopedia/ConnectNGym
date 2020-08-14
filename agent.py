#!/usr/bin/env python
import random

from ConnectNGym import ConnectNGym
from minimax.PlannedStrategy import PlannedMinimaxStrategy
from PyGameConnectN import PyGameBoard
from ConnectNGame import ConnectNGame, Pos
from minimax.strategy import Strategy


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
        result, move = self.strategy.action(game.connect_n_game)
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
    planned_minimax_agent = AIAgent(PlannedMinimaxStrategy(env.pygameBoard.connectNGame))
    play(env, HumanAgent(), planned_minimax_agent)


def play_ai_vs_ai(env: ConnectNGym):
    planned_minimax_agent = AIAgent(PlannedMinimaxStrategy(env.pygameBoard.connectNGame))
    play(env, planned_minimax_agent, planned_minimax_agent)


def play(env: ConnectNGym, agent1: BaseAgent, agent2: BaseAgent, render=True) -> int:
    agents = [agent1, agent2]

    env.reset()
    done = False
    agent_id = -1
    while not done:
        agent_id = (agent_id + 1) % 2
        agent = agents[agent_id]
        action = agent.get_action(board)
        _, reward, done, info = env.step(action)
        env.render(render)

        if done:
            print(f'result={reward}')
            return reward


if __name__ == '__main__':
    board = PyGameBoard(connect_n_game=ConnectNGame(board_size=3, n=3))
    env = ConnectNGym(board)
    env.render(True)

    play_ai_vs_ai(env)
# play_human_vs_ai(env)
