# -*- coding: utf-8 -*-
from typing import Tuple, List, Iterator, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nptyping import NDArray
from torch.autograd import Variable
import numpy as np

from ConnectNGame import ConnectNGame, Pos

ActionProbs = NDArray[(Any), np.float]
MoveWithProb = Tuple[Pos, ActionProbs]
NetGameState = NDArray[(4, Any, Any), np.int]


def convert_game_state(game: ConnectNGame) -> NetGameState:
    """
    Converts game state to type NetGameState as ndarray.

    :param game:
    :return:
        Of shape 4 * board_size * board_size.
        [0] is current player positions.
        [1] is opponent positions.
        [2] is last move location.
        [3] is the color to play.
    """
    state_matrix = np.zeros((4, game.board_size, game.board_size))

    if game.action_stack:
        actions = np.array(game.action_stack)
        move_curr = actions[::2]
        move_oppo = actions[1::2]
        for move in move_curr:
            state_matrix[0][move] = 1.0
        for move in move_oppo:
            state_matrix[1][move] = 1.0
        # indicate the last move location
        state_matrix[2][actions[-1]] = 1.0
    if len(game.action_stack) % 2 == 0:
        state_matrix[3][:, :] = 1.0  # indicate the colour to play
    return state_matrix[:, ::-1, :]


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=0)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet:
    def __init__(self, board_width: int, board_height: int, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch: List[NetGameState]):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board: ConnectNGame) -> Tuple[Iterator[MoveWithProb], float]:
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        avail_pos_list = board.get_avail_pos()
        game_state = convert_game_state(board)
        current_state = np.ascontiguousarray(game_state.reshape(-1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            pos_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).float())
            pos_probs = np.exp(log_act_probs.data.numpy().flatten())
        value = float(value.data[0][0])
        return zip(avail_pos_list, pos_probs), value

    def backward_step(self, state_batch: List[NetGameState], probs_batch: List[ActionProbs],
                      value_batch: List[NDArray[(Any), np.float]], lr) -> Tuple[float, float]:
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            probs_batch = Variable(torch.FloatTensor(probs_batch).cuda())
            value_batch = Variable(torch.FloatTensor(value_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            probs_batch = Variable(torch.FloatTensor(probs_batch))
            value_batch = Variable(torch.FloatTensor(value_batch))

        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        log_act_probs, value = self.policy_value_net(state_batch)
        # loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), value_batch)
        policy_loss = -torch.mean(torch.sum(probs_batch * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
