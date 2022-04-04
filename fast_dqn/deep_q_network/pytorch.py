import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, Flatten, Linear, ReLU
import torch.nn.functional as F
from torch.optim import Adam


class PytorchDeepQNetwork:
    def __init__(self, env, discount):
        self.discount = discount

        def model():
            return nn.Sequential(*[
                Conv2d(env.observation_space.shape[-1], 32, kernel_size=8, stride=4),
                ReLU(),
                Conv2d(32, 64, kernel_size=4, stride=2),
                ReLU(),
                Conv2d(64, 64, kernel_size=3, stride=1),
                ReLU(),
                Flatten(),
                Linear(3136, 512),
                ReLU(),
                Linear(512, env.action_space.n)
            ])
        self._main_net = model()
        self._target_net = model()
        self._exec_net = model()  # Used by Fast DQN for concurrent training/execution

        self.optimizer = Adam(self._main_net.parameters(), lr=2.5e-4)

    def _preprocess_states(self, states):
        # TODO: This is hard coded for CNNs, we should add a condition here
        states = states.transpose(0, 3, 1, 2)
        return torch.tensor(states.astype(np.float32) / 255.0)

    def predict(self, states, network):
        return {
            'main': self._main_net,
            'target': self._target_net,
            'exec': self._exec_net,
        }[network](self._preprocess_states(states))

    def greedy_actions(self, states, network):
        Q = self.predict(states, network)
        return torch.argmax(Q, dim=1)

    def train(self, states, actions, rewards, next_states, dones):
        # TODO: We shouldn't hardcode these type casts here
        actions = actions.astype(np.int64)
        dones = dones.astype(np.float32)
        actions, rewards, dones = map(torch.tensor, (actions, rewards, dones))

        with torch.no_grad():
            next_Q = self.predict(next_states, network='target')
            done_mask = 1.0 - dones
            targets = rewards + self.discount * done_mask * torch.max(next_Q, dim=1)[0]

            action_mask = F.one_hot(actions, num_classes=next_Q.shape[1])

        Q = self.predict(states, network='main')
        Q = torch.sum(action_mask * Q, dim=1)
        loss = torch.mean(huber_loss(targets - Q))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self._copy_network(self._main_net, self._target_net)

    def update_exec_net(self):
        self._copy_network(self._main_net, self._exec_net)

    def _copy_network(self, src_net, dst_net):
        dst_net.load_state_dict(src_net.state_dict())


def huber_loss(x):
    # Huber loss: This is the correct implementation of DQN's partial derivative clipping
    # See https://github.com/devsisters/DQN-tensorflow/issues/16
    return torch.where(
        torch.abs(x) <= 1.0,    # Condition
        0.5 * torch.square(x),  # True
        torch.abs(x) - 0.5      # False
    )
