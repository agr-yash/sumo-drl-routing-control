import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn.functional as F

from dqn import DQN
from replay_buffer import ReplayBuffer

BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.001
TARGET_UPDATE_FREQUENCY = 100
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPS_START


        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % TARGET_UPDATE_FREQUENCY

        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    
    def act(self, state, eps=None):

        if eps is None:
            eps = self.epsilon

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    
    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1).unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)

    
    def soft_update(self, local_model, target_model, tau=1e-3):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

