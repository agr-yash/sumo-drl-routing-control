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
TOTAL_EPISODES = 250

class DQNAgent:
    def __init__(self, state_size, action_size):

        self.total_episodes = TOTAL_EPISODES
        self.epsilon = EPS_START
        self.eps_start = EPS_START
        self.eps_end = EPS_END

        self.state_size = state_size
        self.action_size = action_size
        self.last_loss = None  # Track last loss

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[INFO] Using Apple Metal (MPS) backend for GPU acceleration.")
        else:
            self.device = torch.device("cpu")
            print("[WARNING] MPS not available — using CPU.")

        self.qnetwork_local = DQN(state_size, action_size).to(self.device)
        self.qnetwork_target = DQN(state_size, action_size).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.t_step = 0

    # ---------------------------------------------------------------------
    def step(self, state, action, reward, next_state, done):
        """Store experience and possibly learn."""

        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % TARGET_UPDATE_FREQUENCY

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        if self.t_step % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()

    # ---------------------------------------------------------------------
    def act(self, state, eps=None):
        """Choose action using epsilon-greedy policy."""
        if eps is None:
            eps = self.epsilon

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        greedy_action = np.argmax(action_values.cpu().data.numpy())
        random_action = random.choice(np.arange(self.action_size))

        if random.random() > eps:
            return greedy_action
        else:
            return random_action

    # ---------------------------------------------------------------------
    def learn(self, experiences, gamma):
        """Update network weights."""

        states, actions, rewards, next_states, dones = experiences

        # Compute Q targets for next states
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.last_loss = loss.item()


        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, current_episode):
            decay_rate = (self.eps_start - self.eps_end) / self.total_episodes
            self.epsilon = max(self.eps_end, self.eps_start - decay_rate * current_episode)

    def update_target_network(self):
        """Hard update: copy weights directly from local to target network."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

