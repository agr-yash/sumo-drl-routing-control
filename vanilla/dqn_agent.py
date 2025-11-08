import torch.nn.functional as F
import torch.optim as optim

from common.base_agent import BaseDQNAgent

from .dqn import DQN
from .replay_buffer import ReplayBuffer


class DQNAgent(BaseDQNAgent):
    def __init__(self, state_size, action_size, agent_config, training_config):
        super().__init__(state_size, action_size, agent_config, training_config)

        self.qnetwork_local = DQN(state_size, action_size).to(self.device)
        self.qnetwork_target = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=agent_config["lr"]
        )

        self.memory = ReplayBuffer(
            agent_config["buffer_size"], agent_config["batch_size"], self.device
        )
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Store experience and possibly learn."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            if experiences:
                self.learn(experiences, self.agent_config["gamma"])

        if self.t_step % self.agent_config["target_update_frequency"] == 0:
            self.update_target_network()

    def learn(self, experiences, gamma):
        """Update network weights."""
        states, actions, rewards, next_states, dones = experiences

        # Compute Q targets for next states
        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.last_loss = loss.item()

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
