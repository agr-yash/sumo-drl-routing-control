import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from common.base_agent import BaseDQNAgent

from .dqn import DuelingDQN
from .replay_buffer import PrioritizedReplayBuffer


class DQNAgent(BaseDQNAgent):
    def __init__(self, state_size, action_size, agent_config, training_config):
        super().__init__(state_size, action_size, agent_config, training_config)

        self.qnetwork_local = DuelingDQN(state_size, action_size).to(self.device)
        self.qnetwork_target = DuelingDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=agent_config["lr"]
        )

        self.memory = PrioritizedReplayBuffer(
            buffer_size=agent_config["buffer_size"],
            batch_size=agent_config["batch_size"],
            device=self.device,
            alpha=agent_config["alpha"],
            beta_start=agent_config["beta_start"],
            beta_frames=agent_config["beta_frames"],
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
        """Update network weights using Double DQN and PER."""
        states, actions, rewards, next_states, dones, is_weights, indices = experiences

        # --- Double DQN ---
        # 1. Get best action from local network for next_states
        with torch.no_grad():
            best_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
        # 2. Get Q-values for those actions from target network
        Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
        # --- End Double DQN ---

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # --- PER ---
        # Compute TD errors for priority updates
        errors = (Q_expected - Q_targets).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, errors.flatten())

        # Compute loss with importance sampling weights
        loss = (is_weights * F.mse_loss(Q_expected, Q_targets, reduction="none")).mean()
        # --- End PER ---

        self.last_loss = loss.item()

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
