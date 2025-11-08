import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseDQNAgent(ABC):
    """Abstract Base Class for a DQN Agent."""

    def __init__(self, state_size, action_size, agent_config, training_config):
        self.state_size = state_size
        self.action_size = action_size
        self.last_loss = None

        self.total_episodes = training_config["num_episodes"]
        self.epsilon = agent_config["eps_start"]
        self.eps_start = agent_config["eps_start"]
        self.eps_end = agent_config["eps_end"]

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.qnetwork_local = None
        self.qnetwork_target = None
        self.optimizer = None
        self.memory = None
        self.t_step = 0

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """Store experience and trigger learning."""
        raise NotImplementedError

    @abstractmethod
    def learn(self, experiences, gamma):
        """Update network weights."""
        raise NotImplementedError

    def act(self, state, eps=None):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state.
            eps (float, optional): Epsilon value. Defaults to the agent's current epsilon.

        Returns:
            int: The chosen action.
        """
        if eps is None:
            eps = self.epsilon

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update_epsilon(self, current_episode):
        """
        Decay epsilon based on the current episode.
        """
        decay_rate = (self.eps_start - self.eps_end) / self.total_episodes
        self.epsilon = max(self.eps_end, self.eps_start - decay_rate * current_episode)

    def update_target_network(self):
        """Hard update: copy weights from local to target network."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def save(self, checkpoint_path):
        """
        Save model, optimizer, and epsilon.

        Args:
            checkpoint_path (str): Path to save the checkpoint.
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.qnetwork_local.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path):
        """
        Load model, optimizer, and epsilon if a checkpoint exists.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not os.path.exists(checkpoint_path):
            return False
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.qnetwork_local.load_state_dict(checkpoint["model_state_dict"])
            self.qnetwork_target.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            print(
                f"[INFO] ✅ Checkpoint loaded from {checkpoint_path}. Resuming training."
            )
            return True
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint from {checkpoint_path}: {e}")
            return False
