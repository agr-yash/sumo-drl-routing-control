import random
import torch
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None

        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([e.action for e in experiences]))
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e.reward for e in experiences]))
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e.next_state for e in experiences]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8))
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        size = len(self.memory)
        return size
