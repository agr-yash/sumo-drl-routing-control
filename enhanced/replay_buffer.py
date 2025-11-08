import random
from collections import deque, namedtuple

import numpy as np
import torch


# --- New SumTree class ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


# --- Modified ReplayBuffer to PrioritizedReplayBuffer ---
class PrioritizedReplayBuffer:
    def __init__(
        self,
        buffer_size,
        batch_size,
        device,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
    ):
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_increment_per_sampling = (1.0 - beta_start) / beta_frames
        self.epsilon = 0.01  # for priority calculation
        self.max_priority = 1.0
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.tree.add(self.max_priority, e)

    def sample(self):
        if self.tree.n_entries < self.batch_size:
            return None

        experiences = []
        indices = []
        priorities = []
        segment = self.tree.total() / self.batch_size

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            (idx, p, data) = self.tree.get(s)

            if isinstance(data, int) and data == 0:  # SumTree initialized with 0
                # This can happen if segments are empty. Resample.
                s = random.uniform(0, self.tree.total())
                (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            experiences.append(data)
            indices.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        is_weights_tensor = (
            torch.from_numpy(is_weights).float().to(self.device).unsqueeze(1)
        )

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            is_weights_tensor,
            indices,
        )

    def update_priorities(self, batch_indices, errors):
        for i, idx in enumerate(batch_indices):
            p = self._get_priority(errors[i])
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries
