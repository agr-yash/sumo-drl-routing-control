import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.action_size = action_size

        # Shared layers
        self.layer1 = nn.Linear(state_size, 150)
        self.layer2 = nn.Linear(150, 128)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_size)
        )

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage streams
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        output = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return output
