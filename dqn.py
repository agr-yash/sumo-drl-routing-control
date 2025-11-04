import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_size, 150)
        self.layer2 = nn.Linear(150, 100)
        self.layer3 = nn.Linear(100, action_size)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            # print(f"[WARN] Expected torch.Tensor, got {type(state)}. Converting to tensor.")
            state = torch.tensor(state, dtype=torch.float32)

        x = F.relu(self.layer1(state))

        x = F.relu(self.layer2(x))

        output = self.layer3(x)

        return output
