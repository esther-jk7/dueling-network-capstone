import torch
import torch.nn as nn

class SingleStreamNetwork(nn.Module):
    """
    Standard single-stream Q-network.
    Baseline network from Wang et al. (2016).
    Outputs one Q-value per action.
    """
    def __init__(self, state_dim, hidden_dim, num_actions):
        super(SingleStreamNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state):
        return self.net(state)  # shape: [batch_size, num_actions]