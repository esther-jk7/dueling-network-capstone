import torch
import torch.nn as nn

class DuelingNetwork(nn.Module):
    """
    Dueling Network Architecture - Wang et al. (2016).
    Q(s,a) = V(s) + (A(s,a) - mean(A(s, all actions)))
    """
    def __init__(self, state_dim, num_actions, shared_dim, stream_dim):
        super(DuelingNetwork, self).__init__()

        # shared trunk
        self.shared = nn.Sequential(
            nn.Linear(state_dim, shared_dim),
            nn.ReLU()
        )

        # V(s): single number per state
        self.value_stream = nn.Sequential(
            nn.Linear(shared_dim, stream_dim),
            nn.ReLU(),
            nn.Linear(stream_dim, 1)
        )

        # A(s,a): one number per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(shared_dim, stream_dim),
            nn.ReLU(),
            nn.Linear(stream_dim, num_actions)
        )

    def forward(self, state):
        shared_out = self.shared(state)
        value = self.value_stream(shared_out)
        advantage = self.advantage_stream(shared_out)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        return value + advantage
