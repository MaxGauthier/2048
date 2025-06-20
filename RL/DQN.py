import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_states, hidden_layer, out_actions):
        super().__init__()

        self.connected_layer = nn.Linear(in_states, hidden_layer) 
        self.out = nn.Linear(hidden_layer, out_actions)

    def forward(self, x):
        x = F.relu(self.connected_layer(x))
        x = self.out(x)
        return x