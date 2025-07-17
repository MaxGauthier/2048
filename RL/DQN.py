import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_states, out_actions, hidden_layer=128):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(in_states, hidden_layer) 
        self.out = nn.Linear(hidden_layer, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)