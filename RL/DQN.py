import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_states, out_actions, hidden_layer):
        super.__init__()

        self.connected_layer = nn.Linear(in_states, hidden_layer) 
        self.out = nn.Linear(hidden_layer, out_actions)

    def foward(self, data_set):
        data_set = F.relu(self.connected_layer(data_set))
        data_set = self.out(data_set)
        return data_set   