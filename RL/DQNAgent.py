import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQNAgent:
    # gamma: Factor for short (0) or long (1) term reward
    # epsilon: Randomness factor of actions
    # epsilon_min: Lowest value of randomness allowed
    # epsilon_decay: Speed at which the randomness becomes less random
    def __init__(self, state_shape, num_of_actions, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.num_of_actions = num_of_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = DQN(self.device)
        self.target_model = DQN(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_of_actions)
        state = T.FloatTensor(state).unsqueeze(0).to(self.device)
        with T.no_grad():
            q_values = self.model(state)
        return T.argmax(q_values, dim=1).item()
    
    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        
        state_branch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
        state_branch = T.FloatTensor(state_branch).to(self.device)
        next_state_batch = T.FloatTensor(next_state_batch).to(self.device)
        action_batch = T.FloatTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = T.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        done_batch = T.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        q_values = self.model(state_batch).gather(1, action_batch)

        with T.no_grad():
            max_next_q_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
        
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        