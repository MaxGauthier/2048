import sys
import random
import os
import pickle
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from collections import deque
from RL.DQN import DQN
from RL.Env import Env
from RL.ReplayMemory import ReplayMemory
from game.model.Grid import Grid
from utils.constants import *
from RL.Plot import Plot

class Agent:
    def __init__(self):
        self.alpha = 0.001
        self.gamma = 0.99
        self.sync_rate = 500
        self.replay_memory_size = 30000                      # Size of the deque (replay_memory) 
        self.batch_size = 32
        self.MINUMUM_REPLAY_MEMORY_SIZE = 5000
        self.hidden_size = 256

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.loss_fn = nn.SmoothL1Loss() 

        self.plot = Plot()

    def train(self, episodes):
        epsilon = 1.0
        final_epsilon = 0.1
        decay_until_episode = 20000
        decay_rate = (final_epsilon / epsilon) ** (1 / decay_until_episode)

        grid = Grid(ROWS, COLS, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.env = Env(grid)

        num_states = np.prod(self.env.get_state_shape())    # 16
        num_actions = self.env.get_action_space()           # 4

        self.actions = list(range(num_actions))             # [0,1,2,3]
        self.memory = ReplayMemory(self.replay_memory_size) # List of tuples (state, action, new_state, reward, terminated)

        best_reward = float('-inf')
        best_grid = None

        self.policy_net = DQN(num_states, num_actions, self.hidden_size).to(self.device)
        self.target_net = DQN(num_states, num_actions, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        checkpoint_path = "checkpoint.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = T.load(checkpoint_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epsilon = checkpoint['epsilon']
            start_episode = checkpoint['episode'] + 1
            print(epsilon)
            print(f"Loaded checkpoint from episode {checkpoint['episode']}, continuing training.")
            self.load_replay_memory()

            print("Loaded existing model weights, continuing training.")
        else:
            start_episode = 0
            self.optimizer = optim.Adam(self.policy_net.parameters(), self.alpha)
            print("No checkpoint found, training from scratch.")

        print('Training DQN agent...')
        print('Policy (random, before training):')
        self.print_dqn(self.policy_net)

        self.optimizer = optim.Adam(self.policy_net.parameters(), self.alpha)
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0

        for i in range(start_episode, start_episode + episodes):
            state = self.env.reset()
            terminated = False
            total_reward = 0
            while not terminated:
                if random.random() < epsilon:
                    action = random.choice(self.actions)
                else:
                    with T.no_grad():
                        input_tensor = self.state_to_dqn_input(state, num_states).to(self.device)
                        action = self.policy_net(input_tensor).argmax().item()

                new_state, reward, terminated, _ = self.env.step(action)
                self.memory.append((state, action, new_state, reward, terminated))
                state = new_state
                total_reward += reward

                if terminated and total_reward > best_reward:
                    best_reward = total_reward
                    best_grid = [row[:] for row in self.env.grid.grid]
                    best_episode = i
                step_count += 1

            rewards_per_episode[i - start_episode] = total_reward
            if len(self.memory) < self.MINUMUM_REPLAY_MEMORY_SIZE:
                continue
            batch = self.memory.sample(self.batch_size)
            self.optimize(batch)
            if i > start_episode + 1000:
                epsilon = max(final_epsilon, epsilon * decay_rate ** (i - start_episode))
            epsilon_history.append(epsilon)
            if step_count > self.sync_rate:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                step_count = 0

            if i % 1000 == 0 or i == start_episode + episodes - 1:
                checkpoint = {
                    'episode': i,
                    'model_state_dict': self.policy_net.state_dict(),
                    'target_state_dict': self.target_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': epsilon
                }
                T.save(checkpoint, "checkpoint.pth")
                self.save_replay_memory()
                print(f"Saved checkpoint at episode {i}")

        if best_grid is not None:
            self.env.grid.grid = best_grid
            print(f"Best grid at episode {best_episode}: with a total reward of {best_reward}")
            self.env.grid.print_grid(self.env.grid.grid)

        self.plot.plot_training(rewards_per_episode)

    def state_to_dqn_input(self, state: np.ndarray, num_states: int) -> T.Tensor:
        if isinstance(state, np.ndarray):
            return T.tensor(state, dtype=T.float32).view(1, num_states)
        else:
            raise TypeError("Expected state to be a numpy array")

    def optimize(self, mini_batch):
        num_states = self.policy_net.fc1.in_features
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            state_tensor = self.state_to_dqn_input(state, num_states).to(self.device)
            new_state_tensor = self.state_to_dqn_input(new_state, num_states).to(self.device)

            with T.no_grad():
                if terminated:
                    target = T.tensor([reward], dtype=T.float32, device=self.device)
                else:
                    best_action = self.policy_net(new_state_tensor).argmax().item()
                    next_q_val = self.target_net(new_state_tensor)[0][best_action].item()
                    target = T.tensor([reward + self.gamma * next_q_val], device=self.device)

            current_q = self.policy_net(state_tensor)
            current_q_list.append(current_q)

            target_q = current_q.clone().detach()
            target_q[0][action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(T.cat(current_q_list), T.cat(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def print_dqn(self, dqn):
        num_states = dqn.fc1.in_features

        for s in range(num_states):
            dummy_input = T.zeros(1, num_states).to(self.device)
            dummy_input[0][s] = 1.0  # One-hot style

            q_values_tensor = dqn(dummy_input)
            q_values = ' '.join("{:+.2f}".format(q) for q in q_values_tensor.tolist()[0])
            best_action = self.actions[q_values_tensor.argmax().item()]

            print(f'{s:02},{best_action},[{q_values}]', end=' ')
            if (s + 1) % 4 == 0:
                print()


    def save_replay_memory(self, filename="replay_memory.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.memory.memory, f)

    def load_replay_memory(self, filename="replay_memory.pkl"):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "rb") as f:
                memory_list = pickle.load(f)
            self.memory.memory = deque(memory_list, maxlen=self.replay_memory_size)
        else:
            print("No valid replay memory file found, starting fresh.")

