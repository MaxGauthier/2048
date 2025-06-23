import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from RL.DQN import DQN
from RL.Env import Env
from RL.ReplayMemory import ReplayMemory
from game.model.Grid import Grid
from utils.constants import *
import matplotlib.pyplot as plt

class TrainingAgent:
    def __init__(self, hidden_size=128):
        self.alpha = 0.001
        self.gamma = 0.9
        self.sync_rate = 10
        self.replay_memory_size = 1000
        self.batch_size = 32
        self.hidden_size = hidden_size

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.loss_fn = nn.MSELoss()

    def train(self, episodes):
        grid = Grid(ROWS, COLS, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.env = Env(grid)

        num_states = np.prod(self.env.get_state_shape())
        num_actions = self.env.get_action_space()
        self.actions = list(range(num_actions))
        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        self.policy_net = DQN(num_states, self.hidden_size, num_actions).to(self.device)
        self.target_net = DQN(num_states, self.hidden_size, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Training DQN agent...')
        print('Policy (random, before training):')
        self.print_dqn(self.policy_net)

        self.optimizer = optim.Adam(self.policy_net.parameters(), self.alpha)
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0

        for i in range(episodes):
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
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                total_reward += reward
                step_count += 1

            rewards_per_episode[i] = total_reward

            if len(memory) > self.batch_size:
                batch = memory.sample(self.batch_size)
                self.optimize(batch)

                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                if step_count > self.sync_rate:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    step_count = 0

        T.save(self.policy_net.state_dict(), "policy_net.pth")
        self.env.grid.print_grid(self.env.grid.grid)

        plt.plot(rewards_per_episode, label='Reward per Episode')
        plt.plot(np.convolve(rewards_per_episode, np.ones(10)/10, mode='valid'), label='10-Episode Average')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('DQN Training Reward Progress')
        plt.legend()
        plt.show()

    def state_to_dqn_input(self, state: np.ndarray, num_states: int) -> T.Tensor:
        """
        Assumes `state` is a flattened numpy array (e.g. 16,) of floats.
        """
        if isinstance(state, np.ndarray):
            return T.tensor(state, dtype=T.float32).view(1, num_states)
        else:
            raise TypeError("Expected state to be a numpy array")

    def optimize(self, mini_batch):
        num_states = self.policy_net.connected_layer.in_features
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            state_tensor = self.state_to_dqn_input(state, num_states).to(self.device)
            new_state_tensor = self.state_to_dqn_input(new_state, num_states).to(self.device)

            with T.no_grad():
                if terminated:
                    target = T.tensor([reward], device=self.device)
                else:
                    target = T.tensor([reward + self.gamma * self.target_net(new_state_tensor).max().item()],
                                      device=self.device)

            current_q = self.policy_net(state_tensor)
            current_q_list.append(current_q)

            target_q = current_q.clone().detach()
            target_q[0][action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(T.cat(current_q_list), T.cat(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes):
        grid = Grid(ROWS, COLS, SCREEN_WIDTH, SCREEN_HEIGHT)
        env = Env(grid)

        num_states = np.prod(env.get_state_shape())
        num_actions = env.get_action_space()

        self.policy_net = DQN(num_states, self.hidden_size, num_actions).to(self.device)
        self.policy_net.load_state_dict(T.load("policy_net.pth", map_location=self.device))
        self.policy_net.eval()

        print('Testing DQN agent...')
        print("Policy (trained): ")
        self.print_dqn(self.policy_net)

        for i in range(episodes):
            state = env.reset()
            terminated = False

            while not terminated:
                with T.no_grad():
                    input_tensor = self.state_to_dqn_input(state, num_states).to(self.device)
                    action = self.policy_net(input_tensor).argmax().item()

                state, reward, terminated, _ = env.step(action)

    def print_dqn(self, dqn):
        num_states = dqn.connected_layer.in_features

        for s in range(num_states):
            dummy_input = T.zeros(1, num_states).to(self.device)
            dummy_input[0][s] = 1.0  # One-hot style

            q_values_tensor = dqn(dummy_input)
            q_values = ' '.join("{:+.2f}".format(q) for q in q_values_tensor.tolist()[0])
            best_action = self.actions[q_values_tensor.argmax().item()]

            print(f'{s:02},{best_action},[{q_values}]', end=' ')
            if (s + 1) % 4 == 0:
                print()
