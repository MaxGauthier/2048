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

class TrainingAgent:
    def __init__(self, hidden_size=128):
        self.alpha = 0.001                 # Learning rate
        self.gamma = 0.9                   # Discount factor
        self.sync_rate = 10                # Sync policy and target net every N steps
        self.replay_memory_size = 1000     # Replay buffer size
        self.batch_size = 32               # Training batch size
        self.hidden_size = hidden_size

        self.loss_fn = nn.MSELoss()
        self.optimizer = None  # Will be initialized after model creation

        self.actions = list(range(self.num_actions))  # [0, 1, 2, 3]

    def train(self, episodes):
        grid = Grid(ROWS, COLS, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.env = Env(grid)

        num_states = np.prod(self.env.get_state_shape())
        num_actions = self.env.get_action_space()
        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policy_net = DQN(num_states, num_states, num_actions)
        target_net = DQN(num_states, num_states, num_actions)
        target_net.load_state_dict(policy_net.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_net)

        self.optimizer = T.optim.Adam(policy_net.parameters(), self.alpha)
        rewards_per_episode = np.zeros(episodes)
        
        epsilon_history = []

        step_count = 0

        for i in range(episodes):
            state = self.env.reset()[0]
            terminated = False
            truncated = False

            while(not terminated and not truncated):
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with T.no_grad():
                        action = policy_net(self.state_to_dqn_input(state, num_states)).argmax().item()

                new_state, reward, terminated, truncated, _ = self.env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1

            if reward == 1:
                rewards_per_episode[i] = 1
            
            if len(memory) > self.batch_size and np.sum(rewards_per_episode) > 0:
                batch = memory.sample(self.batch_size)
                self.optimize(batch, policy_net, target_net)

                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                if step_count > self.sync_rate:
                    target_net.load_state_dict(policy_net.state_dict())
                    step_count = 0

            T.save(policy_net.state_dict(), "SOMETHING?????")
            




    def state_to_dqn_input(self, state:int, num_states:int)->T.Tensor:
        input_tensor = T.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = T.FloatTensor([reward])
            else:
                # Calculate target q value 
                with T.no_grad():
                    target = T.FloatTensor(
                        reward + self.gamma * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(T.stack(current_q_list), T.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def test(self, episodes):
        grid = Grid(ROWS, COLS, SCREEN_WIDTH, SCREEN_HEIGHT)
        env = Env(grid)

        num_states = np.prod(env.get_state_shape())
        num_actions = env.get_action_space()

        policy_net = DQN(num_states, num_states, num_actions)
        policy_net.load_state_dict(T.load("SOMETHING??????"))
        policy_net.eval()

        print("Policy (trained): ")
        self.print_dqn(policy_net)

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                with T.no_grad():
                    action = policy_net(self.state_to_dqn_input(state, num_states)).argmax().item()

                state,reward,terminated,truncated,_ = env.step(action)

    def print_dqn(self, dqn):
        num_states = dqn.conected_layers.in_features

        for s in range(num_states):
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).toList():
                q_values += "{:+.2f}".format(q)+' '
            q_values = q_values.rstrip()

            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states