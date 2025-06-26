import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self):
        pass

    def plot_training(self, rewards_per_episode):
        plt.figure(figsize=(12, 6))
        
        plt.plot(rewards_per_episode, color='lightblue', alpha=0.6, label='Reward per Episode')

        # Smoother moving average for long training
        window = 500
        if len(rewards_per_episode) >= window:
            moving_avg = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
            plt.plot(range(window - 1, len(rewards_per_episode)), moving_avg,
                    color='blue', linewidth=2.0, label=f'{window}-Episode Moving Average')

        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('DQN Training Reward Progress (50,000 Episodes)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
