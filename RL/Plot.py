import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self):
        pass

    def plot_training(self, rewards_per_episode):
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(12, 6))

        # Only plot the most recent 50,000 episodes
        max_points = 50000
        rewards = rewards_per_episode[-max_points:]

        # Downsample if needed (e.g., every 10th point)
        downsample = max(1, len(rewards) // 2000)  # ~2000 points max
        rewards_ds = rewards[::downsample]
        episodes_ds = np.arange(len(rewards_per_episode) - len(rewards), len(rewards_per_episode))[::downsample]

        plt.plot(episodes_ds, rewards_ds, color='lightblue', alpha=0.5, label='Reward per Episode')

        # Moving average over 500 episodes
        window = 250
        if len(rewards_ds) >= window:
            moving_avg = np.convolve(rewards_ds, np.ones(window)/window, mode='valid')
            episodes_ma = episodes_ds[:len(moving_avg)]
            plt.plot(episodes_ma, moving_avg, color='blue', linewidth=2.0, label=f'{window}-Episode Moving Average')


        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'DQN Training Reward Progress (Last {len(rewards)} Episodes)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
