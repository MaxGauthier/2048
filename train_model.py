from RL.TrainingAgent import TrainingAgent

if __name__ == "__main__":
    agent = TrainingAgent(hidden_size=128)
    agent.train(episodes=10)  # Adjust the number of episodes as needed