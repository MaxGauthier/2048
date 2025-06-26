from RL.TrainingAgent import TrainingAgent
from game.controller.GameController import GameController

if __name__ == "__main__":
    agent = TrainingAgent(hidden_size=128)
    agent.train(episodes=1000)  # Adjust the number of episodes as needed