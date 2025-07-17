from RL.Agent import Agent
from game.controller.GameController import GameController

if __name__ == "__main__":
    agent = Agent()
    agent.train(episodes=50000)  # Adjust the number of episodes as needed