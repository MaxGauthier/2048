from RL.TrainingAgent import TrainingAgent
from game.controller.GameController import GameController

if __name__ == "__main__":
    #game_controller = GameController()
    #game_controller.run()
    agent = TrainingAgent(hidden_size=128)
    agent.train(episodes=1000)  # Adjust the number of episodes as needed