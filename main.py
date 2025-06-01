from game.controller.GameController import GameController
# TODO: If invalid key pressed, dont crash, just do nothing. Add transitions for a better visual. Add score. Add reset button and its logic (VIEW and MODEL).

if __name__ == "__main__":
    controller = GameController()
    controller.run()