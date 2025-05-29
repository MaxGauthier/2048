import pygame
import sys
from game.view.GridView import GridView      
from game.view.GameView import GameView

class GameController:

    def __init__(self):
        pygame.init()
        width, height = 800, 800
        rows, cols = 4, 4
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Grid MVC Example")
        self.grid_view = GridView(rows, cols, width, height)
        self.game_view = GameView(self.screen, self.grid_view) 
        """
        self.model = GameModel(rows, cols)
        self.view = GameView(self.screen, self.model)
        """


    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.game_view.draw()
    