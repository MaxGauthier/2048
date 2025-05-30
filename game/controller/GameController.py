import pygame
import sys
from utils.constants import *
from game.view.GridView import GridView      
from game.view.GameView import GameView

class GameController:

    def __init__(self):
        pygame.init()
        width, height = SCREEN_WIDTH, SCREEN_HEIGHT
        rows, cols = 4, 4
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2048 GAME")
        self.grid_view = GridView(rows, cols, width, height)
        self.game_view = GameView(self.screen, self.grid_view) 



    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.game_view.draw()
    