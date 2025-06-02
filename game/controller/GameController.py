import pygame
import sys
from utils.constants import *
from game.view.GridView import GridView      
from game.view.GameView import GameView
from game.model.Grid import Grid


class GameController:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("2048 GAME")
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.grid = Grid(ROWS, COLS, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.grid_view = GridView(ROWS, COLS, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.game_view = GameView(self.screen, self.grid_view)



    def run(self):
        self.game_view.draw(self.grid.grid)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if self.grid.game_over:
                        print("move ignored, game over, use reset")
                    else: 
                        if event.key == pygame.K_LEFT:
                            new_grid = self.grid.handle_move("left")
                            self.game_view.draw(new_grid)
                        elif event.key == pygame.K_RIGHT:
                            new_grid = self.grid.handle_move("right")
                            self.game_view.draw(new_grid)
                        elif event.key == pygame.K_UP:
                            new_grid = self.grid.handle_move("up")
                            self.game_view.draw(new_grid)
                        elif event.key == pygame.K_DOWN:
                            new_grid = self.grid.handle_move("down")
                            self.game_view.draw(new_grid)
                        else:
                            continue
            self.clock.tick(60)
    