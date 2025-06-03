import pygame
import sys
from utils.constants import *
from game.view.GridView import GridView      
from game.view.GameView import GameView
from game.view.Button import Button
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

        self.reset_btn = Button(
            x = 550,
            y = 50,
            width = BTN_WIDTH,
            height = BTN_HEIGHT,
            color = ORANGE,
            text = "RESET",
            text_color = WHITE,
            font_size = 30
        )

        self.undo_btn = Button(
            x = 150,
            y = 50,
            width = BTN_WIDTH,
            height = BTN_HEIGHT,
            color = PURPLE,
            text = "UNDO",
            text_color = WHITE,
            font_size = 30 
        ) 

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.reset_btn.is_hovered(mouse_pos):
                        self.grid.reset_game()
                        self.game_view.draw(self.grid.grid)
                    if self.undo_btn.is_hovered(mouse_pos):
                        self.grid.reset_to_previous()
                elif event.type == pygame.KEYDOWN:
                    if self.grid.game_over:
                        print("move ignored, game over, use reset")
                    else: 
                        if event.key == pygame.K_LEFT:
                            self.grid.handle_move("left")
                        elif event.key == pygame.K_RIGHT:
                            self.grid.handle_move("right")
                        elif event.key == pygame.K_UP:
                            self.grid.handle_move("up")
                        elif event.key == pygame.K_DOWN:
                            self.grid.handle_move("down")
                        else:
                            continue

            self.screen.fill(WHITE)  
            self.undo_btn.draw_btn(self.screen)
            self.reset_btn.draw_btn(self.screen)
            self.game_view.draw(self.grid.grid)
            pygame.display.flip()
            self.clock.tick(60)
    