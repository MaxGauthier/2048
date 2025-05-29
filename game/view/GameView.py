import pygame
from utils.constants import *

class GameView:

    def __init__(self, screen, grid_view):
        self.screen = screen
        self.grid_view = grid_view
        self.screen.fill(WHITE)

    def draw_grid(self):
        starting_x = SCREEN_WIDTH // ROWS
        starting_y = SCREEN_HEIGHT // COLS

        grid_width = COLS * CELL_SIZE
        grid_height = ROWS * CELL_SIZE

        for i in range(COLS + 1):
            x_pos = starting_x + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, (x_pos, starting_y), (x_pos, starting_y + grid_height), LINE_WIDTH)

        for i in range(ROWS + 1):
            y_pos = starting_y + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, (starting_x, y_pos), (starting_x + grid_width, y_pos), LINE_WIDTH)

        pygame.display.flip()
    
    def draw(self):
        self.draw_grid()
        pygame.display.flip()