import pygame
from game.view.CellRenderer import CellRenderer
from utils.constants import *

class GameView:

    def __init__(self, screen, grid_view):
        self.screen = screen
        self.grid_view = grid_view
        self.cell_renderer = CellRenderer(screen)
        self.screen.fill(WHITE)

    def draw_grid(self):
        grid_width = COLS * CELL_SIZE
        grid_height = ROWS * CELL_SIZE

        starting_x = (SCREEN_WIDTH - grid_width) // 2
        starting_y = (SCREEN_HEIGHT - grid_height) // 2

        for i in range(COLS + 1):
            x_pos = starting_x + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, (x_pos, starting_y), (x_pos, starting_y + grid_height), LINE_WIDTH)

        for i in range(ROWS + 1):
            y_pos = starting_y + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, (starting_x, y_pos), (starting_x + grid_width, y_pos), LINE_WIDTH)
    
    def draw(self):
        self.draw_grid()

        row = 0
        col = 2
        value = 4
        x = (SCREEN_WIDTH - COLS * CELL_SIZE) // 2 + col * CELL_SIZE
        y = (SCREEN_HEIGHT - ROWS * CELL_SIZE) // 2 + row * CELL_SIZE
        self.cell_renderer.draw_cell(value, x, y)
        self.cell_renderer.draw_number(value, x, y)
        pygame.display.flip()
