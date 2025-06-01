import pygame
from game.view.CellRenderer import CellRenderer
from utils.constants import *

class GameView:

    def __init__(self, screen, grid_view):
        self.screen = screen
        self.grid_view = grid_view
        self.cell_renderer = CellRenderer(screen)
        self.screen.fill(WHITE)
        self.grid_width = COLS * CELL_SIZE
        self.grid_height = ROWS * CELL_SIZE

        self.starting_x = (SCREEN_WIDTH - self.grid_width) // 2
        self.starting_y = (SCREEN_HEIGHT - self.grid_height) // 2

    def draw_grid(self):

        for i in range(COLS + 1):
            x_pos = self.starting_x + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, (x_pos, self.starting_y), (x_pos, self.starting_y + self.grid_height), LINE_WIDTH)

        for i in range(ROWS + 1):
            y_pos = self.starting_y + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, (self.starting_x, y_pos), (self.starting_x + self.grid_width, y_pos), LINE_WIDTH)
    
    def draw(self, grid):
        self.draw_grid()
        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                x = self.starting_x + col_idx * CELL_SIZE
                y = self.starting_y + row_idx * CELL_SIZE
                self.cell_renderer.draw_cell(cell.value, x, y)
                self.cell_renderer.draw_number(cell.value, x, y)
    
        pygame.display.flip()
