import pygame
from utils.constants import *

class CellRenderer:
    def __init__(self, screen):
        self.screen = screen

    def draw_cell(self, value, x, y):
        color = COLOR_MAP.get(value, CELL_COLOR)
        pygame.draw.rect(self.screen, color, (x + 5, y + 5, CELL_SIZE - 10, CELL_SIZE - 10))

    def draw_number(self, number, x, y):
        if number == 0:
            return

        number_str = str(number)
        max_font_size = CELL_SIZE - 30
        shrink_per_digit = 10
        font_size = max(max_font_size - shrink_per_digit * (len(number_str) - 1), 16)

        font = pygame.font.SysFont(None, font_size, bold=False)
        text_surf = font.render(number_str, True, BLACK)
        text_rect = text_surf.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
        self.screen.blit(text_surf, text_rect)
