import pygame
from utils.constants import *

class ScoreView:
    def __init__(self, title, position, font_name='Arial', title_size=24, score_size=16, title_color=BLACK, score_color=BLACK):
        self.title = title
        self.position = position  # (x, y)
        self.score = 0
        self.font_title = pygame.font.SysFont(font_name, title_size)
        self.font_score = pygame.font.SysFont(font_name, score_size)
        self.title_color = title_color
        self.score_color = score_color

        # Pre-render title surface (it's static)
        self.title_surface = self.font_title.render(self.title, True, self.title_color)

    def draw(self, surface):
        # Draw title
        title_rect = self.title_surface.get_rect(topleft=self.position)
        surface.blit(self.title_surface, title_rect)

        # Draw score below title
        score_text = f"{self.score}"
        score_surface = self.font_score.render(score_text, True, self.score_color)
        score_rect = score_surface.get_rect(topleft=(self.position[0], title_rect.bottom + 5))  # 5 pixels below title
        surface.blit(score_surface, score_rect)

    def set_score(self, score):
        self.score = score
