import pygame

class Button:

    def __init__(self, x, y, width, height, color, text, text_color, font_size):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.text_color = text_color
        self.font_size = pygame.font.Font(None, font_size)

    def draw_btn(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surface = self.font_size.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_hovered(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)
