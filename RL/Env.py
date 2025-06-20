import numpy as np
from game.model.Grid import *

class Env:
    def __init__(self, grid):
        self.grid = grid
        self.action_space = 4
        self.state_shape = (self.grid.rows, self.grid.columns)

    def sample_action(self):
        return np.random.randint(0, self.action_space - 1)

    def reset(self):
        self.grid.reset_game()
        return self._get_state()
    
    def step(self, action):
        directions = ['left', 'down', 'right', 'up']
        direction_str = directions[action]

        previous_score = self.grid.score
        previous_grid = self.grid.grid_values(self.grid.grid)
        self.grid.handle_move(direction_str)

        new_grid = self.grid.grid_values(self.grid.grid)
        reward = self.grid.score - previous_score
        done = self.grid.game_over
        truncated = False
        if previous_grid == new_grid:
            reward = -1
        next_state = self._get_state()
        return next_state, reward, done, truncated, {}
    
    def _get_state(self):
        return np.array(self.grid.normalize_grid(), dtype=np.float32).flatten()
    
    def render(self):
        self.grid.print_grid(self.grid.grid)

    def get_action_space(self):
        return self.action_space
    
    def get_state_shape(self):
        return self.state_shape