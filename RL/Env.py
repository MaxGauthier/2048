import math
import numpy as np
from game.model.Grid import *

class Env:
    def __init__(self, grid):
        self.grid = grid
        self.action_space = 4
        self.state_shape = (self.grid.rows, self.grid.columns)

    def sample_action(self):
        return np.random.randint(0, self.action_space)

    def reset(self):
        self.grid.reset_game()
        return self._get_state()
    
    def step(self, action,):
        directions = ['left', 'down', 'right', 'up']
        direction_str = directions[action]
        score_milestones = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

        all_values = [cell.value for row in self.grid.grid for cell in row]

        previous_score = self.grid.score
        previous_grid_values = self.grid.grid_values(self.grid.grid)
        previous_max = max([max(cell.value for cell in row) for row in self.grid.grid])

        self.grid.handle_move(direction_str)

        new_score = self.grid.score
        new_grid_values = self.grid.grid_values(self.grid.grid)
        new_max = max([max(cell.value for cell in row) for row in self.grid.grid])

        next_state = self._get_state()

        # Reward logic
        reward = 0.0
        if previous_grid_values == new_grid_values:
            reward = -1.0
        else:
            if new_score > previous_score:
                reward += 0.2 * (new_score - previous_score)
            else: 
                reward -= 0.01

        if new_max > previous_max and new_max in score_milestones:
            index = score_milestones.index(new_max)
            reward += (index + 1)
            previous_max = new_max

        reward += self.snake_pattern() * 1.5
        done = self.grid.game_over
        if done:
            reward -= 10.0

        return next_state, reward, done, {}
    
    def _get_state(self):
        normalized_grid = np.array(self.grid.normalize_grid(), dtype=np.float32).flatten()
        return normalized_grid
    
    def render(self):
        self.grid.print_grid(self.grid.grid)

    def get_action_space(self):
        return self.action_space
    
    def get_state_shape(self):
        return self.state_shape
    
    def snake_pattern(self):
        # Define the snake-like traversal order
        snake_coords = [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (1, 2), (1, 1), (1, 0),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 3), (3, 2), (3, 1), (3, 0)
        ]

        # Flatten all values with their positions
        cells_with_pos = [
            (cell.value, (i, j))
            for i, row in enumerate(self.grid.grid)
            for j, cell in enumerate(row)
        ]

        # Sort descending by value
        sorted_cells = sorted(cells_with_pos, key=lambda x: x[0], reverse=True)

        # Check if top values are in correct snake positions
        correct_count = 0
        for index, (_, expected_pos) in enumerate(zip(sorted_cells, snake_coords)):
            actual_pos = sorted_cells[index][1]
            if actual_pos == expected_pos:
                correct_count += 1

        return correct_count  # Number of top tiles correctly placed
