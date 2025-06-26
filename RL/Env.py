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
    
    def step(self, action):
        directions = ['left', 'down', 'right', 'up']
        direction_str = directions[action]

        previous_score = self.grid.score
        previous_grid_values = self.grid.grid_values(self.grid.grid)

        self.grid.handle_move(direction_str)

        new_score = self.grid.score
        new_grid_values = self.grid.grid_values(self.grid.grid)
        new_max = max([max(cell.value for cell in row) for row in self.grid.grid])

        next_state = self._get_state()

        # Reward logic
        reward = 0.0
        if previous_grid_values == new_grid_values:
            reward = -5.0
        else:
            score_increase = new_score - previous_score
            reward += score_increase
            empty_cells_before = sum(1 for row in previous_grid_values for cell_val in row if cell_val == 0)
            empty_cells_after = sum(1 for row in new_grid_values for cell_val in row if cell_val == 0)
            reward += (empty_cells_after - empty_cells_before) * 0.5        

            top_right_corner = new_grid_values[0][0]
            if new_max == top_right_corner:
                reward += 10.0

            reward += self.snake_pattern(new_grid_values) * 0.2
        done = self.grid.game_over

        return next_state, reward, done, {}
    
    def _get_state(self):
        normalized_grid = np.array(self.grid.normalize_grid(), dtype=np.float32.flatten())
        return T.from_numpy(normalized_grid)
    
    def render(self):
        self.grid.print_grid(self.grid.grid)

    def get_action_space(self):
        return self.action_space
    
    def get_state_shape(self):
        return self.state_shape
    
    def snake_pattern(self, grid):
        ideal_path = [(0, 0), (0, 1), (0, 2), (0, 3),
                      (1, 3), (1, 2), (1, 1), (1, 0),
                      (2, 0), (2, 1), (2, 2), (2, 3),
                      (3, 3), (3, 2), (3, 1), (3, 0)
                    ]
        values = []
        for row, col in ideal_path:
            val = grid[row][col]
            values.append(math.log2(val) if val > 0 else 0.0)
        score = 0
        for i in range(len(values) - 1):
            if values[i] >= values[i + 1]:
                score += 1
            else:
                break
        return float(score)