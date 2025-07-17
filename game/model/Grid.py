import copy
import numpy as np
import torch as T
from game.model.Cell import Cell
from game.model.Move import Move
from game.model.GenerateCell import GenerateCell

class Grid:
    def __init__(self, rows, columns, width, height):
        self.width = width
        self.height = height
        self.rows = rows
        self.columns = columns
        self.game_over = False
        self.grid = self._create_empty_grid()
        self.previous_grid = None
        self.move = Move(self)
        self.generate_cell = GenerateCell(self.grid, self.rows, self.columns)
        self.generate_cell.starting_board()

    @property
    def score(self):
        return self.move.score

    def _create_empty_grid(self):
        return [[Cell(x, y) for x in range(self.columns)] for y in range(self.rows)]

    def create_grid(self):
        grid = []
        for y in range(0, self.rows):
            row = []
            for x in range(self.columns):
                row.append(Cell(x, y))     
            grid.append(row)

        print(grid)
        return grid
    
    def reset_game(self):
        self.grid = self._create_empty_grid()
        self.previous_grid = None
        self.game_over = False
        self.move = Move(self)
        self.generate_cell = GenerateCell(self.grid, self.rows, self.columns)
        self.generate_cell.starting_board()

    def backup_grid(self):
        return copy.deepcopy(self.grid)

    def print_grid(self, grid):
        for row in grid:
            t = []
            for cell in row:
                t.append(cell.value)
            print(t)

    def get_grid_state(self):
        grid_state = []
        for row in self.grid:
            current_row = []
            for cell in row:
                current_row.append(cell.value)
            grid_state.append(current_row)
        return grid_state

    def reset_to_previous(self):
        if self.previous_grid is None:
            #print("No previous state to reset to.")
            return

        for i in range(self.rows):
            for j in range(self.columns):
                self.grid[i][j].value = self.previous_grid[i][j].value

        self.game_over = False
        self.move.restore_previous_score()

    def grid_values(self, grid):
        return [[cell.value for cell in row] for row in grid]
    
    def normalize_grid(self):
        numeric_grid = np.array(self.grid_values(self.grid), dtype=float)
        with np.errstate(divide='ignore'):
            log_grid = np.log2(numeric_grid)
        log_grid[numeric_grid == 0] = 0  
        normalized_grid = log_grid / 16.0
        return np.round(normalized_grid, 2)


    def handle_move(self, direction):
        #print("MOVE MADE:", direction)
        self.previous_grid = self.backup_grid()
        self.move.save_previous_score()
        self.move.move(direction)
        prev_grid_values = self.grid_values(self.previous_grid)
        current_grid_values = self.grid_values(self.grid)
        if prev_grid_values == current_grid_values:
            if self.move.no_moves_left():
                self.game_over = True
        else: 
            self.generate_cell.generate_new_cell()
            if self.move.no_moves_left():
                self.game_over = True
                #print("Game Over") 
        