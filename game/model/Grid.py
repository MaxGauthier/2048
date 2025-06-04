import copy
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
        self.previous_grid = [[Cell(cell.x, cell.y , cell.value) for cell in row]
        for row in self.grid]
        self.previous_grid = copy.deepcopy(self.grid)
        return self.previous_grid

    def print_grid(self, grid):
        for row in grid:
            t = []
            for cell in row:
                t.append(cell.value)
            print(t)

    def reset_to_previous(self):
        if self.previous_grid is None:
            print("No previous state to reset to.")
            return

        for i in range(self.rows):
            for j in range(self.columns):
                self.grid[i][j].value = self.previous_grid[i][j].value

        self.game_over = False
        self.move.restore_previous_score()

    def grid_values(self, grid):
        return [[cell.value for cell in row] for row in grid]

    def handle_move(self, direction):
        print("MOVE MADE:", direction)
        self.backup_grid()
        self.move.save_previous_score()
        new_grid = self.move.move(direction)
        prev_grid_values = self.grid_values(self.previous_grid)
        new_grid_values = self.grid_values(new_grid)
        if prev_grid_values == new_grid_values:
            print("tempppppppppppppp")
            # Do something
        else: 
            self.generate_cell.generate_new_cell()
            if self.move.no_moves_left():
                self.game_over = True
                print("Game Over") 

        return new_grid
