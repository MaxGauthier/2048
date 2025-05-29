import random
from Cell import Cell

class GenerateCell:

    def __init__(self, grid, rows, columns):
        self.grid = grid
        self.rows = rows
        self.columns = columns
        self.used_positions = set()

    def cell_initial_value(self):
        return 2 if random.random() < 0.9 else 4

    def random_coord(self):
        x = random.randint(0, self.rows - 1)
        y = random.randint(0, self.columns - 1)
        return x, y
    
    def generate_new_cell(self):
        empty_cells = [
            (i, j)
            for i in range(self.rows)
            for j in range(self.columns)
            if self.grid[i][j].value == 0
        ]

        if not empty_cells:
            return  # No space to place a new cell. (Finir boucle devenemnt ici) 

        x, y = random.choice(empty_cells)
        self.grid[x][y].value = self.cell_initial_value()

    def starting_board(self):
        nb_starting_cell = 2

        for _ in range(nb_starting_cell):
            self.generate_new_cell()

        
