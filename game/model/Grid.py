import random
from Cell import Cell
from Move import Move
from GenerateCell import GenerateCell

class Grid:
    """ Generates each cell of the grid rows x columns with each cell value from Cell() 
        Output: [0, 0, 0, 0]
                [0, 0, 0, 0]
                [0, 0, 0, 0]
                [0, 0, 0, 0]
    """

    def __init__(self, rows, columns, width, height):
        self.width = width
        self.height = height
        self.rows = rows
        self.columns = columns 
        self.grid = self.create_grid()
        self.previous_grid = None  # for backup before a move
        self.move = Move(self)
        self.generate_cell = GenerateCell(self.grid, self.rows, self.columns)

    def create_grid(self):
        grid = []
        pos = 0
        for _ in range(0, self.rows):
            row = []
            for _ in range(self.columns):
                row.append(Cell(pos))     
                pos += 1
            grid.append(row)
        return grid

    def backup_grid(self):
        self.previous_grid = [[Cell(cell.pos, cell.value) for cell in row]
        for row in self.grid]

    def print_grid(self):
        for row in g.grid:
            t = []
            for cell in row:
                t.append(cell.value)
            print(t)

    def reset_to_previous(self):
        if self.previous_grid is None:
            print("No previous state to reset to.")
            return

        # Copy values back to current grid
        for i in range(self.rows):
            for j in range(self.columns):
                self.grid[i][j].value = self.previous_grid[i][j].value

    def random_move(self):
        directions = ["left", "right", "up", "down"]

        for _ in range(2):
            direction = random.choice(directions)
            print("MOVE MADE:", direction)
            
            self.backup_grid()  # Save state before move
            
            self.move.move(direction)
            self.generate_cell.generate_new_cell()
            self.print_grid()
        

g = Grid()
g.generate_cell.starting_board()
g.print_grid()
g.random_move()
print("RESET")
g.reset_to_previous()
g.print_grid()



