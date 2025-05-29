from Cell import Cell

class Move:
    def __init__(self, grid):
        self.grid = grid.grid
        self.score = 0

    def remove_empty_cells(self, row):
        non_zeros = []
        for cell in row:
            if cell.value != 0:
                non_zeros.append(cell.value)
        return non_zeros
    
    def pad_left(self, non_zero_tab, row):              # non_zero_tab = remove_empty_cells()
        nb_zeros = len(row) - len(non_zero_tab)     
        padded_row = []
        for _ in range(nb_zeros):
            padded_row.append(0)                        # output: [0,0]
        for value in non_zero_tab:
            padded_row.append(value)                    # output: [0,0,2,4]
        return padded_row                       
    
    def pad_right(self, non_zero_tab, row):       
        nb_zeros = len(row) - len(non_zero_tab)
        padded_row = []
        for value in non_zero_tab:
            padded_row.append(value)                    # output: [2,4]
        for _ in range(nb_zeros):
            padded_row.append(0)                        # output: [2,4,0,0]
        return padded_row

    def combine_right(self, row):               
        for i in range(len(row) - 1, 0, -1):
            if row[i] == row[i - 1] and row[i] != 0:
                row[i] *= 2
                row[i - 1] = 0
                #score += row[i].value

        return row
    
    def combine_left(self, row):               
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                #score += row[i].value

        return row

    def move(self, direction):
        if direction == "up" or direction == "down":
            for col_idx in range(len(self.grid[0])):
                column = [self.grid[row_idx][col_idx] for row_idx in range(len(self.grid))]
                non_zeros = self.remove_empty_cells(column)
                if direction == "down":
                    padded = self.pad_left(non_zeros, column)
                    combined = self.combine_right(padded)
                else:
                    padded = self.pad_right(non_zeros, column)
                    combined = self.combine_left(padded)

                new_values = [val for val in combined if val != 0]

                if direction == "down":
                    final_values = self.pad_left(new_values, column)
                else:
                    final_values = self.pad_right(new_values, column)

                for row_idx in range(len(self.grid)):
                    self.grid[row_idx][col_idx].value = final_values[row_idx]
        else:
            for row in self.grid:
                non_zeros = self.remove_empty_cells(row)
                if direction == "left":
                    padded_row = self.pad_right(non_zeros, row)
                    combined_row = self.combine_left(padded_row)
                else:
                    padded_row = self.pad_left(non_zeros, row)
                    combined_row = self.combine_right(padded_row)

                new_row = [val for val in combined_row if val != 0]
                if direction == "left":
                    final_row = self.pad_right(new_row, row)
                else: 
                    final_row = self.pad_left(new_row, row)
                # Update the row's cell values
                for i in range(len(row)):
                    row[i].value = final_row[i]