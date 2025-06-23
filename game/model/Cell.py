class Cell:
    """ Cell class to instintiate the value of a cell"""
    def __init__(self, x, y, value = 0):
        self.x = x
        self.y = y
        self.value = value

    def __deepcopy__(self, memo):
        """ Create a deep copy of the cell """
        return Cell(self.x, self.y, self.value)