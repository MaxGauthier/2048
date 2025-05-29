class GameModel:
    
    # variables

    # cell logic

    def __init__(self):
        grid = []
        for i in range(0, 16):
            grid.append(i)

        self.grid = grid
        
model = GameModel()

print(model.grid)

        