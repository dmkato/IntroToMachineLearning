import numpy as np

class Data:
    def __init__(self, x, y):
        self.x = np.array(x, dtype=float)
        self.y = int(y)
