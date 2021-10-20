import numpy as np
from abc import ABC, abstractmethod


class BoundaryCondition(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def run(self, x):
        pass
    @abstractmethod
    def __str__(self):
        return 'Boundary Condition: '


class Truncation(BoundaryCondition):
    def __init__(self):
        super().__init__()
    def run(self, x):
        super().run(x)
        x[x < 0] = 0
        x[x > 1] = 1
    def __str__(self):
        return super().__str__() + 'Truncation'


class Reflection(BoundaryCondition):
    def __init__(self):
        super().__init__()
    def run(self, x):
        super().run(x)
        i = np.logical_and(x < 0, np.mod(np.abs(x), 2) < 1)
        j = np.logical_and(x < 0, np.mod(np.abs(x), 2) >= 1)
        x[i] = np.ceil(x[i]) - x[i]
        x[j] = 1 - (np.ceil(x[j])-x[j])
        i = np.logical_and(x > 1, np.mod(x, 2) < 1)
        j = np.logical_and(x > 1, np.mod(x, 2) >= 1)
        x[i] = x[i] - np.floor(x[i])
        x[j] = 1 - (x[j]-np.floor(x[j]))
    def __str__(self):
        return super().__str__() + 'Reflection'