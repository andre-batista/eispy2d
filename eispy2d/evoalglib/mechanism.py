import numpy as np
from abc import ABC, abstractmethod
from numpy.random import randint, permutation


class Mechanism(ABC):
    def __init__(self, boundary_condition):
        self.bc = boundary_condition
        self.xopt = None
        self.fopt = None
    @abstractmethod
    def reset_variables(self, population_size, representation):
        self.xopt = None
        self.fopt = None
    @abstractmethod
    def run(self, population, population_fitness, objective_function,
            current_nevals):
        population = None
        population_fitness = None
        nevals = 0
        return population, population_fitness, nevals
    def best(self):
        return np.copy(self.xopt), self.fopt
    def copy(self, new=None):
        if new is None:
            new = Mechanism(self.bc)
            new.xopt, new.fopt = self.xopt, self.fopt
            return new
        else:
            self.bc = new.bc
            self.xopt, self.fopt = new.xopt, new.fopt
    @abstractmethod
    def __str__(self):
        return "Mechanism: "


def get_indexes(NMAX, size, selection):
    if selection == 'random':
        return randint(NMAX, size=size)
    elif selection == 'permutation':
        return permutation(NMAX)[:size]
