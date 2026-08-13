import numpy as np
from abc import ABC, abstractmethod
from numpy.random import randint, permutation


class Mechanism(ABC):
    """Abstract base class for evolutionary mechanisms.

    Evolutionary mechanisms define how a population evolves from one
    generation to the next.

    Parameters
    ----------
    boundary_condition : BoundaryCondition
        Boundary handling strategy.

    Attributes
    ----------
    bc : BoundaryCondition
        Boundary handling strategy.
    xopt : numpy.ndarray or None
        Best solution found.
    fopt : float or None
        Best fitness value found.

    Methods
    -------
    reset_variables(population_size, representation)
        Reset internal variables for a new run.
    run(population, population_fitness, objective_function, current_nevals)
        Execute one generation of evolution.
    best()
        Return the best solution found.
    """
    def __init__(self, boundary_condition):
        self.bc = boundary_condition
        self.xopt = None
        self.fopt = None
    @abstractmethod
    def reset_variables(self, population_size, representation):
        """Reset internal variables for a new run.

        Parameters
        ----------
        population_size : int
            Size of the population.
        representation : Representation
            Solution representation.
        """
        self.xopt = None
        self.fopt = None
    @abstractmethod
    def run(self, population, population_fitness, objective_function,
            current_nevals):
        """Execute one generation of evolution.

        Parameters
        ----------
        population : numpy.ndarray
            Current population matrix (POP × NVAR).
        population_fitness : numpy.ndarray
            Fitness values of current population.
        objective_function : ObjectiveFunction
            Objective function to evaluate.
        current_nevals : int
            Current number of evaluations.

        Returns
        -------
        tuple
            (population, population_fitness, new_evaluation_count)
        """
        population = None
        population_fitness = None
        nevals = 0
        return population, population_fitness, nevals
    def best(self):
        """Return the best solution found.

        Returns
        -------
        tuple
            (xopt, fopt) where xopt is the best solution and fopt its fitness.
        """
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
    """Generate random indexes for evolutionary operations.

    Parameters
    ----------
    NMAX : int
        Maximum index value (exclusive).
    size : int
        Number of indexes to generate.
    selection : {'random', 'permutation'}
        Selection strategy:
        - 'random': Random choice with replacement.
        - 'permutation': Permutation without replacement.

    Returns
    -------
    numpy.ndarray
        Array of indexes of length `size`.
    """
    if selection == 'random':
        return randint(NMAX, size=size)
    elif selection == 'permutation':
        return permutation(NMAX)[:size]
