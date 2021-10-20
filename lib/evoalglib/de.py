import error
import numpy as np
from numpy import NaN
from evoalglib.mechanism import Mechanism, get_indexes


RAND = 'rand'
BEST = 'best'
CURRENT2BEST = 'current-to-best'
RAND2BEST = 'rand-to-best'


class DifferentialEvolution(Mechanism):
    def __init__(self, boundary_condition, selection, mutation, scaling_factor,
                 crossover, pcross=1., index_selection='random'):
        if (mutation != RAND and mutation != BEST
                and mutation != CURRENT2BEST and mutation != RAND2BEST):
            raise error.WrongValueInput('DifferentialEvolution.__init__',
                                        'mutation', "'rand' or 'best' or "
                                        + "'current-to-best' or "
                                        + "'rand-to-best'", str(mutation))
        if index_selection != 'random' and index_selection != 'permutation':
            raise error.WrongValueInput('DifferentialEvolution.__init__',
                                        'index_selection',
                                        "'random' or 'permutation'",
                                        index_selection)
        super().__init__(boundary_condition)
        self.selection = selection
        self.mutation = mutation
        self.F = scaling_factor
        self.crossover = crossover
        self.pcross = pcross
        self.index_selection = index_selection

    def reset_variables(self, population_size, representation):
        super().reset_variables(population_size, representation)

    def run(self, population, population_fitness, objective_function,
            current_nevals):
        _, _, nevals = super().run(population, population_fitness,
                                   objective_function, current_nevals)
        P, fx, objfun = population, population_fitness, objective_function
        NPOP = P.shape[0]
        if self.mutation == RAND:
            V = self._rand(P)
        elif self.mutation == BEST:
            V = self._best(P, fx)
        elif self.mutation == CURRENT2BEST:
            V = self._current2best(P, fx)
        elif self.mutation == RAND2BEST:
            V = self._rand2best(P, fx)
        POff, fOff = self.crossover.run(P, V, fx, NaN*np.ones(fx.size),
                                        self.pcross)
        self.bc.run(POff)
        for i in range(NPOP):
            if np.isnan(fOff[i]):
                fOff[i] = objfun.eval(POff[i, :])
                nevals += 1
        P, fx = self.selection.run(P, fx, POff, fOff, NPOP) 
        population[:, :] = P[:, :]
        population_fitness[:] = fx[:]
        imin = np.argmin(fx)
        self.xopt = np.copy(P[imin, :])
        self.fopt = fx[imin]
        return population, population_fitness, current_nevals + nevals

    def copy(self, new=None):
        if new is None:
            new = DifferentialEvolution(self.bc, self.selection, self.mutation,
                                        self.F, self.crossover.copy(),
                                        self.pcross, self.index_selection)
            new.xopt = self.xopt
            new.fopt = self.fopt
            return new
        else:
            super().copy(new)
            self.mutation, self.F = new.mutation, new.F
            self.crossover = new.crossover.copy()
            self.pcross = new.pcross
            self.index_selection = new.index_selection
    
    def _rand(self, population):
        x = population
        r1 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        r2 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        r3 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        return x[r1, :] + self.F*(x[r2, :]-x[r3, :])

    def _best(self, population, population_fitness):
        x, fx = population, population_fitness
        best = np.argmin(population_fitness)
        xbest = np.tile(x[best, :], (x.shape[0], 1))
        r1 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        r2 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        return xbest + self.F*(x[r1, :]-x[r2, :])

    def _current2best(self, population, population_fitness):
        x, fx = population, population_fitness
        best = np.argmin(fx)
        xbest = np.tile(x[best, :], (x.shape[0], 1))
        r1 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        r2 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        return x + self.F*(xbest-x) + self.F*(x[r1, :]-x[r2, :])

    def _rand2best(self, population, population_fitness):
        x, fx = population, population_fitness
        best = np.argmin(fx)
        xbest = np.tile(x[best, :], (x.shape[0], 1))
        r1 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        r2 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        r3 = get_indexes(x.shape[0], x.shape[0], self.index_selection)
        return x[r1, :] + self.F*(xbest-x[r1, :]) + self.F*(x[r2, :]-x[r3, :])

    def __str__(self):
        message = super().__str__()
        message += 'Differential Evolution\n'
        message += 'Mutation: ' + self.mutation + ', factor: %.2f\n' % self.F
        message += str(self.crossover) + '\n'
        message += 'Crossover probability: %.2f\n' % self.pcross
        message += ('Index selection for mutation: ' + self.index_selection
                    + '\n')
        message += str(self.bc)
        return message
