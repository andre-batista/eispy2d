import error
import numpy as np
from evoalglib.mechanism import Mechanism, get_indexes


class GeneticAlgorithm(Mechanism):
    def __init__(self, boundary_condition, crossover, pcross, mutation, pmut,
                 selection, pair_selection='random'):
        super().__init__(boundary_condition)
        self.crossover, self.pcross = crossover, pcross
        self.mutation, self.pmut = mutation, pmut
        self.selection = selection
        if pair_selection != 'random' and pair_selection != 'permutation':
            raise error.WrongValueInput('GeneticAlgorithm', 'pair_selection',
                                        "'random' or 'permutation'",
                                        pair_selection)
        self.pair_selection = pair_selection
    def reset_variables(self, population_size, representation):
        super().reset_variables(population_size, representation)
    def run(self, population, population_fitness, objective_function,
            current_nevals):
        _, _, nevals = super().run(population, population_fitness,
                                   objective_function, current_nevals)
        P, fx, objfun = population, population_fitness, objective_function
        NPOP = P.shape[0]
        i = get_indexes(NPOP, NPOP, self.pair_selection)
        j = get_indexes(NPOP, NPOP, self.pair_selection)
        POff, fOff = self.crossover.run(P[i, :], P[j, :], fx[i], fx[j],
                                        self.pcross)
        POff, fOff = self.mutation.run(POff, fOff, self.pmut)
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
            new = GeneticAlgorithm(self.bc, self.crossover.copy(), self.pcross,
                                   self.mutation.copy(), self.pmut,
                                   self.selection.copy(), self.pair_selection)
            new.xopt = np.copy(self.xopt)
            new.fopt = self.fopt
            return new
        else:
            super().copy(new)
            self.crossover, self.pcross = new.crossover.copy(), new.pcross
            self.mutation, self.pmut = new.mutation.copy(), new.pmut
            self.selection = new.selection.copy()
            self.pair_selection = new.pair_selection
    def __str__(self):
        message = super().__str__()
        message += str(self.crossover) + '\n'
        message += 'Crossover probability: %.2f\n' % self.pcross
        message += str(self.mutation) + '\n'
        message += 'Mutation probability: %.2f\n' % self.pmut
        message += str(self.selection) + '\n'
        message += 'Pair selection for operators: ' + self.pair_selection
        message += str(self.bc)
        return message
        
