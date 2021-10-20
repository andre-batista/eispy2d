import copy as cp
import numpy as np
from scipy.stats import cauchy
from evoalglib.mechanism import Mechanism, get_indexes
from numpy.random import randint, rand, normal
from evoalglib.crossover import Binomial


class LSHADE(Mechanism):
    def __init__(self, boundary_condition, maximum_evaluations,
                 index_selection='random', maximum_greediness=.2,
                 archive_size=None, memory_history_size=100,
                 minimum_population_size=4):
        super().__init__(boundary_condition)
        self.index_selection = index_selection
        self.A, self.NA = None, archive_size
        self.MCR, self.MF, self.H = None, None, memory_history_size
        self.pmax = maximum_greediness
        self.MAX_NFE = maximum_evaluations
        self.Nmin = minimum_population_size
        self.Ninit = None
    def reset_variables(self, population_size, representation):
        super().reset_variables(population_size, representation)
        self.A = np.zeros((0, representation.nvar))
        self.MCR, self.MF = [0.5], [0.5]
        self.Ninit = population_size
        if self.Nmin == None:
            self.Nmin = population_size
    def run(self, population, population_fitness, objective_function,
            current_nevals):
        _, _, nevals = super().run(population, population_fitness,
                                   objective_function, current_nevals)
        P, fx, objfun = population, population_fitness, objective_function
        NPOP = P.shape[0]
        p = self._set_p(NPOP)
        NA = self._set_NA(NPOP)
        
        V, F = self._mutation(P, fx, p)
        V, CR = self._crossover(P, V)

        self.bc.run(V)
        fv = np.zeros(NPOP)
        for i in range(NPOP):
            fv[i] = objfun.eval(V[i, :])
            nevals += 1

        P, fx = self._selection(P, fx, V, fv, F, CR, NA)
        
        NPOP = self._update_population_size(current_nevals + nevals)
        
        imin = np.argmin(fx)
        self.xopt = np.copy(P[imin, :])
        self.fopt = fx[imin]

        if imin >= NPOP:
            j = randint(NPOP)
            P[j, :] = P[imin, :]
            fx[j] = fx[imin]
        
        population = P[:NPOP, :]
        population_fitness = fx[:NPOP]
        
        return population, population_fitness, current_nevals + nevals

    def copy(self, new=None):
        if new is None:
            new = LSHADE(self.bc, self.MAX_NFE, self.index_selection,
                         self.pmax, self.NA, self.H, self.Nmin)
            new.Ninit = self.Ninit
            new.A = cp.deepcopy(self.A)
            new.MCR, new.MF = cp.deepcopy(self.MCR), cp.deepcopy(self.MF)
            new.xopt, new.fopt = cp.deepcopy(self.xopt), cp.deepcopy(self.fopt)
            return new
        else:
            super().copy(new)
            self.MAX_NFE = new.MAX_NFE
            self.index_selection = new.index_selection
            self.pmax = new.pmax
            self.NA = new.NA
            self.H = new.H
            self.Nmin = new.Nmin
            self.Ninit = new.Ninit
            self.A = cp.deepcopy(new.A)
            self.MCR, self.MF = cp.deepcopy(new.MCR), cp.deepcopy(new.MF)

    def _set_p(self, population_size):
        NPOP = population_size
        pmin = 2/NPOP
        return int((pmin + (self.pmax-pmin)*rand())*NPOP)
    
    def _set_NA(self, population_size):
        NPOP = population_size
        if self.NA is None:
            return NPOP
        else:
            return self.NA

    def _join_archive(self, population):
        P = population
        if self.A.size == 0:
            return np.copy(P)
        else:
            return np.vstack((P, self.A))

    def _mutation(self, population, population_fitness, p):
        P, fx = population, population_fitness
        NPOP = P.shape[0]

        xpbest = self._get_pbest(P, fx, p)
        r1 = get_indexes(NPOP, NPOP, self.index_selection)

        PA = self._join_archive(P)
        r2 = get_indexes(PA.shape[0], NPOP, 'random')

        F = self._get_F(NPOP, P.shape[1])
        
        V = P + F*(xpbest-P) + F*(P[r1, :]-PA[r2, :])
        return V, F[:, 0].reshape(-1)
        
    def _get_pbest(self, population, population_fitness, p):
        P, fx = population, population_fitness
        NPOP = fx.size
        ipbest = np.argsort(fx)[:p]
        xpbest = P[ipbest[randint(ipbest.size, size=NPOP)], :]
        return xpbest
        
    def _get_F(self, population_size, number_variables):
        NPOP = population_size
        idx = randint(len(self.MF), size=NPOP)
        MFr = [self.MF[i] for i in idx]
        F = cauchy.rvs(loc=MFr, scale=np.sqrt(0.1), size=NPOP)
        F[F > 1] = 1
        for i in range(F.size):
            if F[i] <= 0:
                stop = False
                while not stop:
                    F[i] = cauchy.rvs(loc=MFr[i], scale=np.sqrt(0.1))
                    if F[i] > 1:
                        F[i] = 1
                        stop = True
                    elif F[i] > 0 and F[i] <= 1:
                        stop = True
        F = np.tile(F.reshape((-1, 1)), (1, number_variables))
        return F

    def _crossover(self, population, mutation):
        
        P, V = population, mutation
        NPOP = P.shape[0]
        CR = self._get_CR(NPOP)
        
        for i in range(NPOP):
            V[i, :], _ = Binomial(CR[i]).run(P[i, :], V[i, :], None, None,
                                             None)
        return V, CR

    def _get_CR(self, population_size):
        std = np.sqrt(0.1)
        NPOP = population_size
        idx = get_indexes(len(self.MCR), NPOP, 'random')
        MCRr = [self.MCR[i] for i in idx]
        CR = normal(loc=MCRr, scale=std, size=NPOP)
        CR[CR < 0] = 0
        CR[CR > 1] = 1
        return CR

    def _selection(self, population, population_fitness, offspring,
                   offspring_fitness, mutation_factor, crossover_rate,
                   archive_size):
        P, fx = population, population_fitness
        V, fv = offspring, offspring_fitness
        F, CR = mutation_factor, crossover_rate
        NPOP, NA = P.shape[0], archive_size
        new_P = np.copy(P)
        new_fx = np.copy(fx)

        SCR, SF, df = [], [], []
        for i in range(NPOP):
            if fv[i] <= fx[i]:
                if self.A.shape[0] < NA:
                    self.A = np.vstack((self.A, P[i, :]))
                else:
                    j = randint(NA)
                    self.A[j, :] = P[i, :]
                SCR.append(CR[i])
                SF.append(F[i])
                df.append(np.abs(fv[i]-fx[i]))
                new_P[i, :] = V[i, :]
                new_fx[i] = fv[i]
        
        if len(SCR) != 0:
            SCR, SF, df = np.array(SCR), np.array(SF), np.array(df)
            w = df/np.sum(df)
            MCR = np.sum(w*SCR)
            MF = np.sum(w*SF**2)/np.sum(w*SF)
            if len(self.MCR) < self.H:
                self.MCR.append(MCR)
                self.MF.append(MF)
            else:
                self.MCR = self.MCR[1:] + [MCR]
                self.MF = self.MF[1:] + [MF]
        
        return new_P, new_fx

    def _update_population_size(self, number_evaluations):
        NFE, MAX_NFE = number_evaluations, self.MAX_NFE
        Nmin, Ninit = self.Nmin, self.Ninit
        N = int(round((Nmin-Ninit)/MAX_NFE*NFE + Ninit))
        if N < self.Nmin:
            return self.Nmin
        else:
            return N

    def __str__(self):
        message = super().__str__()
        message += 'L-SHADE\n'
        message += 'Size of parents archive: %d\n' % self.NA
        message += ('Size of historical memory of control parameters: %d\n'
                    % self.H)
        message += 'Maximum value of greediness parameters: %.2f\n' % self.pmax
        message += 'Maximum number of fitness evaluations: %d\n' % self.MAX_NFE
        message += 'Minimum population size: %d\n' % self.Nmin
        message += 'Initial population size: %d\n' % self.Ninit
        message += ('Index selection for mutation: ' + self.index_selection
                    + '\n')
        message += str(self.bc)
        return message

