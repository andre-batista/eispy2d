import error
import numpy as np
from numba import jit
from abc import ABC, abstractmethod
from numpy.random import permutation, randint, rand


class Selection(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def run(self, P1, fx1, P2=None, fx2=None, NPOP=None):
        if P2 is None and NPOP is None:
            raise error.MissingInputError('Selection.run', 'P2 or NPOP')
        new_population = None
        return new_population
    def copy(self, new=None):
        if new is None:
            return Selection()
        else:
            pass
    @abstractmethod
    def __str__(self):
        return 'Selection: '


class BinaryTournament(Selection):
    def __init__(self, elitism=True, pair_selection='random'):
        super().__init__()
        self.pair_selection = pair_selection
        self.elitism = elitism
    def run(self, P1, fx1, P2=None, fx2=None, NPOP=None):
        super().run(P1, fx1, P2=P2, fx2=fx2, NPOP=NPOP)
        if P2 is not None:
            PU, fu = np.vstack((P1, P2)), np.hstack((fx1, fx2))
            if NPOP is None:
                NPOP = min([P1.shape[0], P2.shape[0]])
        else:
            PU, fu = P1, fx1
        if self.pair_selection == 'permutation':
            i = permutation(PU.shape[0])
            if i.size == 2*NPOP:
                i = i.reshape((-1, 2))
            else:
                i = i[:2*NPOP].reshape((-1, 2))
        elif self.pair_selection == 'random':
            i = randint(PU.shape[0], size=2*NPOP).reshape((-1, 2))
        else:
            raise error.Error("BinaryTournament: the attribute "
                              + "'pair_selection' must be either 'random' or "
                              + "'permutation'")
        j = fu[i[:, 0]] < fu[i[:, 1]]
        P = np.zeros((NPOP, PU.shape[1]))
        fx = np.zeros(NPOP)
        P[j, :] = PU[i[j, 0], :]
        fx[j] = fu[i[j, 0]]
        k = np.logical_not(j)
        P[k, :] = PU[i[k, 1], :]
        fx[k] = fu[i[k, 1]]
        if self.elitism:
            if self.pair_selection == 'permutation' and PU.shape[0] == 2*NPOP:
                pass
            else:
                j = np.argmin(fu)
                P[-1, :] = PU[j, :]
                fx[-1] = fu[j]
        return P, fx
    def copy(self, new=None):
        if new is None:
            return BinaryTournament(self.elitism, self.pair_selection)
        else:
            self.elitism = new.elitism
            self.pair_selection = new.pair_selection
    def __str__(self):
        message = super().__str__()
        message += 'Binary Tournament\n'
        message += 'Elitism? '
        if self.elitism:
            message += 'Yes\n'
        else:
            message += 'No\n'
        message += 'Pair selection: ' + self.pair_selection
        return message
        


class Roullete(Selection):
    def __init__(self, elitism=True):
        super().__init__()
        self.elitism = elitism
    def run(self, P1, fx1, P2=None, fx2=None, NPOP=None):
        super().run(P1, fx1, P2=P2, fx2=fx2, NPOP=NPOP)
        if P2 is not None:
            PU, fu = np.vstack((P1, P2)), np.hstack((fx1, fx2))
            if NPOP is None:
                NPOP = min([P1.shape[0], P2.shape[0]])
        else:
            PU, fu = P1, fx1
        fu_min = np.amin(fu)
        if fu_min <= 0:
            fitness = 1/(-fu_min+fu+1)
        else:
            fitness = 1/fu
        p = fitness/np.sum(fitness)
        cumsum = np.cumsum(p)
        P = np.zeros((NPOP, PU.shape[1]))
        fx = np.zeros(NPOP)
        for i in range(NPOP):
            u = rand()
            j = find_edge(u, cumsum)
            P[i, :] = PU[j, :]
            fx[i] = fu[j]
        if self.elitism:
            i = np.argmin(fu)
            P[-1, :] = PU[i, :]
            fx[-1] = fu[i]
        return P, fx
    def copy(self, new=None):
        if new is None:
            return Roullete(self.elitism)
        else:
            self.elitism = new.elitism
    def __str__(self):
        message = super().__str__()
        message += 'Roullete Wheel\n'
        message += 'Elitism? '
        if self.elitism:
            message += 'Yes'
        else:
            message += 'No'
        return message


@jit(nopython=True)
def find_edge(u, cumsum):
    N = cumsum.size
    i = 0
    while u > cumsum[i] and i < N:
        i += 1
    return i
