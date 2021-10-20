import numpy as np
from numpy import NaN
from abc import ABC, abstractmethod
from numpy.random import rand, randint

class Crossover(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def run(self, x1, x2, fx1, fx2, probability=None):
        pass
    def copy(self, new=None):
        if new is None:
            return Crossover()
        else:
            pass
    @abstractmethod
    def __str__(self):
        return "Crossover: "


class Discrete(Crossover):
    def __init__(self):
        super().__init__()
    def run(self, x1, x2, fx1, fx2, probability=None):
        super().run(x1, x2, fx1, fx2, probability=probability)
        pcross = probability
        if x1.ndim == 1:
            if pcross is None or rand() <= pcross:
                u1 = rand(x1.size) <= .5
                x = np.copy(x2)
                x[u1] = x1[u1]
                return x, NaN
            else:
                if rand() <= .5:
                    return np.copy(x1), fx1
                else:
                    return np.copy(x2), fx2
        else:
            if pcross is None:
                u1 = rand(x1.shape[0], x1.shape[1]) <= .5
                x = np.copy(x2)
                x[u1] = x1[u1]
                return x, NaN*np.ones(fx1.size)
            else:
                x = np.zeros(x1.shape, dtype=x1.dtype)
                fx = NaN*np.ones(fx1.size)
                i = rand(x1.shape[0]) <= pcross
                j = np.logical_not(i)
                u1 = rand(np.count_nonzero(i), x1.shape[1]) <= .5
                offs = x2[i, :]
                offs[u1] = x1[i, :][u1]
                x[i, :] = offs
                copies = x2[j, :]
                u1 = rand(np.count_nonzero(j)) <= .5
                copies[u1, :] = x1[j, :][u1, :]
                x[j, :] = copies
                fx[j] = fx2[j]
                fx[j][u1] = fx1[j][u1]
                return x, fx
    def copy(self, new=None):
        if new is None:
            return Discrete()
        else:
            pass
    def __str__(self):
        message = super().__str__()
        message += 'Discrete'
        return message

              
class Convex(Crossover):
    def __init__(self, extrapolation=0):
        super().__init__()
        self.xi = extrapolation
    def run(self, x1, x2, fx1, fx2, probability=None):
        super().run(x1, x2, fx1, fx2, probability=probability)
        pcross = probability
        if x1.ndim == 1:
            if pcross is None or rand() <= pcross:
                u = -self.xi + (1+2*self.xi)*rand(x1.size)
                return u*x1 + (1-u)*x2, NaN
            else:
                if rand() <= .5:
                    return np.copy(x1), fx1
                else:
                    return np.copy(x2), fx2
        else:
            if pcross is None:
                u = -self.xi + (1+2*self.xi)*rand(x1.shape[0], x2.shape[1])
                return u*x1 + (1-u)*x2, NaN*np.ones(fx1.size)
            else:
                x = np.zeros(x1.shape, dtype=x1.dtype)
                fx = NaN*np.ones(fx1.size)
                c = rand(x1.shape[0]) <= pcross
                nc = np.logical_not(c)
                u = (-self.xi + (1+2*self.xi)*rand(np.count_nonzero(c),
                                                       x1.shape[1]))
                offs = u*x1[c, :] + (1-u)*x2[c, :]
                u = rand(np.count_nonzero(nc)) <= .5
                copies = x2[nc, :]
                copies[u, :] = x1[nc, :][u, :]
                x[c, :] = offs
                x[nc, :] = copies
                fx[nc] = fx2[nc]
                fx[nc][u] = fx1[nc][u]
                return x, fx
    def copy(self, new=None):
        if new is None:
            return Convex(self.xi)
        else:
            self.xi = new.xi
    def __str__(self):
        message = super().__str__()
        message += 'Convex, extrapolation factor: %.2f' % self.xi
        return message


class SimulatedBinary(Crossover):
    # quanto maior eta, mais proximo do pai
    def __init__(self, eta):
        super().__init__()
        self.eta = eta
    def run(self, x1, x2, fx1, fx2, probability=None):
        super().run(x1, x2, fx1, fx2, probability=probability)
        pcross = probability
        if x1.ndim == 1:
            if pcross is None or rand() <= pcross:
                u = rand(x1.size)
                beta = (2*(1-u))**(-1/(self.eta+1))
                beta[u <= .5] = (2*u[u <= .5])**(1/(self.eta+1))
                if rand() <= .5:
                    return .5*((1+beta)*x1 + (1-beta)*x2), NaN
                else:
                    return .5*((1-beta)*x1 + (1+beta)*x2), NaN
            else:
                if rand() <= .5:
                    return np.copy(x1), fx1
                else:
                    return np.copy(x2), fx2
        else:
            if pcross is None:
                u = rand(x1.shape[0], x2.shape[1])
                beta = (2*(1-u))**(-1/(self.eta+1))
                beta[u <= .5] = (2*u[u <= .5])**(1/(self.eta+1))
                if rand() <= .5:
                    return (.5*((1+beta)*x1 + (1-beta)*x2),
                            NaN*np.ones(fx1.size))
                else:
                    return (.5*((1-beta)*x1 + (1+beta)*x2),
                            NaN*np.ones(fx1.size))
            else:
                x = np.zeros(x1.shape, dtype=x1.dtype)
                fx = NaN*np.ones(fx1.size)
                c = rand(x1.shape[0]) <= pcross
                nc = np.logical_not(c)
                u = rand(np.count_nonzero(c), x1.shape[1])

                beta = (2*(1-u))**(-1/(self.eta+1))
                beta[u <= .5] = (2*u[u <= .5])**(1/(self.eta+1))
                i = rand(np.count_nonzero(c)) <= .5
                j = np.logical_not(i)

                aux = np.zeros((i.size, x.shape[1]))
                aux[i, :] = .5*((1+beta[i, :])*x1[c, :][i, :]
                                + (1-beta[i, :])*x2[c, :][i, :])
                aux[j, :] = .5*((1-beta[j, :])*x1[c, :][j, :]
                                + (1+beta[j, :])*x2[c, :][j, :])
                x[c, :] = aux

                i = rand(np.count_nonzero(nc)) <= .5
                j = np.logical_not(i)
                aux = np.zeros((i.size, x.shape[1]))
                aux[i, :] = x1[nc, :][i, :]
                aux[j, :] = x2[nc, :][j, :]
                x[nc, :] = aux
                aux = np.zeros(i.size)
                aux[i] = fx1[nc][i]
                aux[j] = fx2[nc][j]
                fx[nc] = aux
                return x, fx
    def copy(self, new=None):
        if new is None:
            return SimulatedBinary(self.eta)
        else:
            self.eta = new.eta
    def __str__(self):
        message = super().__str__()
        message += 'Simulated Binary, factor: %.2f' % self.eta
        return message


class Binomial(Crossover):
    def __init__(self, crossover_rate):
        super().__init__()
        self.CR = crossover_rate
    def run(self, x1, x2, fx1, fx2, probability=None):
        super().run(x1, x2, fx1, fx2, probability=probability)
        pcross = probability
        if x1.ndim == 1:
            if pcross is None or rand() <= pcross:
                u = rand(x1.size) <= self.CR
                jrand = randint(x1.size)
                x = np.copy(x1)
                x[u] = x2[u]
                x[jrand] = x2[jrand]
                return x, NaN
            else:
                if rand() <= .5:
                    return np.copy(x1), fx1
                else:
                    return np.copy(x2), fx2
        else:
            if pcross is None:
                u = rand(x1.shape[0], x1.shape[1]) <= self.CR
                jrand = randint(x1.shape[1], size=x1.shape[0])
                x = np.copy(x1)
                x[u] = x2[u]
                for i in range(x1.shape[0]):
                    x[i, jrand] = x2[i, jrand]
                return x, NaN*np.ones(fx1.size)
            else:
                x = np.zeros(x1.shape, dtype=x1.dtype)
                fx = NaN*np.ones(fx1.size)
                i = rand(x1.shape[0]) <= pcross
                j = np.logical_not(i)
                u = rand(np.count_nonzero(i), x1.shape[1]) <= self.CR
                xrand = randint(x1.shape[1], size=np.count_nonzero(i))
                offs = x1[i, :]
                offs[u] = x2[i, :][u]
                for k in range(xrand.size):
                    offs[k, xrand[k]] = x2[i, :][k, xrand[k]]
                x[i, :] = offs
                x[j, :] = x1[j, :]
                fx[j] = fx1[j]
                return x, fx
    def copy(self, new=None):
        if new is None:
            return Binomial(self.CR)
        else:
            self.CR = new.CR
    def __str__(self):
        message = super().__str__()
        message += 'Binomial, crossover rate: %.2f' % self.CR
        return message
