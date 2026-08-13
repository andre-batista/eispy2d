import numpy as np
from numpy import nan as NaN
from abc import ABC, abstractmethod
from numpy.random import normal, rand


class Mutation(ABC):
    """Abstract base class for mutation operators.

    Mutation operators introduce random variations to solutions.

    Methods
    -------
    run(x, fx, probability=None)
        Apply mutation to solution vector.
    copy(new=None)
        Create a copy of the mutation operator.
    """
    def __init__(self):
        super().__init__()
    @abstractmethod
    def run(self, x, fx, probability=None):
        """Apply mutation to solution vector.

        Parameters
        ----------
        x : numpy.ndarray
            Solution vector to mutate.
        fx : numpy.ndarray or float
            Fitness values of solutions.
        probability : float, optional
            Mutation probability.

        Returns
        -------
        tuple
            (mutated_x, mutated_fx) where mutated_fx may be NaN for new mutations.
        """
        pass
    def copy(self, new=None):
        if new is None:
            return Mutation()
        else:
            pass
    @abstractmethod
    def __str__(self):
        return 'Mutation: '


class Gaussian(Mutation):
    """Gaussian mutation operator.

    Adds Gaussian noise to solution vector.

    Parameters
    ----------
    std : float, default=0.5
        Standard deviation of Gaussian noise.
    """
    def __init__(self, std=.5):
        super().__init__()
        self.sigma = std
    def run(self, x, fx, probability=None):
        super().run(x, fx, probability=None)
        pmut = probability
        if pmut is None:
            if x.ndim == 1:
                fn = NaN
            else:
                fn = NaN*np.ones(fx.size)
            return x + normal(scale=self.sigma, size=x.shape), fn
        else:
            if x.ndim == 1:
                if rand() <= pmut:
                    return x + normal(scale=self.sigma, size=x.shape), NaN
                else:
                    return x, fx
            else:
                m = rand(x.shape[0]) <= pmut
                xm = np.copy(x)
                fm = np.copy(fx)
                xm[m, :] = x[m, :] + normal(scale=self.sigma,
                                                size=np.shape(x[m, :]))
                fm[m] = NaN
                return xm, fm
    def copy(self, new=None):
        if new is None:
            return Gaussian(self.sigma)
        else:
            self.sigma = new.sigma
    def __str__(self):
        message = super().__str__()
        message += 'Gaussian (std: %.1e)' % self.sigma
        return message


class Polynomial(Mutation):
    """Polynomial mutation operator.

    Implements polynomial mutation commonly used in real-coded GAs.

    Parameters
    ----------
    eta : float
        Distribution index. Higher values produce mutations closer to parent.
    """
    def __init__(self, eta):
        super().__init__()
        self.eta = eta
    def run(self, x, fx, probability):
        pmut = probability
        if pmut is None:
            if x.ndim == 1:
                xn = np.zeros(x.size)
                fn = NaN
                u = rand(x.size)
            else:
                xn = np.zeros(x.shape)
                fn = NaN*np.ones(fx.size)
                u = rand(x.shape[0], x.shape[1])
            i = u <= .5
            j = np.logical_not(i)
            xn[i] = x[i] + (2*u[i])**(1/(self.eta+1)) - 1
            xn[j] = x[j] + 1 - (2*(1-u[j]))**(1/(self.eta+1))
            return xn, fn
        else:
            if x.ndim == 1:
                if rand() <= pmut:
                    xn = np.zeros(x.size)
                    u = rand(x.size)
                    i = u <= .5
                    j = np.logical_not(i)
                    xn[i] = x[i] + (2*u[i])**(1/(self.eta+1)) - 1
                    xn[j] = x[j] + 1 - (2*(1-u[j]))**(1/(self.eta+1))
                    return xn, NaN
                else:
                    return np.copy(x), np.copy(fx)
            else:
                xm = np.copy(x)
                fm = np.copy(fx)
                m = rand(x.shape[0]) <= pmut
                u = rand(np.count_nonzero(m), x.shape[1])
                i = u <= .5
                j = np.logical_not(i)
                aux = np.zeros(i.shape)
                aux[i] = x[m, :][i] + (2*u[i])**(1/(self.eta+1)) - 1
                aux[j] = x[m, :][j] + 1 - (2*(1-u[j]))**(1/(self.eta+1))
                xm[m, :] = aux
                fm[m] = NaN
                return xm, fm
    def copy(self, new=None):
        if new is None:
            return Polynomial(self.eta)
        else:
            self.eta = new.eta
    def __str__(self):
        message = super().__str__()
        message += 'Polynomial (eta: %.1e)' % self.eta
        return message

