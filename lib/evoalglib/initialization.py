import error
import evoalglib.representation as rpt
import numpy as np
from abc import ABC, abstractmethod
from numpy.random import rand


class Initialization(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def run(self, population_size, representation, incident_field, inputdata):
        pass
    @abstractmethod
    def __str__(self):
        return 'Initialization: '


class UniformRandomDistribution(Initialization):
    def __init__(self):
        super().__init__()
    def run(self, population_size, representation, incident_field, inputdata):
        super().run(population_size, representation, incident_field, inputdata)
        P = rand(population_size, representation.nvar)
        return P
    def __str__(self):
        return super().__str__() + 'Uniform Random Distribution'


class BornApproximation(Initialization):
    def __init__(self):
        super().__init__()
    def run(self, population_size, representation, incident_field, inputdata):
        if not isinstance(representation, rpt.DiscretizationElementBased):
            raise error.WrongTypeInput('BornApproximation.get',
                                       'representation',
                                       'DiscretizationElementBased',
                                       str(type(representation)))
        super().run(population_size, representation, incident_field, inputdata)
        P = np.zeros((population_size, representation.nvar))
        if representation.xvar_real is not None:
            i, j = representation.xvar_real
            P[:, i:j] = rand(population_size ,j-i)
        if representation.xvar_imag is not None:
            i, j = representation.xvar_imag
            P[:, i:j] = rand(population_size, j-i)
        for p in range(population_size):
            X = representation.contrast(P[p, :])
            E = representation.discretization.solve(incident_field=incident_field, contrast=X, total_field=True)
            x = np.hstack((np.real(E.flatten()), np.imag(E.flatten())))
            lbx = representation.lb[:representation.evar_real[0]]
            x = np.hstack((lbx, x))
            x = representation.real2unit(x)
            i, j = representation.evar_real
            P[p, i:j] = x[i:j]
            i, j = representation.evar_imag
            P[p, i:j] = x[i:j]

        return P
    def __str__(self):
        return super().__str__() + 'Born Approximation'