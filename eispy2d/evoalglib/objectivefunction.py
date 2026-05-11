import error
import numpy as np
import evoalglib.representation as rpt
from abc import ABC, abstractmethod
from numpy import pi


class ObjectiveFunction(ABC):
    def __init__(self):
        self.name = None
    def set_parameters(self, representation, scattered_field,
                       incident_field):
        self.representation = representation
        self.scattered_field = scattered_field
        self.incident_field = incident_field
    @abstractmethod
    def eval(self, x):
        pass
    @abstractmethod
    def __str__(self):
        return "Objective Function: "


class Rastrigin(ObjectiveFunction):
    def __init__(self, amplitude=10):
        super().__init__()
        self.name = 'rastringin'
        self.A = amplitude
        self.xopt = 0.
    def eval(self, x):
        if not isinstance(self.representation, rpt.CanonicalProblems):
            raise error.WrongTypeInput('Rastrigin.eval', 'representation',
                                       'CanonicalProblems',
                                       str(type(self.representation)))
        super().eval(x)
        n = x.size
        x = self.representation.contrast(x)
        return self.A*n + np.sum(x**2-self.A*np.cos(2*pi*x))
    def __str__(self):
        message = super().__str__()
        message += 'Rastringin (Canonical, Nonlinear, Multimodal)\n'
        message += 'Amplitude: %.1f\n' % self.A
        message += 'Optimum solution: %.2f' % self.xopt
        return message


class Ackley(ObjectiveFunction):
    def __init__(self):
        super().__init__()
        self.name = 'ackley'
        self.xopt = 0.
    def eval(self, x):
        if not isinstance(self.representation, rpt.CanonicalProblems):
            raise error.WrongTypeInput('Ackley.eval', 'representation',
                                       'CanonicalProblems',
                                       str(type(self.representation)))
        super().eval(x)
        n = x.size
        x = self.representation.contrast(x)        
        return (-20*np.exp(-.2*np.sqrt(1/n*np.sum(x**2)))
                - np.exp(1/n*np.sum(np.cos(2*pi*x))) + 20 + np.exp(1))
    def __str__(self):
        message = super().__str__()
        message += 'Ackley (Canonical, Nonlinear, Multimodal)\n'
        message += 'Optimum solution: %.2f' % self.xopt
        return message


class Rosenbrock(ObjectiveFunction):
    def __init__(self):
        super().__init__()
        self.name = 'rosenbrock'
        self.xopt = 0.
    def eval(self, x):
        if not isinstance(self.representation, rpt.CanonicalProblems):
            raise error.WrongTypeInput('Rosenbrock.eval', 'representation',
                                       'CanonicalProblems',
                                       str(type(self.representation)))
        super().eval(x)
        n = x.size
        x = self.representation.contrast(x)     
        i = np.arange(n-1)
        return np.sum(100*(x[i]**2-x[i+1])**2 + (x[i]-1)**2)
    def __str__(self):
        message = super().__str__()
        message += 'Rosenbrock (Canonical, Nonlinear, Multimodal)\n'
        message += 'Optimum solution: %.2f' % self.xopt
        return message


class WeightedSum(ObjectiveFunction):
    def __init__(self):
        super().__init__()
        self.name = 'weighted_sum'
    def eval(self, x):
        X = self.representation.contrast(x)
        E = self.representation.total_field(x, self.representation.discretization.elements)
        data_res = self.representation.discretization.residual_data(self.scattered_field,
                                                          contrast=X,
                                                          total_field=E)
        state_res = self.representation.discretization.residual_state(self.incident_field,
                                                            contrast=X,
                                                            total_field=E)
        return (np.sum(np.abs(data_res)**2)/np.sum(np.abs(self.scattered_field)**2)
                + np.sum(np.abs(state_res)**2)
                / np.sum(np.abs(self.incident_field)**2))
    def __str__(self):
        message = super().__str__()
        message += 'Weighted Sum of Data and State Equations Residuals'
        return message

