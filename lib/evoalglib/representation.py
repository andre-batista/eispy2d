import error
import numpy as np
import collocation as clc
from abc import ABC, abstractmethod


class Representation(ABC):
    def __init__(self):
        self.nvar, self.lb, self.ub, self.dtype = None, None, None, None
    def unit2real(self, x):
        if x.ndim == 1:
            return self.lb + x*(self.ub-self.lb)
        elif x.ndim == 2:
            lb = np.tile(self.lb, (x.shape[0], 1))
            ub = np.tile(self.ub, (x.shape[0], 1))
            return self.lb + x*(self.ub-self.lb)
    def real2unit(self, x):
        if x.ndim == 1:
            return (x-self.lb)/(self.ub-self.lb)
        elif x.ndim == 2:
            lb = np.tile(self.lb, (x.shape[0], 1))
            ub = np.tile(self.ub, (x.shape[0], 1))
            return (x-self.lb)/(self.ub-self.lb)
    @abstractmethod
    def contrast(self, x, mode='array'):
        return self.unit2real(x)
    @abstractmethod
    def total_field(self, x, mode='array'):
        return self.unit2real(x)
    @abstractmethod
    def scattered_field(self, x):
        return None
    @abstractmethod
    def current(self, x, mode='array'):
        return None
    @abstractmethod
    def __str__(self):
        return 'Representation: '


class CanonicalProblems(Representation):
    def __init__(self, number_variables, lb, ub):
        super().__init__()
        self.nvar = number_variables
        self.lb = lb*np.ones(self.nvar)
        self.ub = ub*np.ones(self.nvar)
    def contrast(self, x, mode='array'):
        return super().contrast(x, mode=mode)
    def total_field(self, x, mode='array'):
        return None
    def scattered_field(self, x):
        return None
    def current(self, x):
        return None
    def __str__(self):
        message = super().__str__()
        message += 'Canonical Problems\n'
        message += 'Number of variables: %d\n' % self.nvar
        message += ('Variables range: [%.2e, ' % self.lb[0]
                    + '%.2e]' % self.ub[0])
        return message
        

class DiscretizationElementBased(Representation):
    def __init__(self, discretization, contrast_bounds, total_bounds):
        super().__init__()

        self.discretization = discretization
        self.perfect_dielectric = self.discretization.configuration.perfect_dielectric
        self.good_conductor = self.discretization.configuration.good_conductor

        if isinstance(discretization, clc.Collocation):
            NPIXELS = np.prod(self.discretization.elements)
            NS = self.discretization.configuration.NS
            if self.perfect_dielectric:
                self.nvar = (1 + 2*NS)*NPIXELS
                self.xvar_real = [0, NPIXELS]
                self.xvar_imag = None
                self.evar_real = [NPIXELS, NPIXELS + NPIXELS*NS]
                self.evar_imag = [NPIXELS + NPIXELS*NS, NPIXELS + 2*NPIXELS*NS]
            elif self.good_conductor:
                self.nvar = (1 + 2*NS)*NPIXELS
                self.xvar_real = None
                self.xvar_imag = [0, NPIXELS]
                self.evar_real = [NPIXELS, NPIXELS + NPIXELS*NS]
                self.evar_imag = [NPIXELS + NPIXELS*NS, NPIXELS + 2*NPIXELS*NS]
            else:
                self.nvar = (1 + NS)*2*NPIXELS
                self.xvar_real = [0, NPIXELS]
                self.xvar_imag = [NPIXELS, 2*NPIXELS]
                self.evar_real = [2*NPIXELS, 2*NPIXELS + NPIXELS*NS]
                self.evar_imag = [2*NPIXELS + NPIXELS*NS, 2*NPIXELS*(1 + NS)]
            
            self.lb = np.zeros(self.nvar)
            self.ub = np.zeros(self.nvar)
            
        else:
            raise error.Error('Not ready for other discretizations than '
                              + 'Collocation Method')

        if (type(contrast_bounds) is int or type(contrast_bounds) is float
                or type(contrast_bounds) is complex):

            if self.perfect_dielectric:
                i, j = self.xvar_real[0], self.xvar_real[1]
                self.ub[i:j] = np.real(contrast_bounds)
            elif self.good_conductor:
                i, j = self.xvar_imag[0], self.xvar_imag[1]
                self.ub[i:j] = np.imag(contrast_bounds)
            else:
                i, j = self.xvar_real[0], self.xvar_real[1]
                self.ub[i:j] = np.real(contrast_bounds)
                i, j = self.xvar_imag[0], self.xvar_imag[1]
                self.ub[i:j] = np.imag(contrast_bounds)
            
        elif type(contrast_bounds) is tuple or type(contrast_bounds) is list:
            
            if self.perfect_dielectric:
                i, j = self.xvar_real[0], self.xvar_real[1]
                self.lb[i:j] = np.real(contrast_bounds[0])
                self.ub[i:j] = np.real(contrast_bounds[1])
            elif self.good_conductor:
                i, j = self.xvar_imag[0], self.xvar_imag[1]
                self.lb[i:j] = np.imag(contrast_bounds[0])
                self.ub[i:j] = np.imag(contrast_bounds[1])
            else:
                i, j = self.xvar_real[0], self.xvar_real[1]
                self.lb[i:j] = np.real(contrast_bounds[0])
                self.ub[i:j] = np.real(contrast_bounds[1])
                i, j = self.xvar_imag[0], self.xvar_imag[1]
                self.lb[i:j] = np.imag(contrast_bounds[0])
                self.ub[i:j] = np.imag(contrast_bounds[1])

        if type(total_bounds) is int or type(total_bounds) is float:
            i, j = self.evar_real[0], self.evar_real[1]
            self.lb[i:j] = -total_bounds
            self.ub[i:j] = total_bounds
            i, j = self.evar_imag[0], self.evar_imag[1]
            self.lb[i:j] = -total_bounds
            self.ub[i:j] = total_bounds
        elif type(total_bounds) is complex:
            i, j = self.evar_real[0], self.evar_real[1]
            self.lb[i:j] = -np.real(total_bounds)
            self.ub[i:j] = np.real(total_bounds)
            i, j = self.evar_imag[0], self.evar_imag[1]
            self.lb[i:j] = -np.imag(total_bounds)
            self.ub[i:j] = np.imag(total_bounds)
        elif type(total_bounds) is tuple or type(total_bounds) is list:
            i, j = self.evar_real[0], self.evar_real[1]
            self.lb[i:j] = -np.real(total_bounds[0])
            self.ub[i:j] = np.real(total_bounds[1])
            i, j = self.evar_imag[0], self.evar_imag[1]
            self.lb[i:j] = -np.imag(total_bounds[0])
            self.ub[i:j] = np.imag(total_bounds[1])

    def contrast(self, x, mode='array'):
        y = super().contrast(x)
        if self.perfect_dielectric:
            y = y[self.xvar_real[0]:self.xvar_real[1]]
        elif self.good_conductor:
            y = 1j*y[self.xvar_imag[0]:self.xvar_imag[1]]
        else:
            y = (y[self.xvar_real[0]:self.xvar_real[1]]
                    + 1j*y[self.xvar_imag[0]:self.xvar_imag[1]])
        if mode == 'array':
            return y
        else:
            return self.discretization.contrast_image(y, mode)

    def total_field(self, x, mode='array'):
        y = super().total_field(x)
        y = (y[self.evar_real[0]:self.evar_real[1]]
             + 1j*y[self.evar_imag[0]:self.evar_imag[1]])
        if mode == 'array':
            return y
        else:
            return self.discretization.total_image(y, mode)

    def scattered_field(self, x):
        contrast = self.contrast(x)
        total_field = self.total_field(x, mode=self.discretization.elements)
        current = self.current(x, mode=self.discretization.elements)
        return self.discretization.scattered_field(contrast=contrast, total_field=total_field,
                                         current=current)
            
    def current(self, x, mode='array'):
        return super().current(x, mode=mode)

    def __str__(self):
        message = super().__str__()
        message += 'DiscretizationElementBased\n'
        message += 'Discretization Method: ' + str(self.discretization) + '\n'
        message += 'Number of variables: %d\n' % self.nvar
        if self.xvar_real is not None:
            message += ('Range for contrast real variables: [%.2f, '
                        % self.lb[self.xvar_real[0]] + '%.2f]\n'
                        % self.ub[self.xvar_real[0]])
        if self.xvar_imag is not None:
            message += ('Range for contrast imaginary variables: [%.2f, '
                        % self.lb[self.xvar_imag[0]] + '%.2f]\n'
                        % self.ub[self.xvar_imag[0]])
        message += ('Range for total field real variables: [%.2f, '
                        % self.lb[self.evar_real[0]] + '%.2f]\n'
                        % self.ub[self.evar_real[0]])
        message += ('Range for total field imaginary variables: [%.2f, '
                        % self.lb[self.evar_imag[0]] + '%.2f]'
                        % self.ub[self.evar_imag[0]])
        return message