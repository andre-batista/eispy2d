import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from numpy import pi
from scipy.special import jv
from scipy.special import hankel2 as h2v
from scipy.sparse import spdiags, eye
from scipy.interpolate import NearestNDInterpolator


import configuration as cfg
import regularization as reg
import collocation as clc
import error


class Richmond(clc.Collocation):
    def __init__(self, configuration, elements, state=True):
        super().__init__(configuration, 'pulse', elements)
        self.GS = richmond_data(configuration, self.elements)
        if state:
            self.GD = richmond_state(configuration, self.elements)
        else:
            self.GD = None
        self.name = ('Richmond Method (%dx' % self.elements[0] + '%d)'
                     % self.elements[1])
    def residual_data(self, scattered_field, contrast=None, total_field=None,
                      current=None):
        Es, X, E, J = scattered_field, contrast, total_field, current
        super().residual_data(Es, contrast=X, total_field=E, current=J)
        if current is None:
            if X.ndim == 1:
                CHI = spdiags(X, 0, X.size, X.size)
                res = Es - self.GS@CHI@E
            elif X.ndim == 2 and np.prod(X.shape) == self.GS.shape[1]:
                CHI = spdiags(X.flatten(), 0, X.size, X.size)
                res = Es - self.GS@CHI@E
            elif X.ndim == 2 and np.prod(X.shape) == self.GS.shape[1]**2:
                res = Es - self.GS@X@E
        else:
            res = Es - self.GS@J
        return res
    def residual_state(self, incident_field, contrast=None, total_field=None,
                       current=None):
        Ei, X, E, J = incident_field, contrast, total_field, current
        super().residual_state(Ei, contrast=X, total_field=E, current=J)
        if self.GD is None:
            raise error.MissingAttributesError('Richmond', 'GD')
        if current is None:
            if X.ndim == 1:
                CHI = spdiags(X, 0, X.size, X.size)
                res = E - Ei - self.GD@CHI@E
            elif X.ndim == 2 and np.prod(X.shape) == self.GD.shape[0]:
                CHI = spdiags(X.flatten(), 0, X.size, X.size)
                res = E - Ei - self.GD@CHI@E
            elif X.ndim == 2 and np.prod(X.shape) == np.prod(self.GD.shape):
                res = E - Ei - self.GD@X@E
        else:
            if X.ndim == 1:
                CHI = spdiags(X, 0, X.size, X.size)
                res = J - CHI@Ei - CHI@self.GD@J
            elif X.ndim == 2 and np.prod(X.shape) == self.GD.shape[0]:
                CHI = spdiags(X.flatten(), 0, X.size, X.size)
                res = J - CHI@Ei - CHI@self.GD@J
            elif X.ndim == 2 and np.prod(X.shape) == np.prod(self.GD.shape):
                res = J - X@Ei - X@self.GD@J
        return res
    def solve(self, scattered_field=None, incident_field=None, contrast=None, total_field=None,
              current=None, linear_solver=None):
        super().solve(scattered_field=scattered_field, incident_field=incident_field,
                      contrast=contrast, total_field=total_field, current=current)
        Es, Ei, X, E, J = scattered_field, incident_field, contrast, total_field, current
        if scattered_field is not None:
            if (linear_solver is None
                    or not isinstance(linear_solver, reg.Regularization)):
                raise error.WrongTypeInput('Richmond.solve', 'linear_solver',
                                           'Regularization',
                                           str(type(linear_solver)))
            # solve for X
            if total_field is not None and contrast is None and current is None:
                K = clc.kernel_GSE(self.GS, E)
                y = Es.flatten()
                X = linear_solver.solve(K, y)
                return X
            # solve for E, LEMBRETE: SO EH RESOLVIDO PARA DENTRO DO OBJETO
            elif total_field is None and contrast is not None and current is None:
                K = clc.kernel_GSX(self.GS, X)
                y = Es
                E = np.zeros((self.GS.shape[1], self.configuration.NS),
                             dtype=complex)
                for s in range(self.configuration.NS):
                    E[:, s] = linear_solver.solve(K, y[:, s].flatten())
                return E
            # solve for J
            elif total_field is None and contrast is None and current is None:
                K = np.copy(self.GS)
                y = Es
                J = np.zeros((self.GS.shape[1], self.configuration.NS),
                             dtype=complex)
                for s in range(self.configuration.NS):
                    J[:, s] = linear_solver.solve(K, y[:, s].flatten())
                return J
        elif incident_field is not None:
            if self.GD is None:
                raise error.MissingAttributesError('Richmond', 'GD')
            elif (type(total_field) is bool and total_field == True
                    and contrast is not None and current is None):
                if linear_solver is not None:
                    K = clc.kernel_GDX(self.GD, X)
                    y = Ei
                    E = np.zeros((self.GD.shape[1], self.configuration.NS),
                                 dtype=complex)
                    for s in range(self.configuration.NS):
                        E[:, s] = linear_solver.solve(K, y[:, s].flatten())
                else:
                    if X.ndim == 1:
                        CHI = spdiags(X, 0, X.size, X.size)
                    elif X.ndim == 2 and np.prod(X.shape) == self.GD.shape[0]:
                        CHI = spdiags(X.flatten(), 0, X.size, X.size)
                    elif (X.ndim == 2
                            and np.prod(X.shape) == np.prod(self.GD.shape)):
                        CHI = X
                    E = Ei + self.GD@CHI@Ei
                return E
            elif (total_field is None and contrast is not None
                    and type(current) is bool and current == True):
                if (linear_solver is None
                        or not isinstance(linear_solver, reg.Regularization)):
                    raise error.WrongTypeInput('Richmond.solve', 'linear_solver',
                                           'Regularization',
                                           str(type(linear_solver)))
                if X.ndim == 1:
                    CHI = spdiags(X, 0, X.size, X.size)
                    K = eye(self.GD.shape[0]) - CHI@self.GD
                    y = clc.lhs_XEi(X, Ei)
                elif X.ndim == 2 and np.prod(X.shape) == self.GD.shape[0]:
                    CHI = spdiags(X.flatten(), 0, X.size, X.size)
                    K = eye(self.GD.shape[0]) - CHI@self.GD
                    y = clc.lhs_XEi(X.flatten(), Ei)
                elif (X.ndim == 2
                        and np.prod(X.shape) == np.prod(self.GD.shape)):
                    K = eye(self.GD.shape[0]) - X@self.GD
                    y = clc.lhs_XEi(np.diagonal(X), Ei)
                J = np.zeros((self.GD.shape[1], self.configuration.NS),
                             dtype=complex)
                for s in range(self.configuration.NS):
                    J[:, s] = linear_solver.solve(K, y[:, s].flatten())
                return J
            elif total_field is not None and contrast == True and current is None:
                K = clc.kernel_GDE(self.GD, E)
                y = E-Ei
                X = np.zeros((self.GD.shape[1], self.configuration.NS),
                             dtype=complex)
                for s in range(self.configuration.NS):
                    X[:, s] = linear_solver.solve(np.squeeze(K[:, :, s]),
                                                  y[:, s].flatten())
                return X
            elif (total_field is None
                    # and (type(contrast) is bool and contrast == True)
                    and contrast == True
                    and current is not None):
                X = np.zeros((self.GD.shape[1], self.configuration.NS),
                             dtype=complex)
                for s in range(self.configuration.NS):
                    X[:, s] = J[:, s]/(Ei[:, s] + self.GD@J[:, s])
                return X
    def scattered_field(self, contrast=None, total_field=None, current=None):
        super().scattered_field(contrast=contrast, total_field=total_field,
                                current=current)
        if current is None:
            if contrast.ndim == 1:
                X = spdiags(contrast, 0, contrast.size, contrast.size)
            elif (contrast.ndim == 2
                    and np.prod(contrast.shape) == self.GS.shape[1]):
                X = spdiags(contrast.flatten(), 0, contrast.size,
                            contrast.size)
            elif (contrast.ndim == 2
                    and np.prod(contrast.shape) == self.GS.shape[1]**2):
                X = contrast
            Es = self.GS@X@total_field
        else:
            Es = self.GS@current
        return Es
    def contrast_image(self, coefficients, resolution):
        super().contrast_image(coefficients, resolution)
        if (self.elements[0] == resolution[0]
                and self.elements[1] == resolution[1]):
            if coefficients.ndim == 1:
                return coefficients.reshape(resolution)
            elif (coefficients.ndim == 2
                    and np.prod(coefficients.shape) == np.prod(resolution)):
                return np.copy(coefficients)
            elif (coefficients.ndim == 2
                    and np.prod(coefficients.shape) == np.prod(resolution)**2):
                return np.diagonal(coefficients).reshape(resolution)
        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=self.elements)
        xp, yp = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                             resolution=resolution)

        if coefficients.ndim == 1:
            f = NearestNDInterpolator((x.flatten(), y.flatten()), coefficients)
        elif (coefficients.ndim == 2
                and np.prod(coefficients.shape) == np.prod(self.elements)):
            f = NearestNDInterpolator((x.flatten(), y.flatten()),
                                      coefficients.flatten())
        elif (coefficients.ndim == 2
                    and np.prod(coefficients.shape) == np.prod(self.elements)**2):
            f = NearestNDInterpolator((x.flatten(), y.flatten()),
                                      np.diagonal(coefficients).flatten())
        return f(xp, yp)        
    def total_image(self, coefficients, resolution):
        super().total_image(coefficients, resolution)
        if (self.elements[0] == resolution[0]
                and self.elements[1] == resolution[1]):
            if coefficients.ndim == 1:
                return coefficients.reshape((np.prod(resolution),
                                             self.configuration.NS))
            elif (coefficients.ndim == 2
                    and np.prod(coefficients.shape) == np.prod(resolution)):
                return np.copy(coefficients)

        if coefficients.ndim == 1:
            E0 = coefficients.reshape((np.prod(self.elements),
                                       self.configuration.NS))
        else:
            E0 = coefficients

        x, y = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                           resolution=self.elements)
        xp, yp = cfg.get_coordinates_ddomain(configuration=self.configuration,
                                             resolution=resolution)
        E = np.zeros((np.prod(resolution), self.configuration.NS),
                     dtype=complex)
        for s in range(self.configuration.NS):
            f = NearestNDInterpolator((x.flatten(), y.flatten()), E0[:, s])
            E[:, s] = np.reshape(f(xp, yp), (-1))
        return E
    def copy(self, new=None):
        if new is None:
            new = Richmond(self.configuration, self.elements, state=False)
            if self.GD is not None:
                new.GD = np.copy(self.GD)
            return new
        else:
            super().copy(new)
            if new.GS is None:
                self.GS = None
            else:
                self.GS = np.copy(new.GS)
            if new.GD is None:
                self.GD = None
            else:
                self.GD = np.copy(new.GD)
    def __str__(self):
        return self.name


def richmond_data(configuration, elements):
    xm, ym = cfg.get_coordinates_sdomain(configuration.Ro, configuration.NM)
    x, y = cfg.get_coordinates_ddomain(configuration, elements)
    kb = configuration.kb
    dx, dy = x[0, 1]-x[0, 0], y[1, 0]-y[0, 0]
    a = np.sqrt(dx*dy/pi)
    R = cdist(np.transpose(np.vstack((xm, ym))),
              np.transpose(np.vstack((x.flatten(), y.flatten()))))
    return -1j*pi*kb*a/2*jv(1, kb*a)*h2v(0, kb*R)


def richmond_state(configuration, elements):
    x, y = cfg.get_coordinates_ddomain(configuration, elements)
    kb = configuration.kb
    a = np.sqrt((x[0, 1]-x[0, 0])*(y[1, 0]-y[0, 0])/pi)
    R = squareform(pdist(np.transpose(np.vstack((x.flatten(), y.flatten())))))
    G = -1j*pi*kb*a/2*jv(1, kb*a)*h2v(0, kb*R)
    np.fill_diagonal(G, -(1j/2)*(pi*kb*a*h2v(0, kb*a)-2j))
    return G
