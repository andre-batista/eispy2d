import error
import pickle
import numpy as np
import discretization as dct
from numba import jit

TRIAL_FUNCTION = 'trial'
ELEMENTS = 'elements'

class Collocation(dct.Discretization):
    def __init__(self, configuration=None, trial=None, elements=None,
                 name=None, alias='clc', import_filename=None,
                 import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(configuration=configuration, name=name,
                             alias=alias)
            self.trial = trial
            if elements is None:
                raise error.MissingInputError('Collocation.__init__',
                                              'elements')
            elif type(elements) is int:
                self.elements = (elements, elements)
            else:
                self.elements = (elements[0], elements[1])
            self.name = ('Collocation Method (%dx' % self.elements[0] + '%d), '
                         % self.elements[1] + 'trial function: ' + self.trial)
    def copy(self, new=None):
        if new is None:
            return Collocation(self.configuration, self.trial, self.elements,
                               self.name)
        else:
            super().copy(new)
            self.trial = new.trial
            self.elements = new.elements
    def __str__(self):
        message = super().__str__()
        message += 'Discretization:' + self.name + '\n'
        message += 'Alias: ' + self.alias + '\n'
        return message
    def save(self, file_path=''):
        data = super().save(file_path=file_path)
        data[TRIAL_FUNCTION] = self.trial
        data[ELEMENTS] = self.elements
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)
    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.trial = data[TRIAL_FUNCTION]
        self.elements = data[ELEMENTS]

def kernel_GSE(GS, E):
    N, NS = E.shape
    NM = GS.shape[0]
    return _kernel_GSE(GS, E, NM, NS, N)


@jit(nopython=True)
def _kernel_GSE(GS, E, NM, NS, N):
    K = 1j*np.ones((NM*NS, N))
    row = 0
    for m in range(NM):
        for s in range(NS):
            K[row, :] = GS[m, :].flatten()*E[:, s].flatten()
            row += 1
    return K


def kernel_GSX(GS, X):
    NM, N = GS.shape
    if X.ndim == 1:
        return _kernel_GSX(GS, X, NM, N)
    elif X.ndim == 2 and np.prod(X.shape) == N:
        return _kernel_GSX(GS, X.flatten(), NM, N)
    elif X.ndim == 2 and X.shape[0] == N:
        return _kernel_GSX(GS, np.diagonal(X), NM, N)


@jit(nopython=True)
def _kernel_GSX(GS, X, NM, N):
    K = 1j*np.ones((NM, N))
    for m in range(NM):
        K[m, :] = GS[m, :].flatten()*X
    return K


def kernel_GDX(GD, X):
    N = GD.shape[0]
    if X.ndim == 1:
        return _kernel_GDX(GD, X, N)
    elif X.ndim == 2 and np.prod(X.shape) == N:
        return _kernel_GDX(GD, X.flatten(), N)
    elif X.ndim == 2 and X.shape[0] == N:
        return _kernel_GDX(GD, np.diagonal(X), N)


@jit(nopython=True)
def _kernel_GDX(GD, X, N):
    K = 1j*np.ones((N, N))
    for n in range(N):
        K[n, :] = - GD[n, :].flatten()*X
        K[n, n] += 1
    return K


def kernel_GDE(GD, E):
    N, NS = E.shape
    return _kernel_GDE(GD, E, N, NS)


@jit(nopython=True)
def _kernel_GDE(GD, E, N, NS):
    K = 1j*np.ones((N, N, NS))
    row = 0
    for s in range(NS):
        for n in range(N):
            K[n, :, s] = GD[n, :].flatten()*E[:, s].flatten()
            row += 1
    return K


def lhs_XEi(X, Ei):
    N, NS = Ei.shape
    return _lhs_XEi(X, Ei, N, NS)


@jit(nopython=True)
def _lhs_XEi(X, Ei, N, NS):
    lhs = 1j*np.ones((N, NS))
    for s in range(NS):
        lhs[:, s] = X*Ei[:, s]
    return lhs
