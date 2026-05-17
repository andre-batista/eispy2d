import numpy as np
from numpy.fft import fft2, ifft2
from numpy import pi
from scipy.special import jv
from scipy.special import hankel2 as hv2


class FFTProduct:
    @property
    def discretization(self):
        return self._discretization

    @discretization.setter
    def discretization(self, new):
        if new is None:
            self._discretization = None
            self.G = None
        else:
            self._discretization = new
            GD = green(new)
            if self.adjoint:
                GD = GD.conj().T
            NY, NX = new.elements
            G = np.zeros((2*NY-1, 2*NX-1), dtype=complex)
            G[:NY, :NX] = GD[NY-1:2*NY-1, NX-1:2*NX-1]
            G[NY:2*NY-1, NX:2*NX-1] = GD[:NY-1, :NX-1]
            G[NY:2*NY-1, :NX] = GD[:NY-1, NX-1:2*NX-1]
            G[:NY, NX:2*NX-1] = GD[NY-1:2*NY-1, :NX-1]
            if self.conjugate:
                G = G.conj()
            self.G = G
    
    def __init__(self, discretization=None, adjoint=False, conjugate=False):
        self.adjoint = adjoint
        self.conjugate = conjugate
        self.discretization = discretization

    def compute(self, J):
        NY, NX = self.discretization.elements
        NS = self.discretization.configuration.NS
        J = J.reshape((NY, NX, NS))
        G = np.tile(self.G[:, :, np.newaxis], (1, 1, NS))
        Es = ifft2(fft2(G, axes=(0, 1))
                   * fft2(J, axes=(0, 1), s=(2*NY-1, 2*NX-1)),
                   axes=(0, 1))
        Es = Es[:NY, :NX, :].reshape((NX*NY, NS))
        return Es


def green(discretization):
    Nx, Ny = discretization.elements 
    kb = discretization.configuration.kb
    dx = discretization.configuration.Lx/Nx
    dy = discretization.configuration.Ly/Ny

    an = np.sqrt(dx*dy/pi)
    X_dif, Y_dif = np.meshgrid(np.arange(1-Nx, Nx)*dx,
                            np.arange(1-Ny, Ny)*dy)
    R = np.sqrt(X_dif**2 + Y_dif**2)
    G = -1j*pi*kb*an/2*jv(1, kb*an)*hv2(0, kb*R)
    np.fill_diagonal(G, -(1j/2)*(pi*kb*an*hv2(1, kb*an)-2j))

    return G