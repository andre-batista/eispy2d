r"""Method of Moments - Conjugate-Gradient FFT Method.

This module provides the implementation of Method of Moments (MoM) with
the Conjugated-Gradient FFT formulation. It solves the forward problem
following the Forward Solver abstract class.

References
----------
.. [1] P. Zwamborn and P. M. van den Berg, "The three dimensional weak
   form of the conjugate gradient FFT method for solving scattering
   problems," in IEEE Transactions on Microwave Theory and Techniques,
   vol. 40, no. 9, pp. 1757-1766, Sept. 1992, doi: 10.1109/22.156602.

.. [2] Chen, Xudong. "Computational methods for electromagnetic inverse
   scattering". John Wiley & Sons, 2018.
"""

import time
import numpy as np
from numpy import linalg as lag
from numpy import fft
from scipy import constants as ct
from scipy import sparse as spr
import scipy.special as spc
from joblib import Parallel, delayed
import multiprocessing
import forward as fwr
import configuration as cfg

# Predefined constants
MEMORY_LIMIT = 16e9  # [GB]


class MoM_CG_FFT(fwr.ForwardSolver):
    """Method of Moments - Conjugated-Gradient FFT Method.

    This class implements the Method of Moments following the
    Conjugated-Gradient FFT formulation.

    Attributes
    ----------
        MAX_IT : int
            Maximum number of iterations.
        TOL : float
            Tolerance level of error.
    """

    def __init__(self, tolerance=1e-3, maximum_iterations=5000):
        """Create the object.

        Parameters
        ----------
            configuration : string or :class:`Configuration`:Configuration
                Either a configuration object or a string with the name
                of file in which the configuration is saved. In this
                case, the file path may also be provided.

            configuration_filepath : string, optional
                A string with the path to the configuration file (when
                the file name is provided).

            tolerance : float, default: 1e-6
                Minimum error tolerance.

            maximum_iteration : int, default: 10000
                Maximum number of iterations.
        """
        super().__init__()
        self.TOL = tolerance
        self.MAX_IT = maximum_iterations
        self.name = 'Method of Moments - CG-FFT'

    def incident_field(self, resolution, configuration):
        """Compute the incident field matrix.

        Given the configuration information stored in the object, it
        computes the incident field matrix considering plane waves in
        different from different angles.

        Parameters
        ----------
            resolution : 2-tuple
                The image size of D-domain in pixels (y and x).

        Returns
        -------
            ei : :class:`numpy.ndarray`
                Incident field matrix. The rows correspond to the points
                in the image following `C`-order and the columns
                corresponds to the sources.
        """
        NY, NX = resolution
        phi = cfg.get_angles(configuration.NS)
        x, y = cfg.get_coordinates_ddomain(configuration=configuration,
                                           resolution=resolution)
        kb = configuration.kb
        E0 = configuration.E0
        if isinstance(kb, float) or isinstance(kb, complex):
            ei = E0*np.exp(-1j*kb*(x.reshape((-1, 1))
                                   @ np.cos(phi.reshape((1, -1)))
                                   + y.reshape((-1, 1))
                                   @ np.sin(phi.reshape((1, -1)))))
        else:
            ei = np.zeros((NX*NY, configuration.NS, kb.size),
                          dtype=complex)
            for f in range(kb.size):
                ei[:, :, f] = E0*np.exp(-1j*kb[f]*(x.reshape((-1, 1))
                                                   @ np.cos(phi.reshape((1,
                                                                         -1)))
                                                   + y.reshape((-1, 1))
                                                   @ np.sin(phi.reshape((1,
                                                                         -1))))
                                        )
        return ei

    def solve(self, inputdata, noise=None, PRINT_INFO=False,
              COMPUTE_SCATTERED_FIELD=True, SAVE_INTERN_FIELD=True):
        """Solve the forward problem.

        Parameters
        ----------
            inputdata : :class:`inputdata:InputData`
                An object describing the dielectric property map.

            PRINT_INFO : boolean, default: False
                Print iteration information.

            COMPUTE_INTERN_FIELD : boolean, default: True
                Compute the total field in D-domain.

        Return
        ------
            es, et, ei : :class:`numpy:ndarray`
                Matrices with the scattered, total and incident field
                information.

        Examples
        --------
        >>> solver = MoM_CG_FFT(configuration)
        >>> es, et, ei = solver.solve(inputdata)
        >>> es, ei = solver.solve(inputdata, COMPUTE_INTERN_FIELD=False)
        """
        epsilon_r, sigma = super().solve(inputdata)
        # Quick access for configuration variables
        NM = inputdata.configuration.NM
        NS = inputdata.configuration.NS
        Ro = inputdata.configuration.Ro
        epsilon_rb = inputdata.configuration.epsilon_rb
        sigma_b = inputdata.configuration.sigma_b
        f = inputdata.configuration.f
        omega = 2*np.pi*f
        kb = inputdata.configuration.kb
        Lx, Ly = inputdata.configuration.Lx, inputdata.configuration.Ly
        NY, NX = epsilon_r.shape
        N = NX*NY
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)
        dx, dy = Lx/NX, Ly/NY
        xm, ym = cfg.get_coordinates_sdomain(Ro, NM)
        x, y = cfg.get_coordinates_ddomain(dx=dx, dy=dy, xmin=xmin, xmax=xmax,
                                           ymin=ymin, ymax=ymax)
        ei = self.incident_field((NY, NX), inputdata.configuration)
        inputdata.ei = np.copy(ei)
        ei = np.conj(ei)

        if isinstance(f, float):
            MONO_FREQUENCY = True
        else:
            MONO_FREQUENCY = False
            NF = f.size

        deltasn = dx*dy  # area of the cell
        an = np.sqrt(deltasn/np.pi)  # radius of the equivalent circle
        Xr = get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b, omega)

        # Using circular convolution [extended domain (2N-1)x(2N-1)]
        [xe, ye] = np.meshgrid(np.arange(xmin-(NX/2-1)*dx, xmax+NY/2*dx, dx),
                               np.arange(ymin-(NY/2-1)*dy, ymax+NY/2*dy, dy))
        Rmn = np.sqrt(xe**2 + ye**2)  # distance between the cells
        G = self.__get_extended_matrix(Rmn, kb, an, NX, NY)
        b = np.copy(ei)

        if MONO_FREQUENCY:
            tic = time.time()
            et = np.zeros((N, NS), dtype=complex)
            niter = np.zeros(NS)
            error = np.zeros((self.MAX_IT, NS))
            num_cores = multiprocessing.cpu_count()

            results = (Parallel(n_jobs=num_cores)(delayed(self.CG_FFT)
                                                  (G,
                                                   b[:, ns].reshape((-1, 1)),
                                                   NX, NY, 1,
                                                   Xr,
                                                   self.MAX_IT, self.TOL,
                                                   False)
                                                  for ns in range(NS)))

            for ns in range(NS):
                et[:, ns] = results[ns][0].flatten()
                niter[ns] = results[ns][1]
                error[:, ns] = results[ns][2].flatten()

            time_cg_fft = time.time()-tic
            if PRINT_INFO:
                print('Execution time: %.2f' % time_cg_fft + ' [sec]')

        else:
            et = np.zeros((N, NS, NF), dtype=complex)
            niter = np.zeros(NF)
            error = np.zeros((self.MAX_IT, NF))
            num_cores = multiprocessing.cpu_count()

            results = (Parallel(n_jobs=num_cores)(delayed(self.CG_FFT)
                                                  (np.squeeze(G[:, :, nf]),
                                                   np.squeeze(b[:, :, nf]),
                                                   NX, NY, NS,
                                                   np.squeeze(Xr[:, :, nf]),
                                                   self.MAX_IT, self.TOL,
                                                   False)
                                                  for nf in range(NF)))

            for nf in range(NF):
                et[:, :, nf] = results[nf][0]
                niter[nf] = results[nf][1]
                error[:, nf] = results[nf][2]
                print('Frequency: %.3f ' % (f[nf]/1e9) + '[GHz] - '
                      + 'Number of iterations: %d - ' % (niter[nf]+1)
                      + 'Error: %.3e' % error[int(niter[nf]), nf])

        if SAVE_INTERN_FIELD:
            inputdata.total_field = np.conj(et)

        if COMPUTE_SCATTERED_FIELD:
            GS = get_greenfunction(xm, ym, x, y, kb)

            if MONO_FREQUENCY:
                es = GS @ spr.dia_matrix((Xr.flatten(), 0), shape=(N, N)) @ et

            else:
                es = np.zeros((NM, NS, NF), dtype=complex)
                for nf in range(NF):
                    aux = spr.dia_matrix((Xr[:, :, nf].flatten(), 0),
                                         shape=(N, N))
                    es[:, :, nf] = GS[:, :, nf] @ aux @ et[:, :, nf]

            if noise is not None and noise > 0:
                es = fwr.add_noise(es, noise)
            elif inputdata.noise is not None and inputdata.noise > 0:
                es = fwr.add_noise(es, inputdata.noise)
            inputdata.scattered_field = np.conj(es)

            return np.conj(et), np.conj(ei), np.conj(es)

        else:
            return np.conj(et), np.conj(ei)

    def __get_extended_matrix(self, Rmn, kb, an, NX, NY):
        """Return the extended matrix of Method of Moments.

        Parameters
        ----------
            Rmn : :class:`numpy:ndarray`
                Radius matrix.

            kb : float or :class:`numpy:ndarray`
                Wavenumber [1/m]

            an : float
                Radius of equivalent element radius circle.

            Nx, Ny : int
                Number of cells in each direction.

        Returns
        -------
            G : :class:`numpy:ndarray`
                The extent matrix.
        """
        if isinstance(kb, float) or isinstance(kb, complex):

            # Matrix elements for off-diagonal entries (m=/n)
            Gmn = 1j*np.pi*kb*an/2*spc.jv(1, kb*an)*spc.hankel1(0, kb*Rmn)
            # Matrix elements for diagonal entries (m==n)
            Gmn[NY-1, NX-1] = 1j*np.pi*kb*an/2*spc.hankel1(1, kb*an) - 1

            # Extended matrix (2N-1)x(2N-1)
            G = np.zeros((2*NY-1, 2*NX-1), dtype=complex)
            G[:NY, :NX] = Gmn[NY-1:2*NY-1, NX-1:2*NX-1]
            G[NY:2*NY-1, NX:2*NX-1] = Gmn[:NY-1, :NX-1]
            G[NY:2*NY-1, :NX] = Gmn[:NY-1, NX-1:2*NX-1]
            G[:NY, NX:2*NX-1] = Gmn[NY-1:2*NY-1, :NX-1]

        else:

            G = np.zeros((2*NY-1, 2*NX-1, kb.size), dtype=complex)

            for f in range(kb.size):

                # Matrix elements for off-diagonal entries (m=/n)
                Gmn = (1j*np.pi*kb*an/2*spc.jv(1, kb[f]*an)
                       * spc.hankel1(0, kb[f]*Rmn))
                # Matrix elements for diagonal entries (m==n)
                Gmn[NY-1, NX-1] = (1j*np.pi*kb[f]*an/2*spc.hankel1(1, kb[f]*an)
                                   - 1)

                G[:NY, :NX, f] = Gmn[NY-1:2*NY-1, NX-1:2*NX-1]
                G[NY:2*NY-1, NX:2*NX-1, f] = Gmn[:NY-1, :NX-1]
                G[NY:2*NY-1, :NX, f] = Gmn[:NY-1, NX-1:2*NX-1]
                G[:NY, NX:2*NX-1, f] = Gmn[NY-1:2*NY-1, :NX-1]

        return G

    def CG_FFT(self, G, b, NX, NY, NS, Xr, MAX_IT, TOL, PRINT_CONVERGENCE):
        """Apply the Conjugated-Gradient Method to the forward problem.

        Parameters
        ----------
            G : :class:`numpy.ndarray`
                Extended matrix, (2NX-1)x(2NY-1)

            b : :class:`numpy.ndarray`
                Excitation source, (NX.NY)xNi

            NX : int
                Contrast map in x-axis.

            NY : int
                Contrast map in x-axis.

            NS : int
                Number of incidences.

            Xr : :class:`numpy.ndarray`
                Contrast map, NX x NY

            MAX_IT : int
                Maximum number of iterations

            TOL : float
                Error tolerance

            PRINT_CONVERGENCE : boolean
                Print error information per iteration.

        Returns
        -------
            J : :class:`numpy:ndarray`
                Current density, (NX.NY)xNS
        """
        Eo = np.zeros((NX*NY, NS), dtype=complex)  # initial guess
        ro = self.__fft_A(Eo, G, NX, NY, NS, Xr)-b  # ro = A.Jo - b;
        go = self.__fft_AH(ro, G, NX, NY, NS, Xr)  # Complex conjugate AH
        po = -go
        error_res = np.zeros(MAX_IT)

        for n in range(MAX_IT):

            alpha = -1*(np.sum(np.conj(self.__fft_A(po, G, NX, NY, NS, Xr))
                               * (self.__fft_A(Eo, G, NX, NY, NS, Xr)-b),
                               axis=0)
                        / lag.norm(np.reshape(self.__fft_A(po, G, NX, NY, NS,
                                                           Xr), (NX*NY*NS, 1),
                                              order='F'), ord='fro')**2)

            E = Eo + np.tile(alpha, (NX*NY, 1))*po
            r = self.__fft_A(E, G, NX, NY, NS, Xr)-b
            g = self.__fft_AH(r, G, NX, NY, NS, Xr)

            error = lag.norm(r)/lag.norm(b)  # error tolerance
            error_res[n] = error

            if PRINT_CONVERGENCE:
                print('Iteration %d ' % (n+1) + ' - Error: %.3e' % error)

            if error < TOL:  # stopping criteria
                break

            beta = np.sum(np.conj(g)*(g-go), axis=0)/np.sum(np.abs(go)**2,
                                                            axis=0)
            p = -g + np.tile(beta, (NX*NY, 1))*po

            po = np.copy(p)
            Eo = np.copy(E)
            go = np.copy(g)

        return E, n, error_res

    def __fft_A(self, E, G, NX, NY, NS, Xr):
        """Compute Matrix-vector product by using two-dimensional FFT."""
        u = np.tile(Xr.reshape((-1, 1)), (1, NS))*E
        u = np.reshape(u, (NY, NX, NS))
        H = np.tile(G[:, :, np.newaxis], (1, 1, NS))
        e = fft.ifft2(fft.fft2(H, axes=(0, 1))
                      * fft.fft2(u, axes=(0, 1), s=(2*NY-1, 2*NX-1)),
                      axes=(0, 1))
        e = e[:NY, :NX, :]
        e = np.reshape(e, (NX*NY, NS))
        return E - e

    def __fft_AH(self, E, G, NX, NY, NS, Xr):
        """Summarize the method."""
        u = np.reshape(E, (NY, NX, NS))
        H = np.tile(G[:, :, np.newaxis], (1, 1, NS))
        e = fft.ifft2(fft.fft2(np.conj(H), axes=(0, 1))
                      * fft.fft2(u, axes=(0, 1), s=(2*NY-1, 2*NX-1)),
                      axes=(0, 1))
        e = e[:NY, :NX, :]
        e = np.reshape(e, (NX*NY, NS))
        return E - np.conj(np.tile(Xr.reshape((-1, 1)), (1, NS)))*e

    def __str__(self):
        """Print method parametrization."""
        message = super().__str__()
        message = message + "Number of iterations: %d, " % self.MAX_IT
        message = message + "Tolerance level: %.3e" % self.TOL
        return message


def get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b, omega):
    """Compute the contrast function for a given image.

    Parameters
    ----------
        epsilon_r : `:class:numpy.ndarray`
            A matrix with the relative permittivity map.

        sigma : `:class:numpy.ndarray`
            A matrix with the conductivity map [S/m].

        epsilon_rb : float
            Background relative permittivity of the medium.

        sigma_b : float
            Background conductivity of the medium [S/m].

        omega : float or array
            Angular frequency of operation [Hz].
    """
    if isinstance(omega, float):
        return ((epsilon_r - 1j*sigma/omega/ct.epsilon_0)
                / (epsilon_rb - 1j*sigma_b/omega/ct.epsilon_0) - 1)
    else:
        Xr = np.zeros((epsilon_r.shape[0], epsilon_r.shape[1], omega.size),
                      dtype=complex)
        for i in range(omega.size):
            Xr[:, :, i] = ((epsilon_r - 1j*sigma/omega[i]/ct.epsilon_0)
                           / (epsilon_rb - 1j*sigma_b/omega[i]/ct.epsilon_0)-1)
        return Xr


def get_greenfunction(xm, ym, x, y, kb):
    r"""Compute the Green function matrix for pulse basis discre.

    The routine computes the Green function based on a discretization of
    the integral equation using pulse basis functions [1]_.

    Parameters
    ----------
        xm : `numpy.ndarray`
            A 1-d array with the x-coordinates of measumerent points in
            the S-domain [m].

        ym : `numpy.ndarray`
            A 1-d array with the y-coordinates of measumerent points in
            the S-domain [m].

        x : `numpy.ndarray`
            A meshgrid matrix of x-coordinates in the D-domain [m].

        y : `numpy.ndarray`
            A meshgrid matrix of y-coordinates in the D-domain [m].

        kb : float or complex
            Wavenumber of background medium [1/m].

    Returns
    -------
        G : `numpy.ndarray`, complex
            A matrix with the evaluation of Green function at D-domain
            for each measument point, considering pulse basis
            discretization. The shape of the matrix is NM x (Nx.Ny),
            where NM is the number of measurements (size of xm, ym) and
            Nx and Ny are the number of points in each axis of the
            discretized D-domain (shape of x, y).

    References
    ----------
    .. [1] Pastorino, Matteo. Microwave imaging. Vol. 208. John Wiley
       & Sons, 2010.
    """
    Ny, Nx = x.shape
    M = xm.size
    dx, dy = x[0, 1]-x[0, 0], y[1, 0]-y[0, 0]
    an = np.sqrt(dx*dy/np.pi)  # radius of the equivalent circle

    xg = np.tile(xm.reshape((-1, 1)), (1, Nx*Ny))
    yg = np.tile(ym.reshape((-1, 1)), (1, Nx*Ny))
    R = np.sqrt((xg-np.tile(np.reshape(x, (Nx*Ny, 1)).T, (M, 1)))**2
                + (yg-np.tile(np.reshape(y, (Nx*Ny, 1)).T, (M, 1)))**2)

    G = 1j*kb*np.pi*an/2*spc.jv(1, kb*an)*spc.hankel1(0, kb*R)

    return G
