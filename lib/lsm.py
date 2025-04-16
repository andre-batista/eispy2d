import sys
import pickle
import time as tm
import numpy as np
from numpy import pi
from scipy.special import hankel2
from scipy.linalg import svd, norm
from numba import jit
import configuration as cfg
import result as rst
import deterministic as dtm


REGULARIZATION = 'regularization'
TIKHONOV = 'tikhonov'
SV_CUTOFF = 'sv_cutoff'
THRESHOLD = 'threshold'
FAR_FIELD = 'far_field'
INDICATOR = 'indicator'


def standard(x):
    return -np.log10(x)


class LinearSamplingMethod(dtm.Deterministic):
    def __init__(self, alias='', regularization=None, tikhonov=None,
                 sv_cutoff=None, threshold=None, far_field=None,
                 indicator_function=standard, import_filename=None,
                 import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Linear Sampling Method'
            self.regularization = regularization
            self.tikhonov = tikhonov
            self.sv_cutoff = sv_cutoff
            self.threshold = threshold
            self.far_field = far_field
            self.indicator = indicator_function

    def solve(self, inputdata, discretization=None, print_info=True,
              print_file=sys.stdout):
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)
        if self.far_field is not None:
            far_field = self.far_field
        else:
            if (inputdata.configuration.Ro
                    >= 10*inputdata.configuration.lambda_b):
                far_field = True
            else:
                far_field = False
        execution_time = 0.
        tic = tm.time()
        if far_field:
            K = self._far_field_kernel(inputdata)
            rhs = self._far_field_rhs(inputdata)
        else:
            K = self._near_field_kernel(inputdata)
            rhs = self._near_field_rhs(inputdata)
        execution_time += tm.time()-tic
        solution = np.zeros(np.prod(inputdata.resolution))
        tic = tm.time()
        if self.regularization is not None:
            for n in range(solution.size):
                t = self.regularization.solve(K, rhs[:, n].flatten())
                solution[n] = norm(t)
        else:
            U, s, Vh = svd(K, full_matrices=False)
            V = np.conj(Vh).T
            if self.tikhonov is None:
                alpha = 0.
            else:
                alpha = self.tikhonov
            if self.sv_cutoff is not None:
                s = s[s > self.sv_cutoff]
            solve(U, s, V, solution, rhs, alpha)
        solution = self.indicator(solution)
        execution_time += tm.time()-tic
        sol_min, sol_max = np.amin(solution), np.amax(solution)
        contrast = (solution-sol_min)/(sol_max-sol_min)
        contrast = contrast.reshape(inputdata.resolution)
        if self.threshold is not None:
            contrast = contrast > self.threshold
            contrast = contrast.astype(float)
        if not inputdata.configuration.good_conductor:
            result.rel_permittivity = cfg.get_relative_permittivity(
                contrast, inputdata.configuration.epsilon_rb
            )
        if not inputdata.configuration.perfect_dielectric:
            result.conductivity = cfg.get_conductivity(
                contrast, 2*pi*inputdata.configuration.f,
                inputdata.configuration.epsilon_rb,
                inputdata.configuration.sigma_b
            )
        if rst.SHAPE_ERROR in inputdata.indicators:
            groundtruth = cfg.get_contrast_map(
                epsilon_r=inputdata.rel_permittivity,
                sigma=inputdata.conductivity,
                configuration=inputdata.configuration
            )
            result.zeta_s = [rst.compute_zeta_s(groundtruth, contrast)]
        if rst.POSITION_ERROR in inputdata.indicators:
            groundtruth = cfg.get_contrast_map(
                epsilon_r=inputdata.rel_permittivity,
                sigma=inputdata.conductivity,
                configuration=inputdata.configuration
            )
            result.zeta_p = [rst.compute_zeta_p(groundtruth, contrast)]
        if rst.EXECUTION_TIME in inputdata.indicators:
            result.execution_time = execution_time
        return result

    def save(self, file_path=''):
        data = super().save(file_path=file_path)
        data[REGULARIZATION] = self.regularization
        data[TIKHONOV] = self.tikhonov
        data[SV_CUTOFF] = self.sv_cutoff
        data[THRESHOLD] = self.threshold
        data[FAR_FIELD] = self.far_field
        data[INDICATOR] = self.indicator
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.regularization = data[REGULARIZATION]
        self.tikhonov = data[TIKHONOV]
        self.sv_cutoff = data[SV_CUTOFF]
        self.threshold = data[THRESHOLD]
        self.far_field = data[FAR_FIELD]
        self.indicator = data[INDICATOR]

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        if self.regularization is not None:
            print(self.regularization, file=print_file)
        else:
            message = 'Regularization: Standard('
            if self.tikhonov is not None:
                message += 'Tikhonov parameter: %.1e' % self.tikhonov
            else:
                message += 'Tikhonov parameter: 0'
            if self.sv_cutoff is not None:
                message += ', Singular value cut-off: %.1e' % self.sv_cutoff
            message += ')'
            print(message, file=print_file)
        if self.threshold is not None:
            print('Threshold: %.2f' % self.threshold, file=print_file)
        if self.far_field is True:
            print('Field approximation: Far', file=print_file)
        elif self.far_field is False:
            print('Field approximation: Near', file=print_file)
        elif self.far_field is None:
            print('Field approximation: automatic', file=print_file)
        print('Indicator function: ' + self.indicator.__name__,
              file=print_file)

    def copy(self, new=None):
        if new is None:
            return LinearSamplingMethod(alias=self.alias,
                                        regularization=self.regularization,
                                        tikhonov=self.tikhonov,
                                        sv_cutoff=self.sv_cutoff,
                                        threshold=self.threshold,
                                        far_field=self.far_field,
                                        indicator_function=self.indicator)
        else:
            super().copy(new)
            self.regularization = new.regularization
            self.tikhonov = new.tikhonov
            self.sv_cutoff = new.sv_cutoff
            self.threshold = new.threshold
            self.far_field = new.far_field
            self.indicator = new.indicator

    def __str__(self):
        message = super().__str__()
        if self.regularization is not None:
            message += str(self.regularization)
        else:
            message += '\nRegularization: Standard('
            if self.tikhonov is not None:
                message += 'Tikhonov parameter: %.1e' % self.tikhonov
            else:
                message += 'Tikhonov parameter: 0'
            if self.sv_cutoff is not None:
                message += ', Singular value cut-off: %.1e' % self.sv_cutoff
            message += ')'
        if self.threshold is not None:
            message += ('\nThreshold: %.2f' % self.threshold)
        if self.far_field is True:
            message += '\nField approximation: Far'
        elif self.far_field is False:
            message += '\nField approximation: Near'
        elif self.far_field is None:
            message += '\nField approximation: automatic'
        message += '\nIndicator function: ' + self.indicator.__name__
        return message

    def _far_field_kernel(self, inputdata):
        NS = inputdata.configuration.NS
        kb = inputdata.configuration.kb
        rho = inputdata.configuration.Ro
        dphi = 2*pi/NS
        E_inf = np.sqrt(rho)/np.exp(-1j*kb*rho)*inputdata.scattered_field
        return E_inf*dphi

    def _far_field_rhs(self, inputdata):
        NM = inputdata.configuration.NM
        kb = inputdata.configuration.kb
        theta = cfg.get_angles(NM)
        x, y = cfg.get_coordinates_ddomain(
            configuration=inputdata.configuration,
            resolution=inputdata.resolution
        )
        x, y = x.flatten(), y.flatten()
        N = x.size
        r = np.sqrt(x**2 + y**2)
        psi = np.arctan2(y, x)
        psi[psi<0] = 2*pi + psi[psi<0]
        Phi = np.zeros((NM, N), dtype=complex)
        for n in range(N):
            Phi[:, n] = (-1j/4*np.sqrt(2/(pi*kb))
                         * np.exp(1j*pi/4)
                         * np.exp(1j*kb*r[n]*np.cos(theta - psi[n])))
        return Phi

    def _near_field_kernel(self, inputdata):
        NS = inputdata.configuration.NS
        dphi = 2*pi/NS
        return inputdata.scattered_field*dphi

    def _near_field_rhs(self, inputdata):
        NM = inputdata.configuration.NM
        kb = inputdata.configuration.kb
        rho = inputdata.configuration.Ro
        theta = cfg.get_angles(NM)
        x, y = cfg.get_coordinates_ddomain(
            configuration=inputdata.configuration,
            resolution=inputdata.resolution
        )
        x, y = x.flatten(), y.flatten()
        N = x.size
        Phi = np.zeros((NM, N), dtype=complex)
        for n in range(N):
            Phi[:, n] = (
                -1j/4*hankel2(0, kb*np.sqrt((rho*np.cos(theta)-x[n])**2
                                            + (rho*np.sin(theta)-y[n])**2))
            )
        return Phi


@jit(nopython=True)
def solve(U, s, V, solution, rhs, alpha):
    N = solution.size
    P = s.size
    for n in range(N):
        t = s[0]/(s[0]**2+alpha) * np.sum(rhs[:, n]*np.conj(U[:, 0])) * V[:, 0]
        for j in range(1, P):
            t += s[j]/(s[j]**2+alpha) * np.sum(rhs[:, n]*np.conj(U[:, j]))* V[:, j]
        solution[n] = np.sqrt(np.sum(np.abs(t)**2))

