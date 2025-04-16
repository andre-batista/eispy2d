import sys
sys.path.insert(1, '../../eispy2d/library/')
import pickle
import time as tm
import numpy as np
from numpy import linalg as lag
from numpy import pi
from scipy.special import hankel2 as h2v
from scipy.special import jv
from numba import jit
import configuration as cfg
import result as rst
import deterministic as dtm


SV_CUTOFF = 'sv_cutoff'
THRESHOLD = 'threshold'


class OrthogonalitySamplingMethod(dtm.Deterministic):
    def __init__(self, alias='osm', threshold=None, far_field=None,
                 import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Orthogonality Sampling Method'
            self.far_field = far_field
            self.threshold = threshold

    def solve(self, inputdata, discretization=None, print_info=True,
              print_file=sys.stdout):
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)
        execution_time = 0.

        if self.far_field is None:
            if (inputdata.configuration.Ro
                    > 10*inputdata.configuration.lambda_b):
                far_field = True
            else:
                far_field = False
        else:
            far_field = self.far_field

        tic = tm.time()

        if far_field:
            x = far_field_solution(inputdata)
        else:
            x = near_field_solution(inputdata)
        execution_time += tm.time()-tic
        x = (x-np.amin(x))/(np.amax(x)-np.amin(x))
        contrast = discretization.contrast_image(x, inputdata.resolution)
        if self.threshold is not None:
            background = contrast < self.threshold
            contrast[background] = 0.
            # contrast = contrast.astype(float)
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
        data[THRESHOLD] = self.threshold
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.threshold = data[THRESHOLD]

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        message = ''
        if self.far_field is None:
            message += 'Far field: None\n'
        else:
            message += 'Far field: ' + str(self.far_field) + '\n'
        if self.threshold is not None:
            message = 'Threshold: %.2f' % self.threshold
        print(message, file=print_file)

    def copy(self, new=None):
        if new is None:
            return OrthogonalitySamplingMethod(alias=self.alias,
                                               far_field=self.far_field,
                                               threshold=self.threshold)
        else:
            super().copy(new)
            self.far_field = new.far_field
            self.threshold = new.threshold

    def __str__(self):
        message = super().__str__()
        message += '\nFar field: '
        if self.far_field is None:
            message += 'None'
        else:
            message += str(self.far_field)
        if self.threshold is not None:
            message = '\nThreshold: %.2f' % self.threshold
        return message


def far_field_solution(inputdata):
    kb = inputdata.configuration.kb
    NM, NS = inputdata.configuration.NM, inputdata.configuration.NS
    theta = cfg.get_angles(NM)
    dtheta = theta[1]-theta[0]
    x, y = cfg.get_coordinates_ddomain(configuration=inputdata.configuration,
                                       resolution=inputdata.resolution)
    x, y = x.flatten(), y.flatten()
    N = x.size
    r = np.sqrt(x**2 + y**2)
    psi = np.arctan2(y, x)
    psi[psi<0] = 2*np.pi + psi[psi<0]
    u = inputdata.scattered_field
    I = np.zeros(N)
    for n in range(N):
        A = np.tile(np.exp(-1j*np.pi/4)/np.sqrt(8*np.pi*kb)
                    * np.exp(-1j*kb*r[n]*np.cos(theta - psi[n])).reshape((-1,
                                                                          1)),
                    (1, NS))
        I[n] = np.trapz(np.abs(np.trapz(u.T*A.T, dx=dtheta))**2)
    return I.reshape(inputdata.resolution)


def near_field_solution(inputdata):
    kb = inputdata.configuration.kb
    Ro = inputdata.configuration.Ro
    NM = inputdata.configuration.NM
    N = np.prod(inputdata.resolution)
    theta = cfg.get_angles(NM)
    theta = np.tile(theta.reshape((-1, 1)), (1, N))
    x, y = cfg.get_coordinates_ddomain(configuration=inputdata.configuration,
                                       resolution=inputdata.resolution)
    x, y = x.flatten(), y.flatten()
    N = x.size
    r = np.tile(np.sqrt(x**2 + y**2), (NM, 1))
    psi = np.arctan2(y, x)
    psi[psi<0] = 2*np.pi + psi[psi<0]
    psi = np.tile(psi, (NM, 1))
    u = inputdata.scattered_field
    I = np.zeros(N)
    NT = np.arange(-20, 21, dtype=int)
    K = np.zeros((NM, N), dtype=complex)
    for nt in NT:
        K += -2j/(np.pi*Ro)*jv(nt, kb*r)/h2v(nt, kb*Ro)*np.exp(1j*nt*(theta - psi))
    I = np.trapz(np.abs(u.T @ K)**2, axis=0)
    return I.reshape(inputdata.resolution)