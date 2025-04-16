import sys
import pickle
import time as tm
import numpy as np
from numpy import linalg as lag
from numpy import pi
from scipy.special import hankel2
from scipy.linalg import svd, norm
from numba import jit
import configuration as cfg
import result as rst
import deterministic as dtm


SV_CUTOFF = 'sv_cutoff'
THRESHOLD = 'threshold'


class MUSIC(dtm.Deterministic):
    def __init__(self, alias='', sv_cutoff=None, threshold=None,
                 import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Multiple Signal Classification Imaging'
            self.sv_cutoff = sv_cutoff
            self.threshold = threshold

    def solve(self, inputdata, discretization=None, print_info=True,
              print_file=sys.stdout):
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)
        execution_time = 0.
        tic = tm.time()
        U, sv, _ = lag.svd(inputdata.scattered_field, full_matrices=False)
        x = np.zeros(discretization.GS.shape[1])
        if self.sv_cutoff is None:
            pass
        elif type(self.sv_cutoff) is int and self.sv_cutoff < sv.size:
            U = U[:, :self.sv_cutoff]
        elif type(self.sv_cutoff) is float:
            U = U[:, sv >= self.sv_cutoff]
        solve(U, discretization.GS, x)
        execution_time += tm.time()-tic
        x = (x-np.amin(x))/(np.amax(x)-np.amin(x))
        contrast = discretization.contrast_image(x, inputdata.resolution)
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
        data[SV_CUTOFF] = self.sv_cutoff
        data[THRESHOLD] = self.threshold
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.sv_cutoff = data[SV_CUTOFF]
        self.threshold = data[THRESHOLD]

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        message = 'Singular value cut-off: '
        if self.sv_cutoff is None:
            message += 'None'
        elif type(self.sv_cutoff) is int:
            message += 'First %d values' % self.sv_cutoff
        else:
            message += '%.1e' % self.sv_cutoff
        if self.threshold is not None:
            message = '\nThreshold: %.2f' % self.threshold
        print(message, file=print_file)

    def copy(self, new=None):
        if new is None:
            return MUSIC(alias=self.alias, sv_cutoff=self.sv_cutoff,
                         threshold=self.threshold)
        else:
            super().copy(new)
            self.sv_cutoff = new.sv_cutoff
            self.threshold = new.threshold

    def __str__(self):
        message = super().__str__()
        message += '\nSingular value cut-off: '
        if self.sv_cutoff is None:
            message += 'None'
        elif type(self.sv_cutoff) is int:
            message += 'First %d values' % self.sv_cutoff
        else:
            message += '%.1e' % self.sv_cutoff
        if self.threshold is not None:
            message = '\nThreshold: %.2f' % self.threshold
        return message


@jit(nopython=True)
def solve(U, GS, x):
    for n in range(x.size):
        den = 0
        for j in range(U.shape[1]):
            den += np.abs(np.sum(np.conj(U[:, j])*GS[:, n]))**2
        x[n] = den

