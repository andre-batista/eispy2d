# Standard libraries
import sys
import pickle
import time as tm
import numpy as np
from numpy import pi
from numba import jit

# Developed libraries
import deterministic as dtm
import mom_cg_fft as mom
import configuration as cfg
import result as rst
import fftproduct

FORWARD = 'forward'

class BackPropagation(dtm.Deterministic):
    def __init__(self, forward=mom.MoM_CG_FFT(), alias='backprop',
                 import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=False)
            self.name = 'Back-Propagation'
            self.forward = forward

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)

        NY, NX = discretization.elements

        tic = tm.time()
        incident_field = self.forward.incident_field(discretization.elements,
                                                     inputdata.configuration)
        gamma = compute_gamma(inputdata.scattered_field,
                              discretization.GS)
        current = compute_current(inputdata.scattered_field, discretization.GS,
                                  gamma)
        prod = fftproduct.FFTProduct(discretization)
        total_field = incident_field + prod.compute(current)
        contrast = compute_contrast(total_field, current, NX, NY)
        execution_time = tm.time()-tic


        scattered_field = discretization.scattered_field(
            contrast=contrast, total_field=total_field
        )
        contrast = discretization.contrast_image(contrast,
                                                 inputdata.resolution)

        result.update_error(inputdata, scattered_field=scattered_field,
                            total_field=total_field, contrast=contrast)
        result.scattered_field = scattered_field
        result.total_field = total_field

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
        if rst.EXECUTION_TIME in inputdata.indicators:
            result.execution_time = execution_time
        if rst.NUMBER_ITERATIONS in inputdata.indicators:
            result.number_iterations = 1

        return result

    def save(self, file_path=''):
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]

    def copy(self, new=None):
        if new is None:
            return BackPropagation(self.forward, self.alias)
        else:
            super().copy(new)
            self.forward = new.forward

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)

    def __str__(self):
        message = super().__str__()
        message += str(self.forward)
        return message

@jit(nopython=True)
def compute_gamma(Es, GS):
    aux = GS @ GS.T.conjugate() @ Es
    num = np.sum(Es * np.conjugate(aux), axis=0)
    dem = np.sum(np.abs(aux)**2, axis=0)
    return num/dem

@jit(nopython=True)
def compute_current(Es, GS, gamma):
    aux = 0j*np.ones((GS.shape[1], gamma.size))
    for n in range(GS.shape[1]):
        aux[n, :] = gamma
    return aux * (GS.T.conjugate()  @ Es)
    # return np.tile(gamma, (GS.shape[1], 1)) * (GS.T.conjugate()  @ Es)

@jit(nopython=True)
def compute_contrast(E, J, NX, NY):
    Econj = np.conj(E)
    num = np.sum(J * Econj, 1)
    den = np.sum(Econj * E, 1)
    return np.reshape(num/den, (NY, NX))