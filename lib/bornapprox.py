# Standard libraries
import sys
import pickle
import time as tm
from numpy import pi

# Developed libraries
import deterministic as dtm
import mom_cg_fft as mom
import configuration as cfg
import result as rst

REGULARIZATION = 'regularization'
FORWARD = 'forward'

class FirstOrderBornApproximation(dtm.Deterministic):
    def __init__(self, regularization, forward=mom.MoM_CG_FFT(), alias='ba'):
        super().__init__(alias=alias, parallelization=False)
        self.name = 'First-Order Born Approximation'
        self.regularization = regularization
        self.forward = forward

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)

        incident_field = self.forward.incident_field(discretization.elements,
                                                     inputdata.configuration)
        
        tic = tm.time()
        contrast = discretization.solve(scattered_field=inputdata.scattered_field,
                                        total_field=incident_field,
                                        linear_solver=self.regularization)
        execution_time = tm.time()-tic


        scattered_field = discretization.scattered_field(contrast=contrast,
                                                         total_field=incident_field)
        total_field = self.forward.incident_field(inputdata.resolution,
                                                  inputdata.configuration)
        contrast = discretization.contrast_image(contrast,
                                                 inputdata.resolution)

        result.update_error(inputdata, scattered_field=scattered_field,
                            total_field=total_field, contrast=contrast)
        result.scattered_field = scattered_field
        result.total_field = incident_field

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
        data[REGULARIZATION] = self.regularization
        data[FORWARD] = self.forward
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.regularization = data[REGULARIZATION]
        self.forward = data[FORWARD]

    def copy(self, new=None):
        if new is None:
            return FirstOrderBornApproximation(self.regularization, self.forward,
                                               self.alias)
        else:
            super().copy(new)
            self.regularization = new.regularization
            self.forward = new.forward

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)
        print(self.regularization, file=print_file)

    def __str__(self):
        message = super().__str__()
        message += str(self.regularization)
        message += str(self.forward)
        return message

