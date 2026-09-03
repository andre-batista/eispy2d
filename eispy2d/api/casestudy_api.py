import sys
import numpy as np
from joblib import Parallel, delayed
import pickle
import multiprocessing
from matplotlib import pyplot as plt

from eispy2d.api import api
from eispy2d.api import experiment_api as exp
from eispy2d.api import testset_api as ts
from eispy2d.core import error
from eispy2d.core import result as rst

TEST = 'test'
STOCHASTIC_RUNS = 's_nexec'
SAVE_STOCHASTIC_RUNS = 's_save'

PARALLELIZE_ALGORITHM = 'algorithm'
PARALLELIZE_EXECUTIONS = 'executions'
PERMITTIVITY = 'epsilon_r'
CONDUCTIVITY = 'sigma'
BOTH_PROPERTIES = 'both'
CONTRAST = 'contrast'
ALL_EXECUTIONS = 'all'
BEST_EXECUTION = 'best'


class CaseStudy(exp.Experiment):

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, new):
        if new is None:
            self._test = None
            self._test_available = False
        elif type(new) is dict:
            self._test = new.copy()
            self._test_available = True
        elif type(new) is list:
            self._test = [t.copy() for t in new]
            self._test_available = True

    def __init__(self, name=None, algorithm=None, test=None,
                 stochastic_runs=30, save_stochastic_runs=False,
                 import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(name)
            self.test = test
            self._algorithm = algorithm
            self._single_algorithm = None
            self._algorithm_available = False
            self.s_nexec = stochastic_runs
            self.s_save = save_stochastic_runs
            self.results = None

            if algorithm is not None:
                self.algorithm = algorithm

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, new):
        if new is None:
            self._algorithm = None
            self._single_algorithm = None
            self._algorithm_available = False
        elif callable(new):
            self._algorithm = new
            self._single_algorithm = True
            self._algorithm_available = True
        elif type(new) is list and all(callable(a) for a in new):
            self._algorithm = new
            self._single_algorithm = False
            self._algorithm_available = True

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path)
        self.test = data[TEST]
        self.s_nexec = data[STOCHASTIC_RUNS]
        self.s_save = data[SAVE_STOCHASTIC_RUNS]

    def save(self, file_path='', save_test=False):
        data = super().save(file_path)

        if save_test:
            data[TEST] = self.test
        else:
            data[TEST] = self.test

        data[STOCHASTIC_RUNS] = self.s_nexec
        data[SAVE_STOCHASTIC_RUNS] = self.s_save

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def run(self, parallelization=None, save_stochastic_executions=False):
        if not self._test_available:
            raise error.MissingAttributesError('CaseStudy', 'test')
        if not self._algorithm_available:
            raise error.MissingAttributesError('CaseStudy', 'algorithm')

        if self._single_algorithm:
            if self.s_save or save_stochastic_executions:
                self.results = []
                for n in range(self.s_nexec):
                    self.results.append(api.evaluate(self._algorithm, self.test))
            else:
                self.results = api.evaluate(self._algorithm, self.test)
        else:
            self.results = []
            for a in self._algorithm:
                if self.s_save or save_stochastic_executions:
                    algo_results = []
                    for n in range(self.s_nexec):
                        algo_results.append(api.evaluate(a, self.test))
                    self.results.append(algo_results)
                else:
                    self.results.append(api.evaluate(a, self.test))

    def reconstruction(self, image=CONTRAST, axis=None, algorithm=None,
                       file_name=None, file_path='', file_format='eps',
                       show=False, fontsize=10, title=None, indicator=None,
                       include_true=False, mode=ALL_EXECUTIONS):
        if self.results is None:
            raise error.MissingAttributesError('CaseStudy', 'results')

        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()

    def boxplot(self, indicator, axis=None, algorithm=None,
                show=False, file_name=None, file_path='',
                file_format='eps', title=None, fontsize=10, notch=False):
        if self.results is None:
            raise error.MissingAttributesError('CaseStudy', 'results')

        # Versão simplificada
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()

    def __str__(self):
        message = 'CASE STUDY (API)\n'
        message += super().__str__()
        message += 'Algorithm: '
        if self._single_algorithm:
            message += self._algorithm.__name__ + '\n'
        elif self._algorithm is not None:
            message += str([a.__name__ for a in self._algorithm]) + '\n'
        else:
            message += 'None\n'
        message += 'Stochastic runs: %d\n' % self.s_nexec
        message += 'Save stochastic runs: %s\n' % ('yes' if self.s_save else 'no')
        return message