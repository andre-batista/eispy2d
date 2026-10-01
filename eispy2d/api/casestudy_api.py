import sys
import numpy as np
from joblib import Parallel, delayed
import pickle
import multiprocessing
from functools import partial
from matplotlib import pyplot as plt

from eispy2d.api import api
from eispy2d.api import experiment_api as exp
from eispy2d.api import testset_api as ts
from eispy2d.core import error
from eispy2d.core import result as rst

TEST = 'test'
STOCHASTIC_RUNS = 's_nexec'
SAVE_STOCHASTIC_RUNS = 's_save'
ALGORITHM_PARAMS = 'algorithm_params'

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
                 import_filename=None, import_filepath='',
                 fixed_params=None, variable_param=None, variable_values=None,
                 algorithm_params=None):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(name)
            self.test = test
            if test is None and variable_param is not None:
                self.test = self.generate_tests(fixed_params,
                                                variable_param,
                                                variable_values)
            self._algorithm = algorithm
            self._single_algorithm = None
            self._algorithm_available = False
            self.s_nexec = stochastic_runs
            self.s_save = save_stochastic_runs
            self.algorithm_params = algorithm_params
            self.results = None

            if algorithm is not None:
                self.algorithm = algorithm

    @staticmethod
    def generate_tests(fixed_params=None, variable_param=None,
                       variable_values=None):
        fixed_params = {} if fixed_params is None else fixed_params.copy()

        if variable_param is None:
            return [fixed_params.copy()]

        if variable_values is None:
            raise error.MissingAttributesError(
                'CaseStudy', 'variable_values'
            )

        tests = []
        for value in variable_values:
            params = fixed_params.copy()
            params[variable_param] = value
            tests.append(params)
        return tests

    @staticmethod
    def _wrap_algorithm(algorithm, params=None):
        if params is None:
            return algorithm

        wrapped = partial(algorithm, **params)
        if hasattr(algorithm, '__name__'):
            wrapped.__name__ = algorithm.__name__
        return wrapped

    def _get_algorithm_params(self, algorithm_params, n_tests):
        if algorithm_params is None:
            return [None] * n_tests

        if type(algorithm_params) is dict:
            return [algorithm_params.copy() for _ in range(n_tests)]

        if type(algorithm_params) is list:
            if len(algorithm_params) != n_tests:
                raise error.WrongValueInput(
                    'CaseStudy.run', 'algorithm_params',
                    'a list with one dictionary per test',
                    str(algorithm_params)
                )
            if not all(p is None or type(p) is dict for p in algorithm_params):
                raise error.WrongTypeInput(
                    'CaseStudy.run', 'algorithm_params',
                    'list of dictionaries or None',
                    str(type(algorithm_params))
                )
            return [None if p is None else p.copy()
                    for p in algorithm_params]

        raise error.WrongTypeInput(
            'CaseStudy.run', 'algorithm_params',
            'dictionary or list of dictionaries',
            str(type(algorithm_params))
        )

    def _run_algorithm(self, algorithm, tests, algorithm_params=None):
        if algorithm_params is None:
            return api.evaluate(algorithm, tests)

        params = self._get_algorithm_params(algorithm_params, len(tests))
        results = []
        for test, extra_params in zip(tests, params):
            wrapped = self._wrap_algorithm(algorithm, extra_params)
            results.append(api.evaluate(wrapped, test))
        return results

    def run(self, parallelization=None, save_stochastic_executions=False,
            algorithm_params=None):
        if not self._test_available:
            raise error.MissingAttributesError('CaseStudy', 'test')
        if not self._algorithm_available:
            raise error.MissingAttributesError('CaseStudy', 'algorithm')

        if algorithm_params is None:
            algorithm_params = self.algorithm_params
        else:
            self.algorithm_params = algorithm_params

        tests = self.test if type(self.test) is list else [self.test]

        if self._single_algorithm:
            if self.s_save or save_stochastic_executions:
                self.results = []
                for _ in range(self.s_nexec):
                    self.results.append(
                        self._run_algorithm(self._algorithm,
                                            tests,
                                            algorithm_params)
                    )
            else:
                self.results = self._run_algorithm(self._algorithm,
                                                   tests,
                                                   algorithm_params)
        else:
            self.results = []
            for a in self._algorithm:
                current_params = algorithm_params
                if type(algorithm_params) is list and algorithm_params:
                    if all(p is None or type(p) is dict
                           for p in algorithm_params):
                        current_params = algorithm_params
                    else:
                        current_params = algorithm_params[self._algorithm.index(a)]

                if self.s_save or save_stochastic_executions:
                    algo_results = []
                    for _ in range(self.s_nexec):
                        algo_results.append(
                            self._run_algorithm(a, tests, current_params)
                        )
                    self.results.append(algo_results)
                else:
                    self.results.append(
                        self._run_algorithm(a, tests, current_params)
                    )

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