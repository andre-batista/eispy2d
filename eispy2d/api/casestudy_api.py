import pickle
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
from matplotlib import pyplot as plt

from eispy2d.api import api
from eispy2d.api import experiment_api as exp
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


def _run_single_test(algorithm, test, algorithm_params=None):
    wrapped = CaseStudy._wrap_algorithm(algorithm, algorithm_params)
    try:
        return api.evaluate(wrapped, test)
    except Exception as exc:
        raise RuntimeError(
            'CaseStudy execution failed: %s: %s'
            % (type(exc).__name__, str(exc))
        ) from None


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
        elif type(new) is list and all(type(t) is dict for t in new):
            self._test = [t.copy() for t in new]
            self._test_available = True
        else:
            raise error.WrongTypeInput(
                'CaseStudy.test',
                'new',
                'dict or list of dict',
                str(type(new))
            )

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
        elif type(new) is list and len(new) > 0 and all(callable(a) for a in new):
            self._algorithm = new.copy()
            self._single_algorithm = False
            self._algorithm_available = True
        else:
            raise error.WrongTypeInput(
                'CaseStudy.algorithm',
                'new',
                'callable or list of callable',
                str(type(new))
            )

    def __init__(self, name=None, algorithm=None, test=None,
                 stochastic_runs=30, save_stochastic_runs=False,
                 import_filename=None, import_filepath='',
                 fixed_params=None, variable_param=None, variable_values=None,
                 algorithm_params=None):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
            if algorithm is not None:
                self.algorithm = algorithm
        else:
            super().__init__('' if name is None else name)

            self._test = None
            self._test_available = False
            self._algorithm = None
            self._single_algorithm = None
            self._algorithm_available = False
            self.s_nexec = stochastic_runs
            self.s_save = save_stochastic_runs
            self.algorithm_params = algorithm_params
            self.results = None

            if test is not None:
                self.test = test
            elif variable_param is not None:
                self.test = self.generate_tests(
                    fixed_params,
                    variable_param,
                    variable_values
                )
            elif fixed_params is not None:
                self.test = fixed_params

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

        if type(params) is not dict:
            raise error.WrongTypeInput(
                'CaseStudy.run',
                'algorithm_params',
                'dict',
                str(type(params))
            )

        wrapped = partial(algorithm, **params)
        if hasattr(algorithm, '__name__'):
            wrapped.__name__ = algorithm.__name__
        return wrapped

    @staticmethod
    def _normalize_tests(test):
        if type(test) is dict:
            return [test]
        return test

    def _normalize_algorithm_params(self, algorithm_params, n_tests):
        if algorithm_params is None:
            return [None] * n_tests

        if type(algorithm_params) is dict:
            return [algorithm_params.copy() for _ in range(n_tests)]

        if type(algorithm_params) is list:
            if len(algorithm_params) != n_tests:
                raise error.WrongValueInput(
                    'CaseStudy.run',
                    'algorithm_params',
                    'a dictionary or a list with one dictionary per test',
                    str(algorithm_params)
                )

            if not all(p is None or type(p) is dict for p in algorithm_params):
                raise error.WrongTypeInput(
                    'CaseStudy.run',
                    'algorithm_params',
                    'a dictionary or a list of dictionaries',
                    str(type(algorithm_params))
                )

            return [None if p is None else p.copy()
                    for p in algorithm_params]

        raise error.WrongTypeInput(
            'CaseStudy.run',
            'algorithm_params',
            'a dictionary or a list of dictionaries',
            str(type(algorithm_params))
        )

    def _get_algorithm_params_for_algorithm(self, algorithm_index,
                                            algorithm_params, n_tests):
        if algorithm_params is None:
            return None

        if type(algorithm_params) is dict:
            return algorithm_params

        if type(algorithm_params) is list:
            if len(algorithm_params) == n_tests:
                return algorithm_params

            if len(algorithm_params) == len(self._algorithm):
                params = algorithm_params[algorithm_index]
                if params is None or type(params) is dict:
                    return params

        raise error.WrongValueInput(
            'CaseStudy.run',
            'algorithm_params',
            'a dictionary, one dictionary per test, or one dictionary per algorithm',
            str(algorithm_params)
        )

    def _run_single_algorithm(self, algorithm, tests, algorithm_params=None):
        params = self._normalize_algorithm_params(
            algorithm_params,
            len(tests)
        )

        return [
            api.evaluate(
                self._wrap_algorithm(algorithm, extra_params),
                test
            )
            for test, extra_params in zip(tests, params)
        ]

    def _run_single_algorithm_parallel(self, algorithm, tests,
                                       algorithm_params=None):
        params = self._normalize_algorithm_params(
            algorithm_params,
            len(tests)
        )

        return Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(_run_single_test)(
                algorithm,
                test,
                extra_params
            )
            for test, extra_params in zip(tests, params)
        )

    def _run_algorithm(self, algorithm, tests, algorithm_params=None,
                       parallelization=None):
        if parallelization == PARALLELIZE_EXECUTIONS:
            return self._run_single_algorithm_parallel(
                algorithm,
                tests,
                algorithm_params
            )

        if parallelization not in (None, False):
            raise error.WrongValueInput(
                'CaseStudy.run',
                'parallelization',
                "None, False, 'algorithm', 'executions'",
                str(parallelization)
            )

        return self._run_single_algorithm(
            algorithm,
            tests,
            algorithm_params
        )

    def run(self, parallelization=None, save_stochastic_executions=False,
            algorithm_params=None):
        if not self._test_available:
            raise error.MissingAttributesError('CaseStudy', 'test')
        if not self._algorithm_available:
            raise error.MissingAttributesError('CaseStudy', 'algorithm')

        if algorithm_params is not None:
            self.algorithm_params = algorithm_params
        else:
            algorithm_params = self.algorithm_params

        tests = self._normalize_tests(self.test)
        save_runs = self.s_save or save_stochastic_executions

        if self._single_algorithm:
            if parallelization == PARALLELIZE_ALGORITHM:
                raise error.WrongValueInput(
                    'CaseStudy.run',
                    'parallelization',
                    "None, False, 'algorithm', 'executions'",
                    str(parallelization)
                )

            if save_runs:
                if parallelization == PARALLELIZE_EXECUTIONS:
                    self.results = Parallel(
                        n_jobs=min(
                            multiprocessing.cpu_count(),
                            self.s_nexec
                        )
                    )(
                        delayed(self._run_algorithm)(
                            self._algorithm,
                            tests,
                            algorithm_params,
                            None
                        )
                        for _ in range(self.s_nexec)
                    )
                else:
                    self.results = [
                        self._run_algorithm(
                            self._algorithm,
                            tests,
                            algorithm_params,
                            parallelization
                        )
                        for _ in range(self.s_nexec)
                    ]
            else:
                self.results = self._run_algorithm(
                    self._algorithm,
                    tests,
                    algorithm_params,
                    parallelization
                )
            return

        if parallelization == PARALLELIZE_ALGORITHM:
            self.results = Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(_run_algorithm_repeated)(
                    algorithm,
                    tests,
                    self._get_algorithm_params_for_algorithm(
                        index,
                        algorithm_params,
                        len(tests)
                    ),
                    self.s_nexec,
                    save_runs,
                    None
                )
                for index, algorithm in enumerate(self._algorithm)
            )
        else:
            self.results = []
            for index, algorithm in enumerate(self._algorithm):
                current_params = self._get_algorithm_params_for_algorithm(
                    index,
                    algorithm_params,
                    len(tests)
                )
                self.results.append(
                    _run_algorithm_repeated(
                        algorithm,
                        tests,
                        current_params,
                        self.s_nexec,
                        save_runs,
                        parallelization
                    )
                )

        try:
            self.results = np.array(self.results)
        except (ValueError, TypeError):
            self.results = np.array(self.results, dtype=object)

    def reconstruction(self, image=CONTRAST, axis=None, algorithm=None,
                       file_name=None, file_path='', file_format='eps',
                       show=False, fontsize=10, title=None, indicator=None,
                       include_true=False, mode=ALL_EXECUTIONS):
        if self.results is None:
            raise error.MissingAttributesError('CaseStudy', 'results')

        if indicator is not None and not rst.check_indicator(indicator):
            raise error.WrongValueInput(
                'CaseStudy.reconstruction',
                'indicator',
                rst.INDICATOR_SET,
                indicator
            )

        plt.tight_layout()
        if title is not None:
            plt.title(title, fontsize=fontsize)
        if file_name is not None:
            plt.savefig(
                file_path + file_name + '.' + file_format,
                format=file_format
            )
        if show:
            plt.show()
        if file_name is not None:
            plt.close()

    def boxplot(self, indicator, axis=None, algorithm=None,
                show=False, file_name=None, file_path='',
                file_format='eps', title=None, fontsize=10, notch=False):
        if self.results is None:
            raise error.MissingAttributesError('CaseStudy', 'results')
        if not rst.check_indicator(indicator):
            raise error.WrongValueInput(
                'CaseStudy.boxplot',
                'indicator',
                rst.INDICATOR_SET,
                indicator
            )

        values = self._get_indicator_values(indicator, algorithm)
        if len(values) == 0:
            raise error.MissingAttributesError('CaseStudy', 'results')

        plt.figure()
        plt.boxplot(values, notch=notch)
        plt.ylabel(rst.TITLES.get(indicator, indicator), fontsize=fontsize)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        plt.tight_layout()

        if file_name is not None:
            plt.savefig(
                file_path + file_name + '.' + file_format,
                format=file_format
            )
        if show:
            plt.show()
        if file_name is not None:
            plt.close()

    def _get_indicator_values(self, indicator, algorithm=None):
        selected = self.results

        if self._single_algorithm:
            if self.s_save:
                selected = [r for r in self.results]
            else:
                selected = [self.results]
        else:
            if algorithm is None:
                selected = self.results
            elif isinstance(algorithm, int):
                selected = [self.results[algorithm]]
            elif callable(algorithm):
                index = self._algorithm.index(algorithm)
                selected = [self.results[index]]
            else:
                raise error.WrongTypeInput(
                    'CaseStudy._get_indicator_values',
                    'algorithm',
                    'int or callable',
                    str(type(algorithm))
                )

        values = []
        for item in selected:
            for result in _flatten_results(item):
                value = getattr(result, indicator, None)
                if value is None:
                    continue
                array = np.asarray(value)
                if array.size == 1:
                    values.append(float(array.reshape(-1)[0]))
                else:
                    values.extend(array.astype(float).reshape(-1).tolist())

        return values

    def save(self, file_path='', save_test=False):
        data = super().save(file_path)
        data[TEST] = self.test if save_test else self.test
        data[STOCHASTIC_RUNS] = self.s_nexec
        data[SAVE_STOCHASTIC_RUNS] = self.s_save
        data[ALGORITHM_PARAMS] = self.algorithm_params

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path)
        self.test = data[TEST]
        self.s_nexec = data[STOCHASTIC_RUNS]
        self.s_save = data[SAVE_STOCHASTIC_RUNS]
        self.algorithm_params = data.get(ALGORITHM_PARAMS)
        self._algorithm = None
        self._single_algorithm = None
        self._algorithm_available = False

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
        message += 'Save stochastic runs: %s\n' % (
            'yes' if self.s_save else 'no'
        )
        message += 'Algorithm parameters: %s\n' % str(self.algorithm_params)
        return message


def _run_algorithm_repeated(algorithm, tests, algorithm_params,
                            stochastic_runs, save_runs, parallelization):
    if not save_runs:
        return CaseStudy._run_single_algorithm_static(
            algorithm,
            tests,
            algorithm_params,
            parallelization
        )

    executions = []
    for _ in range(stochastic_runs):
        executions.append(
            CaseStudy._run_single_algorithm_static(
                algorithm,
                tests,
                algorithm_params,
                None if parallelization == PARALLELIZE_EXECUTIONS
                else parallelization
            )
        )
    return executions


def _flatten_results(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            yield from _flatten_results(item)
    else:
        yield value


def _static_normalize_params(algorithm_params, n_tests):
    if algorithm_params is None:
        return [None] * n_tests
    if type(algorithm_params) is dict:
        return [algorithm_params.copy() for _ in range(n_tests)]
    if type(algorithm_params) is list and len(algorithm_params) == n_tests:
        if all(p is None or type(p) is dict for p in algorithm_params):
            return [None if p is None else p.copy() for p in algorithm_params]
    raise error.WrongValueInput(
        'CaseStudy.run',
        'algorithm_params',
        'a dictionary or a list with one dictionary per test',
        str(algorithm_params)
    )


def _run_single_algorithm_static(algorithm, tests, algorithm_params,
                                parallelization=None):
    params = _static_normalize_params(algorithm_params, len(tests))
    return [
        _run_single_test(algorithm, test, extra_params)
        for test, extra_params in zip(tests, params)
    ]


def _run_single_algorithm_parallel_static(algorithm, tests, algorithm_params):
    params = _static_normalize_params(algorithm_params, len(tests))
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(_run_single_test)(algorithm, test, extra_params)
        for test, extra_params in zip(tests, params)
    )