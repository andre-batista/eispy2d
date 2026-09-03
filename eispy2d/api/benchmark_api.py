import pickle
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from matplotlib import pyplot as plt

from eispy2d.api import api
from eispy2d.api import testset_api as ts
from eispy2d.core import error
from eispy2d.api import experiment_api as exp
from eispy2d.core import result as rst

TESTSET = "testset"
PARALLELIZE_TESTS = "test"
PARALLELIZE_ALGORITHMS = "algorithm"
LABEL_INSTANCE = 'Instance Index'


class Benchmark(exp.Experiment):

    @property
    def testset(self):
        return self._testset
    
    @testset.setter
    def testset(self, new):
        if new is None:
            self._testset = None
            self._single_testset = None
            self._testset_available = False
        elif type(new) is ts.TestSet:
            self._testset = new.copy()
            self._single_testset = True
            self._testset_available = True
        elif type(new) is str:
            self._testset = new
            self._single_testset = True
            self._testset_available = False
        elif type(new) is list and len(new) == 1:
            self._single_testset = True
            if type(new[0]) is ts.TestSet:
                self._testset = new[0].copy()
                self._testset_available = True
            elif type(new[0]) is str:
                self._testset = new[0]
                self._testset_available = False
        elif type(new) is list and all(isinstance(n, str) for n in new):
            self._testset = new.copy()
            self._single_testset = False
            self._testset_available = False
        elif (type(new) is list
                and all(isinstance(n, ts.TestSet) for n in new)):
            self._testset = [new[i].copy() for i in range(len(new))]
            self._single_testset = False
            self._testset_available = True

    def __init__(self, name='', algorithm=None, testset=None,
                 import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(name)
            self._algorithm = algorithm
            self._testset = testset
            self._single_testset = None
            self._testset_available = False
            self._single_algorithm = None
            self._algorithm_available = False
            self.results = None

            # Configura testset
            if testset is not None:
                self.testset = testset

            # Configura algorithm
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

    def run(self, parallelization=None):
        if not self._testset_available:
            raise error.MissingAttributesError('Benchmark', 'testset')
        if not self._algorithm_available:
            raise error.MissingAttributesError('Benchmark', 'algorithm')

        self.results = []

        if self._single_algorithm and self._single_testset:
            if parallelization == True:
                num_cores = multiprocessing.cpu_count()
                self.results = (
                    Parallel(n_jobs=num_cores)
                    (delayed(_run_single_test)(self._algorithm,
                                               self._testset.test[n])
                     for n in range(self._testset.sample_size))
                )
            else:
                for n in range(self._testset.sample_size):
                    self.results.append(
                        _run_single_test(self._algorithm,
                                         self._testset.test[n])
                    )
            self.results = np.array(self.results)

        elif self._single_algorithm and not self._single_testset:
            if parallelization is None or parallelization == False:
                for t in range(len(self._testset)):
                    self.results.append([])
                    for n in range(self._testset[t].sample_size):
                        self.results[t].append(
                            _run_single_test(self._algorithm,
                                             self._testset[t].test[n])
                        )
            elif parallelization == PARALLELIZE_TESTS:
                for t in range(len(self._testset)):
                    num_cores = multiprocessing.cpu_count()
                    self.results.append(
                        Parallel(n_jobs=num_cores)
                        (delayed(_run_single_test)(self._algorithm,
                                                   self._testset[t].test[n])
                         for n in range(self._testset[t].sample_size))
                    )
            else:
                raise error.WrongValueInput('Benchmark.run', 'parallelization',
                                            "None, False, 'test'",
                                            str(parallelization))

            if all(self._testset[n].sample_size == self._testset[n+1].sample_size
                   for n in range(len(self._testset)-1)):
                self.results = np.array(self.results)
            else:
                self.results = np.array(self.results, dtype=object)

        elif not self._single_algorithm and self._single_testset:
            if parallelization is None or parallelization == False:
                for a in range(len(self._algorithm)):
                    self.results.append([])
                    for n in range(self._testset.sample_size):
                        self.results[a].append(
                            _run_single_test(self._algorithm[a],
                                             self._testset.test[n])
                        )
            elif parallelization == PARALLELIZE_TESTS:
                num_cores = multiprocessing.cpu_count()
                for a in range(len(self._algorithm)):
                    self.results.append(
                        Parallel(n_jobs=num_cores)
                        (delayed(_run_single_test)(self._algorithm[a],
                                                   self._testset.test[n])
                         for n in range(self._testset.sample_size))
                    )
            elif parallelization == PARALLELIZE_ALGORITHMS:
                num_cores = multiprocessing.cpu_count()
                self.results = (Parallel(n_jobs=num_cores)
                                (delayed(_run_testset_algorithm)(
                                    self._testset, self._algorithm[a])
                                 for a in range(len(self._algorithm))))
            else:
                raise error.WrongValueInput('Benchmark.run', 'parallelization',
                                            "None, False, 'test', 'algorithm'",
                                            str(parallelization))

            self.results = np.array(self.results)

        else:
            if parallelization is None or parallelization == False:
                for a in range(len(self._algorithm)):
                    self.results.append([])
                    for t in range(len(self._testset)):
                        self.results[a].append([])
                        for n in range(self._testset[t].sample_size):
                            self.results[a][t].append(
                                _run_single_test(self._algorithm[a],
                                                 self._testset[t].test[n])
                            )
            elif parallelization == PARALLELIZE_TESTS:
                num_cores = multiprocessing.cpu_count()
                for a in range(len(self._algorithm)):
                    self.results.append([])
                    for t in range(len(self._testset)):
                        self.results[a].append(
                            Parallel(n_jobs=num_cores)
                            (delayed(_run_single_test)(self._algorithm[a],
                                                       self._testset[t].test[n])
                             for n in range(self._testset[t].sample_size))
                        )
            elif parallelization == PARALLELIZE_ALGORITHMS:
                num_cores = multiprocessing.cpu_count()
                for t in range(len(self._testset)):
                    output = Parallel(n_jobs=num_cores)
                    (delayed(_run_testset_algorithm)(self._testset[t],
                                                     self._algorithm[a])
                     for a in range(len(self._algorithm)))
            else:
                raise error.WrongValueInput('Benchmark.run', 'parallelization',
                                            "None, False, 'test', 'algorithm'",
                                            str(parallelization))

            try:
                self.results = np.array(self.results)
            except ValueError:
                self.results = np.array(self.results, dtype=object)

    def plot(self, indicator, axis=None, testset=None, algorithm=None,
             yscale=None, show=False, file_name=None, file_path='',
             file_format='eps', title=None, fontsize=10):
        if self.results is None:
            raise error.MissingAttributesError('Benchmark', 'results')
        if indicator is None:
            raise error.WrongTypeInput('Benchmark.plot', 'indicator',
                                       'str or str-list', str(type(indicator)))
        if not rst.check_indicator(indicator):
            raise error.WrongValueInput('Benchmark.plot', 'indicator',
                                        rst.INDICATOR_SET, indicator)

        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()

    def save(self, file_path='', save_testset=False):
        data = super().save(file_path)

        if save_testset:
            data[TESTSET] = self.testset
        elif self._testset_available and self._single_testset:
            data[TESTSET] = self.testset.name
        elif self._testset_available and not self._single_testset:
            data[TESTSET] = [self.testset[n].name
                             for n in range(len(self.testset))]
        else:
            data[TESTSET] = self.testset

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path)
        self.testset = data[TESTSET]

    def __str__(self):
        """Retorna representação string do benchmark."""
        message = 'BENCHMARK (API)\n'
        message += super().__str__()
        message += 'Algorithm: '
        if self._single_algorithm:
            message += self._algorithm.__name__ + '\n'
        elif self._algorithm is not None:
            message += str([a.__name__ for a in self._algorithm]) + '\n'
        else:
            message += 'None\n'
        message += 'Test set: '
        if self._testset_available and self._single_testset:
            message += self.testset.name + '\n'
        elif self._testset_available and not self._single_testset:
            message += str([t.name for t in self.testset]) + '\n'
        else:
            message += str(self.testset) + '\n'
        return message


def _run_single_test(algorithm, params):
    return api.evaluate(algorithm, params)


def _run_testset_algorithm(testset, algorithm):
    results = []
    for n in range(testset.sample_size):
        results.append(api.evaluate(algorithm, testset.test[n]))
    return results