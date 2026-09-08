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
CONFIGURATIONS = "configurations"
PARALLELIZE_TESTS = "test"
PARALLELIZE_CONFIGS = "configs"
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
                 configurations=None, import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(name)
            self._algorithm = algorithm
            self._testset = testset
            self._configurations = configurations
            self._single_testset = None
            self._testset_available = False
            self._single_algorithm = None
            self._algorithm_available = False
            self._single_config = None
            self._config_available = False
            self.results = None

            if testset is not None:
                self.testset = testset

            if algorithm is not None:
                self.algorithm = algorithm
            
            if configurations is not None:
                self.configurations = configurations

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
        else:
            raise error.WrongTypeInput('Benchmark.algorithm', 'new',
                                       'callable or list of callable',
                                       str(type(new)))

    @property
    def configurations(self):
        return self._configurations

    @configurations.setter
    def configurations(self, new):
        if new is None:
            self._configurations = None
            self._single_config = None
            self._config_available = False
        elif type(new) is dict:
            self._configurations = new.copy()
            self._single_config = True
            self._config_available = True
        elif type(new) is list and all(isinstance(c, dict) for c in new):
            self._configurations = [c.copy() for c in new]
            self._single_config = False
            self._config_available = True
        else:
            raise error.WrongTypeInput('Benchmark.configurations', 'new',
                                       'dict or list of dict',
                                       str(type(new)))

    def run(self, parallelization=None):
        if not self._testset_available:
            raise error.MissingAttributesError('Benchmark', 'testset')
        if not self._algorithm_available:
            raise error.MissingAttributesError('Benchmark', 'algorithm')

        self.results = []

        if self._single_algorithm and self._single_testset and self._config_available and not self._single_config:
            if parallelization is None or parallelization == False:
                for c in range(len(self._configurations)):
                    self.results.append([])
                    for n in range(self._testset.sample_size):
                        params = self._testset.test[n].copy()
                        params.update(self._configurations[c])
                        self.results[c].append(
                            _run_single_test(self._algorithm, params)
                        )
            elif parallelization == PARALLELIZE_TESTS:
                for c in range(len(self._configurations)):
                    num_cores = multiprocessing.cpu_count()
                    self.results.append(
                        Parallel(n_jobs=num_cores)(
                            delayed(_run_single_test)(
                                self._algorithm,
                                {**self._testset.test[n], **self._configurations[c]}
                            ) for n in range(self._testset.sample_size)
                        )
                    )
            elif parallelization == PARALLELIZE_CONFIGS:
                num_cores = multiprocessing.cpu_count()
                self.results = Parallel(n_jobs=num_cores)(
                    delayed(_run_config_testset)(
                        self._testset, self._algorithm, self._configurations[c]
                    ) for c in range(len(self._configurations))
                )
            else:
                raise error.WrongValueInput('Benchmark.run', 'parallelization',
                                            "None, False, 'test', 'configs'",
                                            str(parallelization))
            
            try:
                self.results = np.array(self.results)
            except:
                self.results = np.array(self.results, dtype=object)

        elif self._single_algorithm and self._single_testset:
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

            has_multiple_configs = (
                self._config_available and not self._single_config
            )

            if has_multiple_configs:
                if parallelization is None or parallelization == False:
                    for a in range(len(self._algorithm)):
                        algorithm_results = []

                        for c in range(len(self._configurations)):
                            config_results = []

                            for n in range(self._testset.sample_size):
                                params = self._testset.test[n].copy()
                                params.update(self._configurations[c])

                                config_results.append(
                                    _run_single_test(
                                        self._algorithm[a],
                                        params
                                    )
                                )

                            algorithm_results.append(config_results)

                        self.results.append(algorithm_results)

                elif parallelization == PARALLELIZE_TESTS:
                    num_cores = multiprocessing.cpu_count()

                    for a in range(len(self._algorithm)):
                        algorithm_results = []

                        for c in range(len(self._configurations)):
                            algorithm_results.append(
                                Parallel(n_jobs=num_cores)(
                                    delayed(_run_single_test)(
                                        self._algorithm[a],
                                        {
                                            **self._testset.test[n],
                                            **self._configurations[c]
                                        }
                                    )
                                    for n in range(self._testset.sample_size)
                                )
                            )

                        self.results.append(algorithm_results)

                elif parallelization == PARALLELIZE_CONFIGS:
                    num_cores = multiprocessing.cpu_count()

                    for a in range(len(self._algorithm)):
                        algorithm_results = Parallel(n_jobs=num_cores)(
                            delayed(_run_config_testset)(
                                self._testset,
                                self._algorithm[a],
                                self._configurations[c]
                            )
                            for c in range(len(self._configurations))
                        )

                        self.results.append(algorithm_results)

                else:
                    raise error.WrongValueInput(
                        'Benchmark.run',
                        'parallelization',
                        "None, False, 'test', 'configs'",
                        str(parallelization)
                    )

            else:
                # No configuration list: preserve the original behavior.
                if parallelization is None or parallelization == False:
                    for a in range(len(self._algorithm)):
                        self.results.append([])

                        for n in range(self._testset.sample_size):
                            self.results[a].append(
                                _run_single_test(
                                    self._algorithm[a],
                                    self._testset.test[n]
                                )
                            )

                elif parallelization == PARALLELIZE_TESTS:
                    num_cores = multiprocessing.cpu_count()

                    for a in range(len(self._algorithm)):
                        self.results.append(
                            Parallel(n_jobs=num_cores)(
                                delayed(_run_single_test)(
                                    self._algorithm[a],
                                    self._testset.test[n]
                                )
                                for n in range(self._testset.sample_size)
                            )
                        )

                elif parallelization == "algorithms":
                    num_cores = multiprocessing.cpu_count()

                    self.results = Parallel(n_jobs=num_cores)(
                        delayed(_run_testset_algorithm)(
                            self._testset,
                            self._algorithm[a]
                        )
                        for a in range(len(self._algorithm))
                    )

                else:
                    raise error.WrongValueInput(
                        'Benchmark.run',
                        'parallelization',
                        "None, False, 'test', 'algorithms'",
                        str(parallelization)
                    )

            try:
                self.results = np.array(self.results)
            except (ValueError, TypeError):
                self.results = np.array(self.results, dtype=object)

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
            elif parallelization == "algorithms":
                num_cores = multiprocessing.cpu_count()
                for t in range(len(self._testset)):
                    output = Parallel(n_jobs=num_cores)
                    (delayed(_run_testset_algorithm)(self._testset[t],
                                                     self._algorithm[a])
                     for a in range(len(self._algorithm)))
            else:
                raise error.WrongValueInput('Benchmark.run', 'parallelization',
                                            "None, False, 'test', 'algorithms'",
                                            str(parallelization))

            try:
                self.results = np.array(self.results)
            except ValueError:
                self.results = np.array(self.results, dtype=object)

    def plot(self, indicator, axis=None, testset=None, config=None, algorithm=None,
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

        has_configs = self._config_available and not self._single_config
        has_algorithms = not self._single_algorithm
        has_testsets = not self._single_testset

        if has_configs and has_algorithms:
            # Resultados: (configs, algorithms, tests)
            pass
        elif has_configs:
            # Resultados: (configs, tests)
            pass
        elif has_algorithms:
            # Resultados: (algorithms, tests)
            pass
        else:
            # Resultados: (tests,)
            pass

        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()

    def _get_config_label(self, config):
        label_parts = []
        if 'shape' in config:
            label_parts.append(f"shape={config['shape']}")
        if 'background_permittivity' in config:
            label_parts.append(f"eps={config['background_permittivity']}")
        if 'number_measurements' in config:
            label_parts.append(f"NM={config['number_measurements']}")
        if 'number_sources' in config:
            label_parts.append(f"NS={config['number_sources']}")
        if 'noise_level' in config:
            label_parts.append(f"noise={config['noise_level']}")
        return ", ".join(label_parts)

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

        if self._config_available:
            data[CONFIGURATIONS] = self._configurations

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path)
        self.testset = data[TESTSET]
        if CONFIGURATIONS in data:
            self.configurations = data[CONFIGURATIONS]

    def __str__(self):
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
        message += 'Configurations: '
        if self._config_available and self._single_config:
            message += str(self._configurations) + '\n'
        elif self._config_available and not self._single_config:
            message += f"{len(self._configurations)} configurations\n"
        else:
            message += 'None\n'
        return message


def _run_single_test(algorithm, params):
    return api.evaluate(algorithm, params)


def _run_testset_algorithm(testset, algorithm):
    results = []
    for n in range(testset.sample_size):
        results.append(api.evaluate(algorithm, testset.test[n]))
    return results


def _run_config_testset(testset, algorithm, config):
    results = []
    for n in range(testset.sample_size):
        params = testset.test[n].copy()
        params.update(config)
        results.append(api.evaluate(algorithm, params))
    return results