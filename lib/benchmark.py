import sys
import numpy as np
from joblib import Parallel, delayed
import pickle
import multiprocessing
from matplotlib import axes
from matplotlib import pyplot as plt
from matplotlib import colors as mcl
from statsmodels import graphics

import result as rst
import inputdata as ipt
import experiment as exp
import testset as tst
import stochastic as stc
import statistics as sts
import error

TESTSET = 'testset'

PARALLELIZE_TESTSETS = 'testset'
PARALLELIZE_METHODS = 'method'
PARALLELIZE_TESTS = 'test'

LABEL_INSTANCE = 'Instance Index'
CONTRAST = 'contrast'
TOTAL_FIELD = 'total field'
ALL_TESTS = 'all'
BEST_TEST = 'best'


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
        elif type(new) is tst.TestSet:
            self._testset = new.copy()
            self._single_testset = True
            self._testset_available = True
        elif type(new) is str:
            self._testset = new
            self._single_testset = True
            self._testset_available = False
        elif type(new) is list and all(isinstance(n, str) for n in new):
            self._testset = new.copy()
            self._single_testset = False
            self._testset_available = False
        elif (type(new) is list
                and all(isinstance(n, tst.TestSet) for n in new)):
            self._testset = [new[i].copy() for i in range(len(new))]
            self._single_testset = False
            self._testset_available = True

    def __init__(self, name, method=None, discretization=None, testset=None):
        super().__init__(name, method=method, discretization=discretization)
        if (testset is not None and type(testset) is not tst.TestSet
                and type(testset) is not list):
            raise error.WrongTypeInput('Benchmark.__init__',
                                       'testset',
                                       'None, TestSet, or TestSet-list',
                                       str(type(testset)))
        self.testset = testset

    def import_testsets(self, filename, filepath=''):
        if type(filename) is not str and type(filename) is not list:
            raise error.WrongTypeInput('Benchmark.import_testsets',
                                       'filename',
                                       'str or str-list',
                                       str(type(filename)))
        if type(filepath) is not str and type(filepath) is not list:
            raise error.WrongTypeInput('Benchmark.import_testsets',
                                       'filepath',
                                       'str or str-list',
                                       str(type(filepath)))
        if type(filepath) is list and len(filepath) != len(filename):
            raise error.WrongValueInput('Benchmark.import_testsets',
                                        'filepath',
                                        "same length as 'filename'",
                                        str(len(filepath)))
        
        if type(filename) is str:
            self.testset = tst.TestSet(import_filename=filename,
                                       import_filepath=filepath)
        elif type(filepath) is str:
            self.testset = [tst.TestSet(import_filename=filename[n],
                                        import_filepath=filepath)
                            for n in range(len(filename))]
        else:
            self.testset = [tst.TestSet(import_filename=filename[n],
                                        import_filepath=filepath[n])
                            for n in range(len(filename))]
        
    def run(self, parallelization=None):
        if not self._testset_available:
            raise error.MissingAttributesError('Benchmark', 'testset')
        
        self.results = []
        if self._single_method:
            if isinstance(self.method, stc.Stochastic):
                self.method.outputmode.rule = stc.AVERAGE_CASE
        else:
            for m in range(len(self.method)):
                if isinstance(self.method[m], stc.Stochastic):
                    self.method[m].outputmode.rule = stc.AVERAGE_CASE

        if self._single_method and self._single_testset:
            if parallelization == True:
                num_cores = multiprocessing.cpu_count()
                self.results = (
                    Parallel(n_jobs=num_cores)
                    (delayed(self.method.solve)(self.testset.test[n],
                                                self.discretization,
                                                print_info=False)
                     for n in range(self.testset.sample_size))
                )
            else:
                for n in range(self.testset.sample_size):
                    self.results.append(
                        self.method.solve(self.testset.test[n],
                                          self.discretization,
                                          print_info=False)
                    )
            self.results = np.array(self.results)

        elif self._single_method and not self._single_testset:
            if parallelization is None or parallelization == False:
                for t in range(len(self.testset)):
                    self.results.append([])
                    for n in range(self.testset[t].sample_size):
                        self.results[t].append(
                            self.method.solve(self.testset[t].test[n],
                                               self.discretization,
                                               print_info=False)
                        )
            elif parallelization == PARALLELIZE_TESTS:
                for t in range(len(self.testset)):
                    num_cores = multiprocessing.cpu_count()
                    self.results.append(
                        Parallel(n_jobs=num_cores)
                        (delayed(self.method.solve)(self.testset[t].test[n],
                                                    self.discretization,
                                                    print_info=False)
                        for n in range(self.testset[t].sample_size))
                    )
            elif parallelization == PARALLELIZE_TESTSETS:
                num_cores = multiprocessing.cpu_count()
                self.results = (Parallel(n_jobs=num_cores)
                                (delayed(_run_testset)(self.testset[t],
                                                       self.method,
                                                       self.discretization)
                                 for t in range(len(self.testset))))
            else:
                raise error.WrongValueInput('Benchmark.run', 'parallelization',
                                            "None, False, 'test', or "
                                            + "'testset'",
                                            str(parallelization))

            if all(self.testset[n].sample_size == self.testset[n+1].sample_size
                   for n in range(len(self.testset)-1)):
                self.results = np.array(self.results)
            else:
                self.results = np.array(self.results, dtype=object)

        elif not self._single_method and self._single_testset:
            if parallelization is None or parallelization == False:
                for m in range(len(self.method)):
                    self.results.append([])
                    for n in range(self.testset.sample_size):
                        if self._single_discretization:
                            self.results[m].append(
                                self.method[m].solve(self.testset.test[n],
                                                     self.discretization,
                                                     print_info=False)
                            )
                        else:
                            self.results[m].append(
                                self.method[m].solve(self.testset.test[n],
                                                     self.discretization[m],
                                                     print_info=False)
                            )
            
            elif parallelization == PARALLELIZE_TESTS:
                num_cores = multiprocessing.cpu_count()
                for m in range(len(self.method)):
                    if self._single_discretization:
                        self.results.append(
                            Parallel(n_jobs=num_cores)
                            (delayed(self.method[m].solve)(
                                self.testset.test[n], self.discretization,
                                print_info=False
                            ) for n in range(self.testset.sample_size))
                        )
                    else:
                        self.results.append(
                            Parallel(n_jobs=num_cores)
                            (delayed(self.method[m].solve)(
                                self.testset.test[n], self.discretization[m],
                                print_info=False
                            ) for n in range(self.testset.sample_size))
                        )
            
            elif parallelization == PARALLELIZE_METHODS:
                num_cores = multiprocessing.cpu_count()
                if self._single_discretization:
                    self.results = (Parallel(n_jobs=num_cores)
                                    (delayed(_run_testset)(self.testset,
                                                           self.method[m],
                                                           self.discretization)
                                     for m in range(len(self.method))))
                else:
                    self.results = (Parallel(n_jobs=num_cores)
                                    (delayed(_run_testset)(
                                        self.testset, self.method[m],
                                        self.discretization[m]
                                    ) for m in range(len(self.method))))
                
            else:
                raise error.WrongValueInput('Benchmark.run', 'parallelization',
                                            "None, False, 'test', or "
                                            + "'method'",
                                            str(parallelization))
            
            self.results = np.array(self.results)
            
        else:
            if parallelization is None or parallelization == False:
                for m in range(len(self.method)):
                    self.results.append([])
                    for t in range(len(self.testset)):
                        self.results[m].append([])
                        for n in range(self.testset[t].sample_size):
                            if self._single_discretization:
                                self.results[m][t].append(
                                    self.method[m].solve(
                                        self.testset[t].test[n],
                                        self.discretization, print_info=False
                                    )
                                )
                            else:
                                self.results[m][t].append(
                                    self.method[m].solve(
                                        self.testset[t].test[n],
                                        self.discretization[m],
                                        print_info=False
                                    )
                                )
                
            elif parallelization == PARALLELIZE_TESTS:
                num_cores = multiprocessing.cpu_count()
                for m in range(len(self.method)):
                    self.results.append([])
                    for t in range(len(self.testset)):
                        if self._single_discretization:
                            self.results[m].append(
                                Parallel(n_jobs=num_cores)
                                (delayed(self.method[m].solve)(
                                    self.testset[t].test[n],
                                    self.discretization, print_info=False
                                ) for n in range(self.testset[t].sample_size))
                            )
                        else:
                            self.results[m].append(
                                Parallel(n_jobs=num_cores)
                                (delayed(self.method[m].solve)(
                                    self.testset[t].test[n],
                                    self.discretization[m], print_info=False
                                ) for n in range(self.testset[t].sample_size))
                            )
                
            elif parallelization == PARALLELIZE_TESTSETS:
                num_cores = multiprocessing.cpu_count()
                for m in range(len(self.method)):
                    self.results.append([])
                    if self._single_discretization:
                        self.results[m].append(
                            Parallel(n_jobs=num_cores)
                            (delayed(_run_testset)(self.testset[t],
                                                   self.method[m],
                                                   self.discretization)
                             for t in range(len(self.testset)))
                        )
                    else:
                        self.results[m].append(
                            Parallel(n_jobs=num_cores)
                            (delayed(_run_testset)(self.testset[t],
                                                   self.method[m],
                                                   self.discretization[m])
                             for t in range(len(self.testset)))
                        )
                
            elif parallelization == PARALLELIZE_METHODS:
                num_cores = multiprocessing.cpu_count()
                output = []
                for t in range(len(self.testset)):
                    if self._single_discretization:
                        output.append(Parallel(n_jobs=num_cores)
                                      (delayed(_run_testset)(
                                          self.testset[t], self.method[m],
                                          self.discretization
                                        )
                                       for m in range(len(self.method)))
                        )
                    else:
                        output.append(Parallel(n_jobs=num_cores)
                                      (delayed(_run_testset)(
                                          self.testset[t], self.method[m],
                                          self.discretization[m]
                                      ) for m in range(len(self.method)))
                        )
                for m in range(len(self.method)):
                    self.results.append([])
                    for t in range(len(self.testset)):
                        self.results[m].append([])
                        for n in range(self.testset[t].sample_size):
                            self.results[m][t].append(output[t][m][n])

            else:
                raise error.WrongValueInput('Benchmark.run', 'parallelization',
                                            "None, False, 'test', 'testset', "
                                            + "or 'method'",
                                            str(parallelization))

            if all(self.testset[n].sample_size == self.testset[n+1].sample_size
                   for n in range(len(self.testset)-1)):
                self.results = np.array(self.results)
            else:
                self.results = np.array(self.results, dtype=object)
                
    def save(self, file_path='', save_testset=False): 
        """Save object information."""
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

    def plot(self, indicator, axis=None, testset=None, method=None,
             yscale=None, show=False, file_name=None, file_path='',
             file_format='eps', title=None, fontsize=10):
        # Check the existence of results
        if self.results is None:
            raise error.MissingAttributesError('Benchmark', 'results')
        if indicator is None or not(type(indicator) is str
                                        or (type(indicator) is list
                                             and all(isinstance(n, str)
                                                     for n in indicator))):
            raise error.WrongTypeInput('Benchmark.plot', 'indicator',
                                       'str or str-list', str(type(indicator)))
        if not rst.check_indicator(indicator):
            raise error.WrongValueInput('Benchmark.plot', 'indicator',
                                        rst.INDICATOR_SET, indicator)

        if self._single_testset and self._single_method:

            if type(indicator) is str:
                nfig, nlines = 1, 1                    
            else:
                nfig, nlines = len(indicator), 1

            if axis is None:
                _, axis, lgd_size = rst.get_figure(nfig, nlines)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.plot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
            
            x = np.arange(1, self.results.size+1)
            if nfig == 1:
                if title is not None and type(title) is not str:
                    raise error.WrongTypeInput('Benchmark.plot', 'title',
                                               'str', str(type(title)))
                y = exp.final_value(indicator, self.results)
                rst.add_plot(axis, y, x=x, title=title,
                             xlabel=LABEL_INSTANCE,
                             ylabel=rst.indicator_label(indicator),
                             legend=None,
                             legend_fontsize=None,
                             yscale=yscale,
                             fontsize=fontsize)
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))

                for n in range(nfig):
                    y = exp.final_value(indicator[n], self.results)
                    rst.add_plot(axis[n], y, x=x, title=plotstitles[n],
                                 xlabel=LABEL_INSTANCE,
                                 ylabel=rst.indicator_label(indicator[n]),
                                 legend=None,
                                 legend_fontsize=None,
                                 yscale=yscale,
                                 fontsize=fontsize)
                    
        elif self._single_testset and not self._single_method:

            if type(indicator) is str:
                nfig = 1          
            else:
                nfig = len(indicator)
            
            if method is None:
                midx, nlines = range(len(self.method)), len(self.method)
            elif type(method) is int:
                if method >= len(self.method):
                    raise error.WrongValueInput('Benchmark.plot', 'method',
                                                str(len(self.method)), method)
                midx, nlines = method, 1
            elif type(method) is str:
                midx, nlines = self._search_method(method), 1
            elif (type(method) is list
                    and all(isinstance(n, int) for n in method)):
                if any(n >= len(self.method) for n in method):
                    raise error.WrongValueInput('Benchmark.plot', 'method',
                                                '<= %d' %len(self.method),
                                                method)
                midx, nlines = method.copy(), len(method)
            else:
                midx, nlines = self._search_method(method), len(method)
            
            if midx == False:
                raise error.WrongValueInput('Benchmark.plot', 'method',
                                            [self.method[n].alias
                                             for n in range(len(self.method))],
                                            method)

            if axis is None:
                _, axis, lgd_size = rst.get_figure(nfig, nlines)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.plot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
            x = np.arange(1, self.results.shape[1]+1)
            if nfig == 1:
                if title is not None and type(title) is not str:
                    raise error.WrongTypeInput('Benchmark.plot', 'title',
                                               'str', str(type(title)))
                y = exp.final_value(indicator, self.results[midx, :])
                rst.add_plot(axis, y.T, x=x, title=title,
                             xlabel=LABEL_INSTANCE,
                             ylabel=rst.indicator_label(indicator),
                             legend=[self.method[m].alias for m in midx],
                             legend_fontsize=lgd_size,
                             yscale=yscale,
                             fontsize=fontsize)
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))

                for n in range(nfig):
                    y = exp.final_value(indicator[n], self.results[midx, :])
                    rst.add_plot(axis[n], y.T, x=x, title=plotstitles[n],
                                 xlabel=LABEL_INSTANCE,
                                 ylabel=rst.indicator_label(indicator[n]),
                                 legend=[self.method[m].alias for m in midx],
                                 legend_fontsize=lgd_size,
                                 yscale=yscale,
                                 fontsize=fontsize)

        elif not self._single_testset and self._single_method:

            if type(indicator) is list and type(testset) is list:
                raise error.WrongValueInput('Benchmark.plot',
                                            'indicator and testset',
                                            'one must be single and other '
                                            + 'might be list', 'both are list')
            elif type(indicator) is list and type(testset) is None:
                raise error.WrongValueInput('Benchmark.plot',
                                            'indicator and testset',
                                            'one must be single and other '
                                            + 'might be None/list', 'indicator'
                                            + ' is list and testset is None')
            elif (type(indicator) is list
                    and (type(testset) is str or type(testset) is int)):
                nfig = len(indicator)
            elif (type(testset) is list
                    and (type(indicator) is str or type(indicator) is int)):
                nfig = len(testset)
            else:
                nfig = 1
            
            if testset is None:
                tidx = range(len(self.testset))
            elif type(testset) is int:
                if testset >= len(self.testset):
                    raise error.WrongValueInput('Benchmark.plot', 'testset',
                                                str(len(self.testset)),
                                                testset)
                tidx = testset
            elif type(testset) is str:
                tidx = self._search_testset(testset)
            elif (type(testset) is list
                    and all(isinstance(n, int) for n in testset)):
                if any(n >= len(self.testset) for n in testset):
                    raise error.WrongValueInput('Benchmark.plot', 'testset',
                                                '<= %d' %len(self.testset),
                                                testset)
                tidx = testset.copy()
            else:
                tidx = self._search_testset(testset)

            if tidx == False:
                raise error.WrongValueInput('Benchmark.plot', 'testset',
                                            [self.testset[n].name for n in
                                             range(len(self.testset))],
                                            testset)

            if axis is None:
                _, axis, _ = rst.get_figure(nfig)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.plot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
    
            if type(self.results) is np.ndarray and self.results.ndim == 2:
                equal_size = True
            else:
                equal_size = False

            if nfig == 1:
                if title is not None and type(title) is not str:
                    raise error.WrongTypeInput('Benchmark.plot', 'title',
                                               'str', str(type(title)))
                if equal_size:
                    x = np.arange(1, self.results.shape[1]+1)
                    y = exp.final_value(indicator, self.results[tidx, :])
                else:
                    x = range(1, len(self.results[tidx][:])+1)
                    y = exp.final_value(indicator,
                                        np.array(self.results[tidx][:]))

                if type(self.testset[0]) is str:
                    legend = [self.testset[t] for t in tidx]
                else:
                    legend = [self.testset[t].name for t in tidx]

                rst.add_plot(axis, y.T, x=x, title=title,
                             xlabel=LABEL_INSTANCE,
                             ylabel=rst.indicator_label(indicator),
                             legend=legend,
                             legend_fontsize=None,
                             yscale=yscale,
                             fontsize=fontsize)
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    if type(indicator) is str:
                        plotstitles = [self.testset[n].name for n in tidx]
                    else:
                        plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))

                if equal_size:
                        x = range(1, self.results.shape[1]+1)

                for n in range(nfig):
                    
                    if type(indicator) is str:
                        if not equal_size:
                            x = range(1, self.results[tidx[n]]+1)
                            y = exp.final_value(indicator,
                                                self.results[tidx[n]])
                        else:
                            y = exp.final_value(indicator,
                                                self.results[tidx[n], :])
                        rst.add_plot(axis[n], y, x=x, title=plotstitles[n],
                                     xlabel=LABEL_INSTANCE,
                                     ylabel=rst.indicator_label(indicator),
                                     legend=None,
                                     legend_fontsize=None,
                                     yscale=yscale,
                                     fontsize=fontsize)
                    else:
                        if not equal_size:
                            x = range(1, self.results[tidx]+1)
                            y = exp.final_value(indicator[n],
                                                self.results[tidx])
                        else:
                            y = exp.final_value(indicator[n],
                                                self.results[tidx, :])
                        rst.add_plot(axis[n], y, x=x, title=plotstitles[n],
                                     xlabel=LABEL_INSTANCE,
                                     ylabel=rst.indicator_label(indicator[n]),
                                     legend=None,
                                     legend_fontsize=None,
                                     yscale=yscale,
                                     fontsize=fontsize)

        else:
            
            if method is None:
                midx = range(len(self.method))
            elif type(method) is int:
                midx = [method]
            elif type(method) is list and isinstance(method, int):
                midx = method.copy()
            elif type(method) is str:
                midx = [self._search_method(method)]
                if midx == False:
                    raise error.WrongValueInput('Benchmark.plot', 'method',
                                                [self.method[n].alias for n in
                                                 range(len(self.method))],
                                                method)
            else:
                midx = self._search_method(method)
                if midx == False:
                    raise error.WrongValueInput('Benchmark.plot', 'method',
                                                [self.method[n].alias for n in
                                                 range(len(self.method))],
                                                method)
            
            if testset is None:
                tidx = range(len(self.testset))
            elif type(testset) is int:
                if testset >= len(self.testset):
                    raise error.WrongValueInput('Benchmark.plot', 'testset',
                                                str(len(self.testset)),
                                                testset)
                tidx = [testset]
            elif (type(testset) is list
                    and all(isinstance(n, int) for n in testset)):
                if any(n >= len(self.testset) for n in testset):
                    raise error.WrongValueInput('Benchmark.plot', 'testset',
                                                '<= %d' %len(self.testset),
                                                testset)
                tidx = testset.copy()
            elif type(testset) is str:
                tidx = [self._search_testset(testset)]
                if tidx[0] == False:
                    raise error.WrongValueInput('Benchmark.plot', 'testset',
                                                [self.testset[n].name for n in
                                                 range(len(self.testset))],
                                                testset)
            else:
                tidx = self._search_method(testset)
                if tidx == False:
                    raise error.WrongValueInput('Benchmark.plot', 'testset',
                                                [self.testset[n].name for n in
                                                 range(len(self.testset))],
                                                testset)

            nlines = len(midx)
            if type(indicator) is str:
                nfig = len(tidx)
            else:
                nfig = len(tidx)*len(indicator)
            
            if axis is None:
                _, axis, _ = rst.get_figure(nfig, nlines)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.plot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.plot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
            
            if nfig == 1:
                if type(self.results.ndim == 3):
                    x = range(1, self.results.shape[2]+1)
                else:
                    x = range(1, len(self.results[midx[0], tidx[0]])+1)
                y = exp.final_value(indicator, self.results[midx, tidx[0]])
                rst.add_plot(axis, y.T, x=x, title=title,
                             xlabel=LABEL_INSTANCE,
                             ylabel=rst.indicator_label(indicator),
                             legend=[self.method[m].alias for m in midx],
                             legend_fontsize=None,
                             yscale=yscale,
                             fontsize=fontsize)
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    if type(indicator) is str:
                        if self._testset_available:
                            plotstitles = [self.testset[n].name for n in tidx]
                        else:
                            plotstitles = [self.testset[n] for n in tidx]
                    else:
                        plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))                
                
                if type(indicator) is str:
                    ifig = 0
                    for t in tidx:
                        if self._testset_available:
                            x = range(1, self.testset[t].sample_size+1)
                        elif self.results.ndim == 3:
                            x = range(1, self.results.shape[2]+1)
                        else:
                            x = range(1, len(self.results[midx[0], t])+1)
                        y = exp.final_value(indicator, self.results[midx, t])
                        rst.add_plot(axis[ifig], y.T, x=x,
                                     xlabel=LABEL_INSTANCE,
                                     ylabel=rst.indicator_label(indicator),
                                     legend=[self.method[m].alias
                                             for m in midx],
                                     legend_fontsize=None,
                                     yscale=yscale,
                                     fontsize=fontsize,
                                     title=plotstitles[ifig])
                        ifig += 1
                else:
                    ifig = 0
                    for i in indicator:
                        for t in tidx:
                            if self._testset_available:
                                x = range(1, len(self.testset[t].sample_size)+1)
                            elif self.results.ndim == 3:
                                x = range(1, self.results.shape[2]+1)
                            else:
                                x = range(1, len(self.results[midx[0], t])+1)
                            y = exp.final_value(i, self.results[midx, t])
                            rst.add_plot(axis[ifig], y.T, x=x,
                                         xlabel=LABEL_INSTANCE,
                                         ylabel=rst.indicator_label(i),
                                         legend=[self.method[m].alias
                                                 for m in midx],
                                         legend_fontsize=None,
                                         yscale=yscale,
                                         fontsize=fontsize,
                                         title=plotstitles[ifig])
                            ifig += 1
                        
        # Show or save results
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format, transparent=False)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show:
            return axis

    def boxplot(self, indicator, axis=None, testset=None, method=None,
                yscale=None, show=False, file_name=None, file_path='',
                file_format='eps', title=None, fontsize=10, color='b',
                notch=False):
        if self.results is None:
            raise error.MissingAttributesError('Benchmark', 'results')
        if indicator is None or not(type(indicator) is str
                                        or (type(indicator) is list
                                             and all(isinstance(n, str)
                                                     for n in indicator))):
            raise error.WrongTypeInput('Benchmark.boxplot', 'indicator',
                                       'str or str-list', str(type(indicator)))
        if not rst.check_indicator(indicator):
            raise error.WrongValueInput('Benchmark.boxplot', 'indicator',
                                        rst.INDICATOR_SET, indicator)

        if self._single_testset and self._single_method:

            if type(indicator) is str:
                nfig = 1                 
            else:
                nfig = len(indicator)

            if axis is None:
                _, axis, _ = rst.get_figure(nfig)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.boxplot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
            
            x = np.arange(1, self.results.size+1)
            if nfig == 1:
                if title is not None and type(title) is not str:
                    raise error.WrongTypeInput('Benchmark.boxplot', 'title',
                                               'str', str(type(title)))
                elif title is not None:
                    tit = title
                else:
                    tit = rst.TITLES[indicator]

                data = exp.final_value(indicator, self.results)              
                rst.add_box([data], axis=axis, labels=[self.method.alias], # erro no labels
                            xlabel='Algorithms', ylabel=rst.LABELS[indicator],
                            color=color, title=tit, notch=notch,
                            fontsize=fontsize, yscale=yscale)
                
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))

                for n in range(nfig):
                    data = exp.final_value(indicator[n], self.results)                    
                    rst.add_box(data, axis=axis[n], labels=self.method.alias,
                                xlabel='Algorithms',
                                ylabel=rst.LABELS[indicator], color=color,
                                title=plotstitles[n], notch=notch,
                                fontsize=fontsize, yscale=yscale)
                    
        elif self._single_testset and not self._single_method:

            if type(indicator) is str:
                nfig = 1          
            else:
                nfig = len(indicator)
            
            if method is None:
                midx, nboxes = range(len(self.method)), len(self.method)
            elif type(method) is int:
                if method >= len(self.method):
                    raise error.WrongValueInput('Benchmark.plot', 'method',
                                                str(len(self.method)), method)
                midx, nboxes = method, 1
            elif type(method) is str:
                midx, nboxes = self._search_method(method), 1
            elif (type(method) is list
                    and all(isinstance(n, int) for n in method)):
                if any(n >= len(self.method) for n in method):
                    raise error.WrongValueInput('Benchmark.boxplot', 'method',
                                                '<= %d' %len(self.method),
                                                method)
                midx, nboxes = method.copy(), len(method)
            else:
                midx, nlines = self._search_method(method), len(method)
            
            if midx == False:
                raise error.WrongValueInput('Benchmark.boxplot', 'method',
                                            [self.method[n].alias
                                             for n in range(len(self.method))],
                                            method)

            if axis is None:
                _, axis, _ = rst.get_figure(nfig)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.boxplot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)

            x = np.arange(1, self.results.shape[1]+1)
            if nfig == 1:
                if (title is not None and type(title) is not str
                        and type(title) is not bool):
                    raise error.WrongTypeInput('Benchmark.boxplot', 'title',
                                               'str', str(type(title)))
                elif title is not None and type(title) is str:
                    tit = title
                elif title is not None and title == False:
                    tit = ''
                else:
                    tit = rst.TITLES[indicator]
                
                data = exp.final_value(indicator, self.results[midx, :])
                rst.add_box(data, axis=axis,
                            labels=[self.method[m].alias for m in midx],
                            xlabel='Algorithms',
                            ylabel=rst.indicator_label(indicator),
                            color=color, title=tit, notch=notch,
                            fontsize=fontsize, yscale=yscale)
                
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))

                for n in range(nfig):
                    data = exp.final_value(indicator[n], self.results[midx, :])
                    rst.add_box(data, axis=axis[n],
                                labels=[self.method[m].alias for m in midx],
                                xlabel='Algorithms',
                                ylabel=rst.indicator_label(indicator),
                                color=color, title=plotstitles[n], notch=notch,
                                fontsize=fontsize, yscale=yscale)

        elif not self._single_testset and self._single_method:

            if type(indicator) is list and type(testset) is list:
                raise error.WrongValueInput('Benchmark.boxplot',
                                            'indicator and testset',
                                            'one must be single and other '
                                            + 'might be list', 'both are list')
            elif type(indicator) is list and type(testset) is None:
                raise error.WrongValueInput('Benchmark.boxplot',
                                            'indicator and testset',
                                            'one must be single and other '
                                            + 'might be None/list', 'indicator'
                                            + ' is list and testset is None')
            elif (type(indicator) is list
                    and (type(testset) is str or type(testset) is int)):
                nfig = len(indicator)
            else:
                nfig = 1
            
            if testset is None:
                tidx = range(len(self.testset))
            elif type(testset) is int:
                if testset >= len(self.testset):
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                str(len(self.testset)),
                                                testset)
                tidx = testset
            elif type(testset) is str:
                tidx = self._search_testset(testset)
            elif (type(testset) is list
                    and all(isinstance(n, int) for n in testset)):
                if any(n >= len(self.testset) for n in testset):
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                '<= %d' %len(self.testset),
                                                testset)
                tidx = testset.copy()
            else:
                tidx = self._search_testset(testset)

            if tidx == False:
                raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                            [self.testset[n].name for n in
                                             range(len(self.testset))],
                                            testset)

            if axis is None:
                _, axis, _ = rst.get_figure(nfig)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.boxplot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
    
            if type(self.results) is np.ndarray and self.results.ndim == 2:
                equal_size = True
            else:
                equal_size = False

            if nfig == 1:
                if (title is not None and type(title) is not str
                        and type(title) is not bool):
                    raise error.WrongTypeInput('Benchmark.plot', 'title',
                                               'str', str(type(title)))
                elif title is None:
                    tit = rst.TITLES[indicator]
                elif title == False:
                    tit = ''
                else:
                    tit = title
                if equal_size:
                    x = np.arange(1, self.results.shape[1]+1)
                    y = exp.final_value(indicator, self.results[tidx, :])
                else:
                    x = range(1, len(self.results[tidx][:])+1)
                    y = exp.final_value(indicator,
                                        np.array(self.results[tidx][:]))
                if type(self.testset[0]) is str:
                    labels = [self.testset[t] for t in tidx]
                else:
                    labels = [self.testset[t].name for t in tidx],

                rst.add_box(y, axis=axis,
                            labels=labels,
                            xlabel='Test sets',
                            ylabel=rst.indicator_label(indicator),
                            color=color, title=tit, notch=notch,
                            fontsize=fontsize, yscale=yscale)
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.boxplot',
                                                   'title', 'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.boxplot',
                                                    'title', '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.boxplot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))

                if equal_size:
                    x = range(1, self.results.shape[1]+1)

                for n in range(nfig):
                    if not equal_size:
                        x = range(1, self.results[tidx]+1)
                        y = exp.final_value(indicator[n], self.results[tidx])
                    else:
                        y = exp.final_value(indicator[n],
                                            self.results[tidx, :])

                    rst.add_box(y, axis=axis[n],
                                labels=[self.testset[t].name for t in tidx],
                                xlabel='Test sets',
                                ylabel=rst.indicator_label(indicator),
                                color=color, title=plotstitles[n], notch=notch,
                                fontsize=fontsize, yscale=yscale)

        else:
            
            if method is None:
                midx = range(len(self.method))
            elif type(method) is int:
                midx = [method]
            elif type(method) is list and isinstance(method, int):
                midx = method.copy()
            elif type(method) is str:
                midx = [self._search_method(method)]
                if midx == False:
                    raise error.WrongValueInput('Benchmark.boxplot', 'method',
                                                [self.method[n].alias for n in
                                                 range(len(self.method))],
                                                method)
            else:
                midx = self._search_method(method)
                if midx == False:
                    raise error.WrongValueInput('Benchmark.boxplot', 'method',
                                                [self.method[n].alias for n in
                                                 range(len(self.method))],
                                                method)
            
            if testset is None:
                tidx = range(len(self.testset))
            elif type(testset) is int:
                if testset >= len(self.testset):
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                str(len(self.testset)),
                                                testset)
                tidx = [testset]
            elif (type(testset) is list
                    and all(isinstance(n, int) for n in testset)):
                if any(n >= len(self.testset) for n in testset):
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                '<= %d' %len(self.testset),
                                                testset)
                tidx = testset.copy()
            elif type(testset) is str:
                tidx = [self._search_testset(testset)]
                if tidx[0] == False:
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                [self.testset[n].name for n in
                                                 range(len(self.testset))],
                                                testset)
            else:
                tidx = self._search_method(testset)
                if tidx == False:
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                [self.testset[n].name for n in
                                                 range(len(self.testset))],
                                                testset)

            nboxes = len(midx)
            if nboxes == 1:
                if type(indicator) is str:
                    nfig = 1
                else:
                    nfig = len(indicator)
            elif type(indicator) is str:
                nfig = len(tidx)
            else:
                nfig = len(tidx)*len(indicator)
            
            if axis is None:
                _, ax, _ = rst.get_figure(nfig)
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.boxplot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
            elif nfig == 1 and isinstance(axis, axes.Axes):
                ax = [axis]
            
            if nfig == 1:
                if len(tidx) == 1:
                    data = exp.final_value(indicator,
                                           self.results[midx, tidx[0]])
                    names = [self.method[m].alias for m in midx]
                    xlabel = 'Algorithms'
                    if title is None:
                        tit = (self.testset[tidx[0]].name + ' - '
                               + rst.TITLES[indicator])
                else:
                    data = exp.final_value(indicator,
                                           self.results[midx[0], tidx])
                    names = [self.testset[t].name for t in tidx]
                    xlabel = 'Test sets'
                    if title is None:
                        tit = (self.method[midx[0]].alias + ' - '
                               + rst.TITLES[indicator])
                
                if title == False:
                    tit = ''
                elif title is not None:
                    tit = title

                rst.add_box(data, axis=ax[0], labels=names, xlabel=xlabel,
                            ylabel=rst.indicator_label(indicator), color=color,
                            title=tit, notch=notch, fontsize=fontsize,
                            yscale=yscale)
                
            else:
                if title == False:
                    tit = [None for n in range(nfig)]
                elif type(title) is str:
                    tit = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        tit = title.copy()
                elif title is not None and title != True:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))                
                
                if type(indicator) is str:
                    ifig = 0
                    if title is None or title == True:
                        if self._testset_available:
                            tit = [self.testset[n].name for n in tidx]
                        else:
                            tit = [self.testset[n] for n in tidx]
                            
                    for t in tidx:
                        y = exp.final_value(indicator, self.results[midx, t])
                        rst.add_box(y, axis=axis[ifig],
                                    labels=[self.method[m].alias
                                            for m in midx],
                                    xlabel='Algorithms',
                                    ylabel=rst.indicator_label(indicator),
                                    color=color, title=tit[ifig], notch=notch,
                                    fontsize=fontsize, yscale=yscale)
                        ifig += 1
                else:
                        
                    ifig = 0
                    for i in indicator:
                        for t in tidx:
                            if title is None or title == True:
                                tit = self.testset[t].name + rst.TITLES[i]
                            y = exp.final_value(i, self.results[midx, t])
                            rst.add_box(y, axis=axis[ifig],
                                        labels=[self.method[m].alias
                                                for m in midx],
                                        xlabel='Algorithms',
                                        ylabel=rst.indicator_label(indicator),
                                        color=color, title=tit[ifig],
                                        notch=notch, fontsize=fontsize,
                                        yscale=yscale)
                            ifig += 1
                        
        # Show or save results
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format, transparent=False)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show:
            return axis

    def violinplot(self, indicator, axis=None, testset=None, method=None,
                yscale=None, show=False, file_name=None, file_path='',
                file_format='eps', title=None, fontsize=10, color='b',
                notch=False):
        if self.results is None:
            raise error.MissingAttributesError('Benchmark', 'results')
        if indicator is None or not(type(indicator) is str
                                        or (type(indicator) is list
                                             and all(isinstance(n, str)
                                                     for n in indicator))):
            raise error.WrongTypeInput('Benchmark.boxplot', 'indicator',
                                       'str or str-list', str(type(indicator)))
        if not rst.check_indicator(indicator):
            raise error.WrongValueInput('Benchmark.boxplot', 'indicator',
                                        rst.INDICATOR_SET, indicator)

        if self._single_testset and self._single_method:

            if type(indicator) is str:
                nfig = 1                 
            else:
                nfig = len(indicator)

            if axis is None:
                _, axis, _ = rst.get_figure(nfig)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.boxplot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
            
            x = np.arange(1, self.results.size+1)
            if nfig == 1:
                if title is not None and type(title) is not str:
                    raise error.WrongTypeInput('Benchmark.boxplot', 'title',
                                               'str', str(type(title)))
                elif title is not None:
                    tit = title
                else:
                    tit = rst.TITLES[indicator]

                data = exp.final_value(indicator, self.results)                
                rst.add_violin([data], axis=axis, labels=[self.method.alias],
                               xlabel='Algorithms',
                               ylabel=rst.LABELS[indicator], color=color,
                               title=tit, fontsize=fontsize,
                               yscale=yscale)
                
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))

                for n in range(nfig):
                    data = exp.final_value(indicator[n], self.results)                    
                    rst.add_violin(data, axis=axis[n],
                                   labels=self.method.alias,
                                   xlabel='Algorithms',
                                   ylabel=rst.LABELS[indicator], color=color,
                                   title=plotstitles[n], fontsize=fontsize,
                                   yscale=yscale)

        elif self._single_testset and not self._single_method:

            if type(indicator) is str:
                nfig = 1          
            else:
                nfig = len(indicator)
            
            if method is None:
                midx, nboxes = range(len(self.method)), len(self.method)
            elif type(method) is int:
                if method >= len(self.method):
                    raise error.WrongValueInput('Benchmark.plot', 'method',
                                                str(len(self.method)), method)
                midx, nboxes = method, 1
            elif type(method) is str:
                midx, nboxes = self._search_method(method), 1
            elif (type(method) is list
                    and all(isinstance(n, int) for n in method)):
                if any(n >= len(self.method) for n in method):
                    raise error.WrongValueInput('Benchmark.boxplot', 'method',
                                                '<= %d' %len(self.method),
                                                method)
                midx, nboxes = method.copy(), len(method)
            else:
                midx, nlines = self._search_method(method), len(method)
            
            if midx == False:
                raise error.WrongValueInput('Benchmark.boxplot', 'method',
                                            [self.method[n].alias
                                             for n in range(len(self.method))],
                                            method)

            if axis is None:
                _, axis, _ = rst.get_figure(nfig)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.boxplot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)

            x = np.arange(1, self.results.shape[1]+1)
            if nfig == 1:
                if (title is not None and type(title) is not str
                        and type(title) is not bool):
                    raise error.WrongTypeInput('Benchmark.boxplot', 'title',
                                               'str', str(type(title)))
                elif title is not None and type(title) is str:
                    tit = title
                elif title is not None and title == False:
                    tit = ''
                else:
                    tit = rst.TITLES[indicator]
                
                data = exp.final_value(indicator, self.results[midx, :])
                rst.add_violin(data, axis=axis,
                               labels=[self.method[m].alias for m in midx],
                               xlabel='Algorithms',
                               ylabel=rst.indicator_label(indicator),
                               color=color, title=tit, fontsize=fontsize,
                               yscale=yscale)
                
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))

                for n in range(nfig):
                    data = exp.final_value(indicator[n], self.results[midx, :])
                    rst.add_violin(data, axis=axis[n],
                                   labels=[self.method[m].alias for m in midx],
                                   xlabel='Algorithms',
                                   ylabel=rst.indicator_label(indicator),
                                   color=color, title=plotstitles[n],
                                   fontsize=fontsize, yscale=yscale)

        elif not self._single_testset and self._single_method:

            if type(indicator) is list and type(testset) is list:
                raise error.WrongValueInput('Benchmark.boxplot',
                                            'indicator and testset',
                                            'one must be single and other '
                                            + 'might be list', 'both are list')
            elif type(indicator) is list and type(testset) is None:
                raise error.WrongValueInput('Benchmark.boxplot',
                                            'indicator and testset',
                                            'one must be single and other '
                                            + 'might be None/list', 'indicator'
                                            + ' is list and testset is None')
            elif (type(indicator) is list
                    and (type(testset) is str or type(testset) is int)):
                nfig = len(indicator)
            else:
                nfig = 1
            
            if testset is None:
                tidx = range(len(self.testset))
            elif type(testset) is int:
                if testset >= len(self.testset):
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                str(len(self.testset)),
                                                testset)
                tidx = testset
            elif type(testset) is str:
                tidx = self._search_testset(testset)
            elif (type(testset) is list
                    and all(isinstance(n, int) for n in testset)):
                if any(n >= len(self.testset) for n in testset):
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                '<= %d' %len(self.testset),
                                                testset)
                tidx = testset.copy()
            else:
                tidx = self._search_testset(testset)

            if tidx == False:
                raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                            [self.testset[n].name for n in
                                             range(len(self.testset))],
                                            testset)

            if axis is None:
                _, axis, _ = rst.get_figure(nfig)
                if nfig == 1:
                    axis = axis[0]
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.boxplot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
            if type(self.results) is np.ndarray and self.results.ndim == 2:
                equal_size = True
            else:
                equal_size = False
            if type(self.testset[0]) is str:
                labels=[self.testset[t] for t in tidx]
            else:
                labels=[self.testset[t].name for t in tidx]

            if nfig == 1:
                if (title is not None and type(title) is not str
                        and type(title) is not bool):
                    raise error.WrongTypeInput('Benchmark.plot', 'title',
                                               'str', str(type(title)))
                elif title is None:
                    tit = rst.TITLES[indicator]
                elif title == False:
                    tit = ''
                else:
                    tit = title
                if equal_size:
                    x = np.arange(1, self.results.shape[1]+1)
                    y = exp.final_value(indicator, self.results[tidx, :])
                else:
                    x = range(1, len(self.results[tidx][:])+1)
                    y = exp.final_value(indicator,
                                        np.array(self.results[tidx][:]))
                rst.add_violin(y, axis=axis,
                               labels=labels,
                               xlabel='Test sets',
                               ylabel=rst.indicator_label(indicator),
                               color=color, title=tit, fontsize=fontsize,
                               yscale=yscale)
            else:
                if title == False or title == None:
                    plotstitles = [None for n in range(nfig)]
                elif title == True:
                    plotstitles = [rst.TITLES[n] for n in indicator]
                elif type(title) is str:
                    plotstitles = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.boxplot',
                                                   'title', 'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.boxplot',
                                                    'title', '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        plotstitles = title.copy()
                else:
                    raise error.WrongValueInput('Benchmark.boxplot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))

                if equal_size:
                    x = range(1, self.results.shape[1]+1)

                for n in range(nfig):
                    if not equal_size:
                        x = range(1, self.results[tidx]+1)
                        y = exp.final_value(indicator[n], self.results[tidx])
                    else:
                        y = exp.final_value(indicator[n],
                                            self.results[tidx, :])
                    rst.add_violin(y, axis=axis[n],
                                   labels=labels,
                                   xlabel='Test sets',
                                   ylabel=rst.indicator_label(indicator),
                                   color=color, title=plotstitles[n],
                                   fontsize=fontsize, yscale=yscale)

        else:
            
            if method is None:
                midx = range(len(self.method))
            elif type(method) is int:
                midx = [method]
            elif type(method) is list and isinstance(method, int):
                midx = method.copy()
            elif type(method) is str:
                midx = [self._search_method(method)]
                if midx == False:
                    raise error.WrongValueInput('Benchmark.boxplot', 'method',
                                                [self.method[n].alias for n in
                                                 range(len(self.method))],
                                                method)
            else:
                midx = self._search_method(method)
                if midx == False:
                    raise error.WrongValueInput('Benchmark.boxplot', 'method',
                                                [self.method[n].alias for n in
                                                 range(len(self.method))],
                                                method)
            
            if testset is None:
                tidx = range(len(self.testset))
            elif type(testset) is int:
                if testset >= len(self.testset):
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                str(len(self.testset)),
                                                testset)
                tidx = [testset]
            elif (type(testset) is list
                    and all(isinstance(n, int) for n in testset)):
                if any(n >= len(self.testset) for n in testset):
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                '<= %d' %len(self.testset),
                                                testset)
                tidx = testset.copy()
            elif type(testset) is str:
                tidx = [self._search_testset(testset)]
                if tidx[0] == False:
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                [self.testset[n].name for n in
                                                 range(len(self.testset))],
                                                testset)
            else:
                tidx = self._search_method(testset)
                if tidx == False:
                    raise error.WrongValueInput('Benchmark.boxplot', 'testset',
                                                [self.testset[n].name for n in
                                                 range(len(self.testset))],
                                                testset)

            nboxes = len(midx)
            if nboxes == 1:
                if type(indicator) is str:
                    nfig = 1
                else:
                    nfig = len(indicator)
            elif type(indicator) is str:
                nfig = len(tidx)
            else:
                nfig = len(tidx)*len(indicator)
            
            if axis is None:
                _, ax, _ = rst.get_figure(nfig)
            elif (not isinstance(axis, axes.Axes)
                    and not type(axis, np.ndarray)):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes or Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig == 1 and type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes', str(type(axis)))
            elif nfig > 1 and not type(axis, np.ndarray):
                raise error.WrongTypeInput('Benchmark.boxplot', 'axis',
                                           'Axes-numpy.ndarray',
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('Benchmark.boxplot', 'axis',
                                           '%d-numpy.ndarray' %nfig,
                                           '%d-numpy.ndarray' %axis.size)
            elif nfig == 1 and isinstance(axis, axes.Axes):
                ax = [axis]
            
            if nfig == 1:
                if len(tidx) == 1:
                    data = exp.final_value(indicator,
                                           self.results[midx, tidx[0]])
                    names = [self.method[m].alias for m in midx]
                    xlabel = 'Algorithms'
                    if title is None:
                        tit = (self.testset[tidx[0]].name + ' - '
                               + rst.TITLES[indicator])
                else:
                    data = exp.final_value(indicator,
                                           self.results[midx[0], tidx])
                    names = [self.testset[t].name for t in tidx]
                    xlabel = 'Test sets'
                    if title is None:
                        tit = (self.method[midx[0]].alias + ' - '
                               + rst.TITLES[indicator])
                
                if title == False:
                    tit = ''
                elif title is not None:
                    tit = title

                rst.add_violin(data, axis=ax[0], labels=names, xlabel=xlabel,
                               ylabel=rst.indicator_label(indicator),
                               color=color, title=tit, fontsize=fontsize,
                               yscale=yscale)

            else:
                if title == False:
                    tit = [None for n in range(nfig)]
                elif type(title) is str:
                    tit = [title for n in range(nfig)]
                elif type(title) is list:
                    if not all(isinstance(n, str) for n in title):
                        raise error.WrongTypeInput('Benchmark.plot', 'title',
                                                   'str-list',
                                                   str(type(title)))
                    elif nfig != len(title):
                        raise error.WrongValueInput('Benchmark.plot', 'title',
                                                    '%d-list' %nfig,
                                                    '%d-list' %axis.size)
                    else:
                        tit = title.copy()
                elif title is not None and title != True:
                    raise error.WrongValueInput('Benchmark.plot', 'title',
                                                'None, False, True, str, '
                                                + 'str-list', str(type(title)))                
                
                if type(indicator) is str:
                    ifig = 0
                    if title is None or title == True:
                        if self._testset_available:
                            tit = [self.testset[n].name for n in tidx]
                        else:
                            tit = [self.testset[n] for n in tidx]
                            
                    for t in tidx:
                        y = exp.final_value(indicator, self.results[midx, t])
                        rst.add_violin(y, axis=ax[ifig],
                                       labels=[self.method[m].alias
                                               for m in midx],
                                       xlabel='Algorithms',
                                       ylabel=rst.indicator_label(indicator),
                                       color=color, title=tit[ifig],
                                       fontsize=fontsize, yscale=yscale)
                        ifig += 1
                else:
                        
                    ifig = 0
                    for i in indicator:
                        for t in tidx:
                            if title is None or title == True:
                                if self._testset_available:
                                    tit = self.testset[t].name + rst.TITLES[i]
                                else:
                                    tit = self.testset[t] + rst.TITLES[i]
                            y = exp.final_value(i, self.results[midx, t])
                            rst.add_violin(y, axis=axis[ifig],
                                           labels=[self.method[m].alias
                                                   for m in midx],
                                           xlabel='Algorithms',
                                           ylabel=rst.indicator_label(indicator),
                                           color=color, title=tit[ifig],
                                           fontsize=fontsize, yscale=yscale)
                            ifig += 1
                        
        # Show or save results
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format, transparent=False)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show:
            return axis

    def evolution(self, indicator, axis=None, method=None, testset=None,
                  labels=None, xlabel=None, title=None, fontsize=10,
                  yscale=None, variable='testset', show=False, file_name=None,
                  file_path='', file_format='eps'):
        if self._single_method and self._single_testset:
            raise error.Error('Benchmark.evolution: this function is not '
                              + 'available for benchmarks with single method '
                              + 'and test set.')
        if self._single_method:
            if testset is None:
                tidx = range(len(self.testset))
            else:
                tidx = self._search_testset(testset)
            if len(tidx) == 1:
                raise error.Error('Benchmark.evolution: the number of test'
                                  + ' sets should be greater than 1.')
            if type(indicator) is str:
                nfig = 1
            else:
                nfig = len(indicator)
            if axis is None:
                fig, axis, _ = rst.get_figure(nfig)
                if nfig == 1:
                    axis = axis[0]
            else:
                if len(axis) != nfig:
                    raise error.Error('Benchmark.evolution: the number of axis'
                                      + ' objects(%d)' % len(axis) + ' must be'
                                      + ' equal to the number of figures (%d).'
                                      % nfig)
                fig = plt.gcf()
            if type(indicator) is str:
                data = exp.final_value(indicator, self.results[tidx])
                if labels is None:
                    if self._testset_available:
                        labels = [self.testset[t].name for t in tidx]
                    else:
                        labels = [self.testset[t] for t in tidx]
                elif type(labels) is list:
                    if len(labels) != tidx:
                        raise error.Error('Benchmark.evolution: the number of'
                                          + ' labels should be equal to the '
                                          + 'number of test sets.')  
                if xlabel is None:
                    xlabel = 'Test Sets'
                if title is None:
                    title = rst.TITLES[indicator]
                elif title == False:
                    title = ''
                color = 'b'
                rst.add_box(data, axis=axis, labels=labels, meanline=True,
                            xlabel=xlabel, ylabel=rst.LABELS[indicator],
                            color=color, title=title, notch=False,
                            fontsize=fontsize, yscale=yscale)
            else:
                ifig = 0
                if labels is None:
                    labels = [self.testset[t].name for t in tidx]
                elif type(labels) is list:
                        if len(labels) != len(tidx):
                            raise error.Error('Benchmark.evolution: the number'
                                              + ' of labels should be equal to'
                                              + ' the number of test sets.') 
                if xlabel is None:
                    xlabel = 'Test Sets'
                color = 'b'
                for ind in indicator:
                    data = exp.final_value(indicator, self.results[tidx])
                    if title is None:
                        tit = rst.TITLES[ind]
                    elif title == False:
                        tit = ''
                    elif type(title) is list:
                        tit = title[ifig]
                    elif type(title) is str:
                        tit = title
                    rst.add_box(data, axis=axis[ifig], labels=labels,
                                meanline=True, xlabel=xlabel,
                                ylabel=rst.LABELS[indicator],
                                color=color, title=tit, notch=False,
                                fontsize=fontsize, yscale=yscale)
                    ifig += 1
        elif self._single_testset:
            if method is None:
                midx = range(len(self.method))
            elif type(method) is not list:
                raise error.WrongTypeInput('Benchmark.evolution', 'method',
                                           'list', str(type(method)))
            elif all(type(m) is int for m in method):
                midx = method
            else:
                midx = self._search_method(method)
            if type(indicator) is str:
                nfig = 1
            else:
                nfig = len(indicator)
            if axis is None:
                fig, axis, _ = rst.get_figure(nfig)
                if nfig == 1:
                    axis = axis[0]
            else:
                if len(axis) != nfig:
                    raise error.Error('Benchmark.evolution: the number of axis'
                                      + ' objects(%d)' % len(axis) + ' must be'
                                      + ' equal to the number of figures (%d).'
                                      % nfig)
                fig = plt.gcf()
            if type(indicator) is str:
                data = exp.final_value(indicator, self.results[midx])
                if labels is None:
                    labels = [self.method[m].alias for m in midx]
                elif type(labels) is list:
                    if len(labels) != midx:
                        raise error.Error('Benchmark.evolution: the number of'
                                          + ' labels should be equal to the '
                                          + 'number of methods.')  
                if xlabel is None:
                    xlabel = 'Methods'
                if title is None:
                    title = rst.TITLES[indicator]
                elif title == False:
                    title = ''
                color = 'b'
                rst.add_box(data, axis=axis, labels=labels, meanline=True,
                            xlabel=xlabel, ylabel=rst.LABELS[indicator],
                            color=color, title=title, notch=False,
                            fontsize=fontsize, yscale=yscale)
            else:
                ifig = 0
                if labels is None:
                    labels = [self.method[m].alias for m in midx]
                elif type(labels) is list:
                        if len(labels) != len(midx):
                            raise error.Error('Benchmark.evolution: the number'
                                              + ' of labels should be equal to'
                                              + ' the number of methods.') 
                if xlabel is None:
                    xlabel = 'Methods'
                color = 'b'
                for ind in indicator:
                    data = exp.final_value(indicator, self.results[midx])
                    if title is None:
                        tit = rst.TITLES[ind]
                    elif title == False:
                        tit = ''
                    elif type(title) is list:
                        tit = title[ifig]
                    elif type(title) is str:
                        tit = title
                    rst.add_box(data, axis=axis[ifig], labels=labels,
                                meanline=True, xlabel=xlabel,
                                ylabel=rst.LABELS[indicator],
                                color=color, title=tit, notch=False,
                                fontsize=fontsize, yscale=yscale)
                    ifig += 1
        else:
            if method is None:
                midx = range(len(self.method))
            elif type(method) is int:
                midx = [method]
            else:
                midx = self._search_method(method)
            if testset is None:
                tidx = range(len(self.testset))
            elif type(testset) is int:
                tidx = [testset]
            else:
                tidx = self._search_testset(testset)
            if type(indicator) is str:
                nfig = 1
                inds = [indicator]
            else:
                nfig = len(indicator)
                inds = indicator
            if variable == 'testset':
                nlines = len(midx)
            else:
                nlines = len(tidx)
            if axis is None:
                fig, axis, lgd_size = rst.get_figure(nfig, nlines)
            else:
                if nfig != len(axis):
                    raise error.Error('Benchmark.evolution: the number of axis'
                                      + ' objects(%d)' % len(axis) + ' must be'
                                      + ' equal to the number of figures (%d).'
                                      % nfig)
                fig = plt.gcf()
                _, _, lgd_size = rst.get_figure(nfig, nlines)
            ifig = 0
            color = list(mcl.TABLEAU_COLORS.keys())
            for ind in inds:
                if title is None or title == True:
                    tit = rst.TITLES[ind]
                elif title == False:
                    tit = ''
                elif type(title) is list:
                    tit = title[ifig]
                else:
                    tit = title
                if variable == 'testset':
                    if labels is None:
                        if self._testset_available:
                            labels = [self.testset[t].name for t in tidx]
                        else:
                            labels = [self.testset[t] for t in tidx]
                    elif len(labels) != len(tidx):
                        raise error.Error('Benchmark.evolution: the number'
                                          + ' of labels (%d)' % len(labels)
                                          + ' should be equal to the number '
                                          + 'of test sets (%d).' % len(tidx))
                    if xlabel is None:
                        xlabel = 'Test Sets'
                    icol = 0
                    for m in midx:
                        if self.results.ndim == 3:
                            data = exp.final_value(ind,
                                                   self.results[m, tidx, :])
                        else:
                            data = exp.final_value(ind, self.results[m, tidx])
                        rst.add_box(data, axis=axis[ifig], labels=labels,
                                    xlabel=xlabel, ylabel=rst.LABELS[ind],
                                    color=color[icol], title=tit, notch=False,
                                    fontsize=fontsize, yscale=yscale,
                                    legend_fontsize=lgd_size)
                        icol += 1
                else:
                    if labels is None:
                        labels = [self.method[m].alias for m in midx]
                    elif len(labels) != len(midx):
                        raise error.Error('Benchmark.evolution: the number'
                                          + ' of labels (%d)' % len(labels)
                                          + ' should be equal to the number '
                                          + 'of methods (%d).' % len(midx))
                    if xlabel is None:
                        xlabel = 'Methods'
                    icol = 0
                    for t in tidx:
                        if self.results.ndim == 3:
                            data = exp.final_value(ind,
                                                   self.results[midx, t, :])
                        else:
                            data = exp.final_value(ind, self.results[midx, t])
                        rst.add_box(data, axis=axis[ifig], labels=labels,
                                    xlabel=xlabel, ylabel=rst.LABELS[ind],
                                    color=color[icol], title=tit, notch=False,
                                    fontsize=fontsize, yscale=yscale,
                                    legend_fontsize=lgd_size)
                        icol += 1
                ifig += 1
                    
        # Show or save results
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format, transparent=False)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show:
            return axis          

    def reconstruction(self, image=CONTRAST, mode=None, indicator=None,
                       axis=None, testset=None, method=None, test=None,
                       show=False, file_name=None, file_path='',
                       file_format='eps', title=None, fontsize=10,
                       source=None):
        # Check result
        if self.results is None:
            raise error.MissingAttributesError('Benchmark', 'results')

        # Set method
        if not self._single_method:
            if method is None:
                method = range(len(self.method))
            elif type(method) is int:
                method = [method]
            elif type(method) is str:
                method = self._search_method(method)
            elif type(method) is list:
                if any([type(m) is str for m in method]):
                    method = self._search_method(method)
                else:
                    pass

        # Set test set
        if not self._single_testset:
            if testset is None:
                testset = range(len(self.testset))
            elif type(testset) is int:
                testset = [testset]
            elif type(testset) is str:
                testset = self._search_testset(testset)
            elif type(testset) is list:
                if any([type(t) is str for t in testset]):
                    testset = self._search_testset(testset)
                else:
                    pass

        # Set test
        if test is None:
            if self._single_method and self._single_testset:
                test = range(self.results.size)
            elif not self._single_method and self._single_testset:
                test = range(self.results.shape[1])
            elif self._single_method and not self._single_testset:
                test = []
                for t in testset:
                    if self.results.ndim == 1:
                        test.append(range(len(self.results[t])))
                    else:
                        test.append(range(self.results.shape[1]))
            else:
                test = []
                for t in testset:
                    if self.results.ndim == 2:
                        test.append(range(len(self.results[0, t])))
                    elif self.results.ndim == 3:
                        test.append(range(self.results.shape[2]))
        elif type(test) is int:
            if self._single_testset:
                test = [test]
            else:
                test = [test for i in range(len(testset))]
        elif type(test) is list and not self._single_testset:
            test = [test for i in range(len(testset))]

        # Check image
        if image != CONTRAST and image != TOTAL_FIELD:
            raise error.WrongValueInput('Benchmark.reconstruction', 'image',
                                        "'" + CONTRAST + "' or '" + TOTAL_FIELD
                                        + "'", str(image))

        # Check mode
        if mode is not None and mode != ALL_TESTS and mode != BEST_TEST:
            raise error.WrongValueInput('Benchmark.reconstruction', 'mode',
                                        "'" + ALL_TESTS + "' or '" + BEST_TEST
                                        + "'", str(mode))

        # Check total field information and set sources
        if image == TOTAL_FIELD:
            FLAG_MISSING = False
            if self._single_method and self._single_testset:
                if self.results[0].total_field is None:
                    FLAG_MISSING = True
                else:
                    NS = self.results[0].configuration.NS
            elif self._single_method and not self._single_testset:
                if (self.results.ndim == 1
                        and self.results[0][0].total_field is None):
                    FLAG_MISSING = True
                elif self.results.ndim == 1:
                    NS = self.results[0][0].configuration.NS
                elif (self.results.ndim == 2
                        and self.results[0, 0].total_field is None):
                    FLAG_MISSING = True
                else:
                    NS = self.results[0, 0].configuration.NS
            elif not self._single_method and self._single_testset:
                if self.results[0, 0].total_field is None:
                    FLAG_MISSING = True
                else:
                    NS = self.results[0, 0].configuration.NS
            else:
                if (self.results.ndim == 2
                        and self.results[0, 0][0].total_field is None):
                    FLAG_MISSING = True
                elif self.results.ndim == 2:
                    NS = self.results[0, 0][0].configuration.NS
                elif (self.results.ndim == 3
                        and self.results[0, 0, 0].total_field is None):
                    FLAG_MISSING = True
                else:
                    NS = self.results[0, 0, 0].configuration.NS
            if FLAG_MISSING:
                raise error.MissingAttributesError('Result', 'et')
            if source is None:
                source = range(NS)
            elif type(source) is int:
                source = [source]

        # Check indicator
        if mode == BEST_TEST and indicator is None:
            raise error.MissingInputError('Benchmark.reconstruction',
                                          'indicator')
        elif mode == BEST_TEST and type(indicator) is not str:
            raise error.WrongTypeInput('Benchmark.reconstruction', 'indicator',
                                       'str', str(type(indicator)))
        elif mode == BEST_TEST and not rst.check_indicator(indicator):
            raise error.WrongValueInput('Benchmark.reconstruction',
                                        'indicator',
                                        str(rst.INDICATOR_SET), indicator)

        # Calculate the number of figures
        if self._testset_available:
            NFIG = 2
        else:
            NFIG = 1
        if mode is None or mode == ALL_TESTS:
            NFIG = NFIG*len(test)
        elif mode == BEST_TEST:
            pass
        if image == CONTRAST:
            pass
        elif image == TOTAL_FIELD:
            NFIG = NFIG*len(source)
        if self._single_method and self._single_testset:
            pass
        if not self._single_testset:
            NFIG = NFIG*len(testset)
        if not self._single_method:
            NFIG = NFIG*len(method)

        # Set axis
        if axis is None:
            fig, ax, _ = rst.get_figure(NFIG)
        else:
            if NFIG == 1 and isinstance(axis, plt.Axes):
                ax = [axis]
            elif NFIG > 1 and isinstance(axis, plt.Axes):
                raise error.WrongTypeInput('Benchmark.reconstruction', 'axis',
                                           'numpy.ndarray of length %d' % NFIG,
                                           'matplotlib.pyplot.Axes')
            elif NFIG > 1 and axis.size != NFIG:
                raise error.WrongTypeInput('Benchmark.reconstruction', 'axis',
                                           'numpy.ndarray of length %d' % NFIG,
                                           'length %d' % axis.size)
            else:
                fig, ax = plt.gcf(), axis

        # Find best
        if mode == BEST_TEST:
            if self._single_method and self._single_testset:
                data = exp.final_value(indicator, self.results)
                best = np.argmin(data)
            elif not self._single_method and self._single_testset:
                best = []
                for m in method:
                    data = exp.final_value(indicator, self.results[m, :])
                    best.append(np.argmin(data))
            elif self._single_method and not self._single_testset:
                best = []
                for t in testset:
                    data = exp.final_value(indicator, self.results[t])
                    best.append(np.argmin(data))
            else:
                best = []
                for i in range(len(testset)):
                    t = testset[i]
                    best.append([])
                    for m in method:
                        data = exp.final_value(indicator, self.results[m, t])
                        best[i].append(np.argmin(data))

        # Set title
        if type(title) is bool and title is False:
            tit = [None for n in range(NFIG)]
        elif type(title) is str:
            tit = [title for n in range(NFIG)]
        elif type(title) is list:
            if len(title) != NFIG:
                raise error.WrongTypeInput('Benchmark.reconstruction', 'title',
                                           'list of length %d' % NFIG,
                                           'length %d' % len(title))
            else:
                tit = title
        elif title is None or type(title) is bool and title is True:
            tit = []
            if self._single_method and self._single_testset:
                if mode == ALL_TESTS or mode is None:
                    if image == CONTRAST:
                        if self._testset_available:
                            for t in test:
                                tit.append('Test %d' % t)
                                tit.append('Re. %d' % t)
                        else:
                            tit = ['Re. %d' % t for t in test]
                    elif image == TOTAL_FIELD:
                        for t in test:
                            for s in source:
                                if self._testset_available:
                                    tit.append('Test %d' % t + ', s = %d' % s)
                                    tit.append('Re. %d' % t + ', s = %d' % s)
                                else:
                                    tit.append('Re. Test %d' % t
                                               + ', s = %d' % s)
                elif mode == BEST_TEST:
                    if image == CONTRAST:
                        if self._testset_available:
                            tit = ['Instance: %d' % best,
                                   'Best recover: Test %d' % best]
                        else:
                            tit = ['Best recover: Test %d' % best]
                    elif image == TOTAL_FIELD:
                        for s in source:
                            if self._testset_available:
                                tit.append('Test %d' % best + ', s = %d' % s)
                                tit.append('Re. %d' % best + ', s = %d' % s)
                            else:
                                tit.append('Re. of Test %d' % best
                                           + ', s = %d' % s)
            elif not self._single_method and self._single_testset:
                if mode == ALL_TESTS or mode is None:
                    if image == CONTRAST:
                        for t in test:
                            if self._testset_available:
                                tit.append('Test %d' % t)
                            for m in method:
                                if self._testset_available:
                                    tit.append('Re. by '
                                               + self.method[m].alias)
                                else:
                                    tit.append('Re. of Test %d ' % t + 'by '
                                               + self.method[m].alias)
                    elif image == TOTAL_FIELD:
                            for t in test:
                                for s in source:
                                    if self._testset_available:
                                        tit.append('Test %d' % t
                                                   + ', s = %d' % s)
                                    for m in method:
                                        if self._testset_available:
                                            tit.append('Re. by '
                                                       + self.method[m].alias)
                                        else:
                                            tit.append('Re. %d' % t
                                                       + ', s = %d' % s
                                                       + ', by '
                                                       + self.method[m].alias)
                elif mode == BEST_TEST:
                    if image == CONTRAST:
                        for i in range(len(method)):
                            m = method[i]
                            if self._testset_available:
                                tit.append('Test: %d' % best[i])
                                tit.append('Method: ' + self.method[m].alias)
                            else:
                                tit.append('Test: %d, ' % best[i] + 'Method: '
                                           + self.method[m].alias)
                    elif image == TOTAL_FIELD:
                        for i in range(len(method)):
                            m = method[i]
                            for s in source:
                                if self._testset_available:
                                    tit.append('Test %d' % best[i]
                                                   + ', s = %d' % s)
                                    tit.append(self.method[m].alias)
                                else:
                                    tit.apend(self.method[m].alias + ' T %d, '
                                              % best[i] + ', s = %d' % s)
            elif self._single_method and not self._single_testset:
                if mode == ALL_TESTS or mode is None:
                    if image == CONTRAST:
                        for i in range(len(testset)):
                            tst = testset[i]
                            for t in test[i]:
                                if self._testset_available:
                                    tit.append('Test Set %d, ' % tst
                                               + 'Test %d' % t)
                                    tit.append('Recover')
                                else:
                                    tit.append('Re. of Test Set %d, ' % tst
                                               + 'Test %d' % t)
                    elif image == TOTAL_FIELD:
                        for i in range(len(testset)):
                            tst = testset[i]
                            for t in test[i]:
                                for s in source:
                                    if self._testset_available:
                                        tit.append('Test Set %d, ' % tst
                                                   + 'Test %d, ' % t
                                                   + 'Source %d' % s)
                                        tit.append('Recover')
                                    else:
                                        tit.append('Re. of TS %d, '
                                                   % tst + 'T %d, '
                                                   % t + 'S %d' % s)
                elif mode == BEST_TEST:
                    if image == CONTRAST:
                        for i in range(len(testset)):
                            t = testset[i]
                            if self._testset_available:
                                tit.append('Test Set %d, ' % t
                                           + 'Test %d' % best[i])
                                tit.append('Recover')
                            else:
                                tit.append('Re. of Test Set %d, ' % t
                                           + 'Test %d' % best[i])
                    elif image == TOTAL_FIELD:
                        for i in range(len(testset)):
                            t = testset[i]
                            for s in source:
                                if self._testset_available:
                                    tit.append('Test Set %d, ' % t
                                               + 'Test %d, ' % best[i]
                                               + 'Source %d' %s)
                                    tit.append('Recover')
                                else:
                                    tit.append('Re. of Test Set %d, ' % t
                                               + 'Test %d, ' % best[i]
                                               + 'Source %d' %s)
            else:
                if mode == ALL_TESTS or mode is None:
                    if image == CONTRAST:
                        for i in range(len(testset)):
                            tst = testset[i]
                            for t in test[i]:
                                if self._testset_available:
                                    tit.append('Test Set %d, ' % tst
                                               + 'Test %d' % t)
                                    for m in method:
                                        tit.append(self.method[m].alias)
                                else:
                                    for m in method:
                                        tit.append('TS %d, ' % tst
                                                   + 'T %d, ' % t
                                                   + self.method[m].alias)
                    elif image == TOTAL_FIELD:
                        for i in range(len(testset)):
                            tst = testset[i]
                            for t in test[i]:
                                for s in source:
                                    if self._testset_available:
                                        tit.append('Test Set %d, ' % tst
                                                   + 'Test %d, ' % t
                                                   + 'Source %d' % s)
                                        for m in method:
                                            tit.append(self.method[m].alias)
                                    else:
                                        for m in method:
                                            tit.append('TS %d, ' % tst
                                                       + 'T %d, ' % t
                                                       + 'S %d' % s
                                                       + self.method[m].alias)
                elif mode == BEST_TEST:
                    if image == CONTRAST:
                        for i in range(len(testset)):
                            t = testset[i]
                            for j in range(len(method)):
                                m = method[j]
                                if self._testset_available:
                                    tit.append('Test Set %d, ' % t
                                               + 'Test %d' % best[i][j])
                                    tit.append(self.method[m].alias)
                                else:
                                    tit.append('TS %d, ' % t + 'T %d, '
                                               % best[i][j]
                                               + self.method[m].alias)
                    elif image == TOTAL_FIELD:
                        for i in range(len(testset)):
                            t = testset[i]
                            for j in range(len(method)):
                                m = method[j]
                                for s in source:
                                    if self._testset_available:
                                        tit.append('TS %d, ' % t + 'T %d, '
                                                   % best[i][j] + 'S %d' % s)
                                        tit.append(self.method[m].alias)
                                    else:
                                        tit.append('TS %d, ' % t + 'T %d, '
                                                   % best[i][j] + 'S %d, ' % s
                                                   + self.method[m].alias)

        # Plot figures
        n = 0
        if self._single_method and self._single_testset:
            if mode == ALL_TESTS or mode is None:
                if image == CONTRAST:
                    for t in test:
                        if self._testset_available:
                            a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                            groundtruth = self.testset.test[t]
                        else:
                            a, ti, NP = ax[n], tit[n], 1
                            groundtruth = None
                        self.results[t].plot_map(axis=a, image=rst.CONTRAST,
                                                 groundtruth=groundtruth,
                                                 title=ti, fontsize=fontsize)
                        n += NP
                elif image == TOTAL_FIELD:
                    for t in test:
                        for s in source:
                            if self._testset_available:
                                a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                                groundtruth = self.testset.test[t]
                            else:
                                a, ti, NP = ax[n], tit[n], 1
                                groundtruth = None
                            self.results[t].plot_map(
                                axis=a, image=rst.TOTAL_FIELD,
                                groundtruth=groundtruth, title=ti,
                                fontsize=fontsize, source=s
                            )
                            n += NP
            elif mode == BEST_TEST:
                if image == CONTRAST:
                    if self._testset_available:
                        groundtruth = self.testset.test[best]
                    else:
                        groundtruth = None
                    self.results[best].plot_map(
                        axis=ax, image=rst.CONTRAST, groundtruth=groundtruth,
                        title=tit, fontsize=fontsize
                    )
                elif image == TOTAL_FIELD:
                    for s in source:
                        if self._testset_available:
                            a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                            groundtruth = self.testset.test[best]
                        else:
                            a, ti, NP = ax[n], tit[n], 1
                            groundtruth = None
                        self.results[best].plot_map(
                            axis=a, image=rst.TOTAL_FIELD, fontsize=fontsize,
                            groundtruth=groundtruth, title=ti, source=s
                        )
                        n += NP
        elif not self._single_method and self._single_testset:
            if mode == ALL_TESTS or mode is None:
                if image == CONTRAST:
                    for t in test:
                        if self._testset_available:
                            self.testset.test[t].draw(
                                image=ipt.CONTRAST, axis=ax[n],
                                figure_title=tit[n], fontsize=fontsize
                            )
                            n += 1
                        for m in method:
                            self.results[m, t].plot_map(
                                axis=ax[n], image=rst.CONTRAST, title=tit[n],
                                fontsize=fontsize
                            )
                            n += 1
                elif image == TOTAL_FIELD:
                    for t in test:
                        for s in source:
                            if self._testset_available:
                                self.testset.test[t].plot_total_field(
                                    axis=ax[n], source=s, figure_title=tit[n],
                                    fontsize=fontsize
                                )
                                n += 1
                            for m in method:
                                self.results[m, t].plot_map(
                                    axis=ax[n], image=rst.TOTAL_FIELD,
                                    title=tit[n], fontsize=fontsize, source=s
                                )
                                n += 1
            elif mode == BEST_TEST:
                if image == CONTRAST:
                    for i in range(len(method)):
                        m = method[i]
                        if self._testset_available:
                            a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                            groundtruth = self.testset.test[best[i]]
                        else:
                            a, ti, NP = ax[n], tit[n], 1
                            groundtruth = None
                        self.results[m, best[i]].plot_map(
                            axis=a, image=rst.CONTRAST, title=ti,
                            groundtruth=groundtruth, fontsize=fontsize
                        )
                        n += NP
                elif image == TOTAL_FIELD:
                    for i in range(len(method)):
                        m = method[i]
                        for s in source:
                            if self._testset_available:
                                a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                                groundtruth = self.testset.test[best[i]]
                            else:
                                a, ti, NP = ax[n], tit[n], 1
                                groundtruth = None
                            self.results[m, best[i]].plot_map(
                                axis=a, image=rst.TOTAL_FIELD, title=ti,
                                groundtruth=groundtruth, fontsize=fontsize, 
                                source=s
                            )
                            n += NP
        elif self._single_method and not self._single_testset:
            if mode == ALL_TESTS or mode is None:
                if image == CONTRAST:
                    for i in range(len(testset)):
                        tst = testset[i]
                        for t in test[i]:
                            if self._testset_available:
                                a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                                groundtruth = self.testset[tst].test[t]
                            else:
                                a, ti, NP = ax[n], tit[n], 1
                                groundtruth = None
                            self.results[tst][t].plot_map(
                                axis=a, image=rst.CONTRAST, title=ti,
                                fontsize=fontsize, groundtruth=groundtruth
                            )
                            n += NP
                elif image == TOTAL_FIELD:
                    for i in range(len(testset)):
                        tst = testset[i]
                        for t in test[i]:
                            for s in source:
                                if self._testset_available:
                                    a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                                    groundtruth = self.testset[tst].test[t]
                                else:
                                    a, ti, NP = ax[n], tit[n], 1
                                    groundtruth = None
                                self.results[tst][t].plot_map(
                                    axis=a, image=rst.TOTAL_FIELD, title=ti,
                                    fontsize=fontsize, groundtruth=groundtruth,
                                    source=s
                                )
                                n += 2
            elif mode == BEST_TEST:     
                if image == CONTRAST:
                    for i in range(len(testset)):
                        t = testset[i]
                        if self._testset_available:
                            a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                            groundtruth = self.testset[t].test[best[i]]
                        else:
                            a, ti, NP = ax[n], tit[n], 1
                            groundtruth = None
                        self.results[t][best[i]].plot_map(
                            axis=a, image=rst.CONTRAST, title=ti,
                            fontsize=fontsize, groundtruth=groundtruth
                        )
                        n += NP
                elif image == TOTAL_FIELD:                        
                    for i in range(len(testset)):
                        t = testset[i]
                        for s in source:
                            if self._testset_available:
                                a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                                groundtruth = self.testset[t].test[best[i]]
                            else:
                                a, ti, NP = ax[n], tit[n], 1
                                groundtruth = None
                            self.results[t][best[i]].plot_map(
                                    axis=a, image=rst.TOTAL_FIELD,
                                    groundtruth=groundtruth, fontsize=fontsize,
                                    title=ti, source=s
                            )
                            n += NP
        else:
            if mode == ALL_TESTS or mode is None:
                if image == CONTRAST:                        
                    for i in range(len(testset)):
                        t = testset[i]
                        for k in test[i]:
                            if self._testset_available:
                                self.testset[t].test[k].draw(
                                    image=ipt.CONTRAST, axis=ax[n],
                                    figure_title=tit[n], fontsize=fontsize
                                )
                                n += 1
                            for m in method:
                                self.results[m, t][k].plot_map(
                                    axis=ax[n],image=rst.CONTRAST,
                                    title=tit[n], fontsize=fontsize
                                )
                                n += 1
                elif image == TOTAL_FIELD:                        
                    for i in range(len(testset)):
                        t = testset[i]
                        for k in test[i]:
                            for s in source:
                                if self._testset_available:
                                    self.testset[t].test[k].plot_total_field(
                                        axis=ax[n], figure_title=tit[n],
                                        fontsize=fontsize, source=s
                                    )
                                    n += 1
                                for m in method:
                                    self.results[m, t][k].plot_map(
                                        axis=ax[n], image=rst.TOTAL_FIELD,
                                        title=tit[n], fontsize=fontsize,
                                        source=s
                                    )
                                    n += 1
            elif mode == BEST_TEST:
                if image == CONTRAST:                        
                    for i in range(len(testset)):
                        t = testset[i]
                        for j in range(len(method)):
                            m = method[j]
                            if self._testset_available:
                                a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                                groundtruth = self.testset[t].test[best[i][j]]
                            else:
                                a, ti, NP = ax[n], tit[n], 1
                                groundtruth = None
                            self.results[m, t][best[i][j]].plot_map(
                                axis=a, image=rst.CONTRAST, title=ti,
                                fontsize=fontsize, groundtruth=groundtruth
                            )
                            n += NP
                elif image == TOTAL_FIELD:                        
                    for i in range(len(testset)):
                        t = testset[i]
                        for j in range(len(method)):
                            m = method[j]
                            for s in source:
                                if self._testset_available:
                                    a, ti, NP = ax[n:n+2], tit[n:n+2], 2
                                    ptr = self.testset[t]
                                    groundtruth = ptr.test[best[i][j]]
                                else:
                                    a, ti, NP = ax[n], tit[n], 1
                                    groundtruth = None
                                self.results[m, t][best[i][j]].plot_map(
                                    axis=a, image=rst.TOTAL_FIELD, title=ti,
                                    fontsize=fontsize, source=s,
                                    groundtruth=groundtruth
                                )
                                n += NP

        # Show or save results
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format, transparent=False)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show:
            return fig, axis          

    def normality(self, indicator, axis=None, method=None, testset=None,
                  fontsize=None, title=None, show=False, file_name=None,
                  file_path='', file_format='eps'):
        if indicator is None:
            raise error.WrongTypeInput('Benchmark.normality', 'indicator',
                                       'str or str-list',str(type(indicator)))
        elif rst.check_indicator(indicator) is False:
            raise error.WrongValueInput('Benchmark.normality', 'indicator',
                                        str(rst.INDICATOR_SET), str(indicator))
        elif type(indicator) is str:
            indicator = [indicator]

        if self._single_method and self._single_testset:
            nfig = len(indicator)

        elif not self._single_method and self._single_testset:
            if method is None:
                method = range(len(self.method))
                nfig = len(self.method)
            elif type(method) is str:
                idx = self._search_method(method)
                if idx == False:
                    raise error.WrongValueInput('Benchmark.normality',
                                                'method',
                                                str([self.method[m].alias
                                                     for m in
                                                     range(len(self.method))]),
                                                method)
                method = [idx]
                nfig = 1
            elif type(method) is int:
                if method >= len(self.method):
                    raise error.WrongValueInput('Benchmark.normality',
                                                'method','from 0 to %d'
                                                % len(self.method),
                                                str(method))
                method = [method]
                nfig = 1
            elif all(type(m) is str for m in method):
                idx = self._search_method(method)
                if idx == False:
                    raise error.WrongValueInput('Benchmark.normality',
                                                'method',
                                                str([self.method[m].alias
                                                     for m in
                                                     range(len(self.method))]),
                                                str(method))
                method = idx
                nfig = len(idx)
            elif all(type(m) is int for m in method):
                if any(m >= len(self.method) for m in method):
                    raise error.WrongValueInput('Benchmark.normality',
                                                'method','from 0 to %d'
                                                % len(self.method),
                                                str(method))
                nfig = len(method)
            else:
                raise error.WrongTypeInput('Benchamrk.normality', 'method',
                                           'None, int, str, int-list or '
                                           + 'str-list', str(type(method)))   
     
        elif self._single_method and not self._single_testset:
            if self._testset_available:
                if testset is None:
                    testset = range(len(self.testset))
                    nfig = len(self.testset)
                elif type(testset) is str:
                    idx = self._search_testset(testset)
                    if idx == False:
                        raise error.WrongValueInput(
                            'Benchmark.normality', 'testset',
                            str([self.testset[t].alias for t in 
                                 range(len(self.testset))]), testset
                        )
                    testset = [idx]
                    nfig = 1
                elif type(testset) is int:
                    if testset >= len(self.testset):
                        raise error.WrongValueInput('Benchmark.normality',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    testset = [testset]
                    nfig = 1
                elif all(type(t) is str for t in testset):
                    idx = self._search_testset(testset)
                    if idx == False:
                        raise error.WrongValueInput(
                            'Benchmark.normality', 'testset',
                            str([self.testset[t].name for t in
                                 range(len(self.testset))]), str(testset)
                        )
                    testset = idx
                    nfig = len(idx)
                elif all(type(t) is int for t in testset):
                    if any(t >= len(self.testset) for t in testset):
                        raise error.WrongValueInput('Benchmark.normality',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    nfig = len(testset)
                else:
                    raise error.WrongTypeInput('Benchamrk.normality',
                                               'testset', 'None, int, str, '
                                               + 'int-list or str-list',
                                               str(type(testset)))   
            else:
                if (type(testset) is str
                        or (type(testset) is list
                            and all(type(t) is str for t in testset))):
                    raise error.Error(
                        'Benchmark.normality: when test set is not available, '
                        + 'then you must enter an integer or int-list as a '
                        + 'reference'
                    )
                elif type(testset) is int:
                    if testset >= len(self.testset):
                        raise error.WrongValueInput('Benchmark.normality',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    testset = [testset]
                    nfig = 1
                elif all(type(t) is int for t in testset):
                    if any(t >= len(self.testset) for t in testset):
                        raise error.WrongValueInput('Benchmark.normality',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    nfig = len(testset)
                else:
                    raise error.WrongTypeInput('Benchamrk.normality',
                                               'testset', 'None, int, str, '
                                               + 'int-list or str-list',
                                               str(type(testset)))       

        else:
            if method is None:
                method = range(len(self.method))
            elif type(method) is str:
                idx = self._search_method(method)
                if idx == False:
                    raise error.WrongValueInput('Benchmark.normality',
                                                'method',
                                                str([self.method[m].alias
                                                     for m in
                                                     range(len(self.method))]),
                                                method)
                method = [idx]
            elif type(method) is int:
                if method >= len(self.method):
                    raise error.WrongValueInput('Benchmark.normality',
                                                'method','from 0 to %d'
                                                % len(self.method),
                                                str(method))
                method = [method]
            elif all(type(m) is str for m in method):
                idx = self._search_method(method)
                if idx == False:
                    raise error.WrongValueInput('Benchmark.normality',
                                                'method',
                                                str([self.method[m].alias
                                                     for m in
                                                     range(len(self.method))]),
                                                str(method))
                method = idx
            elif all(type(m) is int for m in method):
                if any(m >= len(self.method) for m in method):
                    raise error.WrongValueInput('Benchmark.normality',
                                                'method','from 0 to %d'
                                                % len(self.method),
                                                str(method))
            else:
                raise error.WrongTypeInput('Benchamrk.normality', 'method',
                                           'None, int, str, int-list or '
                                           + 'str-list', str(type(method)))   
            if self._testset_available:
                if testset is None:
                    testset = range(len(self.testset))
                elif type(testset) is str:
                    idx = self._search_testset(testset)
                    if idx == False:
                        raise error.WrongValueInput(
                            'Benchmark.normality', 'testset',
                            str([self.testset[t].alias for t in 
                                 range(len(self.testset))]), testset
                        )
                    testset = [idx]
                elif type(testset) is int:
                    if testset >= len(self.testset):
                        raise error.WrongValueInput('Benchmark.normality',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    testset = [testset]
                elif all(type(t) is str for t in testset):
                    idx = self._search_testset(testset)
                    if idx == False:
                        raise error.WrongValueInput(
                            'Benchmark.normality', 'testset',
                            str([self.testset[t].name for t in
                                 range(len(self.testset))]), str(testset)
                        )
                    testset = idx
                elif all(type(t) is int for t in testset):
                    if any(t >= len(self.testset) for t in testset):
                        raise error.WrongValueInput('Benchmark.normality',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                else:
                    raise error.WrongTypeInput('Benchamrk.normality',
                                               'testset', 'None, int, str, '
                                               + 'int-list or str-list',
                                               str(type(testset)))   
            else:
                if (type(testset) is str
                        or (type(testset) is list
                            and all(type(t) is str for t in testset))):
                    raise error.Error(
                        'Benchmark.normality: when test set is not available, '
                        + 'then you must enter an integer or int-list as a '
                        + 'reference'
                    )
                elif type(testset) is int:
                    if testset >= len(self.testset):
                        raise error.WrongValueInput('Benchmark.normality',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    testset = [testset]
                    nfig = 1
                elif all(type(t) is int for t in testset):
                    if any(t >= len(self.testset) for t in testset):
                        raise error.WrongValueInput('Benchmark.normality',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    nfig = len(testset)
                else:
                    raise error.WrongTypeInput('Benchamrk.normality',
                                               'testset', 'None, int, str, '
                                               + 'int-list or str-list',
                                               str(type(testset)))       
            nfig = len(method)*len(testset)

        if axis is None:
            fig, ax, _ = rst.get_figure(nfig)
        else:
            if nfig > 1 and type(axis) is not np.ndarray:
                raise error.WrongTypeInput('Benchmark.normality', 'axis',
                                           'numpy.ndarray of length %d' % nfig,
                                           str(type(nfig)))
            elif nfig > 1 and axis.size != nfig:
                raise error.WrongTypeInput('Benchmark.normality', 'axis',
                                           'numpy.ndarray of length %d' % nfig,
                                           'length %d' % axis.size)
            elif (nfig == 1 and type(axis) is not np.ndarray
                    and not isinstance(axis, plt.Axes)):
                raise error.WrongTypeInput('Benchmark.normality', 'axis',
                                           'numpy.ndarray of length 1 or '
                                           + 'matplotlib.pyplot.Axes',
                                           str(type(axis)))
            elif isinstance(axis, plt.Axes):
                ax = np.array([axis])
            else:
                ax = axis
            fig = plt.gcf()

        if type(title) is bool and title is False:
            tit = [None for n in range(nfig)]
        elif type(title) is str:
            tit = [title for n in range(nfig)]
        elif type(title) is list:
            if len(title) != nfig or any([type(t) is not str for t in title]):
                raise error.WrongTypeInput('Benchmarking.normality','title',
                                           'str-list of length %d' % nfig,
                                           str(type(title)))
            tit = title
        elif title is None or (type(title) is bool and title is True):
            tit = []

        n = 0
        for ind in indicator:
            if self._single_method and self._single_testset:        
                if title is None or (type(title) is bool and title is True):
                    tit.append(rst.TITLES[ind])
                data = exp.final_value(ind, self.results)
                _ = sts.normalitiyplot(data, axes=ax[n], title=tit[n],
                                       fontsize=fontsize)
                n += 1
            elif not self._single_method and self._single_testset:
                for m in method:
                    if title is None or (type(title) is bool
                                         and title is True):
                        tit.append(rst.TITLES[ind] + ', '
                                   + self.method[m].alias)
                    data = exp.final_value(ind, self.results[m, :])
                    _ = sts.normalitiyplot(data, axes=ax[n], title=tit[n],
                                           fontsize=fontsize)
                    n += 1
            elif self._single_method and not self._single_testset:
                for t in testset:
                    if title is None or (type(title) is bool
                                         and title is True):
                        tit.append(rst.TITLES[ind] + ', '
                                   + self.testset[t].name)
                    data = exp.final_value(ind, self.results[t, :])
                    _ = sts.normalitiyplot(data, axes=ax[n], title=tit[n],
                                           fontsize=fontsize)
                    n += 1
            else:
                for t in testset:
                    for m in method:
                        if title is None or (type(title) is bool
                                             and title is True):
                            tit.append(rst.TITLES[ind] + ', '
                                       + self.testset[t].name + ', '
                                       + self.method[m].alias)
                        data = exp.final_value(ind, self.results[m, t])
                        _ = sts.normalitiyplot(data, axes=ax[n], title=tit[n],
                                               fontsize=fontsize)
                        n += 1

        # Show or save results
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format, transparent=False)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show:
            return axis          

    def compare(self, indicator, method=None, testset=None, reference=None,
                all2all=False, all2one=None, samples='methods'):
        if indicator is None:
            raise error.WrongTypeInput('Benchmark.compare', 'indicator',
                                       str(rst.INDICATOR_SET), 'None')
        elif rst.check_indicator(indicator) is False:
            raise error.WrongTypeInput('Benchmark.compare', 'indicator',
                                       str(rst.INDICATOR_SET), str(indicator)) 
        elif type(indicator) is str:
            indicator = [indicator]
        if not self._single_method:
            if method is None:
                method = range(len(self.method))
            elif type(method) is str:
                idx = self._search_method(method)
                if idx == False:
                    raise error.WrongValueInput('Benchmark.compare',
                                                'method',
                                                str([self.method[m].alias
                                                     for m in
                                                     range(len(self.method))]),
                                                method)
                method = [idx]
            elif type(method) is int:
                if method >= len(self.method):
                    raise error.WrongValueInput('Benchmark.compare',
                                                'method','from 0 to %d'
                                                % len(self.method),
                                                str(method))
                method = [method]
            elif all(type(m) is str for m in method):
                idx = self._search_method(method)
                if idx == False:
                    raise error.WrongValueInput('Benchmark.compare',
                                                'method',
                                                str([self.method[m].alias
                                                     for m in
                                                     range(len(self.method))]),
                                                str(method))
                method = idx
            elif all(type(m) is int for m in method):
                if any(m >= len(self.method) for m in method):
                    raise error.WrongValueInput('Benchmark.compare',
                                                'method','from 0 to %d'
                                                % len(self.method),
                                                str(method))
            else:
                raise error.WrongTypeInput('Benchamrk.compare', 'method',
                                           'None, int, str, int-list or '
                                           + 'str-list', str(type(method)))   
        if not self._single_testset:
            if self._testset_available:
                if testset is None:
                    testset = range(len(self.testset))
                elif type(testset) is str:
                    idx = self._search_testset(testset)
                    if idx == False:
                        raise error.WrongValueInput(
                            'Benchmark.compare', 'testset',
                            str([self.testset[t].alias for t in 
                                 range(len(self.testset))]), testset
                        )
                    testset = [idx]
                elif type(testset) is int:
                    if testset >= len(self.testset):
                        raise error.WrongValueInput('Benchmark.compare',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    testset = [testset]
                elif all(type(t) is str for t in testset):
                    idx = self._search_testset(testset)
                    if idx == False:
                        raise error.WrongValueInput(
                            'Benchmark.compare', 'testset',
                            str([self.testset[t].name for t in
                                 range(len(self.testset))]), str(testset)
                        )
                    testset = idx
                elif all(type(t) is int for t in testset):
                    if any(t >= len(self.testset) for t in testset):
                        raise error.WrongValueInput('Benchmark.compare',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                else:
                    raise error.WrongTypeInput('Benchamrk.compare',
                                               'testset', 'None, int, str, '
                                               + 'int-list or str-list',
                                               str(type(testset)))   
            else:
                if (type(testset) is str
                        or (type(testset) is list
                            and all(type(t) is str for t in testset))):
                    raise error.Error(
                        'Benchmark.compare: when test set is not available, '
                        + 'then you must enter an integer or int-list as a '
                        + 'reference'
                    )
                elif type(testset) is int:
                    if testset >= len(self.testset):
                        raise error.WrongValueInput('Benchmark.compare',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    testset = [testset]
                elif all(type(t) is int for t in testset):
                    if any(t >= len(self.testset) for t in testset):
                        raise error.WrongValueInput('Benchmark.compare',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                else:
                    raise error.WrongTypeInput('Benchamrk.compare',
                                               'testset', 'None, int, str, '
                                               + 'int-list or str-list',
                                               str(type(testset)))       
        
        if self._single_method and self._single_testset:
            if reference is None:
                raise error.Error('Benchmark.compare: when a single method '
                                  + 'and a single test set are available you '
                                  + 'can only use this function when comparing'
                                  + ' them against a reference value (int or '
                                  + 'float).')
            if type(reference) is int or type(reference) is float:
                reference = [reference for i in range(len(indicator))]
            elif type(reference) is list and len(reference) != len(indicator):
                raise error.Error('Benchmark.compare: reference and indicator '
                                  + 'lists must have the same length!')
            message = ''
            for n in range(len(indicator)):
                data = exp.final_value(indicator[n], self.results)
                output = sts.compare1sample(data, offset=reference[n])
                sample_name = indicator[n] + ' of ' + self.method.name
                message += self._print_compare1sample(sample_name, 
                                                      reference[n],
                                                      output)
                message += '\n'
        
        elif not self._single_method and self._single_testset:
            if len(method) == 1:
                if reference is None:
                    raise error.Error('Benchmark.compare: when studying a '
                                      + 'single method, a reference value '
                                      + 'should be given.')
                if type(reference) is int or type(reference) is float:
                    reference = [reference for i in range(len(indicator))]
                elif type(reference) is list and len(reference) != len(indicator):
                    raise error.Error('Benchmark.compare: reference and '
                                      + 'indicator lists must have the same '
                                      + 'length!')
                message = ''
                for n in range(len(indicator)):
                    data = exp.final_value(indicator[n],
                                           self.results[method[0], :])
                    output = sts.compare1sample(data, offset=reference[n])
                    sample_name = (indicator[n] + ' of '
                                   + self.method[method[0]].alias)
                    message += self._print_compare1sample(sample_name,
                                                          reference[n],
                                                          output)
                    message += '\n'
            elif len(method) == 2:
                message = ''
                for n in range(len(indicator)):
                    data1 = exp.final_value(indicator[n],
                                            self.results[method[0], :])
                    data2 = exp.final_value(indicator[n],
                                            self.results[method[1], :])
                    output = sts.compare2samples(data1, data2, paired=True)
                    sample1_name = (indicator[n] + ' of '
                                    + self.method[method[0]].alias)
                    sample2_name = self.method[method[1]].alias
                    message += self._print_compare2sample(sample1_name,
                                                          sample2_name, output,
                                                          True)
                    message += '\n'
            else:
                message = ''
                for n in range(len(indicator)):
                    data = []
                    samples_names = []
                    for m in method:
                        data.append(exp.final_value(indicator[n],
                                                    self.results[m, :]))
                        samples_names.append(self.method[m].alias)
                    output = sts.compare_multiple(data, all2all=all2all,
                                                  all2one=all2one)
                    data_info = indicator[n] + ' of '
                    message += self._print_compare_multiple(
                        samples_names, output, all2one=all2one,
                        extra_data_info=data_info
                    )
                    
        elif self._single_method and not self._single_testset:
            if len(testset) == 1:
                if reference is None:
                    raise error.Error('Benchmark.compare: when studying a '
                                      + 'single test set, a reference value '
                                      + 'should be given.')
                if type(reference) is int or type(reference) is float:
                    reference = [reference for i in range(len(indicator))]
                elif type(reference) is list and len(reference) != len(indicator):
                    raise error.Error('Benchmark.compare: reference and '
                                      + 'indicator lists must have the same '
                                      + 'length!')
                message = ''
                for n in range(len(indicator)):
                    data = exp.final_value(indicator[n],
                                           self.results[testset[0], :])
                    output = sts.compare1sample(data, offset=reference[n])
                    if self._testset_available:
                        sample_name = (indicator[n] + ' of '
                                       + self.testset[testset[0]].name)
                    else:
                        sample_name = (indicator[n] + ' of '
                                       + self.testset[testset[0]])
                    message += self._print_compare1sample(sample_name,
                                                          reference[n],
                                                          output)
                    message += '\n'
            elif len(testset) == 2:
                message = ''
                for n in range(len(indicator)):
                    data1 = exp.final_value(indicator[n],
                                            self.results[testset[0], :])
                    data2 = exp.final_value(indicator[n],
                                            self.results[testset[1], :])
                    output = sts.compare2samples(data1, data2, paired=True)
                    if self._testset_available:
                        sample1_name = (indicator[n] + ' of '
                                        + self.testset[testset[0]].name)
                        sample2_name = self.testset[testset[1]].name
                    else:
                        sample1_name = (indicator[n] + ' of '
                                        + self.testset[testset[0]])
                        sample2_name = self.testset[testset[1]]
                    message += self._print_compare2sample(sample1_name,
                                                          sample2_name, output,
                                                          True)
                    message += '\n'
            else:
                message = ''
                for n in range(len(indicator)):
                    data = []
                    samples_names = []
                    for t in testset:
                        data.append(exp.final_value(indicator[n],
                                                    self.results[t, :]))
                        if self._testset_available:
                            samples_names.append(self.testset[t].name)
                        else:
                            samples_names.append(self.testset[t])
                    output = sts.compare_multiple(data, all2all=all2all,
                                                  all2one=all2one)
                    data_info = indicator[n] + ' of '
                    message += self._print_compare_multiple(
                        samples_names, output, all2one=all2one,
                        extra_data_info=data_info
                    )

        elif not self._single_method and not self._single_testset:
            if samples == 'methods':
                for t in testset:
                    if self._testset_available:
                        test_name = self.testset[t].name
                    else:
                        test_name = self.testset[t]
                    if len(method) == 1:
                        if reference is None:
                            raise error.Error('Benchmark.compare: when '
                                              + 'studying a single method, a '
                                              + 'reference value should be '
                                              + 'given.')
                        if type(reference) is int or type(reference) is float:
                            reference = [reference for i in
                                         range(len(indicator))]
                        elif (type(reference) is list
                                and len(reference) != len(indicator)):
                            raise error.Error('Benchmark.compare: reference '
                                              + 'and indicator lists must have'
                                              + 'the same length!')
                        message = ''
                        for n in range(len(indicator)):
                            data = exp.final_value(
                                indicator[n], self.results[method[0], t, :]
                            )
                            output = sts.compare1sample(data,
                                                        offset=reference[n])
                            sample_name = (indicator[n] + ' of '
                                           + self.method[method[0]].alias
                                           + "for test set '" + test_name
                                           + "'")
                            message += self._print_compare1sample(sample_name,
                                                                  reference[n],
                                                                  output)
                            message += '\n'
                    elif len(method) == 2:
                        message = ''
                        for n in range(len(indicator)):
                            data1 = exp.final_value(
                                indicator[n], self.results[method[0], t, :]
                            )
                            data2 = exp.final_value(
                                indicator[n], self.results[method[1], t, :]
                            )
                            output = sts.compare2samples(data1, data2,
                                                         paired=True)
                            sample1_name = (indicator[n] + ' of '
                                            + self.method[method[0]].alias)
                            sample2_name = (self.method[method[1]].alias
                                            + " for test set '" + test_name
                                            + "'")
                            message += self._print_compare2sample(sample1_name,
                                                                  sample2_name,
                                                                  output, True)
                            message += '\n'
                    else:
                        message = ''
                        for n in range(len(indicator)):
                            data = []
                            samples_names = []
                            for m in method:
                                data.append(
                                    exp.final_value(indicator[n],
                                                    self.results[m, t, :])
                                )
                                samples_names.append(self.method[m].alias)
                            output = sts.compare_multiple(data,
                                                          all2all=all2all,
                                                          all2one=all2one)
                            data_info = (indicator[n] + " in test set'"
                                         + test_name + "' of ")
                            message += self._print_compare_multiple(
                                samples_names, output, all2one=all2one,
                                extra_data_info=data_info
                            )
            elif samples == 'testsets':
                for m in method:
                    method_name = self.method[m].alias
                    if len(testset) == 1:
                        if reference is None:
                            raise error.Error('Benchmark.compare: when '
                                              + 'studying a single test set, '
                                              + 'a reference value should be '
                                              + 'given.')
                        if type(reference) is int or type(reference) is float:
                            reference = [reference for i in
                                         range(len(indicator))]
                        elif (type(reference) is list
                                and len(reference) != len(indicator)):
                            raise error.Error('Benchmark.compare: reference '
                                              + 'and indicator lists must have'
                                              + ' the same length!')
                        message = ''
                        for n in range(len(indicator)):
                            data = exp.final_value(
                                indicator[n], self.results[m, testset[0], :]
                            )
                            output = sts.compare1sample(data,
                                                        offset=reference[n])
                            sample_name = (indicator[n] + 'of method '
                                           + method_name + 'for tests sets ')
                            if self._testset_available:
                                sample_name += self.testset[testset[0]].name
                            else:
                                sample_name += self.testset[testset[0]]
                            message += self._print_compare1sample(sample_name,
                                                                  reference[n],
                                                                  output)
                            message += '\n'
                    elif len(testset) == 2:
                        message = ''
                        for n in range(len(indicator)):
                            data1 = exp.final_value(
                                indicator[n], self.results[m, testset[0], :]
                            )
                            data2 = exp.final_value(
                                indicator[n], self.results[m, testset[1], :]
                            )
                            output = sts.compare2samples(data1, data2,
                                                         paired=True)
                            sample1_name = (indicator[n] + 'of method '
                                            + method_name + 'for tests sets ')
                            if self._testset_available:
                                sample1_name += self.testset[testset[0]].name
                                sample2_name = self.testset[testset[1]].name
                            else:
                                sample1_name += self.testset[testset[0]]
                                sample2_name = self.testset[testset[1]]
                            message += self._print_compare2sample(sample1_name,
                                                                  sample2_name,
                                                                  output, True)
                            message += '\n'
                    else:
                        message = ''
                        for n in range(len(indicator)):
                            data = []
                            samples_names = []
                            for t in testset:
                                data.append(
                                    exp.final_value(indicator[n],
                                                    self.results[m, t, :])
                                )
                                if self._testset_available:
                                    samples_names.append(self.testset[t].name)
                                else:
                                    samples_names.append(self.testset[t])
                            output = sts.compare_multiple(data, all2all=all2all,
                                                  all2one=all2one)
                            data_info = (indicator[n] + 'of method '
                                         + method_name + 'for tests sets ')
                            message += self._print_compare_multiple(
                                samples_names, output, all2one=all2one,
                                extra_data_info=data_info
                            )
            else:
                raise error.WrongValueInput('Benchmark.compare', 'samples',
                                            "'methods' or 'testsets'",
                                            str(samples))
        
        print(message)

    def confint(self, indicator, method=None, testset=None, print_info=True,
                print_obj=sys.stdout, show=False, file_name=None, file_path='',
                file_format='eps', fontsize=10, axis=None, title=None,
                confidence_level=.95, group='methods'):
        if indicator is None:
            raise error.WrongTypeInput('Benchmark.confint', 'indicator',
                                       str(rst.INDICATOR_SET), 'None')
        elif rst.check_indicator(indicator) is False:
            raise error.WrongTypeInput('Benchmark.confint', 'indicator',
                                       str(rst.INDICATOR_SET), str(indicator)) 
        elif type(indicator) is str:
            indicator = [indicator]
        if not self._single_method:
            if method is None:
                method = range(len(self.method))
            elif type(method) is str:
                idx = self._search_method(method)
                if idx == False:
                    raise error.WrongValueInput('Benchmark.confint',
                                                'method',
                                                str([self.method[m].alias
                                                     for m in
                                                     range(len(self.method))]),
                                                method)
                method = [idx]
            elif type(method) is int:
                if method >= len(self.method):
                    raise error.WrongValueInput('Benchmark.confint',
                                                'method','from 0 to %d'
                                                % len(self.method),
                                                str(method))
                method = [method]
            elif all(type(m) is str for m in method):
                idx = self._search_method(method)
                if idx == False:
                    raise error.WrongValueInput('Benchmark.confint',
                                                'method',
                                                str([self.method[m].alias
                                                     for m in
                                                     range(len(self.method))]),
                                                str(method))
                method = idx
            elif all(type(m) is int for m in method):
                if any(m >= len(self.method) for m in method):
                    raise error.WrongValueInput('Benchmark.confint',
                                                'method','from 0 to %d'
                                                % len(self.method),
                                                str(method))
            else:
                raise error.WrongTypeInput('Benchamrk.confint', 'method',
                                           'None, int, str, int-list or '
                                           + 'str-list', str(type(method)))   
        if not self._single_testset:
            if self._testset_available:
                if testset is None:
                    testset = range(len(self.testset))
                elif type(testset) is str:
                    idx = self._search_testset(testset)
                    if idx == False:
                        raise error.WrongValueInput(
                            'Benchmark.confint', 'testset',
                            str([self.testset[t].alias for t in 
                                 range(len(self.testset))]), testset
                        )
                    testset = [idx]
                elif type(testset) is int:
                    if testset >= len(self.testset):
                        raise error.WrongValueInput('Benchmark.confint',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    testset = [testset]
                elif all(type(t) is str for t in testset):
                    idx = self._search_testset(testset)
                    if idx == False:
                        raise error.WrongValueInput(
                            'Benchmark.confint', 'testset',
                            str([self.testset[t].name for t in
                                 range(len(self.testset))]), str(testset)
                        )
                    testset = idx
                elif all(type(t) is int for t in testset):
                    if any(t >= len(self.testset) for t in testset):
                        raise error.WrongValueInput('Benchmark.confint',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                else:
                    raise error.WrongTypeInput('Benchamrk.confint',
                                               'testset', 'None, int, str, '
                                               + 'int-list or str-list',
                                               str(type(testset)))   
            else:
                if (type(testset) is str
                        or (type(testset) is list
                            and all(type(t) is str for t in testset))):
                    raise error.Error(
                        'Benchmark.confint: when test set is not available, '
                        + 'then you must enter an integer or int-list as a '
                        + 'reference'
                    )
                elif type(testset) is int:
                    if testset >= len(self.testset):
                        raise error.WrongValueInput('Benchmark.confint',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                    testset = [testset]
                elif all(type(t) is int for t in testset):
                    if any(t >= len(self.testset) for t in testset):
                        raise error.WrongValueInput('Benchmark.confint',
                                                    'testset','from 0 to %d'
                                                    % len(self.testset),
                                                    str(testset))
                else:
                    raise error.WrongTypeInput('Benchamrk.confint',
                                               'testset', 'None, int, str, '
                                               + 'int-list or str-list',
                                               str(type(testset)))       
        if show or file_name is not None:
            plot_figure = True
        else:
            plot_figure = False

        if self._single_method and self._single_testset:
            message = 'Confidence Intervals\n'
            for ind in indicator:
                data = exp.final_value(ind, self.results)
                output = sts.confint(data, alpha=1-confidence_level)
                normality = output[1]
                name = 'Indicator: ' + ind
                message += self._print_confint(name, output, confidence_level)
                if not print_info and not normality:
                    print(self._print_non_normal_data(name))
            if plot_figure:
                NFIG = len(indicator)
                if axis is None:
                    fig, ax, _ = rst.get_figure(NFIG)
                else:
                    if isinstance(axis, plt.Axes):
                        ax = [axis]
                    if len(ax) != NFIG:
                        raise error.WrongTypeInput('Benchmark.confint', 'axis',
                                                   'numpy.ndarray of length %d'
                                                   % NFIG, 'length %d'
                                                   % len(ax))
                ylabel = self.method.alias
                if type(title) is bool and title is False:
                    tit = [None for n in range(NFIG)]
                elif title is None or (type(title) is bool and title is True):
                    tit = [self.name + ', ' + rst.TITLES[ind] for ind in
                           indicator]
                elif type(title) is str:
                    tit = [title for n in range(NFIG)]
                elif type(title) is list:
                    tit = title
                for n in range(NFIG):
                    data = exp.final_value(indicator[n], self.results)
                    sts.confintplot(data, axes=ax[n],
                                    xlabel=rst.LABELS[indicator[n]],
                                    ylabel=ylabel, title=tit[n],
                                    fontsize=fontsize,
                                    confidence_level=confidence_level)

        elif not self._single_method and self._single_testset:
            message = 'Confidence Intervals\n'
            for ind in indicator:
                message += 'Indicator: ' + ind + '\n'
                for m in method:
                    data = exp.final_value(ind, self.results[m, :])
                    output = sts.confint(data, alpha=1-confidence_level)
                    normality = output[1]
                    message += self._print_confint(self.method[m].alias,
                                                   output, confidence_level)
                    if not print_info and not normality:
                        name = ind + ', ' + self.method[m].alias
                        print(self._print_non_normal_data(name))
            if plot_figure:
                NFIG = len(indicator)
                if axis is None:
                    fig, ax, _ = rst.get_figure(NFIG)
                else:
                    if isinstance(axis, plt.Axes):
                        ax = [axis]
                    if len(ax) != NFIG:
                        raise error.WrongTypeInput('Benchmark.confint', 'axis',
                                                   'numpy.ndarray of length %d'
                                                   % NFIG, 'length %d'
                                                   % len(ax))
                ylabel = [self.method[m].alias for m in method]
                if type(title) is bool and title is False:
                    tit = [None for n in range(NFIG)]
                elif title is None or (type(title) is bool and title is True):
                    tit = [self.name + ', ' + rst.TITLES[ind] for ind in
                           indicator]
                elif type(title) is str:
                    tit = [title for n in range(NFIG)]
                elif type(title) is list:
                    tit = title
                for n in range(NFIG):
                    data = []
                    for m in method:
                        data.append(exp.final_value(indicator[n],
                                                    self.results[m, :]))
                    sts.confintplot(data, axes=ax[n],
                                    xlabel=rst.LABELS[indicator[n]],
                                    ylabel=ylabel, title=tit[n],
                                    fontsize=fontsize)

        elif self._single_method and not self._single_testset:
            message = 'Confidence Intervals\n'
            testset_names = []
            for t in range(len(self.testset)):
                if self._testset_available:
                    testset_names.append(self.testset[t].name)
                else:
                    testset_names.append(self.testset[t])
            for ind in indicator:
                message += 'Indicator: ' + ind + '\n'
                for t in testset:
                    data = exp.final_value(ind, self.results[t, :])
                    output = sts.confint(data, alpha=1-confidence_level)
                    normality = output[1]
                    message += self._print_confint(testset_names[t], output,
                                                   confidence_level)
                    if not print_info and not normality:
                        name = ind + ', ' + testset_names[t]
                        print(self._print_non_normal_data(name))
            if plot_figure:
                NFIG = len(indicator)
                if axis is None:
                    fig, ax, _ = rst.get_figure(NFIG)
                else:
                    if isinstance(axis, plt.Axes):
                        ax = [axis]
                    if len(ax) != NFIG:
                        raise error.WrongTypeInput('Benchmark.confint', 'axis',
                                                   'numpy.ndarray of length %d'
                                                   % NFIG, 'length %d'
                                                   % len(ax))
                ylabel = [testset_names[t] for t in testset]
                if type(title) is bool and title is False:
                    tit = [None for n in range(NFIG)]
                elif title is None or (type(title) is bool and title is True):
                    tit = [self.name + ', ' + rst.TITLES[ind] for ind in
                           indicator]
                elif type(title) is str:
                    tit = [title for n in range(NFIG)]
                elif type(title) is list:
                    tit = title
                for n in range(NFIG):
                    data = []
                    for t in testset:
                        data.append(exp.final_value(indicator[n],
                                                    self.results[t, :]))
                    sts.confintplot(data, axes=ax[n],
                                    xlabel=rst.LABELS[indicator[n]],
                                    ylabel=ylabel, title=tit[n],
                                    fontsize=fontsize)

        elif not self._single_method and not self._single_testset:
            message = 'Confidence Intervals\n'
            testset_names = []
            for t in range(len(self.testset)):
                if self._testset_available:
                    testset_names.append(self.testset[t].name)
                else:
                    testset_names.append(self.testset[t])
            for ind in indicator:
                if group == 'methods':
                    for t in testset:
                        message += ('Indicator: ' + ind + ', Test Set: '
                                    + testset_names[t] + '\n')
                        for m in method:
                            data = exp.final_value(ind, self.results[m, t])
                            output = sts.confint(data,
                                                 alpha=1-confidence_level)
                            normality = output[1]
                            message += self._print_confint(
                                self.method[m].alias, output, confidence_level
                            )
                            if not print_info and not normality:
                                name = (ind + ', ' + testset_names[t] + ', '
                                        + self.method[m].alias)
                                print(self._print_non_normal_data(name))
                elif group == 'testsets':
                    for m in method:
                        message += ('Indicator: ' + ind + ', Method: '
                                    + self.method[m].alias + '\n')
                        for t in testset:
                            data = exp.final_value(ind, self.results[m, t])
                            output = sts.confint(data,
                                                 alpha=1-confidence_level)
                            normality = output[1]
                            message += self._print_confint(testset_names[t],
                                                           output,
                                                           confidence_level)
                            if not print_info and not normality:
                                name = (ind + ', ' + self.method[m].alias
                                        + ', ' + testset_names[t])
                                print(self._print_non_normal_data(name))
                else:
                    raise error.WrongValueInput('Benchmark.confint', 'group',
                                                "'methods' or 'testsets'",
                                                str(group))
            if plot_figure:
                if group == 'methods':
                    NFIG = len(indicator)*len(testset)
                    ylabel = [self.method[m].alias for m in method]
                elif group == 'testsets':
                    NFIG = len(indicator)*len(method)
                    ylabel = [testset_names[t] for t in testset]
                if axis is None:
                    fig, ax, _ = rst.get_figure(NFIG)
                else:
                    if isinstance(axis, plt.Axes):
                        ax = [axis]
                    if len(ax) != NFIG:
                        raise error.WrongTypeInput('Benchmark.confint', 'axis',
                                                   'numpy.ndarray of length %d'
                                                   % NFIG, 'length %d'
                                                   % len(ax))
                if type(title) is bool and title is False:
                    tit = [None for n in range(NFIG)]
                elif title is None or (type(title) is bool and title is True):
                    tit = []
                    for ind in indicator:
                        if group == 'methods':
                            for t in testset:
                                tit.append(rst.TITLES[ind] + ', '
                                           + testset_names[t])
                        elif group == 'testsets':
                            for m in method:
                                tit.append(rst.TITLES[ind] + ', '
                                           + self.method[m].alias)
                elif type(title) is str:
                    tit = [title for n in range(NFIG)]
                elif type(title) is list:
                    tit = title
                n = 0
                for ind in indicator:
                    if group == 'methods':
                        for t in testset:
                            data = []
                            for m in method:
                                data.append(
                                    exp.final_value(indicator[n],
                                                    self.results[m, t])
                                )
                            sts.confintplot(data, axes=ax[n],
                                            xlabel=rst.LABELS[indicator[n]],
                                            ylabel=ylabel, title=tit[n],
                                            fontsize=fontsize)
                            n += 1
                    elif group == 'testset':
                        for m in method:
                            data = []
                            for t in testset:
                                data.append(
                                    exp.final_value(indicator[n],
                                                    self.results[m, t])
                                )
                            sts.confintplot(data, axes=ax[n],
                                            xlabel=rst.LABELS[indicator[n]],
                                            ylabel=ylabel, title=tit[n],
                                            fontsize=fontsize)
                            n += 1

        if print_info:
            print(message, file=print_obj)
        if plot_figure:
            if file_name is not None:
                plt.savefig(file_path + file_name + '.' + file_format,
                            format=file_format, transparent=False)
            if show:
                plt.show()
            if file_name is not None:
                plt.close()
            elif not show:
                return fig, axis

    def _search_testset(self, name):
        if type(name) is not str and type(name) is not list:
            raise error.WrongTypeInput('Benchmark._check_testset', 'name',
                                       'str or str-list', str(type(name)))
        if type(name) is str:
            if self._single_testset:
                return self.testset.name == name
            else:
                for n in range(len(self.method)):
                    if name == self.method[n].name:
                        return n
                return False
        else:
            if self._single_testset:
                for n in range(len(name)):
                    if name[n] == self.testset.name:
                        return n
                return False
            else:
                idx = []
                for m in range(len(name)):
                    for n in range(len(self.testset)):
                        if name[m] == self.testset[n].name:
                            idx.append(n)
                            break
                if len(idx) > 0:
                    return idx
                else:
                    return False


def _run_testset(testset, method, discretization):
    results = []
    for n in range(testset.sample_size):
        results.append(method.solve(testset.test[n], discretization,
                                    print_info=False))
    return results
