import sys
import numpy as np
from joblib import Parallel, delayed
import pickle
import multiprocessing
from matplotlib import pyplot as plt
from matplotlib import colors

import inputdata as ipt
import result as rst
import experiment as exp
import deterministic as dtm
import stochastic as stc
import statistics as sts
import error

TEST = 'test'
STOCHASTIC_RUNS = 's_nexec'
SAVE_STOCHASTIC_RUNS = 's_save'

PARALLELIZE_METHOD = 'method'
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
        elif type(new) is ipt.InputData:
            self._test = new.copy()
            self._test_available = True
        elif type(new) is str:
            self._test = new
            self._test_available = False
        else:
            raise error.WrongTypeInput('CaseStudy.test', 'new object',
                                       'None or InputData or str',
                                       str(type(new)))

    def __init__(self, name, method=None, discretization=None, test=None,
                 stochastic_runs=30, save_stochastic_runs=False):
        super().__init__(name, method, discretization)
        self.test = test
        self.s_nexec = stochastic_runs
        self.s_save = save_stochastic_runs
    
    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path)
        self.test = data[TEST]
        self.s_nexec = data[STOCHASTIC_RUNS]
        self.s_save = data[SAVE_STOCHASTIC_RUNS]

    def save(self, file_path='', save_test=False):
        data = super().save(file_path)

        if save_test:
            data[TEST] = self.test
        elif self._test_available:
            data[TEST] = self.test.name
        else:
            data[TEST] = self.testset
        
        data[STOCHASTIC_RUNS] = self.s_nexec
        data[SAVE_STOCHASTIC_RUNS] = self.s_save

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def run(self, parallelization=None, save_stochastic_executions=False):
        if not self._test_available:
            raise error.MissingAttributesError('CaseStudy', 'test')
        if self._single_method:
            if isinstance(self.method, dtm.Deterministic):
                if parallelization == True:
                    self.method.parallelization = True
                else:
                    self.method.parallelization = False
                self.results = self.method.solve(self.test,
                                                 self.discretization)
                    
            elif isinstance(self.method, stc.Stochastic):
                if self.s_save:
                    self.method.outputmode.rule = stc.EACH_EXECUTION
                else:
                    self.method.outputmode.rule = stc.AVERAGE_CASE
                self.method.nexec = self.s_nexec
                if parallelization == True:
                    self.method.parallelization = True
                else:
                    self.method.parallelization = False
                self.results = self.method.solve(self.test,
                                                 self.discretization)
        else:
            self.results = []
            if parallelization == False:
                for m in range(len(self.method)):
                    self.method[m].parallelization = False
                    if isinstance(self.method[m], dtm.Deterministic):
                        if self._single_discretization:
                            self.results.append(
                                self.method[m].solve(self.test,
                                                     self.discretization)
                            )
                        else:
                            self.results.append(
                                self.method[m].solve(self.test,
                                                     self.discretization[m])
                            )
                    elif isinstance(self.method[m], stc.Stochastic):
                        if self.s_save:
                            self.method[m].outputmode.rule = stc.EACH_EXECUTION
                        self.method[m].nexec = self.s_nexec
                        if self._single_discretization:
                            self.results.append(
                                self.method[m].solve(self.test,
                                                     self.discretization)
                            )
                        else:
                            self.results.append(
                                self.method[m].solve(self.test,
                                                     self.discretization[m])
                            )
            elif (parallelization == True or parallelization is None
                    or parallelization == PARALLELIZE_EXECUTIONS):
                for m in range(len(self.method)):
                    self.method[m].parallelization = True
                    if isinstance(self.method[m], stc.Stochastic):
                        if self.s_save:
                            self.method[m].outputmode.rule = stc.EACH_EXECUTION
                        self.method[m].nexec = self.s_nexec
                    if self._single_discretization:
                        self.results.append(
                            self.method[m].solve(self.test,
                                                 self.discretization)
                        )
                    else:
                        self.results.append(
                            self.method[m].solve(self.test,
                                                 self.discretization[m])
                        )
            elif parallelization == PARALLELIZE_METHOD:
                for m in range(len(self.method)):
                    self.method[m].parallelization = False
                    if isinstance(self.method[m], stc.Stochastic):
                        self.method[m].nexec = self.s_nexec
                        if self.s_save:
                            self.method[m].outputmode.rule = stc.EACH_EXECUTION
                num_cores = multiprocessing.cpu_count()
                if self._single_discretization:
                    self.results = (Parallel(n_jobs=num_cores)
                                    (delayed(self.method[m].solve)
                                     (self.test, self.discretization,
                                      print_info=False) for m in
                                     range(len(self.method))))
                else:
                    self.results = (Parallel(n_jobs=num_cores)
                                    (delayed(self.method[m].solve)
                                     (self.test, self.discretization[m],
                                      print_info=False) for m in
                                     range(len(self.method))))
    
    def reconstruction(self, image=CONTRAST, axis=None, method=None,
                       file_name=None, file_path='', file_format='eps',
                       show=False, fontsize=10, title=None, indicator=None,
                       include_true=False, mode=ALL_EXECUTIONS):
        if (image != PERMITTIVITY and image != CONDUCTIVITY
                and image != BOTH_PROPERTIES and image != CONTRAST):
            raise error.WrongValueInput('CaseStudy.reconstruction',
                                        'image', "'"+ PERMITTIVITY + "' or '"
                                        + CONDUCTIVITY + "' or '"
                                        + BOTH_PROPERTIES + "' or '"
                                        + CONTRAST + "'", image)
        elif image == PERMITTIVITY and self.configuration.good_conductor:
            raise error.WrongValueInput('CaseStudy.reconstruction',
                                        'image', "'" + CONDUCTIVITY + "' or '"
                                        + CONTRAST + "' for good "
                                        + "conductors", image)
        elif image == CONDUCTIVITY and self.configuration.perfect_dielectric:
            raise error.WrongValueInput('CaseStudy.reconstruction',
                                        'image', "'" + PERMITTIVITY + "' or '"
                                        + CONTRAST + "' for perfect "
                                        + "dielectrics", image)

        if title == False:
            figure_title = ''

        if self._single_method:
            
            if (isinstance(self.method, dtm.Deterministic)
                or (isinstance(self.method, dtm.Deterministic)
                    and self.s_save)):

                if include_true:
                    if image == BOTH_PROPERTIES:
                        nfig = 4
                    else:
                        nfig = 2
                else:
                    if image == BOTH_PROPERTIES:
                        nfig = 2
                    else:
                        nfig = 1

                if axis is None:
                    fig, ax, _ = rst.get_figure(nfig)
                else:
                    if nfig == 1 and isinstance(axis, plt.Axes):
                        ax = np.ndarray([axis])
                    elif (nfig == 1 and type(axis) is np.ndarray
                            and axis.size != 1):
                        raise error.WrongValueInput(
                            'CaseStudy.reconstruction', 'axis',
                            'matplotlib.axes.Axes or 1-numpy.ndarray',
                            '%d-numpy.ndarray' % axis.size
                        )
                    elif type(axis) is np.ndarray and axis.size != nfig:
                        raise error.WrongValueInput(
                            'CaseStudy.reconstruction', 'axis',
                            '%d-numpy.ndarray' % nfig,
                            '%d-numpy.ndarray' % axis.size
                        )
                    else:
                        ax = axis
                    fig = plt.gcf()
                
                if include_true:
                    if image == BOTH_PROPERTIES:
                        if title != False:
                            figure_title = 'Ground-Truth'
                        self.test.draw(image=ipt.BOTH_PROPERTIES, 
                                       axis=ax[:2],
                                       figure_title=figure_title,
                                       fontsize=fontsize)
                        ifig = 2
                    else:
                        if title != False:
                            figure_title = 'Ground-Truth'
                        self.test.draw(image=image, 
                                       axis=ax[0],
                                       figure_title=figure_title,
                                       fontsize=fontsize)
                        ifig = 1
                else:
                    ifig = 0

                if title is None or title == True:
                    figure_title = 'Recovered'
                elif title is not None and title != False:
                    figure_title = title

                if image == BOTH_PROPERTIES:
                    self.results.plot_map(image=rst.BOTH_PROPERTIES,
                                          axis=ax[ifig:ifig+2],
                                          title=figure_title,
                                          fontsize=fontsize)
                else:
                    self.results.plot_map(image=image,
                                          axis=ax[ifig],
                                          title=figure_title,
                                          fontsize=fontsize)
                        
            else:

                if include_true:
                    if image == BOTH_PROPERTIES:
                        nfig = 2 + 2*self.s_nexec
                    else:
                        nfig = 1 + self.s_nexec
                else:
                    if image == BOTH_PROPERTIES:
                        nfig = 2*self.s_nexec
                    else:
                        nfig = self.s_nexec

                if axis is None:
                    fig, ax, _ = rst.get_figure(nfig)
                else:
                    if type(axis) is np.ndarray and axis.size != nfig:
                        raise error.WrongValueInput(
                            'CaseStudy.reconstruction', 'axis',
                            '%d-numpy.ndarray' % nfig,
                            '%d-numpy.ndarray' % axis.size
                        )
                    else:
                        ax = axis
                    fig = plt.gcf()
                
                if include_true:
                    if image == BOTH_PROPERTIES:
                        if title != False:
                            figure_title = 'Ground-Truth'
                        self.test.draw(image=ipt.BOTH_PROPERTIES, 
                                       axis=ax[:2],
                                       figure_title=figure_title,
                                       fontsize=fontsize)
                        ifig = 2
                    else:
                        if title != False:
                            figure_title = 'Ground-Truth'
                        self.test.draw(image=image, 
                                       axis=ax[0],
                                       figure_title=figure_title,
                                       fontsize=fontsize)
                        ifig = 1
                else:
                    ifig = 0

                if title is not None and title != False:
                    figure_title = title

                if image == BOTH_PROPERTIES:
                    for n in range(self.s_nexec):
                        if title is None or title == True:
                            figure_title = 'Recovered %d' % (n+1)
                        self.results[n].plot_map(image=rst.BOTH_PROPERTIES,
                                                 axis=ax[ifig:ifig+2],
                                                 title=figure_title,
                                                 fontsize=fontsize)
                        ifig += 2
                else:
                    for n in range(self.s_nexec):
                        if title is None or title == True:
                            figure_title = 'Recovered %d' % (n+1)
                        self.results[n].plot_map(image=image,
                                                 axis=ax[n+1],
                                                 title=figure_title,
                                                 fontsize=fontsize)
                    
        else:

            if include_true:
                if image == BOTH_PROPERTIES:
                    nfig = 2
                else:
                    nfig = 1
            else:
                nfig = 0
            
            if method is None:
                midx = range(len(self.method))
            elif type(method) is int:
                if method >= len(self.method) or method < 0:
                    raise error.WrongValueInput(
                        'CaseStudy.reconstruction', 'method',
                        'int < %d' % len(self.method), '%d' % method
                    )
                else:
                    midx = [method]
            elif type(method) is list and all(type(m) is int for m in method):
                if any(m < 0 or m >= len(self.method) for m in method):
                    raise error.WrongValueInput(
                        'CaseStudy.reconstruction', 'method',
                        '0 <= int-list < %d' % len(self.method), '%d' % method
                    )
                else:
                    midx = method
            elif type(method) is str:
                midx = self._search_method(method)
                if type(midx) is bool and midx == False:
                    raise error.WrongValueInput(
                        'CaseStudy.reconstruction', 'method',
                        str([self.method[m].alias
                             for m in range(len(self.method))]), method
                    )
                else:
                    midx = [midx]
            else:
                if not all(any(m == self.method[n].alias
                               for n in range(len(self.method)))
                           for m in method):
                    raise error.WrongValueInput(
                        'CaseStudy.reconstruction', 'method',
                        str([self.method[m].alias
                             for m in range(len(self.method))]), method
                    )
                else:
                    midx = self._search_method(method)
            
            for m in midx:
                if (isinstance(self.method[m], dtm.Deterministic)
                        or (isinstance(self.method[m], stc.Stochastic)
                            and not self.s_save)):
                    if image == BOTH_PROPERTIES:
                        nfig += 2
                    else:
                        nfig += 1
                else:
                    if image == BOTH_PROPERTIES:
                        if mode == ALL_EXECUTIONS:
                            nfig += 2*self.s_nexec
                        elif mode == BEST_EXECUTION:
                            nfig += 2
                    else:
                        if mode == ALL_EXECUTIONS:
                            nfig += self.s_nexec
                        else:
                            nfig += 1
            
            if axis is None:
                fig, ax, _ = rst.get_figure(nfig)
            else:
                if isinstance(axis, plt.Axes):
                    if nfig != 1:
                        raise error.WrongValueInput(
                            'CaseStudy.reconstruction', 'axis',
                            '%d-numpy.ndarray' % nfig,
                            'matplotlib.axes.Axes'
                        )
                    else:
                        ax = np.ndarray([axis])
                elif axis.size != nfig:
                    raise error.WrongValueInput(
                            'CaseStudy.reconstruction', 'axis',
                            '%d-numpy.ndarray' % nfig,
                            '%d-numpy.ndarray' % axis.size
                    )
                else:
                    ax = axis
                fig = plt.gcf()

            if include_true:
                if title != False:
                    figure_title = 'Ground-Truth'
                elif type(title) is str:
                    figure_title = title
                if image == BOTH_PROPERTIES:
                    self.test.draw(image=ipt.BOTH_PROPERTIES,
                                   axis=ax[:2],
                                   figure_title=figure_title,
                                   fontsize=fontsize)
                    ifig = 2
                else:
                    self.test.draw(image=image,
                                   axis=ax[0],
                                   figure_title=figure_title,
                                   fontsize=fontsize)
                    ifig = 1
            else:
                ifig = 0

            for m in midx:
                if title is None or title == True:
                    figure_title = self.method[m].alias
                if isinstance(self.method[m], dtm.Deterministic):
                    if image == BOTH_PROPERTIES:
                        self.results[m].plot_map(image=rst.BOTH_PROPERTIES,
                                                 axis=ax[ifig:ifig+2],
                                                 title=figure_title,
                                                 fontsize=fontsize)
                        ifig += 2
                    else:
                        self.results[m].plot_map(image=image,
                                                 axis=ax[ifig],
                                                 title=figure_title,
                                                 fontsize=fontsize)
                        ifig += 1
                elif not self.s_save:
                    figure_title += ' - ' + self.method[m].output + ' case'
                    if image == BOTH_PROPERTIES:
                        self.results[m].plot_map(image=rst.BOTH_PROPERTIES,
                                                 axis=ax[ifig:ifig+2],
                                                 title=figure_title,
                                                 fontsize=fontsize)
                        ifig += 2
                    else:
                        self.results[m].plot_map(image=image,
                                                 axis=ax[ifig],
                                                 title=figure_title,
                                                 fontsize=fontsize)
                        ifig += 1
                else:
                    if mode == ALL_EXECUTIONS:
                        for n in range(len(self.results[m])):
                            if image == BOTH_PROPERTIES:
                                self.results[m][n].plot_map(
                                    image=rst.BOTH_PROPERTIES,
                                    axis=ax[ifig:ifig+2],
                                    title=figure_title+ ' - %d' % (n+1),
                                    fontsize=fontsize)
                                ifig += 2
                            else:
                                self.results[m][n].plot_map(
                                    image=image, axis=ax[ifig],
                                    title=figure_title+ ' - %d' % (n+1),
                                    fontsize=fontsize)
                                ifig += 1
                    elif mode == BEST_EXECUTION:
                        data = exp.final_value(indicator, self.results[m])
                        n = np.argmin(data)
                        if image == BOTH_PROPERTIES:
                            self.results[m][n].plot_map(
                                image=rst.BOTH_PROPERTIES,
                                axis=ax[ifig:ifig+2],
                                title=figure_title+ ' - %d' % (n+1),
                                fontsize=fontsize
                            )
                            ifig += 2
                        else:
                            self.results[m][n].plot_map(
                                image=image, axis=ax[ifig],
                                title=figure_title+ ' - %d' % (n+1),
                                fontsize=fontsize
                            )
                            ifig += 1
                            
        
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show and axis is None:
            return axis

    def convergence(self, indicator, axis=None, method=None, file_name=None,
                    file_path='', file_format='eps', show=False, fontsize=10,
                    title=None, mean=False, yscale=None, sample_rate=None,
                    widths=None, color=None):
        if (type(indicator) is not str
                or (type(indicator) is list
                    and not all(type(i) is str for i in indicator))):
            raise error.WrongTypeInput('CaseStudy.convergence', 'indicator',
                                       'str or str-list', str(type(indicator)))
        if not rst.check_indicator(indicator):
            raise error.WrongValueInput('CaseStudy.convergence', 'indicator',
                                        str(rst.INDICATOR_SET), indicator)

        if self._single_method or type(method) is int or type(method) is str or (type(method) is list and len(method) == 1):
            
            if type(method) is int and (method < 0
                                        or method >= len(self.method)):
                raise error.WrongValueInput('CaseStudy.convergence', 'method',
                                            'int < %d' % len(self.method),
                                            '%d' % method)
            elif type(method) is str and self._search_method(method) is False:
                raise error.WrongValueInput('CaseStudy.converegence', 'method',
                                            str([self.method[m].alias for m in
                                                 range(len(self.method))]),
                                            method)
            elif type(method) is int:
                midx = method
            elif type(method) is str:
                midx = self._search_method(method)    
            elif type(method) is list:        
                if type(method[0]) is int:
                    midx = method[0]
                else:
                    midx = self._search_method(method[0])
            
            if type(indicator) is str:
                ind = [indicator]
            else:
                ind = indicator
            nfig = len(ind)
            
            if axis is None:
                fig, ax, _ = rst.get_figure(nfig)
            else:
                if type(axis) is np.ndarray and axis.size != nfig:
                    raise error.WrongValueInput('CaseStudy.convergence',
                                                'axis',
                                                '%d-numpy.ndarray' % nfig,
                                                '%d-numpy.ndarray' % axis.size)
                if nfig == 1 and not isinstance(axis, plt.Axes):
                    raise error.WrongTypeInput('CaseStudy.convergence', 'axis',
                                               'matplotlib.axes.Axes',
                                               str(type(axis)))
                if nfig == 1 and isinstance(axis, plt.Axes):
                    fig, ax = plt.gcf(), [axis]
                else:
                    fig, ax = plt.gcf(), axis
            
            if title == False:
                figure_title = ''
            elif type(title) is str:
                figure_title = title
            elif (type(title) is list
                    and not all(type(t) is str for t in title)):
                raise error.WrongTypeInput('CaseStudy.convergence', 'title',
                                           'None, True, False, str or '
                                           + 'str-list', 'list where not all'
                                           + 'are str')
            elif (type(title) is list and len(title) != nfig):
                raise error.WrongValueInput('CaseStudy.convergence', 'title',
                                            'str-list of size %d' % nfig,
                                            'str-list of size' % len(title))
            
            ifig = 0
            for i in ind:
                if title is None or title == True:
                    figure_title = rst.TITLES[i]
                elif type(title) is list:
                    figure_title = title[ifig]
            
                if ((self._single_method
                     and isinstance(self.method, dtm.Deterministic)) 
                        or (not self._single_method
                            and isinstance(self.method[midx],
                                           dtm.Deterministic))):
                    y = np.ndarray(getattr(self.results, i))
                    x = np.arange(1, y.size+1)
                    rst.add_plot(ax[ifig], y, x=x, title=figure_title,
                                 xlabel='Iterations',
                                 ylabel=rst.indicator_label(i),
                                 yscale=yscale, fontsize=fontsize)
                elif ((self._single_method
                        and isinstance(self.method, stc.Stochastic)
                        and not self.s_save) 
                      or (not self._single_method
                          and isinstance(self.method[midx],
                                         stc.Stochastic) and not self.s_save)):
                    if self._single_method:
                        y = np.ndarray(getattr(self.results, i))
                    else:
                        y = np.ndarray(getattr(self.results[midx], i))
                    x = np.linspace(0, 100, y.size)
                    rst.add_plot(ax[ifig], y, x=x, title=figure_title,
                                 xlabel='Iterations [%]',
                                 ylabel=rst.indicator_label(i),
                                 yscale=yscale, fontsize=fontsize)
                elif mean:
                    if sample_rate is None:
                            sample_rate = 20
                    percent = np.append(np.arange(0, 100, sample_rate), 100)
                    if self._single_method:
                        data = np.zeros((len(self.results), percent.size))
                        for n in range(len(self.results)):
                            y = np.array(getattr(self.results[n], i))
                            j = percent/100*(y.size-1)
                            j = j.astype(int)
                            data[n, :] = y[j]
                    else:
                        data = np.zeros((len(self.results[midx]), percent.size))
                        for n in range(len(self.results[midx])):
                            y = np.array(getattr(self.results[midx][n], i))
                            j = percent/100*(y.size-1)
                            j = j.astype(int)
                            data[n, :] = y[j]

                    if color is None:
                        color = 'tab:blue'
                    x = percent
                    rst.add_box(data.T,
                                axis=ax[ifig],
                                meanline='pointwise',
                                xlabel='Iterations [%]',
                                ylabel=rst.indicator_label(i),
                                color=color,
                                title=figure_title,
                                fontsize=fontsize,
                                positions=x,
                                widths=widths,
                                yscale=yscale)
                else:
                    if self._single_method:
                        N = len(self.results)
                    else:
                        N = len(self.results[midx])
                    for n in range(N):
                        if self._single_method:
                            y = np.array(getattr(self.results[n], i))
                        else:
                            y = np.array(getattr(self.results[midx][n], i))
                        x = np.arange(1, y.size+1)
                        rst.add_plot(ax[ifig], y, x=x, title=figure_title,
                                     xlabel='Iterations',
                                     ylabel=rst.indicator_label(i),
                                     yscale=yscale, fontsize=fontsize,
                                     style='--')
 
                ifig += 1
        else:

            if method is None:
                midx = range(len(self.method))
            elif type(method) is list and all(type(m) is int for m in method):
                if any(m < 0 or m >= len(self.method) for m in method):
                    raise error.WrongValueInput(
                        'CaseStudy.plot_reconstruction', 'method',
                        '0 <= int-list < %d' % len(self.method), '%d' % method
                    )
                else:
                    midx = method
            else:
                if not all(any(m == self.method[n].alias
                               for n in range(len(self.method)))
                           for m in method):
                    raise error.WrongValueInput(
                        'CaseStudy.plot_reconstruction', 'method',
                        str([self.method[m].alias
                             for m in range(len(self.method))]), method
                    )
                else:
                    midx = self._search_method(method)
            
            if type(indicator) is str:
                ind = [indicator]
            else:
                ind = indicator
            
            nfig, nlines = len(ind), len(midx)
            if axis is None:
                fig, ax, lgd_size = rst.get_figure(nfig, nlines)
            else:
                if type(axis) is np.ndarray and axis.size != nfig:
                    raise error.WrongValueInput('CaseStudy.convergence',
                                                'axis',
                                                '%d-numpy.ndarray' % nfig,
                                                '%d-numpy.ndarray' % axis.size)
                if nfig == 1 and not isinstance(axis, plt.Axes):
                    raise error.WrongTypeInput('CaseStudy.convergence', 'axis',
                                               'matplotlib.axes.Axes',
                                               str(type(axis)))
                if nfig == 1 and isinstance(axis, plt.Axes):
                    fig, ax, lgd_size = plt.gcf(), [axis], None
                else:
                    fig, ax, lgd_size = plt.gcf(), axis, None

            if title == False:
                figure_title = ''
            elif type(title) is str:
                figure_title = title
            elif (type(title) is list
                    and not all(type(t) is str for t in title)):
                raise error.WrongTypeInput('CaseStudy.convergence', 'title',
                                           'None, True, False, str or '
                                           + 'str-list', 'list where not all'
                                           + 'are str')
            elif (type(title) is list and len(title) != nfig):
                raise error.WrongValueInput('CaseStudy.convergence', 'title',
                                            'str-list of size %d' % nfig,
                                            'str-list of size' % len(title))

            all_deterministic = all(isinstance(self.method[m],
                                               dtm.Deterministic)
                                    for m in midx) 

            ifig = 0
            for i in ind:
                if title is None or title == True:
                    figure_title = rst.TITLES[i]
                elif type(title) is list:
                    figure_title = title[i]
                cols = list(colors.TABLEAU_COLORS.keys())
                icol = 0
                for m in midx:                    
                    if all_deterministic:
                        y = np.array(getattr(self.results[m], i))
                        x = np.arange(1, y.size+1)
                        rst.add_plot(ax[ifig], y, x=x, title=figure_title,
                                     xlabel='Iterations',
                                     ylabel=rst.indicator_label(i),
                                     legend=self.method[m].alias,
                                     legend_fontsize=lgd_size,
                                     color=cols[icol],
                                     yscale=yscale, fontsize=fontsize)
                    elif (isinstance(self.method[m], dtm.Deterministic)
                            or not self.s_save):
                        y = np.array(getattr(self.results[m], i))
                        x = np.arange(100/y.size, 101, 100/y.size)
                        rst.add_plot(ax[ifig], y, x=x, title=figure_title,
                                     xlabel='Iterations [%]',
                                     ylabel=rst.indicator_label(i),
                                     legend=self.method[m],
                                     legend_fontsize=lgd_size,
                                     color=cols[icol],
                                     yscale=yscale, fontsize=fontsize)
                    else:
                        if sample_rate is None:
                            sample_rate = 20
                        percent = np.append(np.arange(0, 100, sample_rate), 100)
                        data = np.zeros((len(self.results[m]), percent.size))
                        for n in range(len(self.results[m])):
                            y = np.array(getattr(self.results[m][n], i))
                            j = percent/100*(y.size-1)
                            j = j.astype(int)
                            data[n, :] = y[j]

                        x = percent
                        rst.add_box(data.T,
                                    axis=ax[ifig],
                                    meanline='pointwise',
                                    xlabel='Iterations [%]',
                                    ylabel=rst.indicator_label(i),
                                    color=cols[icol],
                                    legend=self.method[m].alias,
                                    legend_fontsize=lgd_size,
                                    title=figure_title,
                                    fontsize=fontsize,
                                    positions=x,
                                    yscale=yscale,
                                    widths=widths)
                    icol += 1
                ifig += 1

        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show and axis is None:
            return fig, axis
        
    def boxplot(self, indicator, axis=None, method=None, file_name=None,
                file_path='', file_format='eps', show=False, fontsize=10,
                title=None, mean=False, yscale=None, notch=False):
        if (self._single_method
                and (isinstance(self.method, dtm.Deterministic)
                     or not self.s_save)):
            raise error.Error('This method can be called only there is at '
                              + 'least one stochastic method with saved runs.')
        elif (not self._single_method
                and (not self.s_save
                     or all(isinstance(self.method[m], dtm.Deterministic)
                            for m in range(len(self.method))))):
            raise error.Error('This method can be called only there is at '
                              + 'least one stochastic method with saved runs.')
        elif (type(indicator) is not str
                and (type(indicator) is not list
                     or not all(type(i) is str for i in indicator))):
            raise error.WrongTypeInput('CaseStudy.boxplot', 'indicator',
                                       'str or str-list', str(type(indicator)))
        elif not rst.check_indicator(indicator):
            raise error.WrongValueInput('CaseStudy.boxplot', 'indicator',
                                        str(rst.INDICATOR_SET), str(indicator))

        if type(indicator) is int or type(indicator) is str:
            ind, nfig = [indicator], 1
        else:
            ind, nfig = indicator, len(indicator)

        if title == False:
            figure_title = ''
        elif type(title) is list and len(title) != nfig:
            raise error.WrongValueInput('CaseStudy.boxplot', 'title',
                                        '%d-list' % nfig,
                                        '%d-list' % len(title))

        if axis is None:
            fig, ax, _ = rst.get_figure(nfig)
        else:
            if nfig == 1 and isinstance(axis, plt.Axes):
                fig, ax = plt.gcf(), [axis]
            elif nfig == 1 and isinstance(axis, np.ndarray) and axis.size != nfig:
                raise error.Error("'axis' must be an object of "
                                  + "matplotlib.axes.Axes or 1D-numpy.ndarray")
            elif nfig == 1:
                fig, ax = plt.gcf(), axis
            elif nfig > 1 and isinstance(axis, plt.Axes):
                raise error.WrongTypeInput('CaseStudy.boxplot', 'axis',
                                           '%d-numpy.ndarray' % nfig,
                                           str(type(axis)))
            elif nfig != axis.size:
                raise error.WrongValueInput('CaseStudy.boxplot', 'axis',
                                            '%d-numpy.ndarray' % nfig,
                                            '%d-numpy.ndarray' % axis.size)
            else:
                fig, ax = plt.gcf(), axis

        if self._single_method:
            ifig = 0
            for i in ind:
                if title is None or title == True:
                    figure_title = rst.TITLES[i]
                elif type(title) is str:
                    figure_title = title
                elif type(title) is list:
                    figure_title = title[ifig]

                data = np.zeros(len(self.results))
                for n in range(len(self.results)):
                    data[n] = exp.final_value(i, self.results[n])

                rst.add_box(data,
                            axis=ax[fig],
                            meanline=False,
                            xlabel='Algorithms',
                            ylabel=rst.indicator_label(i),
                            labels=[self.method.alias],
                            title=figure_title,
                            fontsize=fontsize,
                            yscale=yscale,
                            notch=notch)
                ifig += 1
                
        else:
            if method is None:
                midx = range(len(self.method))
            elif type(method) is int:
                if method > len(self.method):
                    raise error.WrongValueInput(
                        'CaseStudy.boxplot', 'method',
                        '0 <= int < %d' % len(self.method), '%d' % method
                    )
                else:
                    midx = [method]
            elif type(method) is list and all(type(m) is int for m in method):
                if any(m < 0 or m >= len(self.method) for m in method):
                    raise error.WrongValueInput(
                        'CaseStudy.boxplot', 'method',
                        '0 <= int-list < %d' % len(self.method), '%d' % method
                    )
                elif not any(isinstance(self.method[m], stc.Stochastic)
                             for m in method):
                    raise error.Error('None of the given methods is a '
                                      + 'stochastic one.')
                else:
                    midx = method
            else:
                midx = self._search_method(method)
                if type(midx) is int:
                    midx = [midx]
                if type(midx[0]) is bool and midx[0] == False:
                    raise error.WrongValueInput(
                        'CaseStudy.boxplot', 'method',
                        str([self.method[m].alias
                             for m in range(len(self.method))]), method
                    )
                elif not all(isinstance(self.method[m], stc.Stochastic)
                             for m in midx):
                    raise error.Error('None of the given methods is a '
                                      + 'stochastic one.')
            
            ifig = 0
            for i in ind:
                if title is None or title == True:
                    figure_title = rst.TITLES[i]
                elif type(title) is str:
                    figure_title = title
                elif type(title) is list:
                    figure_title = title[ifig]
                
                jm = 1
                for m in midx:
                    if isinstance(self.method[m], dtm.Deterministic):
                        data = exp.final_value(i, self.results[m])
                        rst.add_plot(ax[ifig],
                                     data,
                                     x=jm,
                                     title=figure_title,
                                     xlabel='Algorithms',
                                     ylabel=rst.indicador_label(i),
                                     style='s',
                                     markersize=20,
                                     yscale=yscale,
                                     fontsize=fontsize,
                                     color='k')
                    else:
                        data = np.zeros(len(self.results[m]))
                        for n in range(len(self.results[m])):
                            data[n] = exp.final_value(i, self.results[m][n])
                        rst.add_box(data,
                                    axis=ax[ifig],
                                    meanline=False,
                                    xlabel='Algorithms',
                                    ylabel=rst.indicator_label(i),
                                    labels=[self.method[m].alias],
                                    title=figure_title,
                                    fontsize=fontsize,
                                    yscale=yscale,
                                    notch=notch,
                                    positions=[jm],
                                    color=colors.TABLEAU_COLORS['tab:blue'])
                    jm += 1
                ifig += 1
    
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show and axis is None:
            return fig, ax

    def compare(self, indicator, method=None, all2all=False, all2one=None):
        if type(indicator) is not str and type(indicator) is not list:
            raise error.WrongTypeInput('CaseStudy.compare', 'indicator',
                                       'str or str-list', str(type(indicator)))
        elif (type(indicator) is list
                and any(type(i) is not str for i in indicator)):
            raise error.WrongTypeInput('CaseStudy.compare', 'indicator',
                                       'str or str-list', str(type(indicator)))
        elif rst.check_indicator(indicator) == False:
            raise error.WrongValueInput('CaseStudy.compare', 'indicator',
                                        str([i for i in rst.INDICATOR_SET]),
                                        str(indicator))
        elif method is not None and type(method) is not list:
            raise error.WrongTypeInput('CaseStudy.compare', 'method',
                                       'None, int-list or str-list',
                                       str(type(method)))
        elif (type(method) is list and not all(type(m) is int for m in method)
                and not all(type(m) is str for m in method)):
            raise error.WrongTypeInput('CaseStudy.compare', 'method',
                                       'None, int-list or str-list',
                                       str(type(method)))
        elif self._single_method:
            raise error.Error('CaseStudy.compare is not valid for single '
                              + 'method cases.')
        elif not self.s_save:
            raise error.Error('CaseStudy.compare is not valid when stochastic'
                              + ' executions are not available.')
        elif (type(method) is list and all(type(m) is str for m in method)
                and self._search_method(method) is False):
            raise error.WrongValueInput('CaseStudy.compare', 'method',
                                        str([self.method[n].alias for n in
                                             range(len(self.method))]),
                                        str(method))
        elif (method is not None and all(type(m) is int for m in method)
                and any(m < 0 or m >= len(self.method) for m in method)):
            raise error.WrongValueInput('CaseStudy.compare', 'method',
                                        'int-list where 0 <= int < %d'
                                        % len(self.method), str(method))

        if method is None:
            midx = range(len(self.method))
        elif all(type(m) is str for m in method):
            midx = self._search_method(method)
        else:
            midx = method
        if type(indicator) is str:
            indicator = [indicator]

        if len(midx) == 2:
            message = ''
            if (isinstance(self.method[midx[0]], stc.Stochastic)
                    and isinstance(self.method[midx[1]], stc.Stochastic)):
                for ind in indicator:
                    x1 = exp.final_value(ind, np.array(self.results[midx[0]]))
                    x2 = exp.final_value(ind, np.array(self.results[midx[1]]))
                    output = sts.compare2samples(x1, x2, paired=False)
                    sample1_name = ind + ' of ' + self.method[midx[0]].alias
                    sample2_name = self.method[midx[1]].alias
                    message += self._print_compare2sample(sample1_name,
                                                          sample2_name, output,
                                                          False)
            elif isinstance(self.method[midx[0]], stc.Stochastic):
                message = ''
                for ind in indicator:
                    x0 = exp.final_value(ind, self.results[midx[0]])
                    x1 = exp.final_value(ind, self.results[midx[1]])
                    output = sts.compare1sample(x0, offset=x1)
                    sample_name = (ind + ' of '
                                   + self.method[midx[0]].alias)
                    message += self._print_compare1sample(
                        sample_name, self.method[midx[1]].alias, output
                    )
            elif isinstance(self.method[midx[1]], stc.Stochastic):
                for ind in indicator:
                    x0 = exp.final_value(ind, self.results[midx[0]])
                    x1 = exp.final_value(ind, self.results[midx[1]])
                    output = sts.compare1sample(x1, offset=x0)
                    sample_name = (ind + ' of '
                                   + self.method[midx[1]].alias)
                    message += self._print_compare1sample(
                        sample_name, self.method[midx[0]].alias, output
                    )
            else:
                raise error.Error('Only Stochastic-Stochastic and '
                                  + 'Deterministic-Stochastic pair '
                                  + 'comparisons are allowed!')

        elif len(midx) > 2:
            if not all(isinstance(self.method[n],
                                  stc.Stochastic) for n in midx):
                raise error.Error('For multiple comparisons, only Stochastic '
                                  + 'methods are supported!')
            samples_names = []
            for m in range(len(midx)):
                samples_names.append(self.method[midx[m]].alias)
            message = ''
            for ind in indicator:
                data = []
                for m in midx:
                    data.append(exp.final_value(ind, self.results[m]))
                output = sts.compare_multiple(data, all2all, all2one)
                data_info = ind + ' of '
                message += self._print_compare_multiple(
                    samples_names, output, all2one=all2one,
                    extra_data_info=data_info
                )

        print(message)   

    def confint(self, indicator, method=None, axis=None, file_name=None,
                file_path='', file_format='eps', show=False, fontsize=10,
                title=None, print_info=True, print_obj=sys.stdout,
                confidence_level=.95):
        if type(indicator) is not str and type(indicator) is not list:
            raise error.WrongTypeInput('CaseStudy.confint', 'indicator',
                                       'str or str-list', str(type(indicator)))
        elif (type(indicator) is list
                and any(type(i) is not str for i in indicator)):
            raise error.WrongTypeInput('CaseStudy.confint', 'indicator',
                                       'str or str-list', str(type(indicator)))
        elif rst.check_indicator(indicator) == False:
            raise error.WrongValueInput('CaseStudy.confint', 'indicator',
                                        str([i for i in rst.INDICATOR_SET]),
                                        str(indicator))
        if type(indicator) is not list:
            indicator = [indicator]
        if self._single_method:
            if isinstance(self.method, dtm.Deterministic):
                raise error.Error('CaseStudy.confint is available only for '
                                  + 'stochastic methods')
            if axis is None:
                fig, axis = rst.get_figure(len(indicator))
            else:
                fig = plt.gcf()

            n = 0
            if print_info:
                message = 'Confidence Intervals\n'

            for ind in indicator:
                data = exp.final_value(ind, self.results)
                output = sts.confint(data, alpha=1-confidence_level)
                normality = output[1]
                name = 'Indicator: ' + ind
                if print_info:
                    message += self._print_confint(name, output,
                                                   confidence_level)
                elif not print_info and not normality:
                    print(self._print_non_normal_data(name))

                if title is None:
                    tit = rst.TITLES[ind]
                elif type(title) is str:
                    tit = title
                elif type(title) is list:
                    tit = title[n]
                elif title == False:
                    tit = ''

                sts.confintplot(data, axes=axis[n], xlabel=rst.LABELS[ind],
                                ylabel=self.method.alias, fontsize=fontsize,
                                title=tit, confidence_level=confidence_level)
                n += 1
        else:
            if (method is not None and type(method) is not list
                  and type(method) is not str and type(method) is not int):
                raise error.WrongTypeInput('CaseStudy.confint', 'method',
                                           'None, int-list or str-list',
                                           str(type(method)))
            elif (type(method) is list
                    and not all(type(m) is int for m in method)
                    and not all(type(m) is str for m in method)):
                raise error.WrongTypeInput('CaseStudy.confint', 'method',
                                           'None, int-list or str-list',
                                           str(type(method)))
            
            elif method is not None and self._search_method(method) == False:
                raise error.WrongValueInput('CaseStudy.confint', 'method',
                                            str([self.method[n].alias for n in
                                                 range(len(self.method))]),
                                            str(method))
            elif (type(method) is list and all(type(m) is int for m in method)
                    and any(m < 0 or m >= len(self.method) for m in method)):
                raise error.WrongValueInput('CaseStudy.confint', 'method',
                                            'int-list where 0 <= int < %d'
                                            % len(self.method), str(method))

            if method is None:
                midx = range(len(self.method))
            elif type(method) is int:
                midx = [method]
            elif type(method) is str or all(type(m) is str for m in method):
                midx = self._search_method(method)
            else:
                midx = method
            
            if (not self.s_save
                    and any(isinstance(self.method[m], stc.Stochastic)
                            for m in midx)):
                raise error.Error('CaseStudy.confint is not valid when '
                                  + 'stochastic executions are not available.')

            if axis is None:
                fig, axis, _ = rst.get_figure(len(indicator))
            else:
                fig = plt.gcf()

            if print_info:
                message = 'Confidence Intervals\n'
            n = 0
            for ind in indicator:
                data = []
                names = []
                for m in midx:
                    data.append(exp.final_value(ind, self.results[m]))
                    names.append(self.method[m].alias)
                    output = sts.confint(data[-1], alpha=1-confidence_level)
                    normality = output[1]
                    if print_info:
                        message += self._print_confint(names[-1], output,
                                                       confidence_level)
                    if not print_info and not normality:
                        name = ind + ', ' + names[-1]
                        print(self._print_non_normal_data(name))

                if title is None:
                    tit = rst.TITLES[ind]
                elif type(title) is str:
                    tit = title
                elif type(title) is list:
                    tit = title[n]
                elif title == False:
                    tit = ''
                sts.confintplot(data, axes=axis[n], xlabel=rst.LABELS[ind],
                                ylabel=names, title=tit, fontsize=fontsize)
                n += 1

        if print_info:
            print(message, file=print_obj)

        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show and axis is None:
            return fig, axis

    def __str__(self):
        message = 'CASE STUDY\n' + super().__str__()
        message += 'Test: '
        if self._test_available:
            message += self.test.name + '\n'
        elif self.test is not None:
            message += self.test + '\n'
        else:
            message += 'empty\n'
        message += 'Save stochastic runs? '
        if self.s_save:
            message += 'yes\n'
        else:
            message += 'no\n'
        message += 'Number of stochastic runs: %d\n' % self.s_nexec            
        return message

