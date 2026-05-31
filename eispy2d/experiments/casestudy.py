"""
Case Study Module for Electromagnetic Inverse Scattering

This module provides the CaseStudy class, which extends the Experiment class
to handle comprehensive case studies for electromagnetic inverse scattering
problems. A case study includes test data, method configuration, and tools
for statistical analysis, visualization, and comparison of reconstruction
results.

The CaseStudy class supports both deterministic and stochastic methods,
parallel execution strategies, and comprehensive result analysis including
convergence plots, boxplots, statistical comparisons, and confidence intervals.

Classes
-------
CaseStudy : Extends experiment.Experiment
    Comprehensive case study framework for inverse scattering problems

Constants
---------
TEST : str
    Dictionary key for test data storage
STOCHASTIC_RUNS : str
    Dictionary key for number of stochastic runs
SAVE_STOCHASTIC_RUNS : str
    Dictionary key for stochastic run saving flag
PARALLELIZE_METHOD : str
    Parallelization strategy for methods
PARALLELIZE_EXECUTIONS : str
    Parallelization strategy for executions
PERMITTIVITY : str
    Identifier for permittivity property
CONDUCTIVITY : str
    Identifier for conductivity property
BOTH_PROPERTIES : str
    Identifier for both electromagnetic properties
CONTRAST : str
    Identifier for contrast property
ALL_EXECUTIONS : str
    Mode for displaying all stochastic executions
BEST_EXECUTION : str
    Mode for displaying best stochastic execution

Examples
--------
>>> # Create a case study with single method
>>> case = CaseStudy(name='test_case', method=my_method, 
...                  discretization=my_disc, test=my_test)
>>> case.run(parallelization=True)
>>> case.reconstruction(show=True)

>>> # Create a case study with multiple methods
>>> case = CaseStudy(name='comparison', method=[method1, method2], 
...                  discretization=my_disc, test=my_test)
>>> case.run()
>>> case.boxplot('total_error', show=True)
>>> case.compare('total_error')
"""

import sys
import numpy as np
from joblib import Parallel, delayed
import pickle
import multiprocessing
from matplotlib import pyplot as plt
from matplotlib import colors

from eispy2d.data import inputdata as ipt
from eispy2d.data import result as rst
from eispy2d.experiments import experiment as exp
from eispy2d.solvers.base import deterministic as dtm
from eispy2d.solvers.base import stochastic as stc
from eispy2d.utils import statisticsutils as sts
from eispy2d.core import error

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
    """
    Comprehensive case study framework for electromagnetic inverse scattering.
    
    This class extends the Experiment class to provide a complete framework
    for conducting case studies in electromagnetic inverse scattering problems.
    It supports both deterministic and stochastic methods, parallel execution,
    and comprehensive result analysis including statistical comparisons and
    visualizations.
    
    A case study includes:
    - Test data (ground truth) for validation
    - One or more inverse scattering methods
    - Discretization parameters
    - Stochastic execution configuration
    - Result analysis and visualization tools
    
    Parameters
    ----------
    name : str, optional
        Name identifier for the case study
    method : object or list of objects, optional
        Inverse scattering method(s) to be evaluated
    discretization : object or list of objects, optional
        Discretization parameters for the methods
    test : InputData or str, optional
        Test data for validation or filename if string
    stochastic_runs : int, default=30
        Number of stochastic executions for statistical analysis
    save_stochastic_runs : bool, default=False
        Whether to save individual stochastic execution results
    import_filename : str, optional
        Filename to import case study configuration from
    import_filepath : str, default=''
        Path to import file
    
    Attributes
    ----------
    test : InputData or str
        Test data for validation
    s_nexec : int
        Number of stochastic executions
    s_save : bool
        Flag for saving stochastic runs
    results : object or list
        Results from method execution(s)
    
    Methods
    -------
    run(parallelization=None, save_stochastic_executions=False)
        Execute the case study with specified parallelization
    reconstruction(image='contrast', **kwargs)
        Visualize reconstruction results
    convergence(indicator, **kwargs)
        Plot convergence analysis
    boxplot(indicator, **kwargs)
        Create boxplot for statistical analysis
    compare(indicator, method=None, all2all=False, all2one=None)
        Perform statistical comparison between methods
    confint(indicator, **kwargs)
        Calculate and visualize confidence intervals
    
    Examples
    --------
    >>> # Single method case study
    >>> case = CaseStudy(name='bim_test', method=bim_method, 
    ...                  discretization=disc, test=test_data)
    >>> case.run(parallelization=True)
    >>> case.reconstruction(show=True)
    
    >>> # Multiple method comparison
    >>> methods = [bim_method, born_method]
    >>> case = CaseStudy(name='comparison', method=methods, 
    ...                  discretization=disc, test=test_data)
    >>> case.run()
    >>> case.compare('total_error')
    >>> case.boxplot('total_error', show=True)
    
    >>> # Stochastic analysis
    >>> case = CaseStudy(name='stochastic', method=pso_method, 
    ...                  discretization=disc, test=test_data,
    ...                  stochastic_runs=50, save_stochastic_runs=True)
    >>> case.run()
    >>> case.convergence('total_error', mean=True, show=True)
    """

    @property
    def test(self):
        """
        Get the test data for validation.
        
        Returns
        -------
        InputData or str or None
            Test data object, filename string, or None if not set
        """
        return self._test

    @test.setter
    def test(self, new):
        """
        Set the test data for validation.
        
        Parameters
        ----------
        new : InputData, str, or None
            New test data. Can be:
            - InputData object: copied and marked as available
            - str: filename stored but not loaded (marked as unavailable)
            - None: test data cleared
            
        Raises
        ------
        error.WrongTypeInput
            If new is not InputData, str, or None
        """
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

    def __init__(self, name=None, method=None, discretization=None, test=None,
                 stochastic_runs=30, save_stochastic_runs=False,
                 import_filename=None, import_filepath=''):
        """Initialize a case study for electromagnetic inverse scattering.

        Creates a new case study with specified parameters or imports from
        an existing saved case study file.

        Parameters
        ----------
        name : str, optional
            Name identifier for the case study
        method : object or list of objects, optional
            Inverse scattering method(s) to be evaluated
        discretization : object or list of objects, optional
            Discretization parameters for the methods
        test : InputData or str, optional
            Test data for validation or filename if string
        stochastic_runs : int, default=30
            Number of stochastic executions for statistical analysis
        save_stochastic_runs : bool, default=False
            Whether to save individual stochastic execution results
        import_filename : str, optional
            Filename to import case study configuration from
        import_filepath : str, default=''
            Path to import file

        Examples
        --------
        >>> # Create new case study
        >>> case = CaseStudy(name='test', method=my_method, 
        ...                  discretization=my_disc, test=my_test)

        >>> # Import from saved file
        >>> case = CaseStudy(import_filename='saved_case.pkl')
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(name, method, discretization)
            self.test = test
            self.s_nexec = stochastic_runs
            self.s_save = save_stochastic_runs
    
    def importdata(self, file_name, file_path=''):
        """
        Import case study configuration from a saved file.
        
        Loads a previously saved case study configuration including test data,
        stochastic execution parameters, and all inherited experiment settings.
        
        Parameters
        ----------
        file_name : str
            Name of the file to import from
        file_path : str, default=''
            Path to the file location
            
        Examples
        --------
        >>> case = CaseStudy()
        >>> case.importdata('my_case.pkl', '/path/to/files/')
        """
        data = super().importdata(file_name, file_path)
        self.test = data[TEST]
        self.s_nexec = data[STOCHASTIC_RUNS]
        self.s_save = data[SAVE_STOCHASTIC_RUNS]

    def save(self, file_path='', save_test=False):
        """
        Save the case study configuration to a file.
        
        Saves the complete case study configuration including test data,
        stochastic execution parameters, and all inherited experiment settings
        using pickle serialization.
        
        Parameters
        ----------
        file_path : str, default=''
            Path where the file will be saved
        save_test : bool, default=False
            Whether to save the complete test data object or just the name
            
        Examples
        --------
        >>> case.save('/path/to/save/', save_test=True)
        """
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
        """Execute the case study with the configured methods.

        Runs the inverse scattering method(s) on the test data with specified
        parallelization strategy. Supports both deterministic and stochastic
        methods with different parallelization approaches.

        Parameters
        ----------
        parallelization : bool, str, or None, default=None
            Parallelization strategy:
            - None or True: Enable parallelization (default strategy)
            - False: Disable parallelization
            - 'method': Parallelize across methods
            - 'executions': Parallelize across executions
        save_stochastic_executions : bool, default=False
            Whether to save individual stochastic execution results
            (parameter name kept for backward compatibility)

        Raises
        ------
        error.MissingAttributesError
            If test data is not available

        Notes
        -----
        The method automatically detects single vs. multiple methods and
        applies appropriate parallelization strategies:

        - For single deterministic methods: Simple parallel execution
        - For single stochastic methods: Parallel stochastic executions
        - For multiple methods: Choice between method-level or execution-level parallelization

        Examples
        --------
        >>> # Run with default parallelization
        >>> case.run()

        >>> # Run without parallelization
        >>> case.run(parallelization=False)

        >>> # Run with method-level parallelization
        >>> case.run(parallelization='method')
        """
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
                        if self._single_discretization is None:
                            self.results.append(
                                self.method[m].solve(self.test)
                            )
                        elif self._single_discretization:
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
                        if self._single_discretization is None:
                            self.results.append(
                                self.method[m].solve(self.test)
                            )
                        elif self._single_discretization:
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
                    if self._single_discretization is None:
                        self.results.append(
                            self.method[m].solve(self.test)
                        )
                    elif self._single_discretization:
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
                if self._single_discretization is None:
                    self.results = (Parallel(n_jobs=num_cores)
                                    (delayed(self.method[m].solve)
                                     (self.test, print_info=False) for m in
                                     range(len(self.method))))
                elif self._single_discretization:
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
        """
        Visualize reconstruction results from the case study.
        
        Creates visualization plots of the reconstructed electromagnetic
        properties, with options to include ground truth, compare multiple
        methods, and handle stochastic results.
        
        Parameters
        ----------
        image : str, default='contrast'
            Property to visualize:
            - 'epsilon_r': Relative permittivity
            - 'sigma': Conductivity
            - 'both': Both properties
            - 'contrast': Contrast function
        axis : matplotlib.axes.Axes or ndarray, optional
            Axes for plotting. If None, new figure is created
        method : int, str, or list, optional
            Method indices or names to plot. If None, all methods are used
        file_name : str, optional
            Name for saving the figure
        file_path : str, default=''
            Path for saving the figure
        file_format : str, default='eps'
            Format for saving the figure
        show : bool, default=False
            Whether to display the figure
        fontsize : int, default=10
            Font size for labels and titles
        title : str, bool, or None, optional
            Figure title. If None, automatic titles are used
        indicator : str, optional
            Performance indicator for selecting best execution
        include_true : bool, default=False
            Whether to include ground truth in the visualization
        mode : str, default='all'
            Mode for stochastic results:
            - 'all': Show all executions
            - 'best': Show only best execution
            
        Returns
        -------
        matplotlib.axes.Axes or None
            Axes object if axis is None and show is False
            
        Raises
        ------
        error.WrongValueInput
            If image type is invalid for the problem configuration
            
        Examples
        --------
        >>> # Basic reconstruction plot
        >>> case.reconstruction(show=True)
        
        >>> # Include ground truth with both properties
        >>> case.reconstruction(image='both', include_true=True, show=True)
        
        >>> # Compare specific methods
        >>> case.reconstruction(method=[0, 1], show=True)
        
        >>> # Save reconstruction plot
        >>> case.reconstruction(file_name='reconstruction', 
        ...                    file_path='/results/', show=True)
        """
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
                                       title=figure_title,
                                       fontsize=fontsize)
                        ifig = 2
                    else:
                        if title != False:
                            figure_title = 'Ground-Truth'
                        self.test.draw(image=image, 
                                       axis=ax[0],
                                       title=figure_title,
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
                                   title=figure_title,
                                   fontsize=fontsize)
                    ifig = 2
                else:
                    self.test.draw(image=image,
                                   axis=ax[0],
                                   title=figure_title,
                                   fontsize=fontsize)
                    ifig = 1
            else:
                ifig = 0

            for m in midx:
                if title is None or title == True:
                    figure_title = self.method[m].alias
                elif type(title) is str:
                    figure_title = title
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
                        if title is None or title is True:
                            figure_title+ ' - %d' % (n+1)
                        if image == BOTH_PROPERTIES:
                            self.results[m][n].plot_map(
                                image=rst.BOTH_PROPERTIES,
                                axis=ax[ifig:ifig+2],
                                title=figure_title,
                                fontsize=fontsize
                            )
                            ifig += 2
                        else:
                            self.results[m][n].plot_map(
                                image=image, axis=ax[ifig],
                                title=figure_title,
                                fontsize=fontsize
                            )
                            ifig += 1
                            
        
        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format, bbox_inches='tight')
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show and axis is None:
            return axis

    def convergence(self, indicator, axis=None, method=None, file_name=None,
                    file_path='', file_format='eps', show=False, fontsize=10,
                    title=None, mean=False, yscale=None, sample_rate=None,
                    widths=None, color=None, legend_size=None):
        """
        Plot convergence analysis for performance indicators.
        
        Visualizes the convergence behavior of specified performance indicators
        throughout the iterative process. Supports both deterministic and
        stochastic methods with options for mean curves and boxplots.
        
        Parameters
        ----------
        indicator : str or list of str
            Performance indicator(s) to plot (e.g., 'total_error', 'data_error')
        axis : matplotlib.axes.Axes or ndarray, optional
            Axes for plotting. If None, new figure is created
        method : int, str, or list, optional
            Method indices or names to plot. If None, all methods are used
        file_name : str, optional
            Name for saving the figure
        file_path : str, default=''
            Path for saving the figure
        file_format : str, default='eps'
            Format for saving the figure
        show : bool, default=False
            Whether to display the figure
        fontsize : int, default=10
            Font size for labels and titles
        title : str, bool, list, or None, optional
            Figure title(s). If None, automatic titles are used
        mean : bool, default=False
            Whether to show mean curves for stochastic methods
        yscale : str, optional
            Y-axis scale ('linear', 'log', etc.)
        sample_rate : int, optional
            Sampling rate for stochastic convergence (default=20)
        widths : float or array-like, optional
            Width of boxplots for stochastic methods
        color : str, optional
            Color for plots
        legend_size : int, optional
            Font size for legend
            
        Returns
        -------
        tuple or None
            (fig, axis) if axis is None and show is False
            
        Raises
        ------
        error.WrongTypeInput
            If indicator is not string or list of strings
        error.WrongValueInput
            If indicator is not in the valid indicator set
            
        Examples
        --------
        >>> # Basic convergence plot
        >>> case.convergence('total_error', show=True)
        
        >>> # Multiple indicators
        >>> case.convergence(['total_error', 'data_error'], show=True)
        
        >>> # Mean convergence for stochastic methods
        >>> case.convergence('total_error', mean=True, show=True)
        
        >>> # Compare methods
        >>> case.convergence('total_error', method=[0, 1], show=True)
        """
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
                fig, ax, _ = rst.get_figure(nfig, nlines)
                lgd_size = legend_size
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
                    fig, ax, lgd_size = plt.gcf(), [axis], legend_size
                else:
                    fig, ax, lgd_size = plt.gcf(), axis, legend_size

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
        """
        Create boxplot visualization for stochastic method results.
        
        Generates boxplot visualizations of performance indicators for
        stochastic methods to show statistical distributions of results
        across multiple executions.
        
        Parameters
        ----------
        indicator : str or list of str
            Performance indicator(s) to plot (e.g., 'total_error', 'data_error')
        axis : matplotlib.axes.Axes or ndarray, optional
            Axes for plotting. If None, new figure is created
        method : int, str, or list, optional
            Method indices or names to plot. If None, all stochastic methods are used
        file_name : str, optional
            Name for saving the figure
        file_path : str, default=''
            Path for saving the figure
        file_format : str, default='eps'
            Format for saving the figure
        show : bool, default=False
            Whether to display the figure
        fontsize : int, default=10
            Font size for labels and titles
        title : str, bool, list, or None, optional
            Figure title(s). If None, automatic titles are used
        mean : bool, default=False
            Whether to show mean indicators (unused parameter)
        yscale : str, optional
            Y-axis scale ('linear', 'log', etc.)
        notch : bool, default=False
            Whether to show notches in boxplots
            
        Returns
        -------
        tuple or None
            (fig, ax) if axis is None and show is False
            
        Raises
        ------
        error.Error
            If no stochastic methods with saved runs are available
        error.WrongTypeInput
            If indicator is not string or list of strings
        error.WrongValueInput
            If indicator is not in the valid indicator set
            
        Examples
        --------
        >>> # Basic boxplot
        >>> case.boxplot('total_error', show=True)
        
        >>> # Multiple indicators
        >>> case.boxplot(['total_error', 'data_error'], show=True)
        
        >>> # Compare stochastic methods
        >>> case.boxplot('total_error', method=[0, 1], show=True)
        
        >>> # Boxplot with notches
        >>> case.boxplot('total_error', notch=True, show=True)
        """
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
                                     ylabel=rst.indicator_label(i),
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
                        format=file_format, bbox_inches='tight')
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show and axis is None:
            return fig, ax

    def compare(self, indicator, method=None, all2all=False, all2one=None):
        """
        Perform statistical comparison between methods.
        
        Conducts statistical tests to compare performance indicators between
        different methods. Supports pairwise comparisons and multiple comparison
        procedures for stochastic methods.
        
        Parameters
        ----------
        indicator : str or list of str
            Performance indicator(s) to compare (e.g., 'total_error', 'data_error')
        method : int, str, or list, optional
            Method indices or names to compare. If None, all methods are used
        all2all : bool, default=False
            Whether to perform all-to-all comparisons in multiple comparison
        all2one : int, optional
            Index of reference method for all-to-one comparisons
            
        Raises
        ------
        error.WrongTypeInput
            If indicator is not string or list of strings
        error.WrongValueInput
            If indicator is not in the valid indicator set
        error.Error
            If case study configuration is invalid for comparison
            
        Notes
        -----
        The method automatically selects appropriate statistical tests:
        - Two-sample tests for stochastic vs. stochastic comparisons
        - One-sample tests for stochastic vs. deterministic comparisons
        - Multiple comparison procedures for more than two methods
        
        Only stochastic methods with saved runs can be compared statistically.
        Deterministic methods are included as reference points.
        
        Examples
        --------
        >>> # Compare two methods
        >>> case.compare('total_error', method=[0, 1])
        
        >>> # Compare multiple indicators
        >>> case.compare(['total_error', 'data_error'])
        
        >>> # All-to-one comparison
        >>> case.compare('total_error', all2one=0)
        
        >>> # All-to-all comparison
        >>> case.compare('total_error', all2all=True)
        """
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
                confidence_level=.95, xscale=None):
        """
        Calculate and visualize confidence intervals for performance indicators.

        Computes confidence intervals for performance indicators from stochastic
        method executions and creates visualization plots with optional
        statistical information output.

        Parameters
        ----------
        indicator : str or list of str
            Performance indicator(s) to analyze (e.g., 'total_error', 'data_error')
        method : int, str, or list, optional
            Method indices or names to analyze. If None, all methods are used
        axis : matplotlib.axes.Axes or ndarray, optional
            Axes for plotting. If None, new figure is created
        file_name : str, optional
            Name for saving the figure
        file_path : str, default=''
            Path for saving the figure
        file_format : str, default='eps'
            Format for saving the figure
        show : bool, default=False
            Whether to display the figure
        fontsize : int, default=10
            Font size for labels and titles
        title : str, bool, list, or None, optional
            Figure title(s). If None, automatic titles are used
        print_info : bool, default=True
            Whether to print statistical information
        print_obj : file-like object, default=sys.stdout
            Object to print statistical information to
        confidence_level : float, default=0.95
            Confidence level for interval calculation (0 < confidence_level < 1)
        xscale : str, optional
            X-axis scale ('linear', 'log', etc.)

        Returns
        -------
        matplotlib.axes.Axes or None
            Axes object if axis is None and show is False

        Raises
        ------
        error.WrongTypeInput
            If indicator is not string or list of strings
        error.WrongValueInput
            If indicator is not in the valid indicator set

        Notes
        -----
        Confidence intervals are calculated using appropriate statistical methods
        based on the distribution of the performance indicator values. The method
        requires stochastic methods with saved executions.

        Examples
        --------
        >>> # Basic confidence interval plot
        >>> case.confint('total_error', show=True)

        >>> # Multiple indicators with custom confidence level
        >>> case.confint(['total_error', 'data_error'], 
        ...              confidence_level=0.99, show=True)

        >>> # Specific method analysis
        >>> case.confint('total_error', method=0, show=True)
        """
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
                                title=tit, confidence_level=confidence_level,
                                xscale=xscale)
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
                                ylabel=names, title=tit, fontsize=fontsize,
                                xscale=xscale)
                n += 1

        if print_info:
            print(message, file=print_obj)

        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format, bbox_inches='tight')
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        elif not show and axis is None:
            return fig, axis

    def __str__(self):
        """
        Return string representation of the case study.
        
        Creates a formatted string containing comprehensive information about
        the case study configuration, including inherited experiment details,
        test data status, and stochastic execution parameters.
        
        Returns
        -------
        str
            Formatted string representation of the case study
            
        Examples
        --------
        >>> print(case)
        CASE STUDY
        Name: my_case
        Method: Born Iterative Method
        Test: test_scenario_1
        Save stochastic runs? yes
        Number of stochastic runs: 50
        """
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

