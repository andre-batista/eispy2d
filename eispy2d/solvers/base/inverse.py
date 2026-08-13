"""Abstract Inverse Scattering Solver model.

This module provides the abstract class for implementation of any method
which solve the nonlinear inverse scattering problem. Therefore, this
class aims to compute the dielectric map and the total intern field.
"""

# Standard libraries
from abc import ABC, abstractmethod
import sys

# Developed libraries
from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst

NAME = 'name'
ALIAS = 'alias'
PARALLELIZATION = 'parallelization'


class InverseSolver(ABC):
    """Abstract base class for inverse scattering solvers.

    This class defines the basic interface for any implementation of an
    inverse solver for electromagnetic scattering problems.

    Attributes
    ----------
    name : str
        The name of the solver.
    alias : str
        Short identifier for the solver.
    parallelization : bool
        Whether parallel processing is enabled.
    execution_time : float
        Execution time for a single run of the method (set by derived classes).

    Methods
    -------
    solve(inputdata, discretization, print_info=True, print_file=sys.stdout)
        Solve the inverse scattering problem.
    save(file_path='')
        Save solver state to file.
    importdata(file_name, file_path='')
        Load solver state from file.
    copy(new=None)
        Create a copy of the solver instance.
    """

    def __init__(self, alias='', parallelization=False, import_filename=None,
                 import_filepath=''):
        """Create an inverse solver object.

        Parameters
        ----------
        alias : str, default=''
            Short identifier for the solver.
        parallelization : bool, default=False
            Whether to enable parallel processing.
        import_filename : str, optional
            If provided, imports solver configuration from this file.
        import_filepath : str, default=''
            Path to the import file.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            self.name = ''
            self.alias = alias
            self.parallelization = parallelization


    @abstractmethod
    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        """Solve the inverse scattering problem.

        This is the main routine for any method implementation. The input
        may include additional arguments, but the output must always be a
        Result object.

        Parameters
        ----------
        inputdata : InputData
            Input data object defining the problem instance.
        discretization : Discretization
            Discretization scheme to use.
        print_info : bool, default=True
            Whether to display progress information.
        print_file : file-like object, default=sys.stdout
            Output stream for printed information.

        Returns
        -------
        Result
            Result object containing the reconstruction results.
        """
        if print_info:
            self._print_title(inputdata, discretization, print_file=print_file)

        return rst.Result(inputdata.name + '_' + self.alias,
                          method_name=self.alias,
                          configuration=inputdata.configuration)

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        """Print the execution title header.

        Parameters
        ----------
        inputdata : InputData
            Input data object for the problem.
        discretization : Discretization
            Discretization scheme being used.
        print_file : file-like object, default=sys.stdout
            Output stream for printed information.
        """
        print("==============================================================",
              file=print_file)
        print('Method: ' + self.name, file=print_file)
        if self.alias != '':
            print('Alias: ' + self.alias, file=print_file)
        print('Input Data: ' + inputdata.name, file=print_file)
        if discretization is not None:
            print('Discretization: ' + discretization.name, file=print_file)
        if self.parallelization is not None:
            print('Parallelization: ' + str(self.parallelization),
                  file=print_file)

    @abstractmethod
    def save(self, file_path=''):
        """Save solver configuration to file.

        Parameters
        ----------
        file_path : str, default=''
            Base path for saving the configuration.

        Returns
        -------
        dict
            Dictionary containing the serialized solver data.
        """
        return {NAME: self.name,
                ALIAS: self.alias,
                PARALLELIZATION: self.parallelization}
    
    @abstractmethod
    def importdata(self, file_name, file_path=''):
        """Import solver configuration from file.

        Parameters
        ----------
        file_name : str
            Name of the file to import from.
        file_path : str, default=''
            Path to the import file.

        Returns
        -------
        dict
            Dictionary containing the imported data.
        """
        data = cfg.import_dict(file_name, file_path)
        self.name = data[NAME]
        self.alias = data[ALIAS]
        self.parallelization = data[PARALLELIZATION]
        return data

    def copy(self, new=None):
        """Create a copy of the solver instance.

        Parameters
        ----------
        new : InverseSolver, optional
            If provided, copies configuration into this instance.
            If None, creates a new instance.

        Returns
        -------
        InverseSolver or None
            New instance if new=None, otherwise None.
        """
        if new is None:
            return InverseSolver(self.alias, self.parallelization)
        else:
            self.alias = new.alias
            self.parallelization = new.parallelization

    def __str__(self):
        message = 'Inverse Solver: ' + self.name + '\n'
        message += 'Alias: ' + self.alias + '\n'
        if self.parallelization is not None:
            message += 'Parallelization: ' + str(self.parallelization) + '\n'
        return message
