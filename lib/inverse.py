"""Abstract Inverse Scattering Solver model.

This module provides the abstract class for implementation of any method
which solve the nonlinear inverse scattering problem. Therefore, this
class aims to compute the dielectric map and the total intern field.
"""

# Standard libraries
from abc import ABC, abstractmethod
import sys

# Developed libraries
import configuration as cfg
import result as rst

NAME = 'name'
ALIAS = 'alias'
PARALLELIZATION = 'parallelization'


class InverseSolver(ABC):
    """Abstract inverse solver class.

    This class defines the basic defintion of any implementation of
    inverse solver.

    Attributes
    ----------
        name : str
            The name of the solver.

        version : str
            The version of method.

        config : :class:`configuration.Configuration`
            An object of problem configuration.

        execution_time : float
            The amount of time for a single execution of the method.

    Notes
    -----
        The name of the method should be defined by default.
    """

    def __init__(self, alias='', parallelization=False, import_filename=None,
                 import_filepath=''):
        """Create the object.

        Parameters
        ----------
            configuration : :class:`configuration.Configuration`
                An object of problem configuration.
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

        This is the model routine for any method implementation. The
        input may include other arguments. But the output must always be
        an object of :class:`results.Results`.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An object of the class which defines an instance.

            print_info : bool
                A flag to indicate if information should be displayed or
                not on the screen.

        Returns
        -------
            :class:`results.Results`
        """
        if print_info:
            self._print_title(inputdata, discretization, print_file=print_file)

        return rst.Result(inputdata.name + '_' + self.alias,
                          method_name=self.alias,
                          configuration=inputdata.configuration)

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        """Print the title of the execution.

        Parameters
        ----------
            instance : :class:`results.Results`
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
        return {NAME: self.name,
                ALIAS: self.alias,
                PARALLELIZATION: self.parallelization}
    
    @abstractmethod
    def importdata(self, file_name, file_path=''):
        data = cfg.import_dict(file_name, file_path)
        self.name = data[NAME]
        self.alias = data[ALIAS]
        self.parallelization = data[PARALLELIZATION]
        return data

    def copy(self, new=None):
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
