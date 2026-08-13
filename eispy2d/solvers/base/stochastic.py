import sys
import numpy as np
import copy as cp
from abc import abstractmethod

from eispy2d.core import error
from eispy2d.solvers.base import inverse as inv
from eispy2d.core import result as rst


NEXECUTIONS = 'nexec'
OUTPUTMODE = 'outputmode'


class Stochastic(inv.InverseSolver):
    """Base class for stochastic inverse scattering solvers.

    This class provides the foundation for stochastic inverse scattering
    methods in the eispy2d library. Stochastic solvers use random processes
    during execution and may produce different results on different runs
    even with identical inputs.

    Attributes
    ----------
    nexec : int
        Number of executions to perform.
    outputmode : OutputMode
        Strategy for processing multiple execution results.
    _single_execution : bool
        Whether only one execution is required.
    """
    @property
    def nexec(self):
        """Set the number of executions.

        Parameters
        ----------
        new : int or None
            Number of executions. If None, defaults to 1.
            Must be >= 1.

        Raises
        ------
        WrongValueInput
            If new < 1.
        WrongTypeInput
            If new is not None or int.
        """
        return self._nexec
    @nexec.setter
    def nexec(self, new):
        if new is None:
            self._nexec = 1
            self._single_execution = True
        elif type(new) is int:
            if new < 1:
                raise error.WrongValueInput('Stochastic.nexec', 'new value',
                                            'None or int > 0', str(new))
            elif new == 1:
                self._nexec = 1
                self._single_execution = True
            else:
                self._nexec = new
                self._single_execution = False
        else:
            raise error.WrongTypeInput('Stochastic.nexec', 'new value',
                                       'None or int > 0', str(new))
            
    def __init__(self, outputmode, alias=None, parallelization=False,
                 number_executions=1):
        super().__init__(alias=alias, parallelization=parallelization)
        self.nexec = number_executions
        self.outputmode = outputmode

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        return super().solve(inputdata, discretization, print_info=print_info,
                             print_file=print_file)
    
    @abstractmethod
    def save(self, file_path=''):
        data = super().save(file_path)
        data[NEXECUTIONS] = self.nexec
        data[OUTPUTMODE] = self.outputmode
        return data

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path)
        self.nexec = data[NEXECUTIONS]
        self.outputmode = data[OUTPUTMODE]
    
    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        print('Number of executions: %d' % self.nexec, file=print_file)
        print('Output rule: ' + str(self.outputmode))

    def copy(self, new=None):
        if new is None:
            return Stochastic(self.outputmode, alias=self.alias,
                              parallelization=self.parallelization,
                              number_executions=self.nexec)
        else:
            super().copy(new)
            self.nexec = new.nexec
            self.outputmode = new.outputmode.copy()

    def __str__(self):
        message = super().__str__()
        message += 'Number of execution: %d\n' % self.nexec
        message += 'Output mode: ' + str(self.outputmode) + '\n'
        return message


EACH_EXECUTION = 'each'
BEST_CASE = 'best'
WORST_CASE = 'worst'
AVERAGE_CASE = 'average'


class OutputMode:
    """Processing strategy for stochastic method results.

    Defines how multiple execution results are combined into a single
    output result.

    Parameters
    ----------
    rule : {'each', 'best', 'worst', 'average'}
        Processing rule:
        - 'each': Return all individual results
        - 'best': Return the result with best reference indicator
        - 'worst': Return the result with worst reference indicator
        - 'average': Return averaged results using reference indicator
    reference : str, optional
        Indicator name used for best/worst/average selection (required
        unless rule='each').
    sample_rate : float, default=5
        Sampling rate for averaging (0-100%).

    Methods
    -------
    make(name, method_name, results)
        Process results according to the specified rule.
    copy(new=None)
        Create a copy of the OutputMode object.
    """
    def __init__(self, rule, reference=None, sample_rate=5):
        if (rule != EACH_EXECUTION and rule != BEST_CASE
                and rule != WORST_CASE and rule != AVERAGE_CASE):
            raise error.WrongValueInput('OutputMode.__init__', 'rule', "'each'"
                                        + " or 'best' or 'worst' or 'average'",
                                        rule)
        elif (rule != EACH_EXECUTION
                and (reference is None or not rst.check_indicator(reference))):
            raise error.WrongValueInput('OutputMode.__init__', 'reference',
                                        'a valid indicator', reference)
        elif sample_rate < 0 or sample_rate > 100:
            raise error.WrongValueInput('OutputMode.__init__', 'sample_rate',
                                        'between 0 and 100%%',
                                        str(sample_rate))
        self.rule = rule
        self.reference = reference
        self.sample_rate = sample_rate

    def make(self, name, method_name, results):
        """Process results according to the specified rule.

        Parameters
        ----------
        name : str
            Name for the result object.
        method_name : str
            Name of the method that generated the results.
        results : list of Result or Result
            Results to process.

        Returns
        -------
        Result or list of Result
            Processed result(s) according to the rule.
        """
        if type(results) is rst.Result:
            results.name = name
            results.method_name = method_name
            return results
        elif len(results) == 1:
            results[0].name = name
            results[0].method_name = method_name
            return results[0]
        elif self.rule == EACH_EXECUTION:
            return results
        elif self.rule == AVERAGE_CASE:
            N = len(results)
            data = np.zeros(N)
            for n in range(N):
                data[n] = results[n].final_value(self.reference)
            dist = np.abs(data-np.mean(data))
            i = np.argmin(dist)
            result = rst.Result(
                name=name, method_name=method_name,
                configuration=results[0].configuration.copy(),
                scattered_field=cp.deepcopy(results[i].scattered_field),
                total_field=cp.deepcopy(results[i].total_field),
                rel_permittivity=cp.deepcopy(results[i].rel_permittivity),
                conductivity=cp.deepcopy(results[i].conductivity)
            )
            rates = np.arange(0, 100+self.sample_rate, self.sample_rate)/100
            for indicator in rst.INDICATOR_SET:
                data = getattr(results[0], indicator)
                if data is None or (type(data) is list and len(data) == 0):
                    continue
                elif type(data) is int or type(data) is float:
                    values = np.zeros(N)
                    values[0] = data
                    for n in range(1, N):
                        values[n] = getattr(results[n], indicator)
                    setattr(result, indicator, np.mean(values))
                else:
                    data = np.zeros((N, rates.size))
                    for n in range(N):
                        values = getattr(results[n], indicator)
                        indexes = rates*(len(values)-1)
                        indexes = indexes.astype(int)
                        data[n, :] = [values[i] for i in indexes]
                    setattr(result, indicator, np.mean(data, axis=0))
            return result
        elif self.rule == WORST_CASE:
            N = len(results)
            values = np.zeros(N)
            for n in range(N):
                values[n] = results[n].final_value(self.reference)
            n = np.argmax(values)
            result = results[n].copy()
            result.name = name
            result.method_name = method_name
            return result
        elif self.rule == BEST_CASE:
            N = len(results)
            values = np.zeros(N)
            for n in range(N):
                values[n] = results[n].final_value(self.reference)
            n = np.argmin(values)
            result = results[n].copy()
            result.name = name
            result.method_name = method_name
            return result

    def copy(self, new=None):
        if new is None:
            return OutputMode(self.rule, self.reference,
                              sample_rate=self.sample_rate)
        else:
            self.rule = new.rule
            self.reference = new.reference
            self.sample_rate = new.sample_rate

    def __str__(self):
        message = self.rule
        if self.rule != EACH_EXECUTION:
            message += (', reference: ' + self.reference
                        + ', sample rate: %.1f %%' % self.sample_rate)
        return message