from abc import ABC, abstractmethod
import numpy as np
import configuration as cfg
import result as rst
import inverse as inv
import discretization as dct
import error

NAME = 'name'
METHOD = 'method'
DISCRETIZATION = 'discretization'
RESULTS = 'results'

RANDOM_POLYGONS_PATTERN = 'random_polygons'
REGULAR_POLYGONS_PATTERN = 'regular_polygons'
SURFACES_PATTERN = 'surfaces'
FIXED_CONTRAST = 'fixed'
MAXIMUM_CONTRAST = 'maximum'
FIXED_SIZE = 'fixed'
MAXIMUM_SIZE = 'maximum'
SINGLE_OBJECT = 'single'
FIXED_NUMBER = 'fixed'
MAXIMUM_NUMBER = 'max_number'
MINIMUM_DENSITY = 'min_density'
LABEL_INSTANCE = 'Instance Index'


class Experiment(ABC):

    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, new):
        if new is None:
            self._method = None
            self._single_method = None
        elif issubclass(type(new), inv.InverseSolver):
            self._method = new.copy()
            self._single_method = True
        elif type(new) is list:
            self._method = [new[i].copy() for i in range(len(new))]
            self._single_method = False

    @property
    def discretization(self):
        return self._discretization

    @discretization.setter
    def discretization(self, new):
        if new is None:
            self._discretization = None
            self._single_discretization = None
        elif issubclass(type(new), dct.Discretization):
            self._discretization = new.copy()
            self._single_discretization = True
        elif type(new) is list:
            self._discretization = [new[i].copy() for i in range(len(new))]
            self._single_discretization = False

    def __init__(self, name, method=None, discretization=None):
        if type(name) is not str:
            raise error.WrongTypeInput('Experiment.__init__',
                                       'name',
                                       'str',
                                       str(type(name)))
        if (method is not None
                and not issubclass(type(method), inv.InverseSolver)
                and type(method) is not list):
            raise error.WrongTypeInput('Experiment.__init__',
                                       'methods',
                                       'None, Solver, or Solver-list',
                                       str(type(method)))
        if (discretization is not None
                and not issubclass(type(discretization), dct.Discretization)
                and type(discretization) is not list):
            raise error.WrongTypeInput('Experiment.__init__',
                                       'discretization',
                                       'None, Discretization, '
                                       + 'or Discretization-list',
                                       str(type(discretization)))
        if (type(discretization) is list and type(method) is list
                and len(discretization) != len(method)):
            raise error.Error('Experiment.__init__: discretization list must'
                              + ' have the same size as method list.')
        self.name = name
        self.method = method
        self.discretization = discretization
        self.results = None

    @abstractmethod
    def save(self, file_path=''):
        return {NAME: self.name,
                METHOD: self.method,
                DISCRETIZATION: self.discretization,
                RESULTS: self.results}

    @abstractmethod
    def importdata(self, file_name, file_path=''):
        data = cfg.import_dict(file_name, file_path)
        self.name = data[NAME]
        self.method = data[METHOD]
        self.discretization = data[DISCRETIZATION]
        self.results = data[RESULTS]
        return data

    def _search_method(self, alias):
        if type(alias) is not str and type(alias) is not list:
            raise error.WrongTypeInput('Experiment._check_method', 'alias',
                                       'str or str-list', str(type(alias)))
        if type(alias) is str:
            if self._single_method:
                return self.method.alias == alias
            else:
                for n in range(len(self.method)):
                    if alias == self.method[n].alias:
                        return n
                return False            
        else:
            if self._single_method:
                for n in range(len(alias)):
                    if alias[n] == self.method.alias:
                        return n
                return False
            else:
                idx = []
                for m in range(len(alias)):
                    for n in range(len(self.method)):
                        if alias[m] == self.method[n].alias:
                            idx.append(n)
                            break
                if len(idx) > 0:
                    return idx
                else:
                    return False

    def _print_compare1sample(self, sample_name, reference, output):
        statistic, pvalue, alternative, nonnormal, transf, delta = output
        if not nonnormal:
            message = 'T-Test 1 Sample'
            if transf is not None:
                message += '(' + transf + ' transformation)\n'
            else:
                message += '\n'
        else:
            message = 'Wilcoxon signed-rank test (non-normal data)\n'
        message += 'Data: ' + sample_name + '\n'
        message += 'Statistic: %.3f' % statistic + ', p-value: %.1e\n' % pvalue
        message += 'Alternative hypothesis: '
        if not nonnormal:
            message += 'true difference in means is not '
        else:
            message += 'true location shift is not '
        if alternative == 'two-sided':
            message += 'equal to '
        elif alternative == 'greater':
            message += alternative + ' than '
        else:
            message += alternative + ' than '
        message += '%.2f\n' % reference
        if delta is not None:
            message += 'Effect size for power of 0.8: %.3e\n' % delta
        return message

    def _print_compare2sample(self, sample1_name, sample2_name, output,
                              paired):
        statistic = output[0]
        pvalue = output[1]
        alternative = output[2]
        delta = output[3]
        nonnormal = output[4]
        transformation = output[5]
        equal_var = output[6]
        message = ''
        if not nonnormal and not paired and equal_var:
            message += 'Two Sample T-Test\n'
        elif not nonnormal and not paired and not equal_var:
            message += 'Welch Two Sample T-Test\n'
        elif not nonnormal and paired and equal_var:
            message += 'Paired T-Test\n'
        elif not nonnormal and not paired and not equal_var:
            message += 'Welch Paired T-Test\n'
        elif not nonnormal and nonnormal:
            message += 'Mann-Whitney U test (Non-parametric)\n'
        message += 'Data: ' + sample1_name + ' and ' + sample2_name
        if transformation is not None:
            message += ' (Transformation: ' + transformation + ')\n'
        else:
            message += '\n'
        message += 'Statistic: %.3f' % statistic + ', p-value: %.3e\n' % pvalue
        message += 'Alternative Hypothesis: '
        if not nonnormal:
            message += 'true difference in means is not '
        else:
            message += 'true location shift is not '
        if alternative == 'two-sided':
            message += 'equal to '
        elif alternative == 'greater':
            message += 'greater than '
        elif alternative == 'less':
            message += 'less than '
        message += '0\n'
        if delta is not None:
            message += 'Effect size for 0.8 power: %.3e\n' % delta
        return message

    def _print_compare_multiple(self, samples_names, output, all2one=None,
                                extra_data_info=None):
        statistic = output[0]
        pvalue=output[1]
        nonnormal=output[2]
        transformation=output[3]
        homocedascity=output[4]
        all2all_out=output[5]
        all2one_out=output[6]
        message = ''
        if not nonnormal and homocedascity:
            message += 'One-Way Analysis of Variance\n'
        elif not nonnormal and not homocedascity:
            message += 'Welch One-Way Analysis of Variance\n'
        elif nonnormal:
            message += 'Kruskal-Wallis H-Test\n'
        message += 'Data: '
        if extra_data_info is not None:
            message += extra_data_info + ' '
        for n in range(len(samples_names)-1):
            message += samples_names[n] + ', '
        message += samples_names[-1]
        if transformation is not None:
            message += ' (Transformation: ' + transformation + ')\n'
        else:
            message += '\n'
        message += 'Statistic: %.4f' % statistic + ', p-value: %.3e\n' % pvalue
        if all2all_out is not None:
            message += 'All-to-all comparison method: '
            if not nonnormal and homocedascity:
                message += "Tukey's Honest Significant Difference\n"
            elif not nonnormal and not homocedascity:
                message += ('Multiple Welch Two Sample T-Test with Bonferroni '
                            + 'correction\n')
            elif nonnormal:
                message += 'Multiple Mann-Whitney U test (Non-parametric)\n'
            n = 0
            for i in range(len(samples_names)-1):
                for j in range(i+1, len(samples_names)):
                    message += ('* ' + samples_names[i] + ' = '
                                + samples_names[j])
                    if not nonnormal:
                        H0, pvalue, confint = all2all_out[n]
                        message += (', H0: ' + str(H0)
                                    + ', p-value: %.3e, ' % pvalue
                                    + 'Confi. Inter. (%.2e, ' % confint[0]
                                    + '%.2e)\n' % confint[1])
                    elif nonnormal:
                        message += ', p-value: %.3e\n' % all2all_out[n]
                    n += 1
        if all2one_out is not None:
            if all2one is None:
                raise error.MissingInputError(
                    'Experiment._print_compare_multiple', 'all2one'
                )
            elif type(all2one) is int:
                ref = all2one
            elif type(all2one) is str:
                for n in range(len(samples_names)):
                    if samples_names[n] == all2one:
                        ref = n
                        break
                    else:
                        ref = False
                if type(ref) is bool and ref is False:
                    raise error.WrongValueInput(
                        'Experiment._print_compare_multiple', 'all2one',
                        str(samples_names), all2one
                    )
            else:
                raise error.WrongTypeInput(
                    'Experiment._print_compare_multiple', 'all2one',
                    'int or str', str(type(all2one))
                )
            message += 'All-to-one comparison method: '
            if not nonnormal and homocedascity:
                message += "Dunnett's Test\n"
            elif not nonnormal and not homocedascity:
                message += ('Multiple Welch Two Sample T-Test with Bonferroni '
                            + 'correction\n')
            elif nonnormal:
                message += 'Multiple Mann-Whitney U test (Non-parametric)\n'
            n = 0
            for i in range(len(samples_names)):
                if i == ref:
                    continue
                message += ('* ' + samples_names[i] + ' = '
                                + samples_names[ref])
                if not nonnormal and homocedascity:
                    message += ', H0: ' + str(all2one_out[n]) + '\n'
                elif not nonnormal and not homocedascity:
                    H0, pvalue, confint = all2one_out[n]
                    message += (', H0: ' + str(H0)
                                + ', p-value: %.3e, ' % pvalue
                                + 'Confi. Inter. (%.2e, ' % confint[0]
                                + '%.2e)\n' % confint[1])
                elif nonnormal:
                    message += ', p-value: %.3e\n' % all2one_out[n]
                n += 1
        return message

    def _print_confint(self, sample_name, output, confidence_level):
        cfi, normality, transformation = output
        message = ''
        message += '* ' + sample_name + ', '
        message += 'Normality: ' + str(normality)
        if transformation is not None:
            message += ' (Transformation: ' + transformation + '), '
        else:
            message += ', '
        message += ('%.1f Confi. In.: ' % (confidence_level*100)
                    + '(%.2e, ' % cfi[0] + '%.2e)\n' % cfi[1])
        return message

    def _print_non_normal_data(self, sample_name):
        return '* ' + sample_name + ': not succeed on normality test.'

    def __str__(self):
        message = 'Name: ' + self.name + '\n'
        message += 'Method: '
        if self.method is None:
            message += 'None\n'
        elif self._single_method:
            message += self.method.alias + '\n'
        else:
            for m in range(len(self.method)-1):
                message += self.method[m].alias + ', '
            message += self.method[-1].alias + '\n'
        message += 'Discretization: '
        if self.discretization is None:
            message += 'None\n'
        elif self._single_discretization:
            message += self.discretization.name + '\n'
        else:
            for m in range(len(self.discretization)-1):
                message += self.discretization[m].name + ', '
            message += self.discretization[-1].name + '\n'
        message += 'Results: '
        if self.results is None:
            message += 'None\n'
        elif len(self.results) == 0:
            message += 'empty\n'
        else:
            message += 'done\n'
        return message       
    
def final_value(indicator, result):
    if type(indicator) is not str and not(type(indicator) is list
                                          and all(isinstance(n, str)
                                                  for n in indicator)):
        raise error.WrongTypeInput('experiment.final_value', 'indicator',
                                   'str or str-list', str(type(indicator)))
    if not rst.check_indicator(indicator):
        raise error.WrongValueInput('experiment.final_value', 'indicator',
                                    str(rst.INDICATOR_SET), indicator)
    if (type(result) is not rst.Result
            and type(result) is not np.ndarray 
                     and type(result) is not list):
        raise error.WrongTypeInput('experiment.final_value', 'result',
                                   'Result or Result-numpy.ndarray '
                                   + 'or Result-list', str(type(result)))
    if type(indicator) is str:
        if type(result) is rst.Result:
            return result.final_value(indicator)                    
        elif type(result) is list:
            return np.array([result[n].final_value(indicator) 
                             for n in range(len(result))])
        else:
            output = np.array([result.flatten()[n].final_value(indicator)
                               for n in range(result.size)])
            return output.reshape(result.shape)
    else:
        if type(result) is rst.Result:
            return np.array([result.final_value(n) for n in indicator])
        elif type(result) is list:
            return np.array([[result[n].final_value(m) for n in
                              range(result.size)]
                             for m in indicator])    
        else:
            return np.array([[result.flatten()[n].final_value(m)
                              for n in result.size] for m in indicator])    
                    