from abc import ABC, abstractmethod
import numpy as np

from eispy2d.api import api
from eispy2d.core import error, configuration as cfg

NAME = "name"
WAVELENGTH = "wavelength"
IMAGE_SIZE = "image_size"
NUMBER_MEASUREMENTS = "number_measurements"
NUMBER_SOURCES = "number_sources"
OBSERVATION_RADIUS = "observation_radius"
RESOLUTION = "resolution"
BACKGROUND_PERMITTIVITY = "background_permittivity"
NOISE_LEVEL = "noise_level"
SHAPE = "shape"
SAMPLE_SIZE = "sample_size"
RESULTS = "results"


class Experiment(ABC):

    def __init__(self, name):
        if type(name) is not str:
            raise error.WrongTypeInput('Experiment.__init__',
                                        'name',
                                        'str',
                                        str(type(name)))

        self.name = name
        self.wavelength = None
        self.image_size = None
        self.number_measurements = None
        self.number_sources = None
        self.observation_radius = None
        self.resolution = None
        self.background_permittivity = None
        self.noise_level = None
        self.shape = None
        self.sample_size = None
        self.results = None

    @abstractmethod
    def save(self, file_path=''):
        return {
            NAME: self.name,
            WAVELENGTH: self.wavelength,
            IMAGE_SIZE: self.image_size,
            NUMBER_MEASUREMENTS: self.number_measurements,
            NUMBER_SOURCES: self.number_sources,
            OBSERVATION_RADIUS: self.observation_radius,
            RESOLUTION: self.resolution,
            BACKGROUND_PERMITTIVITY: self.background_permittivity,
            NOISE_LEVEL: self.noise_level,
            SHAPE: self.shape,
            SAMPLE_SIZE: self.sample_size,
            RESULTS: self.results
        }

    @abstractmethod
    def importdata(self, file_name, file_path=''):
        data = cfg.import_dict(file_name, file_path)
        self.name = data[NAME]
        self.wavelength = data[WAVELENGTH]
        self.image_size = data[IMAGE_SIZE]
        self.number_measurements = data[NUMBER_MEASUREMENTS]
        self.number_sources = data[NUMBER_SOURCES]
        self.observation_radius = data[OBSERVATION_RADIUS]
        self.resolution = data[RESOLUTION]
        self.background_permittivity = data[BACKGROUND_PERMITTIVITY]
        self.noise_level = data[NOISE_LEVEL]
        self.shape = data[SHAPE]
        self.sample_size = data[SAMPLE_SIZE]
        self.results = data[RESULTS]
        return data

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

    def _print_compare2sample(self, sample1_name, sample2_name, output, paired):
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
        elif not paired and nonnormal:
            message += 'Mann-Whitney U test (Non-parametric)\n'
        elif paired and nonnormal:
            message += 'Wilcoxon Signed-Rank test (Non-parametric)\n'
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
                                extra_data_info=None, paired=False):
        statistic = output[0]
        pvalue = output[1]
        nonnormal = output[2]
        transformation = output[3]
        homocedascity = output[4]
        all2all_out = output[5]
        all2one_out = output[6]
        message = ''
        if not nonnormal and paired:
            message += 'Randomized Complete Block Design\n'
        elif not nonnormal and homocedascity:
            message += 'One-Way Analysis of Variance\n'
        elif not nonnormal and not homocedascity:
            message += 'Welch One-Way Analysis of Variance\n'
        elif nonnormal and paired:
            message += 'Friedman Rank Sum Test\n'
        elif nonnormal and not paired:
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
            if not nonnormal and paired:
                message += ('Multiple Paired T-Test with Bonferroni '
                            + 'correction\n')
            elif not nonnormal and homocedascity:
                message += "Tukey's Honest Significant Difference\n"
            elif not nonnormal and not homocedascity:
                message += ('Multiple Welch Two Sample T-Test with Bonferroni '
                            + 'correction\n')
            elif nonnormal and paired:
                message += ('Multiple Wilcoxon Signed-Rank test '
                            + '(Non-parametric)\n')
            elif nonnormal and not paired:
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
            if not nonnormal and paired:
                message += ('Multiple Paired T-Test with Bonferroni '
                            + 'correction\n')
            elif not nonnormal and homocedascity:
                message += "Dunnett's Test\n"
            elif not nonnormal and not homocedascity:
                message += ('Multiple Welch Two Sample T-Test with Bonferroni '
                            + 'correction\n')
            elif nonnormal and paired:
                message += ('Multiple Wilcoxon Signed-Rank test '
                            + '(Non-parametric)\n')
            elif nonnormal and not paired:
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
        
        from scipy.special import inv_boxcox
        cfi, normality, transformation = output
        message = ''
        message += '* ' + sample_name + ', '
        message += 'Normality: ' + str(normality)
        if transformation == 'log':
            a = np.exp(cfi[0])
            b = np.exp(cfi[1])
        elif transformation == 'sqrt':
            a = cfi[0]**2
            b = cfi[1]**2
        elif transformation is not None and transformation.startswith('boxcox'):
            lambda_value = float(transformation.split('=')[1].split(')')[0])
            a = inv_boxcox(cfi[0], lambda_value)
            b = inv_boxcox(cfi[1], lambda_value)
        else:
            a = cfi[0]
            b = cfi[1]
        message += ('%.1f Confi. In.: ' % (confidence_level*100)
                    + '(%.2e, ' % a + '%.2e)\n' % b)
        return message

    def _print_non_normal_data(self, sample_name):
        
        return '* ' + sample_name + ': not succeed on normality test.'

    def __str__(self):
        message = 'Name: ' + self.name + '\n'
        message += 'Wavelength: ' + str(self.wavelength) + '\n'
        message += 'Image size: ' + str(self.image_size) + '\n'
        message += 'Number measurements: ' + str(self.number_measurements) + '\n'
        message += 'Number sources: ' + str(self.number_sources) + '\n'
        message += 'Observation radius: ' + str(self.observation_radius) + '\n'
        message += 'Resolution: ' + str(self.resolution) + '\n'
        message += 'Background permittivity: ' + str(self.background_permittivity) + '\n'
        message += 'Noise level: ' + str(self.noise_level) + '\n'
        message += 'Shape: ' + str(self.shape) + '\n'
        message += 'Sample size: ' + str(self.sample_size) + '\n'
        message += 'Results: ' + ('done' if self.results is not None else 'None') + '\n'
        return message