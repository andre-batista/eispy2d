"""A module for results information.

The results module provides the :class:`Results` which contains the
resultant information of a single execution of a method for a given
input data and the corresponding problem configuration. The class is also
a tool for plotting results. The following class is defined

:class:`Results`
    a class for storing results information of a single execution.

The list of routines...
"""

from math import pi
import pickle
import copy as cp
import numpy as np
from scipy.stats import linregress
from skimage import measure
from statsmodels.graphics.boxplots import violinplot
import matplotlib.pyplot as plt
import error
import configuration as cfg


# Strings for easier implementation of plots
XLABEL_STANDARD = r'x [$\lambda_b$]'
YLABEL_STANDARD = r'y [$\lambda_b$]'
XLABEL_UNDEFINED = r'x [$L_x$]'
YLABEL_UNDEFINED = r'y [$L_y$]'
COLORBAR_REL_PERMITTIVITY = r'$\epsilon_r$'
COLORBAR_CONDUCTIVITY = r'$\sigma$ [S/m]'
TITLE_REL_PERMITTIVITY = 'Relative Permittivity'
TITLE_CONDUCTIVITY = 'Conductivity'
TITLE_RECOVERED_REL_PERMITTIVITY = ('Recovered '
                                         + TITLE_REL_PERMITTIVITY)
TITLE_RECOVERED_CONDUCTIVITY = 'Recovered ' + TITLE_CONDUCTIVITY
TITLE_ORIGINAL_REL_PERMITTIVITY = ('Original '
                                        + TITLE_REL_PERMITTIVITY)
TITLE_ORIGINAL_CONDUCTIVITY = 'Original ' + TITLE_CONDUCTIVITY
LABEL_ZETA_RN = r'$\zeta_{RN} [V/m]$'
LABEL_ZETA_RPAD = r'$\zeta_{RPAD}$ [\%/sample]'
LABEL_ZETA_EPAD = r'$\zeta_{\epsilon PAD}$ [\%/pixel]'
LABEL_ZETA_EBE = r'$\zeta_{\epsilon BE}$ [\%/pixel]'
LABEL_ZETA_EOE = r'$\zeta_{\epsilon OE}$ [\%/pixel]'
LABEL_ZETA_SAD = r'$\zeta_{\sigma AD}$ [S/pixel]'
LABEL_ZETA_SBE = r'$\zeta_{\sigma BE}$ [S/pixel]'
LABEL_ZETA_SOE = r'$\zeta_{\sigma OE}$ [S/pixel]'
LABEL_ZETA_TV = r'$\zeta_{TV}$'
LABEL_ZETA_P = r'$\zeta_{P}$ [\%]'
LABEL_ZETA_S = r'$\zeta_{S}$ [\%]'
LABEL_ZETA_TFMPAD = r'$\zeta_{TFMPAD}$ [\%/pixel]'
LABEL_ZETA_TFPPAD = r'$\zeta_{TFPPAD}$ [\%/rad]'
LABEL_EXECUTION_TIME = r'$t_{exe}$ [sec]'
LABEL_OBJECTIVE_FUNCTION = r'$f(\chi, E_z^s)$'
LABEL_NUMBER_EVALUATIONS = 'Evaluations'
LABEL_NUMBER_ITERATIONS = 'Iterations'

IMAGE_SIZE_SINGLE = (6., 5.)
IMAGE_SIZE_1x2 = (9., 4.) # 9 x 5
IMAGE_SIZE_2X2 = (9., 9.)

# Constant string for easier access of dictionary fields
NAME = 'name'
CONFIGURATION = 'configuration'
INPUT_FILENAME = 'input_filename'
INPUT_FILEPATH = 'input_filepath'
METHOD_NAME = 'method_name'
TOTAL_FIELD = 'total_field'
SCATTERED_FIELD = 'scattered_field'
REL_PERMITTIVITY = 'rel_permittivity'
CONDUCTIVITY = 'conductivity'
EXECUTION_TIME = 'execution_time'
NUMBER_EVALUATIONS = 'number_evaluations'
NUMBER_ITERATIONS = 'number_iterations'
OBJECTIVE_FUNCTION = 'objective_function'
RESIDUAL_NORM_ERROR = 'zeta_rn'
RESIDUAL_PAD_ERROR = 'zeta_rpad'
REL_PERMITTIVITY_PAD_ERROR = 'zeta_epad'
CONDUCTIVITY_AD_ERROR = 'zeta_sad'
TOTAL_VARIATION = 'zeta_tv'
POSITION_ERROR = 'zeta_p'
SHAPE_ERROR = 'zeta_s'
REL_PERMITTIVITY_BACKGROUND_ERROR = 'zeta_ebe'
REL_PERMITTIVITY_OBJECT_ERROR = 'zeta_eoe'
CONDUCTIVITY_BACKGROUND_ERROR = 'zeta_sbe'
CONDUCTIVITY_OBJECT_ERROR = 'zeta_soe'
TOTALFIELD_MAGNITUDE_PAD = 'zeta_tfmpad'
TOTALFIELD_PHASE_AD = 'zeta_tfpad'
PERMITTIVITY = 'epsilon_r'
CONDUCTIVITY = 'sigma'
BOTH_PROPERTIES = 'both'
CONTRAST = 'contrast'
TOTAL_FIELD = 'total field'

INDICATOR_SET = [RESIDUAL_NORM_ERROR, RESIDUAL_PAD_ERROR,
                 REL_PERMITTIVITY_PAD_ERROR, REL_PERMITTIVITY_BACKGROUND_ERROR,
                 REL_PERMITTIVITY_OBJECT_ERROR, CONDUCTIVITY_AD_ERROR,
                 CONDUCTIVITY_OBJECT_ERROR, CONDUCTIVITY_BACKGROUND_ERROR,
                 TOTALFIELD_MAGNITUDE_PAD, TOTALFIELD_PHASE_AD,
                 TOTAL_VARIATION, SHAPE_ERROR, POSITION_ERROR, EXECUTION_TIME,
                 OBJECTIVE_FUNCTION, NUMBER_EVALUATIONS, NUMBER_ITERATIONS]

LABELS = {RESIDUAL_NORM_ERROR: r'$\zeta_{RN}$ (V/m)',
          RESIDUAL_PAD_ERROR: r'$\zeta_{RPAD}$ [%/sample]',
          REL_PERMITTIVITY_PAD_ERROR: r'$\zeta_{\epsilon PAD}$ [%/pixel]',
          REL_PERMITTIVITY_BACKGROUND_ERROR: r'$\zeta_{\epsilon BE}$ [%/pixel]',
          REL_PERMITTIVITY_OBJECT_ERROR: r'$\zeta_{\epsilon OE}$ [%/pixel]',
          CONDUCTIVITY_AD_ERROR: r'$\zeta_{\sigma AD}$ [S/m]',
          CONDUCTIVITY_OBJECT_ERROR: r'$\zeta_{\sigma OE}$ [S/m]',
          CONDUCTIVITY_BACKGROUND_ERROR: r'$\zeta_{\sigma BE}$ [S/m]',
          TOTALFIELD_MAGNITUDE_PAD: r'$\zeta_{TFMPAD}$ [%/pixel]',
          TOTALFIELD_PHASE_AD: r'$\zeta_{TFPAD}$ [rad/pixel]',
          TOTAL_VARIATION: r'$\zeta_{tv}$',
          SHAPE_ERROR: r'$\zeta_{S}$ [%]',
          POSITION_ERROR: r'$\zeta_{P}$ [%]',
          EXECUTION_TIME: 'Execution Time [sec]',
          OBJECTIVE_FUNCTION: 'Objective Function',
          NUMBER_EVALUATIONS: 'Evaluations',
          NUMBER_ITERATIONS: 'Iterations'}

TITLES = {RESIDUAL_NORM_ERROR: 'Residual Norm',
          RESIDUAL_PAD_ERROR: 'Residual PAD',
          REL_PERMITTIVITY_PAD_ERROR: 'Rel. Per. PAD',
          REL_PERMITTIVITY_BACKGROUND_ERROR: 'Background Rel. Per. PAD',
          REL_PERMITTIVITY_OBJECT_ERROR: 'Object Rel. Per. PAD',
          CONDUCTIVITY_AD_ERROR: 'Conductivity AD',
          CONDUCTIVITY_OBJECT_ERROR: 'Object Con. AD',
          CONDUCTIVITY_BACKGROUND_ERROR: 'Background Con. AD',
          TOTALFIELD_MAGNITUDE_PAD: 'To. Field Mag. PAD',
          TOTALFIELD_PHASE_AD: 'To. Field Phase AD',
          TOTAL_VARIATION: 'Total Variation',
          SHAPE_ERROR: 'Shape error',
          POSITION_ERROR: 'Position error',
          EXECUTION_TIME: 'Execution Time',
          OBJECTIVE_FUNCTION: 'Ob. Func. Evaluation',
          NUMBER_EVALUATIONS: 'Evaluations',
          NUMBER_ITERATIONS: 'Iterations'}


class Result:
    """A class for storing results information of a single execution.

    Each executation of method for a giving input data with
    corresponding configuration will result in information which will be
    stored in this class.

    Attributes
    ----------
        name
            A string identifying the stored result. It may be a
            combination of the method, input data and configuration
            names.

        method_name
            A string with the name of the method which yield this result.

        configuration_filename
            A string containing the file name in which configuration
            info is stored.

        configuration_filepath
            A string containing the path to the file which stores the
            configuration info.

        input_filename
            A string containing the file name in which instance info is
            stored.

        input_filepath
            A string containing the path to the file which stores the
            instance info.

        et
            The total field matrix which is estimated by the inverse
            nonlinear solver. Unit: [V/m]

        es
            The scattered field matrix resulting from the estimation of
            the total field and contrast map. Unit: [V/m]

        epsilon_r
            The image matrix containing the result of the relative
            permittivity map estimation.

        sigma
            The image matrix containing the result of the conductivity
            map estimation. Unit: [S/m]

        execution_time
            The amount of time for running the method.

        number_evaluations
            Number of times in which the objective function was
            evaluated (for stochastic algorithms).

        number_iterations
            Number of iterations.

        objective_function
            The array with the recorded evaluations of the objective
            function throughout the iterations.
    """
    def __init__(self, name=None, method_name=None,
                 configuration=None, scattered_field=None,
                 total_field=None, rel_permittivity=None,
                 conductivity=None, execution_time=None,
                 number_evaluations=None, objective_function=None,
                 number_iterations=None, import_filename=None,
                 import_filepath=''):
        """Build the object.

        You may provide here the value of all attributes. But only name
        is required.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            if name is None:
                raise error.MissingInputError('Results.__init__()', 'name')
            self.name = name
            self.method_name = method_name
            self.configuration = configuration
            self.total_field = total_field
            self.scattered_field = scattered_field
            self.rel_permittivity = rel_permittivity
            self.conductivity = conductivity
            self.execution_time = execution_time
            self.number_evaluations = number_evaluations
            self.number_iterations = number_iterations
            self.zeta_rn, self.zeta_rpad = list(), list()
            self.zeta_epad, self.zeta_sad = list(), list()
            self.zeta_tv, self.zeta_p, self.zeta_s = list(), list(), list()
            self.zeta_ebe, self.zeta_sbe = list(), list()
            self.zeta_eoe, self.zeta_soe = list(), list()
            self.zeta_tfmpad, self.zeta_tfpad = list(), list()
            if objective_function is None:
                self.objective_function = list()
            else:
                self.objective_function = objective_function

    def save(self, file_path=''):
        """Save object information."""
        data = {
            NAME: self.name,
            CONFIGURATION: self.configuration,
            METHOD_NAME: self.method_name,
            TOTAL_FIELD: self.total_field,
            SCATTERED_FIELD: self.scattered_field,
            REL_PERMITTIVITY: self.rel_permittivity,
            CONDUCTIVITY: self.conductivity,
            EXECUTION_TIME: self.execution_time,
            NUMBER_EVALUATIONS: self.number_evaluations,
            NUMBER_ITERATIONS: self.number_iterations,
            OBJECTIVE_FUNCTION: self.objective_function,
            RESIDUAL_NORM_ERROR: self.zeta_rn,
            RESIDUAL_PAD_ERROR: self.zeta_rpad,
            REL_PERMITTIVITY_PAD_ERROR: self.zeta_epad,
            REL_PERMITTIVITY_BACKGROUND_ERROR: self.zeta_ebe,
            REL_PERMITTIVITY_OBJECT_ERROR: self.zeta_eoe,
            CONDUCTIVITY_AD_ERROR: self.zeta_sad,
            CONDUCTIVITY_BACKGROUND_ERROR: self.zeta_sbe,
            CONDUCTIVITY_OBJECT_ERROR: self.zeta_soe,
            TOTAL_VARIATION: self.zeta_tv,
            SHAPE_ERROR: self.zeta_s,
            POSITION_ERROR: self.zeta_p,
            TOTALFIELD_MAGNITUDE_PAD: self.zeta_tfmpad,
            TOTALFIELD_PHASE_AD: self.zeta_tfpad
        }

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import data from a saved object."""
        with open(file_path + file_name, 'rb') as datafile:
            data = pickle.load(datafile)
        self.name = data[NAME]
        self.configuration = data[CONFIGURATION]
        self.method_name = data[METHOD_NAME]
        self.total_field = data[TOTAL_FIELD]
        self.scattered_field = data[SCATTERED_FIELD]
        self.rel_permittivity = data[REL_PERMITTIVITY]
        self.conductivity = data[CONDUCTIVITY]
        self.execution_time = data[EXECUTION_TIME]
        self.number_evaluations = data[NUMBER_EVALUATIONS]
        self.number_iterations = data[NUMBER_ITERATIONS]
        self.objective_function = data[OBJECTIVE_FUNCTION]
        self.zeta_rn = data[RESIDUAL_NORM_ERROR]
        self.zeta_rpad = data[RESIDUAL_PAD_ERROR]
        self.zeta_epad = data[REL_PERMITTIVITY_PAD_ERROR]
        self.zeta_ebe = data[REL_PERMITTIVITY_BACKGROUND_ERROR]
        self.zeta_eoe = data[REL_PERMITTIVITY_OBJECT_ERROR]
        self.zeta_sad = data[CONDUCTIVITY_AD_ERROR]
        self.zeta_sbe = data[CONDUCTIVITY_BACKGROUND_ERROR]
        self.zeta_soe = data[CONDUCTIVITY_OBJECT_ERROR]
        self.zeta_tv = data[TOTAL_VARIATION]
        self.zeta_tfmpad = data[TOTALFIELD_MAGNITUDE_PAD]
        self.zeta_tfpad = data[TOTALFIELD_PHASE_AD]
        self.zeta_p = data[POSITION_ERROR]
        self.zeta_s = data[SHAPE_ERROR]

    def plot_map(self, axis=None, image=CONTRAST, groundtruth=None, title=None,
                 show=False, save=False, file_path='', file_format='eps',
                 fontsize=10, file_name=None, source=None, interpolation=None):
        """Plot map results.

        Call signatures::

            plot_map(show=False, filepath='', file_format='eps')

        Parameters
        ----------
            show : boolean
                If `False`, a figure will be saved with the name
                attribute of the object will be save at the specified
                path with the specified format. If `True`, a plot window
                will be displayed.

            file_path : string
                The location in which you want save the figure. Default:
                ''

            file_format : string
                The format of the figure to be saved. The possible
                formats are the same of the command `pyplot.savefig()`.
                Default: 'eps'

        """
        xlabel, ylabel = r'x [$\lambda_b$]', r'y [$\lambda_b$]'
        xmin, xmax = cfg.get_bounds(self.configuration.Lx)
        ymin, ymax = cfg.get_bounds(self.configuration.Ly)
        extent = [xmin/self.configuration.lambda_b,
                  xmax/self.configuration.lambda_b,
                  ymin/self.configuration.lambda_b,
                  ymax/self.configuration.lambda_b]
        clb_epsilon_r = r'$\epsilon_r$'
        clb_sigma = r'$\sigma$ [S/m]'
        clb_contrast = r'$|\chi|$'
        clb_total = r'$|E_z|$ [V/m]'

        if image == TOTAL_FIELD:
            if self.total_field is None:
                raise error.MissingAttributesError('Result', 'et')
            elif source is None:
                source = range(self.configuration)
            elif type(source) is int:
                if source >= self.configuration.NS:
                    raise error.WrongValueInput('Result.plot_map', 'source',
                                                '0 to %d', str(source))
                source = [source]
            elif type(source) is list:
                if any([s >= self.configuration.NS for s in
                        range(self.configuration.NS)]):
                    raise error.WrongValueInput('Result.plot_map', 'source',
                                                '0 to %d', str(source))
            else:
                raise error.WrongTypeInput('Result.plot_map', 'source',
                                           'None, int or int-list',
                                           str(type(source)))

        if groundtruth is not None:
            if image == BOTH_PROPERTIES:
                nfig = 4
            elif image == TOTAL_FIELD:
                nfig = 2*len(source)
            else:
                nfig = 2
        else:
            if image == BOTH_PROPERTIES:
                nfig = 2
            elif image == TOTAL_FIELD:
                nfig = len(source)
            else:
                nfig = 1

        if axis is None:
            fig, ax, _ = get_figure(nfig)
        else:
            if type(axis) is np.ndarray and axis.size != nfig:
                raise error.WrongValueInput('Result.plot_map', 'axis',
                                            '%d-numpy.ndarray' % nfig,
                                            '%d-numpy.ndarray' % axis.size)
            elif isinstance(axis, plt.Axes) and nfig != 1:
                raise error.WrongValueInput('Result.plot_map', 'axis',
                                            '%d-numpy.ndarray' % nfig,
                                            'matplotlib.axes.Axes')
            fig = plt.gcf()
            if type(axis) is not np.ndarray:
                ax = [axis]
            else:
                ax = axis

        if title == False:
            figure_title = ''
        elif type(title) is list:
            figure_title = title[0]
        
        ifig = 0
        if groundtruth is not None:
            if title is None or title is True:
                figure_title = 'Ground-Truth'
            if image == BOTH_PROPERTIES:
                groundtruth.draw(image=BOTH_PROPERTIES,
                                 axis=ax[:2],
                                 title=figure_title,
                                 show=False,
                                 save=False,
                                 fontsize=fontsize,)
                ifig = 2
            elif image != TOTAL_FIELD:
                groundtruth.draw(image=image,
                                 axis=ax[0],
                                 title=figure_title,
                                 show=False,
                                 save=False,
                                 fontsize=fontsize)
                ifig = 1
            elif image == TOTAL_FIELD:
                groundtruth.plot_total_field(axis=ax[:len(source)],
                                             source=source,
                                             figure_title=figure_title,
                                             fontsize=fontsize)
                ifig = len(source)

        if title is None or title == True:
            figure_title = 'Recovered'
        elif type(title) is str:
            figure_title = title
        elif type(title) is list:
            figure_title = title[1]

        if image == PERMITTIVITY:
            add_image(ax[ifig], self.rel_permittivity, figure_title,
                      clb_epsilon_r, bounds=extent, xlabel=xlabel,
                      ylabel=ylabel, fontsize=fontsize,
                      interpolation=interpolation)

        elif image == CONDUCTIVITY:
            add_image(ax[ifig], self.conductivity, figure_title, clb_sigma,
                      bounds=extent, xlabel=xlabel, ylabel=ylabel,
                      fontsize=fontsize, interpolation=interpolation)
            
        elif image == CONTRAST:
            X = cfg.get_contrast_map(epsilon_r=self.rel_permittivity,
                                     sigma=self.conductivity,
                                     configuration=self.configuration)
            add_image(ax[ifig], np.abs(X), figure_title, clb_contrast,
                      bounds=extent, xlabel=xlabel, ylabel=ylabel,
                      fontsize=fontsize, interpolation=interpolation)
        elif image == TOTAL_FIELD:
            for s in source:
                E = np.abs(
                    self.total_field[:, s].reshape(self.rel_permittivity.shape)
                )
                add_image(ax[ifig], E, figure_title, clb_total,
                          bounds=extent, xlabel=xlabel, ylabel=ylabel,
                          fontsize=fontsize, interpolation=interpolation)
                ifig += 1
        
        else:
            if title is None or title == True:
                figure_title = 'Recovered Rel. Per.'
            add_image(ax[ifig], self.rel_permittivity, figure_title,
                      clb_epsilon_r, bounds=extent, xlabel=xlabel,
                      ylabel=ylabel, fontsize=fontsize,
                      interpolation=interpolation)
            if title is None or title == True:
                figure_title = 'Recovered Con.'
            add_image(ax[ifig+1], self.conductivity, figure_title, clb_sigma,
                      bounds=extent, xlabel=xlabel, ylabel=ylabel,
                      fontsize=fontsize, interpolation=interpolation)

        if save:
            plt.tight_layout()
            if file_name is None:
                plt.savefig(file_path + self.name + '.' + file_format,
                            format=file_format)
            else:
                plt.savefig(file_path + file_name + '.' + file_format,
                            format=file_format)
        if show:
            plt.tight_layout()
            plt.show()
        if save:
            plt.close()
        elif not show and axis is None:
            return fig, ax

    def update_error(self, inputdata, scattered_field=None, total_field=None,
                     rel_permittivity=None, conductivity=None,
                     contrast=None, objective_function=None):
        """Compute errors for a given set of variables.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                An object of InputData representing an instance.

            scattered_field, total_field : :class:`numpy.ndarray`
                Fields estimated by the solver.

            REL_PERMITTIVITY : :class:`numpy.ndarray`
                Relative permittivity image recovered by the solver.

            CONDUCTIVITY : :class:`numpy.ndarray`
                Conductivity image recovered by the solver.
        """
        if RESIDUAL_NORM_ERROR in inputdata.indicators:
            if scattered_field is None:
                raise error.MissingInputError('Result.update_error',
                                              'scattered_field')
            elif inputdata.scattered_field is None:
                raise error.MissingAttributesError('InputData', 'es')
            else:
                self.zeta_rn.append(compute_zeta_rn(inputdata.scattered_field,
                                                    scattered_field))

        if RESIDUAL_PAD_ERROR in inputdata.indicators:
            if scattered_field is None:
                raise error.MissingInputError('Result.update_error',
                                              'scattered_field')
            elif inputdata.scattered_field is None:
                raise error.MissingAttributesError('InputData', 'es')
            else:
                self.zeta_rpad.append(
                    compute_zeta_rpad(inputdata.scattered_field,
                                      scattered_field)
                )

        if REL_PERMITTIVITY_PAD_ERROR in inputdata.indicators:
            if rel_permittivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'contrast')
            elif inputdata.rel_permittivity is None:
                raise error.MissingAttributesError('InputData', 'epsilon_r')
            if rel_permittivity is None:
                epsilon_r = cfg.get_relative_permittivity(
                    contrast, self.configuration.epsilon_rb
                )
            else:
                epsilon_r = rel_permittivity
            if epsilon_r.shape != inputdata.rel_permittivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.rel_permittivity'"
                                  + " and 'epsilon_r' must have the same "
                                  + "shape.")
            self.zeta_epad.append(compute_zeta_epad(inputdata.rel_permittivity,
                                                    epsilon_r))
        
        if REL_PERMITTIVITY_OBJECT_ERROR in inputdata.indicators:
            if rel_permittivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'contrast')
            elif inputdata.rel_permittivity is None:
                raise error.MissingAttributesError('InputData', 'epsilon_r')
            if rel_permittivity is None:
                epsilon_r = cfg.get_relative_permittivity(
                    contrast, self.configuration.epsilon_rb
                )
            else:
                epsilon_r = rel_permittivity
            if epsilon_r.shape != inputdata.rel_permittivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.rel_permittivity'"
                                  + " and 'epsilon_r' must have the same "
                                  + "shape.")
            epsilon_rb = self.configuration.epsilon_rb
            self.zeta_eoe.append(compute_zeta_eoe(inputdata.rel_permittivity,
                                                  epsilon_r, epsilon_rb))

        if REL_PERMITTIVITY_BACKGROUND_ERROR in inputdata.indicators:
            if rel_permittivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'contrast')
            elif inputdata.rel_permittivity is None:
                raise error.MissingAttributesError('InputData', 'epsilon_r')
            if rel_permittivity is None:
                epsilon_r = cfg.get_relative_permittivity(
                    contrast, self.configuration.epsilon_rb
                )
            else:
                epsilon_r = rel_permittivity
            if epsilon_r.shape != inputdata.rel_permittivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.rel_permittivity'"
                                  + " and 'epsilon_r' must have the same "
                                  + "shape.")
            epsilon_rb = self.configuration.epsilon_rb
            self.zeta_ebe.append(compute_zeta_ebe(inputdata.rel_permittivity,
                                                  epsilon_r, epsilon_rb))
        
        if CONDUCTIVITY_AD_ERROR in inputdata.indicators:
            if conductivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'conductivity or contrast')
            elif inputdata.conductivity is None:
                raise error.MissingAttributesError('InputData', 'sigma')
            if conductivity is None:
                omega = 2*pi*self.configuration.f
                epsilon_rb = self.configuration.epsilon_rb
                sigma_b = self.configuration.sigma_b
                sigma = cfg.get_conductivity(contrast, omega, epsilon_rb,
                                             sigma_b)
            else:
                sigma = conductivity
            if sigma.shape != inputdata.conductivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.conductivity'"
                                  + " and 'sigma' must have the same "
                                  + "shape.")
            self.zeta_sad.append(compute_zeta_sad(inputdata.conductivity,
                                                  sigma))

        if CONDUCTIVITY_OBJECT_ERROR in inputdata.indicators:
            if conductivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'conductivity or contrast')
            elif inputdata.conductivity is None:
                raise error.MissingAttributesError('InputData', 'sigma')
            if conductivity is None:
                omega = 2*pi*self.configuration.f
                epsilon_rb = self.configuration.epsilon_rb
                sigma_b = self.configuration.sigma_b
                sigma = cfg.get_conductivity(contrast, omega, epsilon_rb,
                                             sigma_b)
            else:
                sigma = conductivity
            if sigma.shape != inputdata.conductivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.conductivity'"
                                  + " and 'sigma' must have the same "
                                  + "shape.")
            sigma_b = self.configuration.sigma_b
            self.zeta_soe.append(compute_zeta_soe(inputdata.conductivity,
                                                  sigma, sigma_b))
        
        if CONDUCTIVITY_BACKGROUND_ERROR in inputdata.indicators:
            if conductivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'conductivity or contrast')
            elif inputdata.conductivity is None:
                raise error.MissingAttributesError('InputData', 'sigma')
            if conductivity is None:
                omega = 2*pi*self.configuration.f
                epsilon_rb = self.configuration.epsilon_rb
                sigma_b = self.configuration.sigma_b
                sigma = cfg.get_conductivity(contrast, omega, epsilon_rb,
                                             sigma_b)
            else:
                sigma = conductivity
            if sigma.shape != inputdata.conductivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.conductivity'"
                                  + " and 'sigma' must have the same "
                                  + "shape.")
            sigma_b = self.configuration.sigma_b
            self.zeta_sbe.append(compute_zeta_sbe(inputdata.conductivity,
                                                  sigma, sigma_b))

        if SHAPE_ERROR in inputdata.indicators:
            if (conductivity is None and rel_permittivity is None
                    and contrast is None):
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'conductivity or contrast')
            elif (inputdata.rel_permittivity is None
                    and inputdata.conductivity is None):
                raise error.MissingAttributesError('InputData',
                                                   'epsilon_r or sigma')
            Xo = cfg.get_contrast_map(epsilon_r=inputdata.rel_permittivity,
                                      sigma=inputdata.conductivity,
                                      configuration=self.configuration)
            if contrast is None:
                Xr = cfg.get_contrast_map(epsilon_r=rel_permittivity,
                                          sigma=conductivity,
                                          configuration=self.configuration)
            else:
                Xr = contrast
            self.zeta_s.append(compute_zeta_s(Xo, Xr))

        if POSITION_ERROR in inputdata.indicators:
            if (conductivity is None and rel_permittivity is None
                    and contrast is None):
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'conductivity or contrast')
            elif (inputdata.rel_permittivity is None
                    and inputdata.conductivity is None):
                raise error.MissingAttributesError('InputData',
                                                   'epsilon_r or sigma')
            Xo = cfg.get_contrast_map(epsilon_r=inputdata.rel_permittivity,
                                      sigma=inputdata.conductivity,
                                      configuration=self.configuration)
            if contrast is None:
                Xr = cfg.get_contrast_map(epsilon_r=rel_permittivity,
                                          sigma=conductivity,
                                          configuration=self.configuration)
            else:
                Xr = contrast
            self.zeta_p.append(compute_zeta_p(Xo, Xr))

        if TOTAL_VARIATION in inputdata.indicators:
            if (conductivity is None and rel_permittivity is None
                    and contrast is None):
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'conductivity or contrast')
            if contrast is None:
                X = cfg.get_contrast_map(epsilon_r=rel_permittivity,
                                          sigma=conductivity,
                                          configuration=self.configuration)
            else:
                X = contrast
            x, y = cfg.get_coordinates_ddomain(
                configuration=self.configuration, resolution=X.shape
            )
            self.zeta_tv.append(compute_zeta_tv(X, x, y))

        if TOTALFIELD_MAGNITUDE_PAD in inputdata.indicators:
            if total_field is None:
                raise error.MissingInputError('Result.update_error',
                                              'total_field')
            elif inputdata.total_field is None:
                raise error.MissingAttributesError('InputData', 'et')
            elif inputdata.total_field.shape != total_field.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.total_field' and"
                                  + " 'total_field' must have the same shape.")
            self.zeta_tfmpad.append(compute_zeta_tfmpad(inputdata.total_field,
                                                        total_field))

        if TOTALFIELD_PHASE_AD in inputdata.indicators:
            if total_field is None:
                raise error.MissingInputError('Result.update_error',
                                              'total_field')
            elif inputdata.total_field is None:
                raise error.MissingAttributesError('InputData', 'et')
            elif inputdata.total_field.shape != total_field.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.total_field' and"
                                  + " 'total_field' must have the same shape.")
            self.zeta_tfpad.append(compute_zeta_tfpad(inputdata.total_field,
                                                      total_field))

        if OBJECTIVE_FUNCTION in inputdata.indicators:
            if objective_function is None:
                raise error.MissingInputError('Result.update_error',
                                              'objective_function')
            self.objective_function.append(objective_function)

    def last_error_message(self, pre_message=None):
        """Summarize the method."""
        if pre_message is not None:
            message = pre_message
        else:
            message = 'Indicators:'

        if self.zeta_rn is not None and len(self.zeta_rn) != 0:
            message += ' Residual norm: %.3e,' % self.zeta_rn[-1]

        if self.zeta_rpad is not None and len(self.zeta_rpad) != 0:
            message += ' Residual PAD: %.2f%%,' % self.zeta_rpad[-1]

        if self.zeta_epad is not None and len(self.zeta_epad) != 0:
            message += ' Rel. Per. PAD: %.2f%%,' % self.zeta_epad[-1]

        if self.zeta_eoe is not None and len(self.zeta_eoe) != 0:
            message += ' Rel. Per. Ob.: %.2f%%,' % self.zeta_eoe[-1]

        if self.zeta_ebe is not None and len(self.zeta_ebe) != 0:
            message += ' Rel. Per. Back.: %.2f%%,' % self.zeta_ebe[-1]

        if self.zeta_sad is not None and len(self.zeta_sad) != 0:
            message += ' Con. AD: %.3e,' % self.zeta_sad[-1]

        if self.zeta_soe is not None and len(self.zeta_soe) != 0:
            message += ' Con. Ob.: %.3e,' % self.zeta_soe[-1]

        if self.zeta_sbe is not None and len(self.zeta_sbe) != 0:
            message += ' Con. Back.: %.3e,' % self.zeta_sbe[-1]

        if self.zeta_s is not None and len(self.zeta_s) != 0:
            message += ' Shape: %.2f,' % self.zeta_s[-1]

        if self.zeta_p is not None and len(self.zeta_p) != 0:
            message += ' Position: %.2f,' % self.zeta_p[-1]

        if self.zeta_tv is not None and len(self.zeta_tv) != 0:
            message += ' Total Variation: %.2f,' % self.zeta_tv[-1]

        if self.zeta_tfmpad is not None and len(self.zeta_tfmpad) != 0:
            message += ' To. Field Mag. PAD: %.2f%%,' % self.zeta_tfmpad[-1]

        if self.zeta_tfpad is not None and len(self.zeta_tfpad) != 0:
            message += ' To. Field Phase AD: %.2f%%,' % self.zeta_tfpad[-1]

        return message

    def valid_indicators(self):
        indicators = []
        if self.zeta_rn is not None and len(self.zeta_rn) != 0:
            indicators.append(RESIDUAL_PAD_ERROR)
        if self.zeta_rpad is not None and len(self.zeta_rpad) != 0:
            indicators.append(RESIDUAL_NORM_ERROR)
        if self.zeta_epad is not None and len(self.zeta_epad) != 0:
            indicators.append(REL_PERMITTIVITY_PAD_ERROR)
        if self.zeta_eoe is not None and len(self.zeta_eoe) != 0:
            indicators.append(REL_PERMITTIVITY_OBJECT_ERROR)
        if self.zeta_ebe is not None and len(self.zeta_ebe) != 0:
            indicators.append(REL_PERMITTIVITY_BACKGROUND_ERROR)
        if self.zeta_sad is not None and len(self.zeta_sad) != 0:
            indicators.append(CONDUCTIVITY_AD_ERROR)
        if self.zeta_soe is not None and len(self.zeta_soe) != 0:
            indicators.append(CONDUCTIVITY_OBJECT_ERROR)
        if self.zeta_sbe is not None and len(self.zeta_sbe) != 0:
            indicators.append(CONDUCTIVITY_BACKGROUND_ERROR)
        if self.zeta_s is not None and len(self.zeta_s) != 0:
            indicators.append(SHAPE_ERROR)
        if self.zeta_p is not None and len(self.zeta_p) != 0:
            indicators.append(POSITION_ERROR)
        if self.zeta_tv is not None and len(self.zeta_tv) != 0:
            indicators.append(TOTAL_VARIATION)
        if self.zeta_tfmpad is not None and len(self.zeta_tfmpad) != 0:
            indicators.append(TOTALFIELD_MAGNITUDE_PAD)
        if self.zeta_tfpad is not None and len(self.zeta_tfpad) != 0:
            indicators.append(TOTALFIELD_PHASE_AD)
        if (self.objective_function is not None
                and len(self.objective_function) != 0):
            indicators.append(OBJECTIVE_FUNCTION)
        return indicators

    def plot_convergence(self, axis=None, indicators=None, show=False,
                         file_name=None, file_path='', file_format='eps',
                         fontsize=10, title=None, style='--*', yscale=None,
                         markersize=None):
        """Summarize the method."""
        if indicators is None:
            indicators = self.valid_indicators()
        elif type(indicators) is str:
            indicators = [indicators]
        nplots = len(indicators)
        if axis is None:
            fig, axis, _ = get_figure(nplots)
            given_axis = False
        else:
            if nplots > 1 and axis.size != nplots:
                raise error.WrongValueInput('Result.plot_convergence', 'axis',
                                            '%dd-ndarray' %nplots,
                                            '%dd' % axis.size)
            fig = plt.gcf()
            given_axis = True

        for n in range(nplots):
            y = getattr(self, indicators[n])
            x = np.arange(len(y))+1
            if title is None:
                figtitle = TITLES[indicators[n]]
            elif type(title) is str:
                figtitle = title
            elif type(title) is list:
                figtitle = title[n]
            else:
                figtitle = None
            if yscale is None:
                figyscale = None
            elif type(yscale) is str:
                figyscale = yscale
            elif type(yscale) is list:
                figyscale = yscale[n]
            add_plot(axis[n], y, x=x, title=figtitle,
                     ylabel=indicator_label(indicators[n]), style=style,
                     yscale=figyscale, fontsize=fontsize,
                     markersize=markersize)

        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        if not given_axis:
            return fig, axis

    def final_value(self, indicator):
        if type(indicator) is not str:
            raise error.WrongTypeInput('Result.final_value', 'indicator',
                                       'str', str(type(indicator)))
        elif not check_indicator(indicator):
            raise error.WrongValueInput('Result.plot', 'indicator',
                                        INDICATOR_SET, indicator)
        
        output = getattr(self, indicator)
        if type(output) is list or type(output) is np.ndarray:
            return output[-1]
        else:
            return output

    def copy(self, new=None):
        if new is None:
            new = Result(
                name=self.name, method_name=self.method_name,
                configuration=self.configuration,
                scattered_field=cp.deepcopy(self.scattered_field),
                total_field=cp.deepcopy(self.total_field),
                rel_permittivity=cp.deepcopy(self.rel_permittivity),
                conductivity=cp.deepcopy(self.conductivity),
                execution_time=self.execution_time,
                number_evaluations=self.number_evaluations,
                number_iterations=self.number_iterations
            )
            new.zeta_rn = cp.deepcopy(self.zeta_rn)
            new.zeta_rpad = cp.deepcopy(self.zeta_rpad)
            new.zeta_epad = cp.deepcopy(self.zeta_epad)
            new.zeta_sad = cp.deepcopy(self.zeta_sad)
            new.zeta_tv = cp.deepcopy(self.zeta_tv)
            new.zeta_p = cp.deepcopy(self.zeta_p)
            new.zeta_s = cp.deepcopy(self.zeta_s)
            new.zeta_ebe = cp.deepcopy(self.zeta_ebe)
            new.zeta_sbe = cp.deepcopy(self.zeta_sbe)
            new.zeta_eoe = cp.deepcopy(self.zeta_eoe)
            new.zeta_soe = cp.deepcopy(self.zeta_soe)
            new.zeta_tfmpad = cp.deepcopy(self.zeta_tfmpad)
            new.zeta_tfpad = cp.deepcopy(self.zeta_tfpad)
            new.objective_function = cp.deepcopy(self.objective_function)
            return new
        else:
            self.name = new.name
            self.method_name = new.method_name
            self.configuration = new.configuration.copy()
            self.scattered_field = np.copy(new.scattered_field)
            self.total_field = np.copy(new.total_field)
            self.rel_permittivity = np.copy(new.rel_permittivity)
            self.conductivity = np.copy(new.conductivity)
            self.execution_time = new.execution_time
            self.number_evaluations = new.number_evaluations
            self.objective_function = cp.deepcopy(new.objective_function)
            self.number_iterations = new.number_iterations
            self.zeta_rn = cp.deepcopy(new.zeta_rn)
            self.zeta_rpad = cp.deepcopy(new.zeta_rpad)
            self.zeta_epad = cp.deepcopy(new.zeta_epad)
            self.zeta_ebe = cp.deepcopy(new.zeta_ebe)
            self.zeta_eoe = cp.deepcopy(new.zeta_eoe)
            self.zeta_sad = cp.deepcopy(new.zeta_sad)
            self.zeta_sbe = cp.deepcopy(new.zeta_sbe)
            self.zeta_soe = cp.deepcopy(new.zeta_soe)
            self.zeta_s = cp.deepcopy(new.zeta_s)
            self.zeta_p = cp.deepcopy(new.zeta_p)
            self.zeta_tfmpad = cp.deepcopy(new.zeta_tfmpad)
            self.zeta_tfpad = cp.deepcopy(new.zeta_tfpad)
            self.zeta_tv = cp.deepcopy(new.zeta_tv)

    def __str__(self):
        """Print object information."""
        message = 'Results name: ' + self.name
        message += '\nConfiguration: ' + self.configuration.name
        if self.scattered_field is not None:
            message = (message + '\nScattered field - measurement samples: %d'
                       % self.scattered_field.shape[0]
                       + '\nScattered field - source samples: %d'
                       % self.scattered_field.shape[1])
        if self.total_field is not None:
            message = (message + '\nTotal field - measurement samples: %d'
                       % self.total_field.shape[0]
                       + '\nTotal field - source samples: %d'
                       % self.total_field.shape[1])
        if self.rel_permittivity is not None:
            if self.rel_permittivity.ndim == 1:
                message += ('\nSolution: ' + str(self.rel_permittivity))
            elif self.rel_permittivity.ndim == 2:
                message += ('\nRelative Permit. map resolution: %dx'
                            % self.rel_permittivity.shape[0] + '%d'
                            % self.rel_permittivity.shape[1])
        if self.conductivity is not None:
            message += ('\nConductivity map resolution: %dx'
                        % self.conductivity.shape[0]
                        + '%d' % self.conductivity.shape[1])
        if self.execution_time is not None:
            print('Execution time: %.2f [sec]' % self.execution_time)
        if len(self.zeta_rn) > 0:
            if len(self.zeta_rn) == 1:
                info = '%.3e' % self.zeta_rn[0]
            elif len(self.zeta_rn) > 30:
                info = '%.3e' % self.zeta_rn[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_rn) + ']')
            message = message + '\nResidual norm error: ' + info
        if len(self.zeta_rpad) > 0:
            if len(self.zeta_rpad) == 1:
                info = '%.2f%%' % self.zeta_rpad[0]
            elif len(self.zeta_rpad) > 30:
                info = '%.2f%%' % self.zeta_rpad[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_rpad) + ']')
            message = message + '\nPercent. Aver. Devi. of Residuals: ' + info
        if len(self.zeta_epad) > 0:
            if len(self.zeta_epad) == 1:
                info = '%.2f%%' % self.zeta_epad[0]
            if len(self.zeta_epad) > 30:
                info = '%.2f%%' % self.zeta_epad[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_epad) + ']')
            message = (message + '\nPercent. Aver. Devi. of Rel. Permittivity:'
                       + ' ' + info)
        if len(self.zeta_sad) > 0:
            if len(self.zeta_sad) == 1:
                info = '%.3e' % self.zeta_sad[0]
            elif len(self.zeta_sad) > 30:
                info = '%.3e' % self.zeta_sad[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_sad) + ']')
            message = (message + '\nAver. Devi. of Conductivity: '
                       + info)
        if len(self.zeta_tv) > 0:
            if len(self.zeta_tv) == 1:
                info = '%.3e' % self.zeta_tv[0]
            elif len(self.zeta_tv) > 30:
                info = '%.3e' % self.zeta_tv[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_tv) + ']')
            message = message + '\nTotal Variation: ' + info
        if len(self.zeta_ebe) > 0:
            if len(self.zeta_ebe) == 1:
                info = '%.2f%%' % self.zeta_ebe[0]
            elif len(self.zeta_ebe) > 30:
                info = '%.2f%%' % self.zeta_ebe[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_ebe) + ']')
            message = message + '\nBackground Rel. Permit. error: ' + info
        if len(self.zeta_sbe) > 0:
            if len(self.zeta_sbe) == 1:
                info = '%.3e' % self.zeta_sbe[0]
            elif len(self.zeta_sbe) > 30:
                info = '%.3e' % self.zeta_sbe[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_sbe) + ']')
            message = message + '\nBackground Conductivity error: ' + info
        if len(self.zeta_eoe) > 0:
            if len(self.zeta_eoe) == 1:
                info = '%.2f%%' % self.zeta_eoe[0]
            elif len(self.zeta_eoe) > 30:
                info = '%.2f%%' % self.zeta_eoe[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_eoe) + ']')
            message = message + '\nObject Rel. Permit. error: ' + info
        if len(self.zeta_soe) > 0:
            if len(self.zeta_soe) == 1:
                info = '%.3e' % self.zeta_soe[0]
            elif len(self.zeta_soe) > 30:
                info = '%.3e' % self.zeta_soe[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_soe) + ']')
            message = message + '\nObject Conduc. error: ' + info
        if len(self.zeta_tfmpad) > 0:
            if len(self.zeta_tfmpad) == 1:
                info = '%.2f%%' % self.zeta_tfmpad[0]
            elif len(self.zeta_tfmpad) > 30:
                info = '%.2f%%' % self.zeta_tfmpad[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_tfmpad) + ']')
            message = (message + '\nTotal Field Mag. Per. Aver. Devi. error: '
                       + info)
        if len(self.zeta_tfpad) > 0:
            if len(self.zeta_tfpad) == 1:
                info = '%.2f%%' % self.zeta_tfpad[0]
            elif len(self.zeta_tfpad) > 30:
                info = '%.2f%%' % self.zeta_tfpad[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_tfpad) + ']')
            message = (message + '\nTotal Field Phase Aver. Devi. error:'
                       + ' ' + info)
        if len(self.zeta_p) > 0:
            if len(self.zeta_p) == 1:
                info = '%.2f%%' % self.zeta_p[0]
            elif len(self.zeta_p) > 30:
                info = '%.2f%%' % self.zeta_p[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_p) + ']')
            message += ('\nPosition error: ' + info)
        if len(self.zeta_s) > 0:
            if len(self.zeta_s) == 1:
                info = '%.2f%%' % self.zeta_s[0]
            elif len(self.zeta_s) > 30:
                info = '%.2f%%' % self.zeta_s[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_s) + ']')
            message += ('\nShape error: ' + info)
        if type(self.objective_function) is float:
            message += ('\nObjective function evaluation: %.3e'
                        % self.objective_function)
        elif len(self.objective_function) > 0:
            if len(self.objective_function) == 1:
                info = '%.3e' % self.objective_function[0]
            if len(self.objective_function) > 30:
                info = '%.3e' % self.objective_function[-1]
            else:
                info = '[' + str(', '.join('{:.2e}'.format(i)
                                           for i in self.objective_function)
                                 + ']')
            message += '\nObjective function:' + ' ' + info
        if self.number_iterations is not None:
            message += '\nNumber of iterations: %d' % self.number_iterations
        if self.number_evaluations is not None:
            message += '\nNumber of evaluations: %d' % self.number_evaluations
        return message


def add_image(axes, image, title, colorbar_name, bounds=(-1., 1., -1., 1.),
              origin='lower', xlabel=XLABEL_STANDARD, ylabel=YLABEL_STANDARD,
              aspect='equal', interpolation=None, fontsize=10):
    """Add a image to the axes.

    A predefined function for plotting image. This is useful for
    standardize plots involving contrast maps and fields.

    Paramaters
    ----------
        axes : :class:`matplotlib.pyplot.Figure.axes.Axes`
            The axes object.

        image : :class:`numpy.ndarray`
            A matrix with image to be displayed. If complex, the
            magnitude will be displayed.

        title : string
            The title to be displayed in the figure.

        colorbar_name : string
            The label for color bar.

        bounds : 4-tuple of floats, default: (-1., 1., -1., 1.)
            The value of the bounds of each axis. Example: (xmin, xmax,
            ymin, ymax).

        origin : {'lower', 'upper'}, default: 'lower'
            Origin of the y-axis.

        xlabel : string, default: XLABEL_STANDARD
            The label of the x-axis.

        ylabel : string, default: YLABEL_STANDARD
            The label of the y-axis.

    """
    if image.dtype == complex:
        im = axes.imshow(np.abs(image),
                         extent=[bounds[0], bounds[1],
                                 bounds[2], bounds[3]],
                         origin=origin, aspect=aspect,
                         interpolation=interpolation)
    else:
        im = axes.imshow(image,
                         extent=[bounds[0], bounds[1],
                                 bounds[2], bounds[3]],
                         origin=origin, aspect=aspect,
                         interpolation=interpolation)
    axes.set_xlabel(xlabel, fontsize=fontsize)
    axes.set_ylabel(ylabel, fontsize=fontsize)
    axes.set_title(title, fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize)
    cbar = plt.colorbar(ax=axes, mappable=im, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_name, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)


def add_plot(axes, data, x=None, title=None, xlabel='Iterations', ylabel=None,
             style='--*', xticks=None, legend=None, legend_fontsize=None,
             yscale=None, fontsize=10, color=None, markersize=None):
    """Add a plot to the axes.

    A predefined function for plotting curves. This is useful for
    standardize plots involving convergence data.

    Paramaters
    ----------
        axes : :class:`matplotlib.pyplot.Figure.axes.Axes`
            The axes object.

        data : :class:`numpy.ndarray`
            The y-data.

        x : :class:`numpy.ndarray`, default: None
            The x-data.

        title : string, default: None
            The title to be displayed in the plot.

        xlabel : string, default: 'Iterations'
            The label of the x-axis.

        ylabel : string, default: None
            The label of the y-axis.

        style : string, default: '--*'
            The style of the curve (line, marker, color).

        yscale : None or {'linear', 'log', 'symlog', 'logit', ...}
            Scale of y-axis. Check some options `here <https://
            matplotlib.org/3.1.1/api/_as_gen/
            matplotlib.pyplot.yscale.html>`
    """
    if x is None:
        if type(data) is list:
            length = len(data)
        else:
            length = data.size
        x = np.arange(1, length+1)

    axes.plot(x, data, style, color=color, markersize=markersize)
    axes.set_xlabel(xlabel, fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize)
    if xticks is not None:
        axes.set_xticks(xticks)
    if ylabel is not None:
        axes.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        axes.set_title(title, fontsize=fontsize)
    if legend is not None:
        if legend_fontsize is not None:
            axes.legend(legend, fontsize=legend_fontsize)
        else:
            axes.legend(legend)
    if yscale is not None:
        axes.set_yscale(yscale)
    axes.grid(True)


def add_box(data, axis=None, meanline=False, labels=None, xlabel=None,
            ylabel=None, color='b', legend=None, title=None, notch=False,
            legend_fontsize=None, fontsize=10, positions=None, yscale=None,
            widths=.5):
    """Improved boxplot routine.

    This routine does not show any plot. It only draws the graphic.

    Parameters
    ----------
        data : list of :class:`numpy.ndarray`
            A list of 1-d arrays meaning the samples.

        axis : :class:`matplotlib.Axes.axes`, default: None
            A specified axis for plotting the graphics. If none is
            provided, then one will be created and returned.

        meanline : bool, default: False
            Draws a line through linear regression of the means among
            the samples.

        labels : list of str, default: None
            Names of the samples.

        xlabel : str, default: None

        ylabel : list of str, default: None

        color : str, default: 'b'
            Color of boxes. Check some `here <https://matplotlib.org/
            3.1.1/gallery/color/named_colors.html>`_

        legend : str, default: None
            Label for meanline.

        title : str, default: None
            A possible title to the plot.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> y3 = np.random.normal(loc=6., size=10)
    >>> boxplot([y1, y2, y3], title='Samples',
                labels=['Sample 1', 'Sample 2', 'Sample 3'],
                xlabel='Samples', ylabel='Unit', color='tab:blue',
                meanline=True, legend='Progression')
    >>> plt.show()
    """
    if (meanline is not None and type(meanline) is not bool
            and meanline != 'regression' and meanline != 'pointwise'):
        raise error.WrongValueInput('result.add_box', 'meanline',
                                    "None, bool, 'regression', 'pointwise'",
                                    str(meanline))
    if axis is None:
        fig, axis = plt.subplots()

    if type(data) is np.ndarray:
        mydata = data.tolist()
    else:
        mydata = data

    if positions is None:
        try:
            _ = len(data[0])
            positions = np.arange(1, len(data)+1)
        except:
            positions = None

    bplot = axis.boxplot(mydata, patch_artist=True, labels=labels,
                         positions=positions, notch=notch, widths=widths)
    for i in range(len(bplot['boxes'])):
        bplot['boxes'][i].set_facecolor(color)

    if meanline == True or meanline == 'regression':
        M = len(mydata)
        x = np.array([positions[0]-.5, positions[-1]+.5])
        means = np.zeros(M)
        for m in range(M):
            means[m] = np.mean(mydata[m])
        a, b = linregress(positions, means)[:2]
        if legend is not None:
            axis.plot(x, a*x + b, '--', color=color, label=legend)
            if legend_fontsize is not None:
                axis.legend(fontsize=legend_fontsize)
            else:
                axis.legend()
        else:
            axis.plot(x, a*x + b, '--', color=color)
    elif meanline == 'pointwise':
        means = np.zeros(len(mydata))
        for m in range(len(mydata)):
            means[m] = np.mean(mydata[m])
        if legend is not None:
            axis.plot(positions, means, '--', color=color, label=legend)
            if legend_fontsize is not None:
                axis.legend(fontsize=legend_fontsize)
            else:
                axis.legend()
        else:
            axis.plot(positions, means, '--', color=color)

    axis.grid(True)
    axis.tick_params(axis='both', which='major', labelsize=fontsize)
    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        axis.set_title(title, fontsize=fontsize)
    if yscale is not None:
        axis.set_yscale(yscale)

    return axis


def add_violin(data, axis=None, meanline=False, labels=None, xlabel=None,
               ylabel=None, color='b', legend=None, title=None,
               legend_fontsize=None, fontsize=10, positions=None, yscale=None):
    """Improved boxplot routine.

    This routine does not show any plot. It only draws the graphic.

    Parameters
    ----------
        data : list of :class:`numpy.ndarray`
            A list of 1-d arrays meaning the samples.

        axis : :class:`matplotlib.Axes.axes`, default: None
            A specified axis for plotting the graphics. If none is
            provided, then one will be created and returned.

        meanline : bool, default: False
            Draws a line through linear regression of the means among
            the samples.

        labels : list of str, default: None
            Names of the samples.

        xlabel : str, default: None

        ylabel : list of str, default: None

        color : str, default: 'b'
            Color of boxes. Check some `here <https://matplotlib.org/
            3.1.1/gallery/color/named_colors.html>`_

        legend : str, default: None
            Label for meanline.

        title : str, default: None
            A possible title to the plot.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> y3 = np.random.normal(loc=6., size=10)
    >>> boxplot([y1, y2, y3], title='Samples',
                labels=['Sample 1', 'Sample 2', 'Sample 3'],
                xlabel='Samples', ylabel='Unit', color='tab:blue',
                meanline=True, legend='Progression')
    >>> plt.show()
    """
    if (meanline is not None and type(meanline) is not bool
            and meanline != 'regression' and meanline != 'pointwise'):
        raise error.WrongValueInput('result.add_violin', 'meanline',
                                    "None, bool, 'regression', 'pointwise'",
                                    str(meanline))

    plot_opts = {'violin_fc': color,
                 'violin_ec': 'w',
                 'violin_alpha': .2}

    if axis is None:
        fig, axis = plt.subplots()

    if type(data) is np.ndarray:
        mydata = data.tolist()
    else:
        mydata = data

    if positions is None:
        positions = np.arange(1, len(data)+1)
    
    violinplot(mydata, ax=axis, labels=labels, positions=positions,
               plot_opts=plot_opts)

    if meanline == True or meanline == 'regression':
        M = len(mydata)
        x = np.array([positions[0]-.5, positions[-1]+.5])
        means = np.zeros(M)
        for m in range(M):
            means[m] = np.mean(mydata[m])
        a, b = linregress(positions, means)[:2]
        if legend is not None:
            axis.plot(x, a*x + b, '--', color=color, label=legend)
            if legend_fontsize is not None:
                axis.legend(fontsize=legend_fontsize)
            else:
                axis.legend()
        else:
            axis.plot(x, a*x + b, '--', color=color)
    elif meanline == 'pointwise':
        means = np.zeros(len(mydata))
        for m in range(len(mydata)):
            means[m] = np.mean(mydata[m])
        if legend is not None:
            axis.plot(positions, means, '--', color=color, label=legend)
            if legend_fontsize is not None:
                axis.legend(fontsize=legend_fontsize)
            else:
                axis.legend()
        else:
            axis.plot(positions, means, '--', color=color)
    axis.tick_params(axis='both', which='major', labelsize=fontsize)
    axis.grid(True)
    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        axis.set_title(title, fontsize=fontsize)
    if yscale is not None:
        axis.set_yscale(yscale)

    return axis


def get_figure(nsubplots=1, number_lines=1):
    """Get a Figure and Axes object with customized sizes.

    Parameters
    ----------
        nsubplots : int, default: 1
            Number of subplots in the figure.

        number_lines : int, default: 1
            Number of lines in each subplot.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

        axes : :class:`matplotlib.axes.Axes`

        legend_fontsize : float
    """
    # Compute number of rows and columns
    nrows = round(np.sqrt(nsubplots))
    ncols = int(np.ceil(nsubplots/nrows))
    legend_fontsize = get_legend_fontsize(number_lines, nrows)

    width, height = 6.4*ncols, 4.8*nrows

    # Figure creation
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(width, height))

    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    if nsubplots < axes.size:
        for i in range(nsubplots, axes.size):
            axes[i].set_visible(False)

    return fig, axes, legend_fontsize


def get_legend_fontsize(number_lines, nrows):
    max_lines = np.array([0, 15, 13, 8, 5, 3, 2, 1, 1, 1, 1, 1])
    if number_lines > max_lines[nrows] and nrows > 2:
        legend_fontsize = 10-(number_lines-max_lines[nrows])*.65
    elif number_lines > max_lines[nrows] and nrows < 3:
        legend_fontsize = 10-(number_lines-max_lines[nrows])*.55
    else:
        legend_fontsize = None
    return legend_fontsize


def compute_zeta_rn(es_o, es_a):
    r"""Compute the residual norm error.

    The zeta_rn error is the residual norm error of the scattered
    field approximation.

    Parameters
    ----------
        es_o : :class:`numpy.ndarray`
            Original scattered field matrix.

        es_a : :class:`numpy.ndarray`
            Approximated scattered field matrix.

    Notes
    -----
        The error is computed through the following relation:

        .. math:: ||E^s-E^{s,\delta}|| = \sqrt{\iint_S(y-y^\delta)
        \overline{(y-y^\delta)}d\theta
    """
    NM, NS = es_o.shape
    theta = cfg.get_angles(NM)
    phi = cfg.get_angles(NS)
    y = (es_o-es_a)*np.conj(es_o-es_a)
    return np.real(np.sqrt(np.trapz(np.trapz(y, x=phi), x=theta)))


def compute_rre(es_o, es_a):
    """Compute the Relative Residual Error (RRE).

    The RRE is a definition found in [1] and it is useful for
    determining the parameter of Tikhonov regularization.

    Parameters
    ----------
        es_o : :class:`numpy.ndarray`
            Original scattered field matrix.

        es_a : :class:`numpy.ndarray`
            Approximated scattered field matrix.

    References
    ----------
    .. [1] Lavarello, Roberto, and Michael Oelze. "A study on the
           reconstruction of moderate contrast targets using the
           distorted Born iterative method." IEEE transactions on
           ultrasonics, ferroelectrics, and frequency control 55.1
           (2008): 112-124.
    """
    return (100*compute_zeta_rn(es_o, es_a)
            / compute_zeta_rn(es_o, np.zeros(es_o.shape, dtype=complex)))


def compute_zeta_rpad(es_o, es_r):
    r"""Compute the residual percentage average deviation.

    The zeta_padr error is the residual percentage average deviation
    of the scattered field approximation.

    Parameters
    ----------
        es_o : :class:`numpy.ndarray`
            Original scattered field matrix.

        es_a : :class:`numpy.ndarray`
            Approximated scattered field matrix.
    """
    y = np.hstack((np.real(es_o.flatten()), np.imag(es_o.flatten())))
    yd = np.hstack((np.real(es_r.flatten()), np.imag(es_r.flatten())))
    return np.mean(np.abs((y-yd)/y))*100


def compute_zeta_epad(epsilon_ro, epsilon_rr):
    """Compute the percent. aver. deviation of relative permit. map.

    The zeta_epad error is the evaluation of the relative
    permittivity estimation error per pixel.

    Parameters
    ----------
        epsilon_ro, epsilon_rr : :class:`numpy.ndarray`
            Original and recovered relative permittivity maps,
            respectively.
    """
    y = epsilon_ro.flatten()
    yd = epsilon_rr.flatten()
    return np.mean(np.abs((y-yd)/y))*100


def compute_zeta_sad(sigma_o, sigma_r):
    """Compute the average deviation of conductivity map.

    The zeta_epad error is the evaluation of the conductivity
    estimation error per pixel.

    Parameters
    ----------
        sigma_o, sigma_r : :class:`numpy.ndarray`
            Original and recovered conductivity maps, respectively.
    """
    y = sigma_o.flatten()
    yd = sigma_r.flatten()
    return np.mean(np.abs((y-yd)))


def compute_zeta_tv(chi, x, y):
    """Compute the total variation.

    The zeta_tv is the of variational functional commonly used as
    regularizer [1].

    Parameters
    ----------
        chi : :class:`numpy.ndarray`
            Constrast map.

        x, y : :class:`numpy.ndarray`
            Meshgrid arrays of x and y coordinates.

    References
    ----------
    .. [1] Lobel, P., et al. "A new regularization scheme for
       inverse scattering." Inverse Problems 13.2 (1997): 403.
    """
    grad_chi = np.gradient(chi, y[:, 0], x[0, :])
    X = np.sqrt(np.abs(grad_chi[1])**2 + np.abs(grad_chi[0])**2)
    return np.trapz(np.trapz(X**2/(X**2+1), x=x[0, :]), x=y[:, 0])


def compute_zeta_ebe(epsilon_ro, epsilon_rr, epsilon_rb):
    """Compute the background relative permit. estimation error.

    The zeta_ebe is an estimation of the error of predicting the
    background region considering specifically the relative
    permittivity information. It is an analogy to the false-positive
    rate.

    Parameters
    ----------
        epsilon_ro, epsilon_rr : :class:`numpy.ndarray`
            Original and recovered relative permittivity maps.
        epsilon_rb : float
            Background relative permittivity.
    """
    background = np.zeros(epsilon_ro.shape, dtype=bool)
    background[epsilon_ro == epsilon_rb] = True
    y = epsilon_ro[background]
    yd = epsilon_rr[background]
    return np.mean(np.abs(y-yd)/y)*100


def compute_zeta_sbe(sigma_o, sigma_r, sigma_b):
    """Compute the background conductivity estimation error.

    The zeta_sbe is an estimation of the error of predicting the
    background region considering specifically the conductivity
    information. It is an analogy to the false-positive
    rate.

    Parameters
    ----------
        sigma_o, sigma_r : :class:`numpy.ndarray`
            Original and recovered conductivity maps.
        sigma_b : float
            Background conductivity.
    """
    background = np.zeros(sigma_o.shape, dtype=bool)
    background[sigma_o == sigma_b] = True
    y = sigma_o[background]
    yd = sigma_r[background]
    return np.mean(np.abs(y-yd))


def compute_zeta_eoe(epsilon_ro, epsilon_rr, epsilon_rb):
    """Compute the object relative permit. estimation error.

    The zeta_eoe is an estimation of the error of predicting the
    object region considering specifically the relative
    permittivity information. It is an analogy to the false-negative
    rate.

    Parameters
    ----------
        epsilon_ro, epsilon_rr : :class:`numpy.ndarray`
            Original and recovered relative permittivity maps.
        epsilon_rb : float
            Background relative permittivity.
    """
    not_background = np.zeros(epsilon_ro.shape, dtype=bool)
    not_background[epsilon_ro != epsilon_rb] = True
    y = epsilon_ro[not_background]
    yd = epsilon_rr[not_background]
    return np.mean(np.abs(y-yd)/y)*100


def compute_zeta_soe(sigma_o, sigma_r, sigma_b):
    """Compute the object conductivity estimation error.

    The zeta_soe is an estimation of the error of predicting the
    object region considering specifically the conductivity
    information. It is an analogy to the false-negative
    rate.

    Parameters
    ----------
        sigma_o, sigma_r : :class:`numpy.ndarray`
            Original and recovered conductivity maps.
        sigma_b : float
            Background conductivity.
    """
    not_background = np.zeros(sigma_o.shape, dtype=bool)
    not_background[sigma_o != sigma_b] = True
    y = sigma_o[not_background]
    yp = sigma_r[not_background]
    return np.mean(np.abs(y-yp))


def compute_zeta_tfmpad(et_o, et_r):
    """Compute the percen. aver. devi. of the total field magnitude.

    The measure estimates the error in the estimation of the
    magnitude of total field.

    Parameters
    ----------
        et_o, et_r : :class:`numpy.ndarray`
            Original and recovered total field, respectively.
    """
    y = np.abs(et_o.flatten())
    yd = np.abs(et_r.flatten())
    return np.mean(np.abs((y-yd)/y))*100


def compute_zeta_tfpad(et_o, et_r):
    """Compute the percen. aver. devi. of the total field phase.

    The measure estimates the error in the estimation of the
    phase of total field.

    Parameters
    ----------
        et_o, et_r : :class:`numpy.ndarray`
            Original and recovered total field, respectively.
    """
    y = np.angle(et_o.flatten())
    yd = np.angle(et_r.flatten())
    return np.mean(np.abs(y-yd))


def compute_zeta_p(chi_o, chi_r):
    Xo, Xr = np.abs(chi_o), np.abs(chi_r)
    threshold = (np.amin(np.abs(Xr))
                 + .5*(np.amax(np.abs(Xr))-np.amin(np.abs(Xr))))

    masko = np.zeros(Xo.shape, dtype=bool)
    maskr = np.zeros(Xr.shape, dtype=bool)

    masko[Xo > 0.] = True
    maskr[Xr >= threshold] = True

    xo, yo = np.meshgrid(np.linspace(0, 1, Xo.shape[1]),
                         np.linspace(0, 1, Xo.shape[0]))

    xr, yr = np.meshgrid(np.linspace(0, 1, Xr.shape[1]),
                         np.linspace(0, 1, Xr.shape[0]))

    if not np.any(maskr) or np.any(np.isnan(maskr)):
        return 100.

    xco = np.sum(masko*xo)/np.sum(masko)
    yco = np.sum(masko*yo)/np.sum(masko)
    xcr = np.sum(maskr*xr)/np.sum(maskr)
    ycr = np.sum(maskr*yr)/np.sum(maskr)

    return np.sqrt((xco-xcr)**2 + (yco-ycr)**2)*100


def compute_zeta_s(chi_o, chi_r):
    Xo, Xr = np.abs(chi_o), np.abs(chi_r)
    threshold = (np.amin(np.abs(Xr))
                 + .5*(np.amax(np.abs(Xr))-np.amin(np.abs(Xr))))

    co = measure.find_contours(Xo, .0, fully_connected='high')
    cr = measure.find_contours(Xr, threshold)

    # Converting scale
    for i in range(len(cr)):
        cr[i][:, 1] = Xo.shape[1]*cr[i][:, 1]/Xr.shape[1]
        cr[i][:, 0] = Xo.shape[0]*cr[i][:, 0]/Xr.shape[0]

    masko = np.zeros(Xo.shape, dtype=bool)
    maskr = np.zeros(Xr.shape, dtype=bool)

    masko[Xo > 0] = True
    maskr[Xr >= threshold] = True

    xo, yo = np.meshgrid(np.arange(0, Xo.shape[1]), np.arange(0, Xo.shape[0]))
    xr, yr = np.meshgrid(np.linspace(0, Xo.shape[1]-1, Xr.shape[1]),
                         np.linspace(0, Xo.shape[0]-1, Xr.shape[0]))

    if np.sum(maskr*Xr) == 0:
        return 100.

    xco = np.sum(masko*Xo*xo)/np.sum(masko*Xo)
    yco = np.sum(masko*Xo*yo)/np.sum(masko*Xo)
    xcr = np.sum(maskr*Xr*xr)/np.sum(maskr*Xr)
    ycr = np.sum(maskr*Xr*yr)/np.sum(maskr*Xr)

    # Centralization
    for i in range(len(co)):
        co[i][:, 0] = co[i][:, 0]-yco+Xo.shape[0]/2
        co[i][:, 1] = co[i][:, 1]-xco+Xo.shape[1]/2

    # Centralization
    for i in range(len(cr)):
        cr[i][:, 0] = cr[i][:, 0]-ycr+Xo.shape[0]/2
        cr[i][:, 1] = cr[i][:, 1]-xcr+Xo.shape[1]/2

    masko = np.zeros(Xo.shape, dtype=bool)
    counter = np.zeros(Xo.shape)
    for i in range(len(co)):
        maskt = measure.grid_points_in_poly(Xo.shape, co[i])
        counter[maskt] += 1
    masko[np.mod(counter, 2) == 1] = True

    maskr = np.zeros(Xo.shape, dtype=bool)
    counter = np.zeros(Xo.shape)
    for i in range(len(cr)):
        maskt = measure.grid_points_in_poly(Xo.shape, cr[i])
        counter[maskt] += 1
    maskr[np.mod(counter, 2) == 1] = True
    
    # Xor operation
    diff = np.logical_xor(masko, maskr)

    # Area of the difference
    area_diff = np.sum(diff)/np.sum(masko)*100

    return area_diff


def check_indicator(indicator):
    if type(indicator) is str:
        return any(indicator == n for n in INDICATOR_SET)
    else:
        return all(any(m == n for n in INDICATOR_SET) for m in indicator)


def indicator_label(indicator):
    if not check_indicator(indicator):
        raise error.WrongValueInput('indicator_label', 'indicator',
                                    INDICATOR_SET, indicator)
    return LABELS[indicator]
    