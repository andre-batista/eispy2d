"""A module to represent problem case.

Based on the same problem configuration, there may be infinite scenarios
describing different geometries and resolutions. So, this module
provides a class in which we may store information about a scenario,
i.e., a problem case in which we may the scattered field measurements
and some other information which will be received by the solver
describing the problem to be solved.

The :class:`InputData` implements a container which will be the standard
input to solvers and include all the information necessary to solve a
inverse scattering problem.

The following class is defined

:class:`InputData`
    The container representing an instance of a inverse scattering
    problem.
"""

import pickle
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

import error
import configuration as cfg
import result as rst
import richmond as ric

# Constants for easier access to fields of the saved pickle file
NAME = 'name'
CONFIGURATION = 'configuration'
RESOLUTION = 'resolution'
SCATTERED_FIELD = 'scattered_field'
TOTAL_FIELD = 'total_field'
INCIDENT_FIELD = 'incident_field'
REL_PERMITTIVITY = 'rel_permittivity'
CONDUCTIVITY_MAP = 'conductivity'
NOISE = 'noise'
INDICATORS = 'indicators'
DNL = 'dnl'

PERMITTIVITY = 'permittivity'
CONDUCTIVITY = 'conductivity'
BOTH_PROPERTIES = 'both'
CONTRAST = 'contrast'


class InputData:
    """The container representing an instance of a problem.

    Attributes
    ----------
        name
            A string naming the instance.

        configuration_filename
            A string for referencing the problem configuration.

        resolution
            A tuple with the size, in pixels, of the recovered image.
            Y-X ordered.

        scattered_field
            Matrix containing the scattered field information at
            S-domain.

        total_field
            Matrix containing the total field information at D-domain.

        incident_field
            Matrix containing the incident field information at
            D-domain.

        relative_permittivity_map
            Matrix with the discretized image of the relative
            permittivity map.

        conductivity_map
            Matrix with the discretized image of the conductivity map.

        noise
            noise level of scattered field data.

        homogeneous_objects : bool
            A flag to indicate if the instance only contains
            homogeneous objects.

        compute_residual_error : bool
            A flag to indicate the measurement of the residual error
            throughout or at the end of the solver executation.

        compute_map_error : bool
            A flag to indicate the measurement of the error in
            predicting the dielectric properties of the image.

        compute_totalfield_error : bool
            A flag to indicate the measurement of the estimation error
            of the total field throughout or at the end of the solver
            executation.
    """

    @property
    def configuration(self):
        """Get the configuration."""
        return self._configuration

    @configuration.setter
    def configuration(self, configuration):
        """Set the configuration attribute."""
        if type(configuration) is str:
            self._configuration = configuration
        else:
            self._configuration = configuration.copy()

    def __init__(self, name=None, configuration=None, resolution=None,
                 scattered_field=None, total_field=None, incident_field=None,
                 rel_permittivity=None, conductivity=None,
                 noise=None, indicators=None, import_filename=None,
                 import_filepath=''):
        r"""Build or import an object.

        You must give either the import file name and path or the
        required variables.

        Call signatures::

            InputData(import_filename='my_file',
                      import_filepath='./data/')
            InputData(name='instance00',
                      configuration_filename='setup00', ...)

        Parameters
        ----------
            name : string
                The name of the instance.

            `configuration_filename` : string
                A string with the name of the problem configuration
                file.

            resolution : 2-tuple
                The size, in pixels, of the image to be recovered. Y-X
                ordered.

            scattered_field : :class:`numpy.ndarray`
                A matrix containing the scattered field information at
                S-domain.

            total_field : :class:`numpy.ndarray`
                A matrix containing the total field information at
                D-domain.

            incident_field : :class:`numpy.ndarray`
                A matrix containing the incident field information at
                D-domain.

            relative_permittivity_map : :class:`numpy.ndarray`
                A matrix with the discretized image of the relative
                permittivity map.

            conductivity_map : :class:`numpy.ndarray`
                A matrix with the discretized image of the conductivity
                map.

            noise : float
                Noise level of scattered field data.

            homogeneous_objects : bool
                A flag to indicate if the instance only contains
                homogeneous objects.

            compute_residual_error : bool
                A flag to indicate the measurement of the residual error
                throughout or at the end of the solver executation.

             compute_map_error : bool
                A flag to indicate the measurement of the error in
                predicting the dielectric properties of the image.

             compute_totalfield_error : bool
                A flag to indicate the measurement of the estimation
                error of the total field throughout or at the end of the
                solver executation.

            import_filename : string
                A string with the name of the saved file.

            import_filepath : string
                A string with the path to the saved file.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)

        else:

            if name is None:
                raise error.MissingInputError('InputData.__init__()', 'name')
            if configuration is None:
                raise error.MissingInputError('InputData.__init__()',
                                              'configuration_filename')
            if (resolution is None and rel_permittivity is None
                    and conductivity is None):
                raise error.MissingInputError('InputData.__init__()',
                                              'resolution')

            self.name = name
            self.configuration = configuration
            self.dnl = None
            
            if indicators is None:
                self.indicators = rst.INDICATOR_SET.copy()
            elif type(indicators) is str:
                self.indicators = [indicators]
            else:
                self.indicators = indicators.copy()

            if resolution is not None:
                self.resolution = resolution
            else:
                self.resolution = None

            if scattered_field is not None:
                self.scattered_field = np.copy(scattered_field)
            else:
                self.scattered_field = None

            if total_field is not None:
                self.total_field = np.copy(total_field)
            else:
                self.total_field = None

            if incident_field is not None:
                self.incident_field = np.copy(incident_field)
            else:
                self.incident_field = None

            if rel_permittivity is not None:
                self.rel_permittivity = rel_permittivity
                if resolution is None:
                    self.resolution = rel_permittivity.shape
            else:
                self.rel_permittivity = None

            if conductivity is not None:
                self.conductivity = conductivity
                if resolution is None:
                    self.resolution = conductivity.shape
            else:
                self.conductivity = None

            if noise is not None:
                self.noise = noise
            else:
                self.noise = None

    def save(self, file_path=None):
        """Save object information."""
        if file_path is not None:
            self.path = file_path
        data = {
            NAME: self.name,
            CONFIGURATION: self.configuration,
            RESOLUTION: self.resolution,
            SCATTERED_FIELD: self.scattered_field,
            TOTAL_FIELD: self.total_field,
            INCIDENT_FIELD: self.incident_field,
            NOISE: self.noise,
            REL_PERMITTIVITY: self.rel_permittivity,
            CONDUCTIVITY: self.conductivity,
            INDICATORS: self.indicators,
            DNL: self.dnl
        }

        with open(self.path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import data from a saved object."""
        with open(file_path + file_name, 'rb') as datafile:
            data = pickle.load(datafile)
        self.name = data[NAME]
        self.configuration = data[CONFIGURATION]
        self.resolution = data[RESOLUTION]
        self.total_field = data[TOTAL_FIELD]
        self.scattered_field = data[SCATTERED_FIELD]
        self.incident_field = data[INCIDENT_FIELD]
        self.rel_permittivity = data[REL_PERMITTIVITY]
        self.conductivity = data[CONDUCTIVITY]
        self.noise = data[NOISE]
        self.indicators = data[INDICATORS]
        self.dnl = data[DNL]

    def draw(self, image=CONTRAST, axis=None, figure_title=None,
             file_path='', file_format='eps', show=False, save=False,
             suptitle=None, fontsize=10):
        """Draw the relative permittivity/conductivity map.

        Parameters
        ----------
            figure_title : str, optional
                A title that you want to give to the figure.

            show : boolean, default: False
                If `True`, a window will be raised to show the image. If
                `False`, the image will be saved.

            file_path : str, default: ''
                A path where you want to save the figure.

            file_format : str, default: 'eps'
                The file format. It must be one of the available ones by
                `matplotlib.pyplot.savefig()`.
        """
        if (image != PERMITTIVITY and image != CONDUCTIVITY
                and image != BOTH_PROPERTIES and image != CONTRAST):
            raise error.WrongValueInput('InputData.draw', 'image',
                                        "'"+ PERMITTIVITY + "' or '"
                                        + CONDUCTIVITY + "' or '"
                                        + BOTH_PROPERTIES + "' or '"
                                        + CONTRAST + "'", image)
        elif image == PERMITTIVITY and self.rel_permittivity is None:
            raise error.MissingAttributesError('InputData', 'epsilon_r')
        elif image == CONDUCTIVITY and self.conductivity is None:
            raise error.MissingAttributesError('InputData', 'sigma')
        elif self.conductivity is None and self.rel_permittivity is None:
            raise error.MissingAttributesError('InputData',
                                               "'epsilon_r' or 'sigma'")

        if axis is None:
            if image == BOTH_PROPERTIES:
                fig, ax = plt.subplots(ncols=2, figsize=[2*6.4, 4.8],
                                       sharey=True)
                fig.subplots_adjust(wspace=.5)
            else:
                fig, ax = plt.subplots()
                ax = np.array([ax])
        else:
            fig = plt.gcf()
            if type(axis) is not np.ndarray:
                ax = np.array([axis])

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

        if figure_title == False:
            title = ['']
        elif figure_title is not None:
            title = figure_title

        if image == PERMITTIVITY:

            if figure_title is None:
                title = 'Relative Permittivity Map'

            rst.add_image(ax[0],
                          self.rel_permittivity,
                          title,
                          clb_epsilon_r,
                          bounds=extent,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          fontsize=fontsize)
        
        elif image == CONDUCTIVITY:

            if figure_title is None:
                title = 'Conductivity Map'

            rst.add_image(ax[0],
                          self.conductivity,
                          title,
                          clb_sigma,
                          bounds=extent,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          fontsize=fontsize)

        elif image == BOTH_PROPERTIES:

            if figure_title is None:
                title = 'Relative Permittivity Map'

            rst.add_image(ax[0],
                          self.rel_permittivity,
                          title,
                          clb_epsilon_r,
                          bounds=extent,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          fontsize=fontsize)
            
            if figure_title is None:
                title = 'Conductivity Map'

            rst.add_image(ax[1],
                          self.conductivity,
                          title,
                          clb_sigma,
                          bounds=extent,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          fontsize=fontsize)
            
            if suptitle is not None:
                fig.suptitle(suptitle, fontsize=1.6*fontsize)
        
        else:
            
            if figure_title is None:
                title = 'Contrast Map'
            
            X = cfg.get_contrast_map(epsilon_r=self.rel_permittivity,
                                     sigma=self.conductivity,
                                     configuration=self.configuration)

            rst.add_image(ax[0],
                          np.abs(X),
                          title,
                          clb_contrast,
                          bounds=extent,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          fontsize=fontsize)

        plt.tight_layout()

        if save:
            plt.savefig(file_path + self.name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if save:
            plt.close()
        elif not show and axis is None:
            return fig, ax

    def plot_scattered_field(self, figure_title=None, file_path='',
                             file_format='eps', show=False,
                             interpolation='spline36', axes=None):
        """Summarize the method."""
        if self.scattered_field is None:
            raise error.EmptyAttribute('InputData', 'es')

        if axes is None:
            figure, ax, _ = rst.get_figure(1)
            no_axes = True
        else:
            figure = plt.gcf()
            if type(axes) is not np.ndarray:
                ax = np.array([axes])
            no_axes = False

        if figure_title is None:
            figure_title = 'Scattered Field Pattern - ' + self.name
        rst.add_image(ax[0], self.scattered_field, figure_title, r'$|E^s_z$| [V/m]',
                      bounds=(0, 360, 0, 360), xlabel='Source angle [deg]',
                      ylabel='Measurement angle [deg]', aspect='auto',
                      interpolation=interpolation)

        plt.tight_layout()

        if no_axes and show:
            plt.show()
        elif no_axes:
            plt.savefig(file_path + self.name + '_es.' + file_format,
                        format=file_format)
            plt.close()

    def plot_total_field(self, axis=None, source=None, figure_title=None,
                         file_path='', file_format='eps', show=False,
                         fontsize=10):
        """Summarize the method."""
        if self.total_field is None:
            raise error.EmptyAttribute('InputData', 'et')
        if self.resolution is None:
            raise error.EmptyAttribute('InputData', 'resolution')
        if self.resolution[0]*self.resolution[1] != self.total_field.shape[0]:
            raise error.WrongValueInput('InputData.plot_total_field',
                                        "'resolution' and 'et'",
                                        'resolution[0]*resolution[1] == '
                                        + 'et.shape[0]', 'resolution[0]*'
                                        + 'resolution[1] != et.shape[0]')
        NS = self.total_field.shape[1]

        if source is None:
            source = range(NS)
        elif type(source) is int:
            if source >= NS:
                raise error.WrongValueInput('InputData.plot_total_field',
                                            'source', '0 to %d' % source,
                                            str(source))
            source = [source]
        elif type(source) is list:
            if any([s >= NS for s in source]):
                raise error.WrongValueInput('InputData.plot_total_field',
                                            'source', '0 to %d' % source,
                                            str(source))
        else:
            raise error.WrongTypeInput('InputData.plot_total_field', 'source',
                                       'None, int our int-list',
                                       str(type(source)))

        if axis is None:
            figure, ax, _ = rst.get_figure(len(source))
        else:
            if len(source) == 1:
                if isinstance(axis, plt.Axes):
                    ax = [axis]
                elif type(axis) is np.ndarray:
                    if axis.size != len(source):
                        raise error.WrongTypeInput(
                            'InputData.plot_total_field', 'axis',
                            '%dD-numpy.ndarray' % len(source),
                            '%dD-numpy.ndarray' % axis.size
                        )
                    ax = axis
                else:
                    raise error.WrongTypeInput(
                        'InputData.plot_total_field', 'axis',
                        'matplotlib.pyplot.Axes or 1D-numpy.ndarray',
                        str(type(axis))
                    )
            else:
                if type(axis) is np.ndarray:
                    if axis.size != len(source):
                        raise error.WrongTypeInput(
                            'InputData.plot_total_field', 'axis',
                            '%dD-numpy.ndarray' % len(source),
                            '%dD-numpy.ndarray' % axis.size
                        )
                    ax = axis
                else:
                    raise error.WrongTypeInput(
                        'InputData.plot_total_field', 'axis',
                        '%D-numpy.ndarray' % len(source),
                        '%D-numpy.ndarray' % axis.size,
                    )

        ifig = 0
        for i in source:
            img = self.total_field[:, i].reshape(self.resolution)
            title = 'Source %d' % (i+1)
            rst.add_image(ax[ifig], img, title, r'$|E_z|$ [V/m]',
                          bounds=(0, 1, 0, 1), xlabel=r'$L_x$',
                          ylabel=r'$L_y$', fontsize=fontsize)
            ifig += 1
        if figure_title is None:
            figure_title = 'Total Field - ' + self.name
        plt.suptitle(figure_title, fontsize=fontsize)

        plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(file_path + self.name + '_et.' + file_format,
                        format=file_format)
            plt.close()

    def compute_dnl(self):
        self.dnl = degrees_nonlinearity(self)

    def copy(self, new=None):
        """Set or return a copy of the object."""
        if new is None:
            new = InputData(
                name=self.name,
                configuration=self.configuration,
                resolution=(self.resolution[0], self.resolution[1]),
                scattered_field=cp.deepcopy(self.scattered_field),
                total_field=cp.deepcopy(self.total_field),
                incident_field=cp.deepcopy(self.incident_field),
                rel_permittivity=cp.deepcopy(self.rel_permittivity),
                conductivity=cp.deepcopy(self.conductivity),
                noise=self.noise,
                indicators=self.indicators.copy(),
            )
            new.dnl = self.dnl
            return new
        else:
            self.name=new.name,
            self.configuration=new.configuration,
            self.resolution=(new.resolution[0], new.resolution[1]),
            self.scattered_field=cp.deepcopy(new.scattered_field),
            self.total_field=cp.deepcopy(new.total_field),
            self.incident_field=cp.deepcopy(new.incident_field),
            self.rel_permittivity=cp.deepcopy(new.rel_permittivity),
            self.conductivity=cp.deepcopy(new.conductivity),
            self.noise=new.noise,
            self.indicators=new.indicators.copy()
            self.dnl = new.dnl

    def __str__(self):
        """Print information."""
        message = 'Input name: ' + self.name
        message += '\nConfiguration file: '
        if type(self.configuration) is str:
            message += self.configuration
        else:
            message += self.configuration.name
        message += ('\nImages Resolution: %dx' % self.resolution[0]
                    + '%d' % self.resolution[1])
        if self.scattered_field is not None:
            message += ('\nScattered field - measurement samples: %d'
                        % self.scattered_field.shape[0]
                        + '\nScattered field - source samples: %d'
                        % self.scattered_field.shape[1])
        if self.total_field is not None:
            message += ('\nTotal field - measurement samples: %d'
                        % self.total_field.shape[0]
                        + '\nTotal field - source samples: %d'
                        % self.total_field.shape[1])
        if self.incident_field is not None:
            message += ('\nIncident field - measurement samples: %d'
                        % self.incident_field.shape[0]
                        + '\nIncident field - source samples: %d'
                        % self.incident_field.shape[1])
        if self.noise is not None:
            message += '\nNoise level: %.2f%%' % self.noise
        if self.rel_permittivity is not None:
            message += ('\nRelative Permit. map shape: %dx'
                        % self.rel_permittivity.shape[0] + '%d'
                        % self.rel_permittivity.shape[1])
        if self.conductivity is not None:
            message += ('\nConductivity map shape: %dx' % self.conductivity.shape[0]
                        + '%d' % self.conductivity.shape[1])
        message += '\nIndicators: ' + str(self.indicators)
        message += '\nDegrees of Non-Linearity: '
        if self.dnl is not None:
            message += '%.4f' % self.dnl
        else:
            message += 'None'
        
        return message


def degrees_nonlinearity(inputdata):
    discretization = ric.Richmond(inputdata.configuration,
                                  inputdata.resolution,
                                  state=True)
    X = cfg.get_contrast_map(epsilon_r=inputdata.rel_permittivity,
                             sigma=inputdata.conductivity,
                             configuration=inputdata.configuration)
    X = X.reshape((-1, 1))
    x, y = cfg.get_coordinates_ddomain(configuration=inputdata.configuration,
                                       resolution=inputdata.resolution)
    dS = (x[0, 1]-x[0, 0])*(y[1, 0]-y[0, 0])
    
    F = discretization.GD @ X
    return np.sum(np.abs(F)**2)*dS

