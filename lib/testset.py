import pickle
import numpy as np
from numpy import pi
from numpy import random as rnd
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import configuration as cfg
import inputdata as ipt
import result as rst
import experiment as exp
import mom_cg_fft as mom
import draw
import error

NAME = 'name'
CONFIGURATION = 'configuration'
CONTRAST = 'contrast'
OBJECT_SIZE = 'object_size'
DENSITY = 'density'
CONTRAST_MODE = 'contrast_mode'
OBJECT_SIZE_MODE = 'object_size_mode'
DENSITY_MODE = 'density_mode'
RESOLUTION = 'resolution'
MAP_PATTERN = 'map_pattern'
SAMPLE_SIZE = 'sample_size'
NOISE = 'noise'
INDICATORS = 'indicators'
TEST = 'test'
MINIMUM_SIZE_PROPORTION = 'min_size_prop'
ROTATE = 'rotate'
RANDOM_POSITION = 'random_position'
TESTSET_CONDITION = 'testset_condition'
_EMPTY = 'Empty'
_MISSING_FIELD_DATA = 'Missing field data'
_READY = 'Ready'


class TestSet:

    @property
    def configuration(self):
        """Get the configuration."""
        return self._configuration

    @configuration.setter
    def configuration(self, configuration):
        """Set the configuration attribute."""
        self._configuration = configuration.copy()

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, new):
        if new is None:
            self._test = None
            self._testset_condition = _EMPTY
        elif type(new) is ipt.InputData:
            self._test = [new.copy()]
            if new.scattered_field is None or np.size(new.scattered_field) == 0:
                self._testset_condition = _MISSING_FIELD_DATA
            else:
                self._testset_condition = _READY
        elif type(new) is list:
            self._test = [new[i].copy() for i in range(len(new))]
            self._testset_condition = _READY
            for i in range(len(new)):
                if self._test[i].scattered_field is None or np.size(self._test[i].scattered_field) == 0:
                    self._testset_condition = _MISSING_FIELD_DATA

    def __init__(self, name=None, configuration=None, contrast=None,
                 object_size=None, resolution=None, density=None,
                 map_pattern=exp.REGULAR_POLYGONS_PATTERN, sample_size=30,
                 noise=1., indicators=None, contrast_mode=exp.FIXED_CONTRAST,
                 object_size_mode=exp.FIXED_SIZE,
                 density_mode=exp.SINGLE_OBJECT, min_size_proportion=40.,
                 allow_rotation=True, random_position=True,
                 import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
            return
        
        if name is None:
            raise error.MissingInputError('TestSet.__init__', 'name')
        elif type(configuration) is not cfg.Configuration:
            raise error.WrongTypeInput('TestSet.__init__', 'configuration',
                                       'Configuration',
                                       str(type(configuration)))
        if contrast is None:
            raise error.MissingInputError('TestSet.__init__', 'contrast')
        elif type(contrast) is not float and type(contrast) is not int:
            raise error.WrongTypeInput('TestSet.__init__', 'contrast',
                                       'int or float', str(type(contrast)))
        if object_size is None:
            raise error.MissingInputError('TestSet.__init__', 'object_size')
        elif type(object_size) is not float and type(object_size) is not int:
            raise error.WrongTypeInput('TestSet.__init__', 'object_size',
                                       'int or float', str(type(object_size)))
        if density is None and density_mode != exp.SINGLE_OBJECT:
            raise error.MissingInputError('TestSet.__init__', 'density')
        elif (type(density) is not float and type(density) is not int
                and density is not None):
            raise error.WrongTypeInput('TestSet.__init__', 'density',
                                       'int or float', str(type(density)))
        if (map_pattern != exp.RANDOM_POLYGONS_PATTERN
                and map_pattern != exp.REGULAR_POLYGONS_PATTERN
                and map_pattern != exp.SURFACES_PATTERN):
            raise error.WrongValueInput('TestSet.__init__', 'map_pattern',
                                        "'random_polygons'"
                                        + " or 'regular_polygons'"
                                        + " or 'surfaces'", str(map_pattern))
        if type(sample_size) is not int:
            raise error.WrongTypeInput('TestSet.__init__', 'sample_size', 'int',
                                       str(type(sample_size)))
        if noise is None:
            noise = 0.
        elif type(noise) is not int and type(noise) is not float:
            raise error.WrongTypeInput('TestSet.__init__', 'noise',
                                       'int or float', str(type(noise)))
        if (indicators is not None and type(indicators) is not str
                and type(indicators) is not list):
            raise error.WrongTypeInput('TestSet.__init__', 'indicators',
                                       'None or str or list of str',
                                       str(type(indicators)))
        if (contrast_mode != exp.FIXED_CONTRAST
                and contrast_mode != exp.MAXIMUM_CONTRAST):
            raise error.WrongValueInput('TestSet.__init__', 'contrast_mode',
                                        "'fixed' or 'maximum'",
                                        str(contrast_mode))
        if (object_size_mode != exp.FIXED_SIZE
                and object_size_mode != exp.MAXIMUM_SIZE):
            raise error.WrongValueInput('TestSet.__init__', 'object_size_mode',
                                        "'fixed' or 'maximum'",
                                        str(object_size_mode))
        if (density_mode != exp.SINGLE_OBJECT
                and density_mode != exp.FIXED_NUMBER
                and density_mode != exp.MAXIMUM_NUMBER
                and density_mode != exp.MINIMUM_DENSITY):
            raise error.WrongValueInput('TestSet.__init__', 'density_mode',
                                        "'single' or 'fixed' or 'max_number' "
                                        + "or 'min_density'",
                                        str(density_mode))
        if (type(min_size_proportion) is not float
                and type(min_size_proportion) is not int):
            raise error.WrongTypeInput('TestSet.__init__',
                                       'min_size_proportion',
                                       'bool', str(type(min_size_proportion)))
        elif min_size_proportion < 0 or min_size_proportion > 100.:
            raise error.WrongValueInput('TestSet.__init__',
                                        'min_size_proportion',
                                        ">0. and <1.",
                                        str(min_size_proportion))
        if type(allow_rotation) is not bool:
            raise error.WrongTypeInput('TestSet.__init__',
                                       'allow_rotation',
                                       'bool', str(type(allow_rotation)))
        if type(random_position) is not bool:
            raise error.WrongTypeInput('TestSet.__init__',
                                       'random_position',
                                       'bool', str(type(random_position)))

        self.name = name
        self.contrast = contrast
        self.object_size = object_size
        self.density = density
        self.contrast_mode = contrast_mode
        self.object_size_mode = object_size_mode
        self.density_mode = density_mode
        self.resolution = resolution
        self.map_pattern = map_pattern
        self.sample_size = sample_size
        self.noise = noise 
        self.test = None
        self.min_size_prop = min_size_proportion
        self.rotate = allow_rotation
        self.random_position = random_position
        self._testset_condition = _EMPTY
        
        if configuration is None:
            dof = cfg.degrees_of_freedom(
                object_radius=self.object_size, wavelength=1.,
                epsilon_r=cfg.get_relative_permittivity(self.contrast, 1.)
            )
            self.configuration = cfg.Configuration(
                name='cfg_' + self.name, number_measurements=3*dof,
                number_sources=3*dof, observation_radius=3*object_size,
                wavelength=1., image_size=[2*object_size, 2*object_size],
                perfect_dielectric=True
            )
        else:
            self.configuration = configuration

        if indicators is not None and not rst.check_indicator(indicators):
            raise error.WrongValueInput('testset.__init__', 'indicators',
                                        "'None' or "
                                        + str(["'" + ind + "' " for ind 
                                               in rst.INDICATOR_SET[:-1]])
                                        + "'" + rst.INDICATOR_SET[-1] + "'",
                                        str(indicators))
        elif indicators is None:
            self.indicators = rst.INDICATOR_SET.copy()
        elif type(indicators) is str:
            self.indicators = [indicators]
        elif type(indicators) is list:
            self.indicators = indicators.copy()

        if resolution is None:
            self.resolution = (
                int(np.ceil(self.configuration.Ly
                            / (self.lambda_b
                               / exp.STANDARD_WAVELENGTH_PROPORTION))),
                int(np.ceil(self.configuration.Lx
                            / (self.lambda_b
                               / exp.STANDARD_WAVELENGTH_PROPORTION)))
            )
        elif type(resolution) is int:
            self.resolution = (resolution, resolution)
        elif type(resolution) is tuple or type(resolution) is list:
            self.resolution = (resolution[0], resolution[1])
        else:
            raise error.WrongTypeInput('TestSet.__init__',
                                       'resolution',
                                       'None, int, 2-tuple, or 2-list',
                                       str(type(resolution)))

    def randomize_tests(self, parallelization=True):
        self.test = []
        N = self.sample_size

        # Create the sample parallely
        if parallelization:
            num_cores = multiprocessing.cpu_count()
            output = Parallel(n_jobs=num_cores)(
                delayed(create_input_image)(
                    self.name + '_%d' % n,
                    self.configuration,
                    self.resolution,
                    self.map_pattern,
                    self.contrast,
                    density=self.density,
                    noise=self.noise,
                    object_size=self.object_size,
                    indicators=self.indicators,
                    contrast_mode=self.contrast_mode,
                    object_size_mode=self.object_size_mode,
                    min_size_prop=self.min_size_prop,
                    density_mode=self.density_mode,
                    rotate=self.rotate,
                    random_position=self.random_position
                ) for n in range(N)
            )
        else:
            output = [
                create_input_image(self.name + '_%d' % n,
                                   self.configuration, 
                                   self.resolution,
                                   self.map_pattern,
                                   self.contrast,
                                   density=self.density,
                                   noise=self.noise,
                                   object_size=self.object_size,
                                   indicators=self.indicators,
                                   contrast_mode=self.contrast_mode,
                                   object_size_mode=self.object_size_mode,
                                   min_size_prop=self.min_size_prop,
                                   density_mode=self.density_mode,
                                   rotate=self.rotate,
                                   random_position=self.random_position) 
                for n in range(N)
            ]

        # Append scenarios into the list
        for n in range(N):
            self.test.append(output[n].copy())
        
        self._testset_condition = _MISSING_FIELD_DATA

    def generate_field_data(self, solver=None):
        if solver is None:
            solver = mom.MoM_CG_FFT()
        if (rst.TOTALFIELD_MAGNITUDE_PAD in self.indicators
                or rst.TOTALFIELD_PHASE_AD in self.indicators):
            SAVE_INTERN_FIELD = True
        else:
            SAVE_INTERN_FIELD = False
        for n in range(self.sample_size):
            solver.solve(self.test[n],
                         SAVE_INTERN_FIELD=SAVE_INTERN_FIELD)
        self._testset_condition = _READY

    def plot(self, tests='all', axis=None, show=False, save=False,
             file_path='', file_format='eps'):
        if (tests != 'all' and type(tests) is not int
                and type(tests) is not list):
            raise error.WrongTypeInput('TestSet.plot()', 'tests',
                                       "'all' or int or list",
                                       str(type(tests)))
        if tests == 'all':
            tests = [i for i in range(self.sample_size)]
        elif type(tests) is int:
            tests = [tests]

        if (self.configuration.perfect_dielectric
                or self.configuration.good_conductor):
            N = len(tests)
        else:
            N = 2*len(tests)

        if axis is None:
            fig, axis, _ = rst.get_figure(nsubplots=N)
        else:
            fig = plt.gcf()

        if (self.configuration.perfect_dielectric
                or self.configuration.good_conductor):
            for i in range(N):
                self.test[tests[i]].draw(axis=axis[i],
                                         figure_title='Sam. %d' % tests[i])
        else:
            for i in range(len(tests)):
                self.test[tests[i]].draw(axis=axis[(2*i):(2*i+2)],
                                         figure_title='Sam. %d' % tests[i],
                                         suptitle=False)

        if show:
            plt.show()
        elif save:
            plt.savefig(file_path + self.name + '.' + file_format,
                        format=file_format)
            plt.close()
        else:
            return fig, axis
                
    def save(self, file_path=None):
        """Save the problem configuration within a pickle file.

        It will only be saved the attribute variables, not the object
        itself. If you want to load these variables, you may use the
        constant string variables for a more friendly usage.
        """
        if file_path is not None:
            self.path = file_path
        data = {
            NAME: self.name,
            CONFIGURATION: self.configuration,
            CONTRAST: self.contrast,
            OBJECT_SIZE: self.object_size,
            DENSITY: self.density,
            CONTRAST_MODE: self.contrast_mode,
            OBJECT_SIZE_MODE: self.object_size_mode,
            DENSITY_MODE: self.density_mode,
            RESOLUTION: self.resolution,
            MAP_PATTERN: self.map_pattern,
            SAMPLE_SIZE: self.sample_size,
            NOISE: self.noise,
            INDICATORS: self.indicators,
            TEST: self.test,
            MINIMUM_SIZE_PROPORTION: self.min_size_prop,
            ROTATE: self.rotate,
            RANDOM_POSITION: self.random_position,
            TESTSET_CONDITION: self._testset_condition
        }

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import data from a saved object."""
        data = cfg.import_dict(file_name, file_path)
        self.name = data[NAME]
        self.configuration = data[CONFIGURATION]
        self.contrast = data[CONTRAST]
        self.object_size = data[OBJECT_SIZE]
        self.density = data[DENSITY]
        self.contrast_mode = data[CONTRAST_MODE]
        self.object_size_mode = data[OBJECT_SIZE_MODE]
        self.density_mode = data[DENSITY_MODE]
        self.resolution = data[RESOLUTION]
        self.map_pattern = data[MAP_PATTERN]
        self.sample_size = data[SAMPLE_SIZE]
        self.noise = data[NOISE]
        self.indicators = data[INDICATORS]
        self.test = data[TEST]
        self.min_size_prop = data[MINIMUM_SIZE_PROPORTION]
        self.rotate = data[ROTATE]
        self.random_position = data[RANDOM_POSITION]
        self._testset_condition = data[TESTSET_CONDITION]

    def copy(self, new=None):
        if new is None:
            obj = TestSet(self.name,
                          self.configuration,
                          contrast=self.contrast,
                          object_size=self.object_size,
                          density=self.density,
                          contrast_mode=self.contrast_mode,
                          object_size_mode=self.object_size_mode,
                          density_mode=self.density_mode,
                          resolution=(self.resolution[0], self.resolution[1]),
                          map_pattern=self.map_pattern,
                          sample_size=self.sample_size,
                          noise=self.noise,
                          indicators=self.indicators,
                          min_size_proportion=self.min_size_prop,
                          allow_rotation=self.rotate,
                          random_position=self.random_position)
            obj.test = [self.test[n].copy() for n in range(len(self.test))]
            return obj
        elif type(new) is TestSet:
            self.name = new.name
            self.configuration = new.configuration.copy()
            self.contrast = new.contrast
            self.object_size = new.object_size
            self.density = new.density
            self.contrast_mode = new.contrast_mode
            self.object_size_mode = new.object_size_mode
            self.density_mode = new.density_mode
            self.resolution = (new.resolution[0], new.resolution[1])
            self.map_pattern = new.map_pattern
            self.sample_size = new.sample_size
            self.noise = new.noise
            self.indicators = new.indicators.copy()
            self.min_size_prop = new.min_size_prop
            self.rotate = new.rotate
            self.random_position = new.random_position
            self.test = [new.test[n].copy() for n in range(len(new.test))]

    def __str__(self):
        message = 'Sample: ' + self.name
        message += '\nConfiguration name: ' + self.configuration.name
        message += '\nContrast: %.3f ' % self.contrast
        if self.contrast_mode == exp.FIXED_CONTRAST:
            message += '(fixed)'
        elif self.contrast_mode == exp.MAXIMUM_CONTRAST:
            message += '(maximum)'
        message += '\nObject size: %.3f [wavelengths] ' % self.object_size
        if self.object_size_mode == exp.FIXED_SIZE:
            message += '(fixed)'
        elif self.object_size_mode == exp.MAXIMUM_SIZE:
            message += '(maximum)'
        message += '\nDensity: '
        if self.density_mode == exp.SINGLE_OBJECT:
            message += 'single object'
        elif self.density_mode == exp.FIXED_NUMBER:
            message += '%d objects (fixed)' % self.density
        elif self.density_mode == exp.MAXIMUM_NUMBER:
            message += '%d objects (maximum)' % self.density
        elif self.density_mode == exp.MINIMUM_DENSITY:
            message += '%.2f [%] (minimum)' % self.density
        message += ('\nResolution: %dx' % self.resolution[0]
                        + '%d' % self.resolution[1])
        message += '\nMap pattern: ' + self.map_pattern
        message += '\nSample size: %d' % self.sample_size
        if self.noise is not None:
            message += '\nNoise: %.2f [%%]' % self.noise
        else:
            message += '\nNoise: 0 [%%]'
        message += '\nIndicators: ' + str(self.indicators)        
        message += '\nMinimum size proportion: %.2f' % self.min_size_prop
        message += '\nTest set condition: ' + self._testset_condition 
        message += '\nAllow rotation? '
        if self.rotate:
            message += 'Yes'
        else:
            message += 'No'
        message += '\nPosition of objects: '
        if self.random_position:
            message += 'Random'
        else:
            message += 'Centered'
        return message
        

def create_input_image(name, configuration, resolution, map_pattern,
                       contrast, density=None, noise=None,
                       object_size=None, indicators=None,
                       contrast_mode=exp.MAXIMUM_CONTRAST,
                       object_size_mode=exp.MAXIMUM_SIZE, min_size_prop=40.,
                       density_mode=exp.SINGLE_OBJECT, rotate=True,
                       random_position=True):
    """Create a single input case.

    Parameters
    ----------
        name : str
            The name of the case.

        configuration : :class:`configuration.Configuration`

        resolution : 2-tuple of int
            Y-X resolution (number of pixels) of the scenario image.

        map_pattern : {'random_polygons', 'regular_polygons', 'surfaces'}
            Pattern of dielectric information on the image.

        contrast : complex
            Reference value for contrast.

        density : int or float, default: None
            When `density_mode` is 'single', `density` might be `None`.
            When `density_mode` is 'fixed', `density` is an integer
            meaning the number of objects in the image. When
            `density_mode` is 'max_number', `density` is an integer
            meaning the maximum number of objects allowed in the image.
            When `density_mode` is 'min_density', `density` is a float
            which is equivalent to the minimum acceptable value for the
            ratio between the average contrast per pixel and the
            reference contrast value. In other words, it is the lower
            percentage limit for the average contrast per pixel compared
            to the reference value. When dealing with surfaces, this
            information is considered for gaussian random functions.

        noise : float, default: None
            Noise level that will be added into the scattered field.

        object_size : float, default: .45*min([Lx, Ly])/2
            Reference value for objects size. *In wavelengths!*

        compute_residual_error : bool, default: None
            A flag to save residual error when running the input.

        compute_map_error : bool, default: None
            A flag to save map error when running the input.

        compute_totalfield_error : bool, default: None
            A flag to save total field error when running the input.

        contrast_mode : str, default: 'maximum'
            How to use the reference value: fixed or maximum allowed.

        object_size_mode : str, default: 'maximum'
            How to use the reference value: fixed or maximum allowed.

        min_size_prop : float, default: 40.
            Proportion (0-100%) of the smallest object allowed in respect to the
            largest.

        density_mode : str, default: 'single'
            Controls the amount of objects which will be added into the
            image. If 'single', then only one object will be added. If
            'fixed', then a fixed amount of objects will be addressed.
            If 'max_number', then a random number of objects will be
            addressed which is not greater than the specified maximum.
            If 'min_density', then objects will be added until the
            average contrast per pixel reaches a minimum percentage
            value in relation to the reference value.

    Returns
    -------
        :class:`inputdata.InputData`
    """
    if (contrast_mode != exp.FIXED_CONTRAST
            and contrast_mode != exp.MAXIMUM_CONTRAST):
        raise error.WrongValueInput('create_input_image', 'contrast_mode',
                                    "'fixed' or 'maximum'", str(contrast_mode))
    if (object_size_mode != exp.FIXED_SIZE
            and object_size_mode != exp.MAXIMUM_SIZE):
        raise error.WrongValueInput('create_input_image', 'object_size_mode',
                                    "'fixed' or 'maximum'",
                                    str(object_size_mode))
    if (density_mode != exp.SINGLE_OBJECT and density_mode != exp.FIXED_NUMBER
            and density_mode != exp.MAXIMUM_NUMBER
            and density_mode != exp.MINIMUM_DENSITY):
        raise error.WrongValueInput('create_input_image', 'density_mode',
                                    "'single' or 'fixed' or 'max_number' "
                                    "or 'min_density'", str(density_mode))
    if (indicators is not None and type(indicators) is not str
                and type(indicators) is not list):
            raise error.WrongTypeInput('create_input_image.__init__',
                                       'indicators', 'None or str or list of'
                                       + ' str', str(type(indicators)))
    elif indicators is not None and not rst.check_indicator(indicators):
            raise error.WrongValueInput('create_input_image.__init__',
                                        'indicators', "'None' or "
                                        + str(["'" + ind + "' " for ind 
                                               in rst.INDICATOR_SET[:-1]])
                                        + "'" + rst.INDICATOR_SET[-1] + "'",
                                        str(indicators))
    elif indicators is None:
        indicators = rst.INDICATOR_SET.copy()
    elif type(indicators) is str:
        indicators = [indicators]

    # Basic parameters of the model
    Lx = configuration.Lx/configuration.lambda_b
    Ly = configuration.Ly/configuration.lambda_b
    epsilon_rb = configuration.epsilon_rb
    sigma_b = configuration.sigma_b
    omega = 2*pi*configuration.f

    # Determining bounds of the conductivity values
    if configuration.perfect_dielectric:
        min_sigma = max_sigma = sigma_b
    elif contrast_mode == exp.FIXED_CONTRAST:
        min_sigma = cfg.get_conductivity(contrast, omega, epsilon_rb, sigma_b)
        max_sigma = min_sigma
    elif contrast_mode == exp.MAXIMUM_CONTRAST:
        min_sigma = 0.
        max_sigma = cfg.get_conductivity(contrast, omega, epsilon_rb, sigma_b)

    # Determining bounds of the relative permittivity values
    if configuration.good_conductor:
        min_epsilon_r = max_epsilon_r = epsilon_rb
    elif contrast_mode == exp.FIXED_CONTRAST:
        min_epsilon_r = cfg.get_relative_permittivity(contrast, epsilon_rb)
        max_epsilon_r = min_epsilon_r
    elif contrast_mode == exp.MAXIMUM_CONTRAST:
        min_epsilon_r = 1.
        max_epsilon_r = cfg.get_relative_permittivity(contrast, epsilon_rb)
        
    if rotate:
        max_rotation = 180.
    else:
        max_rotation = 0.
        
    if not random_position:
        center = [0., 0.]

    # Polygons with random number of edges
    if (map_pattern == exp.RANDOM_POLYGONS_PATTERN
            or map_pattern == exp.REGULAR_POLYGONS_PATTERN):

        if density_mode == exp.FIXED_NUMBER:
            MAX_OBJECTS = density
        elif density_mode == exp.MAXIMUM_NUMBER:
            MAX_OBJECTS = 1 + rnd.randint(density)
        else:
            MAX_OBJECTS = None

        # Defining the maximum object size if it is not defined in the
        # argument.
        if object_size is None:
            maximum_object_size = .45*min([Lx, Ly])/2
        else:
            maximum_object_size = object_size

        if object_size_mode == exp.FIXED_SIZE:
            minimum_object_size = maximum_object_size
        elif object_size_mode == exp.MAXIMUM_SIZE:
            minimum_object_size = min_size_prop/100*maximum_object_size

        # Parameters of the image
        xmin, xmax = cfg.get_bounds(Lx)
        ymin, ymax = cfg.get_bounds(Ly)

        # Initial map
        epsilon_r = epsilon_rb*np.ones(resolution)
        sigma = sigma_b*np.ones(resolution)
        chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                   omega)

        # Add objects until the density is satisfied
        nobjects = 1
        while True:

            # Determine the maximum radius of the edges of the polygon (random)
            radius = minimum_object_size + (maximum_object_size
                                            - minimum_object_size)*rnd.rand()

            # Determine randomly the relative permittivity of the object
            epsilon_ro = min_epsilon_r + (max_epsilon_r
                                          - min_epsilon_r)*rnd.rand()

            # Determine randomly the conductivity of the object
            sigma_o = min_sigma + (max_sigma-min_sigma)*rnd.rand()

            # Determine randomly the position of the object
            if random_position:
                center = [xmin+1.42*radius + (xmax-1.42*radius
                                              - (xmin+1.42*radius))*rnd.rand(),
                          ymin+1.42*radius + (ymax-1.42*radius
                                              - (ymin+1.42*radius))
                          * rnd.rand()]

            if map_pattern == exp.RANDOM_POLYGONS_PATTERN:
                shape = -1
            else:
                # Choose randomly one of the 14 geometric shapes available
                shape = rnd.randint(14)

            if shape == -1:
                # Draw the polygon over the current image (random choice of the
                # number of edges, max: 15)
                epsilon_r, sigma = draw.random(
                    rnd.randint(4, 15), radius, minimum_radius=.5*radius,
                    axis_length_x=Lx, axis_length_y=Ly,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Square
            elif shape == 0:
                epsilon_r, sigma = draw.square(
                    radius*np.sqrt(2), axis_length_x=Lx, axis_length_y=Ly,
                    background_rel_permittivity=epsilon_rb,
                    rotate=rnd.rand()*max_rotation,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Triangle
            elif shape == 1:
                epsilon_r, sigma = draw.triangle(
                    radius*np.sqrt(3), axis_length_x=Lx, axis_length_y=Ly,
                    background_rel_permittivity=epsilon_rb,
                    rotate=rnd.rand()*max_rotation,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Circle
            elif shape == 2:
                epsilon_r, sigma = draw.circle(
                    radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Ring
            elif shape == 3:
                epsilon_r, sigma = draw.ring(
                    (0.2 + 0.5*rnd.rand())*radius, radius, axis_length_x=Lx,
                    axis_length_y=Ly,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Ellipse
            elif shape == 4:
                epsilon_r, sigma = draw.ellipse(
                    (0.3 + .4*rnd.rand())*radius, radius, axis_length_x=Lx,
                    axis_length_y=Ly, rotate=rnd.rand()*max_rotation,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # 4-point star
            elif shape == 5:
                epsilon_r, sigma = draw.star4(
                    radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_rel_permittivity=epsilon_rb,
                    rotate=rnd.rand()*max_rotation,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # 5-point star
            elif shape == 6:
                epsilon_r, sigma = draw.star5(
                    radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_rel_permittivity=epsilon_rb,
                    rotate=rnd.rand()*max_rotation,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # 6-point star
            elif shape == 7:
                epsilon_r, sigma = draw.star6(
                    radius, axis_length_x=Lx, axis_length_y=Ly,
                    background_rel_permittivity=epsilon_rb,
                    rotate=rnd.rand()*max_rotation,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Rhombus
            elif shape == 8:
                epsilon_r, sigma = draw.rhombus(
                    radius, (.4+.3*rnd.rand())*radius, axis_length_x=Lx,
                    axis_length_y=Ly, rotate=rnd.rand()*max_rotation,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Trapezoid
            elif shape == 9:
                l = 2*radius*(.2 + .8*rnd.rand())
                u = 2*radius*(.2 + .8*rnd.rand())
                h = np.sqrt(radius**2-u**2/4) + np.sqrt(radius**2-l**2/4)
                epsilon_r, sigma = draw.trapezoid(
                    u, l, h, axis_length_x=Lx, axis_length_y=Ly,
                    rotate=rnd.rand()*max_rotation*2,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Parallelogram
            elif shape == 10:
                a = 30 + 50*rnd.rand()
                tga = np.tan(np.deg2rad(a))
                h = 2*radius*np.sin(np.deg2rad(a)/2)
                l = (-2*h/tga + np.sqrt((2*h/tga)**2
                                        - 4*(h**2+h**2/tga**2-4*radius**2)))/2
                epsilon_r, sigma = draw.parallelogram(
                    l, h, a, axis_length_x=Lx, axis_length_y=Ly,
                    rotate=rnd.rand()*max_rotation*2,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Regular polygon (pentagon, hexagon, ...)
            elif shape == 11:
                epsilon_r, sigma = draw.polygon(
                    5+rnd.randint(3), radius,
                    axis_length_x=Lx, axis_length_y=Ly,
                    rotate=rnd.rand()*max_rotation,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Cross
            elif shape == 12:
                t = (.4 + rnd.rand()*0.3)*radius
                h = np.sqrt(4*radius**2-t**2)
                w = (.5+.5*rnd.rand())*h
                epsilon_r, sigma = draw.cross(
                    h, w, t,
                    axis_length_x=Lx, axis_length_y=Ly,
                    rotate=rnd.rand()*max_rotation,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            # Line
            elif shape == 13:
                t = (.4 + rnd.rand()*0.3)*radius
                l = np.sqrt(4*radius**2-t**2)
                epsilon_r, sigma = draw.line(
                    l, t, axis_length_x=Lx, axis_length_y=Ly,
                    background_rel_permittivity=epsilon_rb,
                    background_conductivity=sigma_b,
                    rotate=rnd.rand()*max_rotation,
                    object_rel_permittivity=epsilon_ro,
                    object_conductivity=sigma_o, center=center,
                    rel_permittivity=epsilon_r, conductivity=sigma
                )

            if density_mode == exp.SINGLE_OBJECT:
                break
            elif (density_mode == exp.FIXED_NUMBER
                    or density_mode == exp.MAXIMUM_NUMBER):
                if nobjects < MAX_OBJECTS:
                    nobjects += 1
                else:
                    break
            elif density_mode == exp.MINIMUM_DENSITY:
                chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb,
                                           sigma_b, omega)
                if 100*contrast_density(chi)/np.abs(contrast) >= density:
                    break
                else:
                    nobjects += 1

    # Random surfaces (waves of gaussian)
    elif map_pattern == exp.SURFACES_PATTERN:

        # Randomly decide between waves or gaussian functions
        if rnd.rand() < 0.5:
            epsilon_r, sigma = draw.random_waves(
                int(np.ceil(10*rnd.rand())), 5, resolution=resolution,
                rel_permittivity_amplitude=max_epsilon_r-epsilon_rb,
                conductivity_amplitude=max_sigma-sigma_b, axis_length_x=Lx,
                axis_length_y=Ly, background_rel_permittivity=epsilon_rb,
                background_conductivity=sigma_b
            )

        else:

            # When setting gaussian functions, the maximum object size
            # is used as a measure of the variance
            if object_size is None:
                maximum_object_size = .45*min([Lx, Ly])/2
            else:
                maximum_object_size = object_size

            if object_size_mode == exp.FIXED_SIZE:
                minimum_object_size = maximum_object_size
            else:
                minimum_object_size = min_size_prop/100*maximum_object_size

            # Image parameters            
            epsilon_r = epsilon_rb*np.ones(resolution)
            sigma = sigma_b*np.ones(resolution)
            chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb, sigma_b,
                                       omega)

            if density_mode == exp.FIXED_NUMBER:
                MAX_OBJECTS = density
            elif density_mode == exp.MAXIMUM_NUMBER:
                MAX_OBJECTS = 1 + rnd.randint(density)
            else:
                MAX_OBJECTS = None

            # Add gaussian functions until the density criterion is
            # satisfied
            nobjects = 1
            while True:
                epsilon_r, sigma = draw.random_gaussians(
                    1, maximum_spread=4*maximum_object_size,
                    minimum_spread=minimum_object_size,
                    rel_permittivity_amplitude=max_epsilon_r,
                    conductivity_amplitude=max_sigma, axis_length_x=Lx,
                    axis_length_y=Ly, background_conductivity=sigma_b,
                    background_rel_permittivity=epsilon_rb,
                    rel_permittivity=epsilon_r,
                    conductivity=sigma
                )

                if density_mode == exp.SINGLE_OBJECT:
                    break
                elif (density_mode == exp.FIXED_NUMBER
                        or density_mode == exp.MAXIMUM_NUMBER):
                    if nobjects < MAX_OBJECTS:
                        nobjects += 1
                    else:
                        break
                elif density_mode == exp.MINIMUM_DENSITY:
                    chi = cfg.get_contrast_map(epsilon_r, sigma, epsilon_rb,
                                               sigma_b, omega)
                    if 100*contrast_density(chi)/np.abs(contrast) >= density:
                        break
                    else:
                        nobjects += 1

    # Build input object
    inputdata = ipt.InputData(name=name, configuration=configuration,
                              resolution=resolution, noise=noise,
                              indicators=indicators)

    # Set maps
    if not configuration.good_conductor:
        inputdata.rel_permittivity = epsilon_r
    if not configuration.perfect_dielectric:
        inputdata.conductivity = sigma

    inputdata.compute_dnl()

    return inputdata


def contrast_density(contrast_map):
    """Compute the contrast density of a map.

    The contrast density is defined as the mean of the absolute value
    per pixel.

    Parameters
    ----------
        contrast_map : :class:`numpy.ndarray`
            2-d array.
    """
    return np.mean(np.abs(contrast_map))

