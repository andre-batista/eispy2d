import pickle
import copy as cp
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

from eispy2d.core import configuration as cfg
from eispy2d.api import api
from eispy2d.core import error
from eispy2d.core import result as rst

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
TEST = "test"
TESTSET_CONDITION = "testset_condition"

_EMPTY = 'Empty'
_MISSING_SHAPE_DATA = 'Missing field data'
_READY = 'Ready'

SHAPES = [
    "triangle", "square", "circle", "ellipse", "cross", 
    "star5", "star6", "rhombus", "trapezoid", "polygon", 
    "random", "ring", "parallelogram"
]

BACKGROUND_PERMITTIVITIES = [1.0, 2.0, 4.0, 8.0]

NUMBER_VALUES = [8, 16, 32]


class TestSet:
    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, new):
        if new is None:
            self._test = None
            self._testset_condition = _EMPTY
        elif type(new) is list:
            self._test = [t.copy() for t in new]
            self._testset_condition = _READY

    def __init__(self, name=None, wavelength=None, image_size=None, 
                 observation_radius=None, resolution=None,
                 noise_level=1., sample_size=30, import_filename=None,
                 import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
            return

        if name is None:
            raise error.MissingInputError('Api.TestSet.__init__', 'name')
        
        if image_size is None:
            raise error.MissingInputError('Api.TestSet.__init__', 'image_size')
        
        if observation_radius is None:
            raise error.MissingInputError('Api.TestSet.__init__', 'observation_radius')
        
        if resolution is None:
            raise error.MissingInputError('Api.TestSet.__init__', 'resolution')
        
        if noise_level is None:
            raise error.MissingInputError('Api.TestSet.__init__', 'noise_level')

        if type(sample_size) is not int:
            raise error.WrongTypeInput('Api.TestSet.__init__', 'sample_size', 'int',
                                       str(type(sample_size)))

        self.sample_size = sample_size
        self.name = name
        self.wavelength = wavelength
        self.image_size = image_size
        self.observation_radius = observation_radius
        self.resolution = resolution
        self.noise_level = noise_level
        self.params = {
            "wavelength": wavelength,
            "image_size": image_size,
            "observation_radius": observation_radius,
            "resolution": resolution
        }
        self._test = None
        self._testset_condition = _MISSING_SHAPE_DATA

    def randomize_tests(self, parallelization=True):
        self._test = []
        N = self.sample_size

        if parallelization:
            num_cores = multiprocessing.cpu_count()
            output = Parallel(n_jobs=num_cores)(
                delayed(_create_input_params)(
                    n,
                    self.wavelength,
                    self.image_size,
                    self.observation_radius,
                    self.resolution,
                    self.noise_level
                ) for n in range(N)
            )
        else:
            output = [
                _create_input_params(
                    n,
                    self.wavelength,
                    self.image_size,
                    self.observation_radius,
                    self.resolution,
                    self.noise_level
                ) for n in range(N)
            ]

        for n in range(N):
            output[n].update(self.params)
            self._test.append(output[n].copy())

        self._testset_condition = _READY

    def generate_field_data(self, parallelization=False):
        self._testset_condition = _READY

    def save(self, file_path=''):
        if file_path is not None:
            self.path = file_path

        data = {
            NAME: self.name,
            WAVELENGTH: self.wavelength,
            IMAGE_SIZE: self.image_size,
            NUMBER_MEASUREMENTS: self.number_measurements if hasattr(self, 'number_measurements') else None,
            NUMBER_SOURCES: self.number_sources if hasattr(self, 'number_sources') else None,
            OBSERVATION_RADIUS: self.observation_radius,
            RESOLUTION: self.resolution,
            BACKGROUND_PERMITTIVITY: self.background_permittivity if hasattr(self, 'background_permittivity') else None,
            NOISE_LEVEL: self.noise_level,
            SHAPE: self.shape if hasattr(self, 'shape') else None,
            SAMPLE_SIZE: self.sample_size,
            TEST: self._test,
            TESTSET_CONDITION: self._testset_condition
        }

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

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
        self._test = data[TEST]
        self._testset_condition = data[TESTSET_CONDITION]

    def copy(self, new=None):
        if new is None:
            obj = TestSet(
                name=self.name,
                wavelength=self.wavelength,
                image_size=self.image_size,
                observation_radius=self.observation_radius,
                resolution=self.resolution,
                noise_level=self.noise_level,
                sample_size=self.sample_size
            )
            obj._test = [t.copy() for t in self._test] if self._test is not None else None
            obj._testset_condition = self._testset_condition
            return obj
        elif type(new) is TestSet:
            self.name = new.name
            self.wavelength = new.wavelength
            self.image_size = new.image_size
            self.observation_radius = new.observation_radius
            self.resolution = new.resolution
            self.noise_level = new.noise_level
            self.sample_size = new.sample_size
            self._test = [t.copy() for t in new._test] if new._test is not None else None
            self._testset_condition = new._testset_condition

    def __str__(self):
        message = 'Test Set: ' + self.name
        message += '\nWavelength: ' + str(self.wavelength)
        message += '\nImage size: ' + str(self.image_size)
        message += '\nObservation radius: ' + str(self.observation_radius)
        message += '\nResolution: ' + str(self.resolution)
        message += '\nNoise level: ' + str(self.noise_level)
        message += '\nSample size: %d' % self.sample_size
        message += '\nTest set condition: ' + self._testset_condition
        return message


def _create_input_params(control_variation, wavelength, image_size,
                         observation_radius, resolution, noise_level):
    shape_id = (control_variation // 24) % len(SHAPES)
    bg_per_id = (control_variation // 6) % len(BACKGROUND_PERMITTIVITIES)
    measurements_id = (control_variation // 2) % len(NUMBER_VALUES)
    sources_id = control_variation % len(NUMBER_VALUES)

    params = {}
    params["shape"] = SHAPES[shape_id]
    params["background_permittivity"] = BACKGROUND_PERMITTIVITIES[bg_per_id]
    params["number_measurements"] = NUMBER_VALUES[measurements_id]
    params["number_sources"] = NUMBER_VALUES[sources_id]
    params["noise_level"] = noise_level
    params["wavelength"] = wavelength
    params["image_size"] = image_size
    params["observation_radius"] = observation_radius
    params["resolution"] = resolution

    return params