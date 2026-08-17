import pickle
from joblib import Parallel, delayed
import multiprocessing

from eispy2d.core import configuration as cfg
from eispy2d.api import api 
from eispy2d.core import error

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

_EMPTY = 'Empty'
_MISSING_SHAPE_DATA = 'Missing field data'
_READY = 'Ready'

class TestSet:


    def __init__(self, name=None, wavelength=None, image_size=None, 
                observation_radius=None, resolution=None,
                 noise_level=1., sample_size=30):

        if name == None:
            raise error.MissingInputError('Api.TestSet.__init__', 'name')
        
        if image_size == None:
            raise error.MissingInputError('Api.TestSet.__init__', 'image_size')
        
        if observation_radius == None:
            raise error.MissingInputError('Api.TestSet.__init__', 'observation_radius')
        
        if resolution == None:
            raise error.MissingInputError('Api.TestSet.__init__', 'resolution')
        
        if noise_level == None:
            raise error.MissingInputError('Api.TestSet.__init__', 'noise_level')

        if type(sample_size) is not int:
            raise error.WrongTypeInput('Api.TestSet.__init__', 'sample_size', 'int',
                                               str(type(sample_size)))

        self.sample_size = sample_size
        self.name = name
        self.params = {"wavelength": wavelength,"image_size" : image_size,
                       "observation_radius" : observation_radius, "resolution" : resolution}
        
        self._testset_condition = _MISSING_SHAPE_DATA

    def randomize_tests(self, parallelization=True):

        self.test = []
        N = self.sample_size

        if parallelization:
            num_cores = multiprocessing.cpu_count()
            output = Parallel(n_jobs=num_cores)(
                delayed(create_input_image)(
                    self.name,
                    n
                ) for n in range(N)
            )
        else:
            output = [
                create_input_image(self.name, n)
                for n in range(N)
            ]

        for n in range(N):
            output[n].update(self.params)
            self.test.append(output[n].copy())

        self._testset_condition = _READY

        print(output)

    def save(self, file_path=''):
        if file_path is not None:
                    self.path = file_path

        data = {
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
            SAMPLE_SIZE: self.sample_size
        }

        with open(file_path + self.name, 'wb') as datafile:
                    pickle.dump(data, datafile)

    def importdata(self, file_name, file_path):
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

def create_input_image(self, control_variation):

    shapes = [
        "triangle", "square", "circle", "ellipse", "cross", 
        "star5", "star6", "rhombus", "trapezoid", "polygon", 
        "random", "ring", "parallelogram"
    ]

    background_permittivities = [1.0, 2.0, 4.0, 8.0]

    # number of sources and measurements
    number_values = [8, 16, 32]

    shape_id = (control_variation // 24) % len(shapes)

    bg_per_id = (control_variation // 6) % len(background_permittivities)

    measurements_id = (control_variation // 2) % len(number_values)
    sources_id = (control_variation) % len(number_values)

    shape = shapes[shape_id]
    background_permittivity = background_permittivities[bg_per_id]
    number_measurements = number_values[measurements_id] 
    number_source = number_values[sources_id] 

    params = {}

    params["shape"] = shape
    params["background_permittivity"] = background_permittivity
    params["number_measurements"] = number_measurements
    params["number_source"] = number_source

    params["noise_level"] = 1.

    return params