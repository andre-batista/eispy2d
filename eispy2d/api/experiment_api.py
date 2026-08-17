from abc import ABC, abstractmethod

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
                    RESULTS : self.results
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

    def _print_compare1sample(self):
        pass

    def _print_compare2sample(self):
        pass

    def _print_compare_multiple(self):
        pass
    
    def __str__(self):
        message = 'Name: ' + self.name + '\n'


        return message
