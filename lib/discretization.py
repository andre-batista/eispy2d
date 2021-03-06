import error
import configuration as cfg
from abc import ABC, abstractmethod

NAME = 'name'
ALIAS = 'alias'
CONFIGURATION = 'configuration'

class Discretization(ABC):
    def __init__(self, configuration=None, name=None, alias='',
                 import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            if configuration is not None:
                self.configuration = configuration.copy()
            else:
                self.configuration = None
            self.name = name
            self.alias = alias
    @abstractmethod
    def residual_data(self, scattered_field, contrast=None, total_field=None,
                      current=None):
        if contrast is not None and total_field is None:
            raise error.MissingInputError('Discretization.residual_data',
                                          'total_field')
        elif contrast is None and total_field is not None:
            raise error.MissingInputError('Discretization.residual_data',
                                          'contrast')
        elif contrast is None and total_field is None and current is None:
            raise error.Error('Discretization.residual_data: either '
                              + 'contrast-total_field or current must be given!')
    @abstractmethod   
    def residual_state(self, incident_field, contrast=None, total_field=None,
                       current=None):
        if total_field is not None and contrast is None:
            raise error.MissingInputError('Discretization.residual_state',
                                          'contrast')
        elif current is not None and contrast is None:
            raise error.MissingInputError('Discretization.residual_state',
                                          'contrast')
        elif contrast is None and total_field is None and current is None:
            raise error.Error('Discretization.residual_state: either '
                              + 'contrast-total_field or contrast-current must be'
                              + ' given!')
    @abstractmethod
    def solve(self, scattered_field=None, incident_field=None, contrast=None,
              total_field=None, current=None, linear_solver=None):
        pass
    @abstractmethod
    def scattered_field(self, contrast=None, total_field=None, current=None):
        if contrast is not None and total_field is None:
            raise error.MissingInputError('Discretization.scattered_field',
                                          'total_field')
        elif total_field is not None and contrast is None:
            raise error.MissingInputError('Discretization.scattered_field',
                                          'contrast')
        elif total_field is None and contrast is None and current is None:
            raise error.MissingInputError('Discretization.scattered_field',
                                          'contrast')
    @abstractmethod
    def contrast_image(self, coefficients, resolution):
        pass
    @abstractmethod
    def total_image(self, coefficients, resolution):
        pass
    def copy(self, new=None):
        if new is None:
            return Discretization(self.configuration, self.name)
        else:
            self.name = new.name
            self.configuration = new.configuration
    @abstractmethod
    def save(self, file_path=''):
        return {NAME: self.name,
                ALIAS: self.alias,
                CONFIGURATION: self.configuration}
    @abstractmethod
    def importdata(self, file_name, file_path=''):
        data = cfg.import_dict(file_name, file_path)
        self.name = data[NAME]
        self.alias = data[ALIAS]
        self.configuration = data[CONFIGURATION]
        return data
    @abstractmethod
    def __str__(self):
        return "Discretization: "
