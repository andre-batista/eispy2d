import pickle

from eispy2d.api import api, testset_api as ts
from eispy2d.core import error

from eispy2d.api import experiment_api as exp

TESTSET = "testset"

class Benchmark(exp.Experiment):

    @property
    def testset(self):
        return self._testset
    
    @testset.setter
    def testset(self, new):
        if new is None:
            self._testset = None
            self._single_testset = None
            self._testset_available = False
        elif type(new) is ts.TestSet:
            self._testset = new.copy()
            self._single_testset = True
            self._testset_available = True

    def __init__(self, name='', testset=None, import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)

        self.name = name
        self.testset = testset

    def run(self, parallelization=None):
        if not self._testset_available:
            raise error.MissingAttributesError('Benchmark', 'testset')

    def save(self, file_path='', save_testset=False):
        data = super().save(file_path)

        if save_testset:
            data[TESTSET] = self.testset
        elif self._testset_available and self._single_testset:
            data[TESTSET] = self.testset.name
        elif self._testset_available and not self._single_testset:
            data[TESTSET] = [self.testset[n].name
                                for n in range(len(self.testset))]
        else:
            data[TESTSET] = self.testset

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path)
        self.testset = data[TESTSET]