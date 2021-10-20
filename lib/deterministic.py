import sys
import inverse as inv

class Deterministic(inv.InverseSolver):
    def __init__(self, alias='', parallelization=False):
        super().__init__(alias=alias, parallelization=parallelization)
    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        return super().solve(inputdata, discretization, print_info=print_info,
                             print_file=print_file)
    def save(self, file_path=''):
        return super().save(file_path=file_path)
    def importdata(self, file_name, file_path=''):
        return super().importdata(file_name, file_path=file_path)
    def copy(self, new=None):
        if new is None:
            return Deterministic(self.alias, self.parallelization)
        else:
            super().copy(new)
        
