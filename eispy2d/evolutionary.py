import sys
import time as tm
import numpy as np
import multiprocessing
from numpy import pi
from joblib import Parallel, delayed

import result as rst
import inputdata as ipt
import stochastic as stc
import configuration as cfg
import evoalglib.representation as rpt


class EvolutionaryAlgorithm(stc.Stochastic):
    def __init__(self, population_size, initialization, objective_function,
                 representation, mechanism, stop_criteria, outputmode,
                 alias='ea', parallelization=False, number_executions=1,
                forward_solver=None, import_filename=None, import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(outputmode, alias=alias,
                             parallelization=parallelization,
                             number_executions=number_executions)
            self.name = 'Evolutionary Algorithm'
            self.population_size = population_size
            self.initiallization = initialization
            self.objective_function = objective_function
            self.representation = representation
            self.mechanism = mechanism
            self.forward = forward_solver
            self.stop_criteria = stop_criteria

    def solve(self, inputdata, discretization=None, print_info=True,
              print_file=sys.stdout):
        if isinstance(self.representation, rpt.CanonicalProblems):
            CANONICAL = True
        else:
            CANONICAL = False

        if not CANONICAL:
            result = super().solve(inputdata, self.representation.discretization,
                                   print_info=print_info,
                                   print_file=print_file)
            self.forward.configuration = inputdata.configuration
            ei = self.forward.incident_field(self.representation.discretization.elements,
                                             inputdata.configuration)
            self.objective_function.set_parameters(self.representation,
                                                   inputdata.scattered_field, ei)
        else:
            inputdata = ipt.InputData(
                name=(self.objective_function.name
                      + '_n%d' % self.representation.nvar),
                configuration=cfg.Configuration(name='void',
                                                wavelength=1.),
                resolution=(1, 1),
                indicators=rst.OBJECTIVE_FUNCTION
            )
            result = super().solve(inputdata, None, print_info, print_file)
            ei = None
            self.objective_function.set_parameters(self.representation,
                                                   None, None)

        run_names = [result.name + '_exec%d' % ne for ne in range(self.nexec)]

        if self.parallelization:
            if print_info:
                print('Running executions in parallel...', file=print_file)

            num_cores = multiprocessing.cpu_count()
            output = (Parallel(n_jobs=num_cores))(delayed(self.run_algorithm)
                                                  (inputdata, run_names[ne],
                                                   ei, False, None)
                                                  for ne in range(self.nexec))
        else:
            output = []
            for ne in range(self.nexec):
                output.append(self.run_algorithm(inputdata, run_names[ne], ei,
                                                 print_info, print_file))

        result = self.outputmode.make(inputdata.name + '_' + self.alias,
                                      self.alias, output)
        return result
        
    def run_algorithm(self, inputdata, run_name, incident_field,
                      print_info=True, print_file=sys.stdout):
        result = rst.Result(run_name,
                            method_name=self.alias,
                            configuration=inputdata.configuration)
        execution_time = .0
        current_evaluations = 0
        iteration = 0
        NPOP = self.population_size

        stop_criteria = self.stop_criteria.copy()
        mechanism = self.mechanism.copy()

        mechanism.reset_variables(NPOP, self.representation)
        stop_criteria.reset_memory()

        tic = tm.time()
        population = self.initiallization.run(NPOP, self.representation,
                                              incident_field, inputdata)
        mechanism.bc.run(population)
        population_fitness = np.zeros(NPOP)
        for p in range(NPOP):
            population_fitness[p] = self.objective_function.eval(
                population[p, :]
            )
            current_evaluations += 1
        execution_time += tm.time()-tic
        best = np.argmin(population_fitness)
        xopt = population[best, :]
        fopt = population_fitness[best]
        self._update_results(inputdata, result, xopt, fopt)

        if print_info:
            message = 'Generation 0 - fx: %.3e, ' % fopt
            message += 'evaluations: %d' % current_evaluations
            print(message, file=print_file)
            base, power = 1, 1

        while not stop_criteria.stop(current_evaluations, iteration, fopt):
            tic = tm.time()
            iteration += 1
            population, population_fitness, current_evaluations = (
                mechanism.run(population, population_fitness,
                              self.objective_function, current_evaluations)
            )
            execution_time += tm.time()-tic
            xopt, fopt = mechanism.best()
            self._update_results(inputdata, result, xopt, fopt)
            if print_info:
                if iteration >= base*10**power:
                    if base == 9:
                        base = 1
                        power += 1
                    else:
                        base += 1
                    message = 'Generation %d - ' % iteration
                    message += 'fx: %.3e, ' % fopt
                    message += 'evaluations: %d' % current_evaluations
                    print(message, file=print_file)

        if print_info and iteration != base*10**power:
            message = 'Generation %d - ' % iteration
            message += 'fx: %.3e, ' % fopt
            message += 'evaluations: %d' % current_evaluations
            print(message, file=print_file)

        if rst.EXECUTION_TIME in inputdata.indicators:
            result.execution_time = float(execution_time)
        if rst.NUMBER_EVALUATIONS in inputdata.indicators:
            result.number_evaluations = int(current_evaluations)
        if rst.NUMBER_ITERATIONS in inputdata.indicators:
            result.number_iterations = int(iteration)
        return result

    def save(self, file_path):
        super().save(file_path=file_path)
            
    def _update_results(self, inputdata, result, xopt, fopt):
        if not isinstance(self.representation, rpt.CanonicalProblems):
            contrast = self.representation.contrast(xopt,
                                                    mode=inputdata.resolution)
            total_field = self.representation.total_field(xopt,
                                                    mode=inputdata.resolution)
            scattered_field = self.representation.scattered_field(xopt)
            if not inputdata.configuration.good_conductor:
                result.rel_permittivity = cfg.get_relative_permittivity(
                    contrast, inputdata.configuration.epsilon_rb
                )
            if not inputdata.configuration.perfect_dielectric:
                result.conductivity = cfg.get_conductivity(
                    contrast, 2*pi*inputdata.configuration.f,
                    inputdata.configuration.epsilon_rb, inputdata.sigma_b
                )
            result.total_field = total_field
            result.scattered_field = scattered_field
        else:
            contrast, total_field, scattered_field = None, None, None
            result.rel_permittivity = self.representation.contrast(xopt)
            if hasattr(self.objective_function, 'xopt'):
                x = self.representation.contrast(xopt)
                result.zeta_p.append(
                    float(np.sqrt(np.sum((x
                                          -self.objective_function.xopt)**2)))
                )
        result.update_error(inputdata, scattered_field=scattered_field,
                            total_field=total_field, contrast=contrast,
                            objective_function=fopt)

    def copy(self, new=None):
        if new is None:
            return EvolutionaryAlgorithm(self.population_size,
                                         self.initiallization,
                                         self.objective_function,
                                         self.representation,
                                         self.mechanism,
                                         self.stop_criteria,
                                         self.outputmode,
                                         alias=self.alias,
                                         parallelization=self.parallelization,
                                         number_executions=self.nexec,
                                         forward_solver=self.forward)
        else:
            super().copy(new=new)
            self.population_size = new.population_size
            self.initiallization = new.initiallization
            self.objective_function = new.objective_function
            self.representation = new.representation
            self.mechanism = new.mechanism
            self.stop_criteria = new.stop_criteria
            self.forward = new.forward

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        print('Population size: %d' % self.population_size, file=print_file)
        print(self.initiallization, file=print_file)
        print(self.objective_function, file=print_file)
        print(self.representation, file=print_file)
        print(self.mechanism, file=print_file)
        print(self.stop_criteria, file=print_file)
        print(self.forward, file=print_file)

    def __str__(self) -> str:
        message = super().__str__()
        message += 'Population size: %d\n' % self.population_size
        message += str(self.initiallization)
        message += str(self.objective_function)
        message += str(self.representation)
        message += str(self.mechanism)
        message += str(self.stop_criteria)
        message += str(self.forward)
        return message