"""Case Study - Usage Example

This script describes a usage example of `CaseStudy` class. It is intended
to only define the case study and run the algorithms. The result analysis is
left for the Jupyter Notebook with the same name.

First, the domain and source parameters will be defined using `Configuration`
class. Then, the problem will be defined. Next, the inverse methods will be
specified. Finally, an object of `CaseStudy` will be built and the methods will
be executed.

"""

# Import modules
import numpy as np
from numpy import random as rnd
import casestudy as cst
import richmond as ric
import configuration as cfg
import inputdata as ipt
import result as rst
import mom_cg_fft as mom
import stochastic as stc
import evolutionary as evo
import stopcriteria as stp
import draw
from evoalglib import initialization as ini
from evoalglib import objectivefunction as obj
from evoalglib import representation as rpt
from evoalglib import boundary as bc
from evoalglib import selection as slc
from evoalglib import crossover as cross
from evoalglib import mutation as mut
from evoalglib import de
from evoalglib import pso
from evoalglib import ga

# Problem parameters
f0 = 3e8 # linear frequency [m]
Lx, Ly = .8, .8 # D domain size [m]
NS, NM = 10, 9 # number of sources and measurements
RO = 1. # observation radius [m]
epsilon_rb = 4. # background relative permittivity
E0 = 1 # incident wave magnitude [V/m]
resolution = (60, 60) # ground-truth image resolution [pixels]
noise_level = 1. # [%/sample]
indicators = [rst.REL_PERMITTIVITY_PAD_ERROR, rst.OBJECTIVE_FUNCTION]
contrast_level = 1.
object_size = .16 # [m]

# Algorithm parameters
population_size = 250
variables_per_dimension = 7
contrast_max = 1.
total_field_max = 5.
maximum_iterations = 5000

# Define domain and source parameters
config = cfg.Configuration(name='cfg_test',
                           frequency=f0,
                           wavelength_unit=False,
                           number_measurements=NM,
                           number_sources=NS,
                           image_size=[Ly, Lx],
                           observation_radius=RO,
                           background_permittivity=epsilon_rb,
                           magnitude=E0,
                           perfect_dielectric=True)

# Build test object
inputdata = ipt.InputData(name='ipt_test',
                          configuration=config,
                          resolution=resolution,
                          noise=noise_level,
                          indicators=indicators)

# Draw figure
inputdata.rel_permittivity, _ = draw.triangle(
    object_size*np.sqrt(3),
    center=[-.14, .09],
    axis_length_x=config.Lx,
    axis_length_y=config.Ly,
    resolution=resolution,
    background_rel_permittivity=epsilon_rb,
    object_rel_permittivity=(contrast_level+1)*epsilon_rb
)


# Build forward solver object
solver = mom.MoM_CG_FFT(tolerance=.001,
                        maximum_iterations=5000)

# Solve forward problem
_ = solver.solve(inputdata,
                 PRINT_INFO=True,
                 COMPUTE_SCATTERED_FIELD=True,
                 SAVE_INTERN_FIELD=True)

# Define initialization method
initialization = ini.BornApproximation()

# Define objective function
objfun = obj.WeightedSum()

# Define representation scheme
representation = rpt.DiscretizationElementBased(
    ric.Richmond(config, variables_per_dimension, state=True),
    contrast_max,
    total_field_max
)

# Define stop criterium
stopcriterium = stp.StopCriteria(max_iterations=maximum_iterations)

# Configure output of stochastic executions
outputmode = stc.OutputMode(stc.EACH_EXECUTION)

# Differential Evolution
DE = evo.EvolutionaryAlgorithm(population_size, initialization, objfun,
                               representation,
                               de.DifferentialEvolution(
                                   bc.Reflection(),
                                   slc.BinaryTournament(elitism=True),
                                   de.RAND, .5, # mutation factor
                                   cross.Binomial(.5), pcross=1
                               ),
                               stopcriterium, outputmode, alias='de',
                               parallelization=True,
                               forward_solver=mom.MoM_CG_FFT())

# Particle Swarm Optimization
PSO = evo.EvolutionaryAlgorithm(population_size, initialization, objfun,
                                representation,
                                pso.ParticleSwarmOptimization(bc.Reflection(),
                                                              acceleration=2.,
                                                              inertia=.4),
                                stopcriterium, outputmode, alias='pso',
                                parallelization=True,
                                forward_solver=mom.MoM_CG_FFT())

# Genetic Algorithm
GA = evo.EvolutionaryAlgorithm(population_size, initialization, objfun,
                               representation, 
                               ga.GeneticAlgorithm(bc.Reflection(),
                                                   cross.SimulatedBinary(5.),
                                                   1., # crossover probability
                                                   mut.Polynomial(5),
                                                   1., # mutation probability
                                                   slc.BinaryTournament()),
                               stopcriterium, outputmode, alias='ga',
                               parallelization=True,
                               forward_solver=mom.MoM_CG_FFT())

methods = [DE, PSO, GA]

# Case Study definition
mystudy = cst.CaseStudy('cst_basic',
                        method=methods,
                        discretization=ric.Richmond(config, 30, state=False),
                        test=inputdata,
                        stochastic_runs=30,
                        save_stochastic_runs=True)

# Run
mystudy.run(parallelization=cst.PARALLELIZE_EXECUTIONS)
mystudy.save(save_test=True)