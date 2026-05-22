""" Benchmark - Usage example

This script implements a benchmark experiment. It defines the problem domain,
source parameters, test set configuration, and methods.

At the end, the algorithms are run and the results are save for a-posteriori
analysis.
"""

import sys
sys.path.insert(1, '../../eispy2d')

# Import modules
from eispy2d import bim
from eispy2d import benchmark as bmk
from eispy2d import bornapprox as ba
from eispy2d import regularization as reg
from eispy2d import richmond as ric
from eispy2d.core import configuration as cfg
from eispy2d import testset as tst
from eispy2d import result as rst
from eispy2d import experiment as exp
from eispy2d import mom_cg_fft as mom
from eispy2d import stopcriteria as stp
from eispy2d import evolutionary as evo
from eispy2d import stochastic as stc
from eispy2d.evoalglib import initialization as ini
from eispy2d.evoalglib import representation as rpt
from eispy2d.evoalglib import objectivefunction as obj
from eispy2d.evoalglib import de
from eispy2d.evoalglib import boundary as bc
from eispy2d.evoalglib import selection as slc
from eispy2d.evoalglib import crossover as cross

# Problem configuration
f0 = 3e8 # linear frequency [Hz]
Lx = Ly = .8 # D domain size [wavelengths]
Ro = 1. # observation radius [wavelengths]
NS, NM = 10, 9 # number of sources and measurements
E0 = 1. # incident field magnitude
epsilon_rb = 4. # background relative permittivity
contrast_level = 1.
maximum_radius = .32 # [wavelengths] = 0.16 [m]
resolution = (100, 100) # [pixels]
map_pattern = exp.RANDOM_POLYGONS_PATTERN
number_tests = 30
noise_level = 1. # [%/sample]
indicators = [rst.REL_PERMITTIVITY_PAD_ERROR,
              rst.REL_PERMITTIVITY_OBJECT_ERROR]
contrast_mode = exp.FIXED_CONTRAST
density_mode = exp.SINGLE_OBJECT

# Configuration of stochastic algorithms
population_size = 250
variables_per_dimension = 7
contrast_max = 1.
total_max = 5.
max_iterations = 10000

# Build configuration object
config = cfg.Configuration(name='cfg_test',
                           frequency=f0,
                           wavelength_unit=False,
                           number_measurements=NM,
                           number_sources=NS,
                           image_size=[Ly, Lx],
                           observation_radius=Ro,
                           background_permittivity=epsilon_rb,
                           magnitude=E0,
                           perfect_dielectric=True)

# Build test set object
mytestset = tst.TestSet(name='tst_basic',
                        configuration=config,
                        contrast=contrast_level,
                        object_size=maximum_radius,
                        resolution=resolution,
                        density=None,
                        map_pattern=map_pattern,
                        sample_size=number_tests,
                        noise=noise_level,
                        indicators=indicators,
                        contrast_mode=contrast_mode,
                        object_size_mode=exp.FIXED_SIZE,
                        density_mode=density_mode,
                        min_size_proportion=40,
                        allow_rotation=True,
                        random_position=True)

# Generate tests
print('Creating tests...')
mytestset.randomize_tests(parallelization=False)

# Synthesize scattered field data
print('Generating field data...')
mytestset.generate_field_data(solver=mom.MoM_CG_FFT())

# Define methods
methods = [ba.FirstOrderBornApproximation(reg.Tikhonov(1e-1),
                                          alias='ba'),
           bim.BornIterativeMethod(mom.MoM_CG_FFT(),
                                   reg.Tikhonov(1e-1),
                                   stp.StopCriteria(max_iterations=5),
                                   alias='bim'),
           evo.EvolutionaryAlgorithm(population_size,
                                     ini.UniformRandomDistribution(),
                                     obj.WeightedSum(),
                                     rpt.DiscretizationElementBased(
                                         ric.Richmond(config, (7, 7)),
                                         contrast_max, total_max
                                     ),
                                     de.DifferentialEvolution(
                                         bc.Reflection(),
                                         slc.BinaryTournament(),
                                         de.RAND, .5, cross.Binomial(.5)
                                     ),
                                     stp.StopCriteria(max_iterations=max_iterations),
                                     stc.OutputMode(
                                         stc.AVERAGE_CASE,
                                         rst.REL_PERMITTIVITY_PAD_ERROR,
                                         sample_rate=5.
                                     ),
                                     alias='de',
                                     parallelization=True,
                                     number_executions=30,
                                     forward_solver=mom.MoM_CG_FFT())]

# Define discretization
discretization = ric.Richmond(config, 30, state=False)

# Build benchmark object
mybenchmark = bmk.Benchmark('mybenchmark',
                            method=methods,
                            discretization=discretization,
                            testset=mytestset)

# Run benchmark experiment
print('Running benchmark...')
mybenchmark.run(parallelization=bmk.PARALLELIZE_TESTS)

# Save results
mybenchmark.save(save_testset=True)
