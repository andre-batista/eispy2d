""" Benchmark - Usage example

This script implements a benchmark experiment. It defines the problem domain,
source parameters, test set configuration, and methods.

At the end, the algorithms are run and the results are save for a-posteriori
analysis.
"""

# Import modules
import bim
import benchmark as bmk
import bornapprox as ba
import regularization as reg
import richmond as ric
import configuration as cfg
import testset as tst
import result as rst
import experiment as exp
import mom_cg_fft as mom
import stopcriteria as stp
import evolutionary as evo
import stochastic as stc
from evoalglib import initialization as ini
from evoalglib import representation as rpt
from evoalglib import objectivefunction as obj
from evoalglib import de
from evoalglib import boundary as bc
from evoalglib import selection as slc
from evoalglib import crossover as cross

# Problem configuration
lambda_b = 1. # wavelength [m]
Lx = Ly = 4. # D domain size [wavelengths]
Ro = 6. # observation radius [wavelengths]
NS = NM = 15 # number of sources and measurements
E0 = 1. # incident field magnitude
epsilon_rb = 1. # background relative permittivity
contrast_level = .25
maximum_radius = 1. # [wavelengths]
resolution = (100, 100) # [pixels]
map_pattern = exp.RANDOM_POLYGONS_PATTERN
number_tests = 30
noise_level = 1. # [%/sample]
indicators = [rst.REL_PERMITTIVITY_PAD_ERROR]
contrast_mode = exp.FIXED_CONTRAST
density_mode = exp.SINGLE_OBJECT

# Configuration of stochastic algorithms
population_size = 250
variables_per_dimension = 7
contrast_max = 1.
total_max = 5.

# Build configuration object
config = cfg.Configuration(name='cfg_test',
                           wavelength=lambda_b,
                           wavelength_unit=True,
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
                        contrast=.25,
                        object_size=1.,
                        resolution=(100, 100),
                        density=None,
                        map_pattern=exp.RANDOM_POLYGONS_PATTERN,
                        sample_size=30,
                        noise=1.,
                        indicators=rst.REL_PERMITTIVITY_PAD_ERROR,
                        contrast_mode=exp.FIXED_CONTRAST,
                        object_size_mode=exp.FIXED_SIZE,
                        density_mode=exp.SINGLE_OBJECT,
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
                                         ric.Richmond(config, (7, 7)), 1., 5.
                                     ),
                                     de.DifferentialEvolution(
                                         bc.Reflection(),
                                         slc.BinaryTournament(),
                                         de.RAND, .5, cross.Binomial(.5)
                                     ),
                                     stp.StopCriteria(max_iterations=1000),
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