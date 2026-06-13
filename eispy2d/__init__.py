# eispy2d/__init__.py

"""
EISPY2D: Open-Source Python Library for Electromagnetic Inverse Scattering

A comprehensive framework for developing and comparing algorithms for
two-dimensional electromagnetic inverse scattering problems.
"""

__version__ = "1.0.3"
__author__ = "Andre Costa Batista"

# ============================================================================
# Core
# ============================================================================
from eispy2d.core.configuration import Configuration
from eispy2d.core.error import (
    Error,
    MissingInputError,
    ExcessiveInputsError,
    MissingAttributesError,
    WrongTypeInput,
    WrongValueInput,
    EmptyAttribute,
)

# ============================================================================
# Data
# ============================================================================
from eispy2d.core.inputdata import InputData
from eispy2d.core.result import Result
from eispy2d.experiments.testset import TestSet

# ============================================================================
# Solvers - Base Classes
# ============================================================================
from eispy2d.solvers.base.forward import ForwardSolver
from eispy2d.solvers.base.inverse import InverseSolver
from eispy2d.solvers.base.deterministic import Deterministic
from eispy2d.solvers.base.stochastic import Stochastic, OutputMode

# ============================================================================
# Solvers - Forward Methods
# ============================================================================
from eispy2d.solvers.forward.mom_cg_fft import MoM_CG_FFT
from eispy2d.solvers.forward.analytical import Analytical
from eispy2d.solvers.forward.fftproduct import FFTProduct

# ============================================================================
# Solvers - Inverse Methods
# ============================================================================
from eispy2d.solvers.inverse.bim import BornIterativeMethod
from eispy2d.solvers.inverse.dbim import DistortedBornIterativeMethod
from eispy2d.solvers.inverse.csi import ContrastSourceInversion
from eispy2d.solvers.inverse.ecsi import ExtendedContrastSourceInversion
from eispy2d.solvers.inverse.mrcsi import MRContrastSourceInversion
from eispy2d.solvers.inverse.cgm import ConjugatedGradientMethod
from eispy2d.solvers.inverse.bornapprox import FirstOrderBornApproximation
from eispy2d.solvers.inverse.backprop import BackPropagation
from eispy2d.solvers.inverse.lsm import LinearSamplingMethod
from eispy2d.solvers.inverse.osm import OrthogonalitySamplingMethod
from eispy2d.solvers.inverse.som import SubspaceBasedOptimizationMethod
from eispy2d.solvers.inverse.music import MUSIC
from eispy2d.solvers.inverse.evolutionary import EvolutionaryAlgorithm

# ============================================================================
# Discretization
# ============================================================================
from eispy2d.discretization.discretization import Discretization
from eispy2d.discretization.collocation import Collocation
from eispy2d.discretization.richmond import Richmond

# ============================================================================
# Experiments
# ============================================================================
from eispy2d.experiments.experiment import Experiment
from eispy2d.experiments.casestudy import CaseStudy
from eispy2d.experiments.benchmark import Benchmark

# ============================================================================
# Utilities
# ============================================================================
from eispy2d.utils.draw import (
    square, circle, ellipse, triangle, polygon,
    star4, star5, star6, ring, cross, line,
    rhombus, trapezoid, parallelogram, random,
    wave, random_waves, random_gaussians,
)
from eispy2d.solvers.inverse.regularization import (
    Regularization, Tikhonov, Landweber, ConjugatedGradient,
    LeastSquares, SingularValueDecomposition,
    TIK_FIXED, TIK_MOZOROV, TIK_LCURVE,
)
from eispy2d.utils.stopcriteria import StopCriteria
from eispy2d.utils.statisticsutils import (
    compare1sample, compare2samples, compare_multiple,
    confint, confintplot, normalitiyplot, homoscedasticityplot,
    rcbd, factorial_analysis, dunnetttest,
)

# ============================================================================
# Result Indicators (Constants)
# ============================================================================
from eispy2d.core.result import (
    RESIDUAL_NORM_ERROR,
    RESIDUAL_PAD_ERROR,
    REL_PERMITTIVITY_PAD_ERROR,
    REL_PERMITTIVITY_OBJECT_ERROR,
    REL_PERMITTIVITY_BACKGROUND_ERROR,
    CONDUCTIVITY_AD_ERROR,
    CONDUCTIVITY_OBJECT_ERROR,
    CONDUCTIVITY_BACKGROUND_ERROR,
    TOTALFIELD_MAGNITUDE_PAD,
    TOTALFIELD_PHASE_AD,
    TOTAL_VARIATION,
    SHAPE_ERROR,
    POSITION_ERROR,
    EXECUTION_TIME,
    OBJECTIVE_FUNCTION,
    NUMBER_EVALUATIONS,
    NUMBER_ITERATIONS,
    SSIM_ERROR,
    PATH,
)

# ============================================================================
# Evolutionary Algorithm Components
# ============================================================================
from eispy2d.evoalglib.representation import (
    Representation, DiscretizationElementBased, CanonicalProblems,
)
from eispy2d.evoalglib.objectivefunction import (
    ObjectiveFunction, WeightedSum, Rastrigin, Rosenbrock, Ackley,
)
from eispy2d.evoalglib.initialization import (
    Initialization, UniformRandomDistribution, BornApproximation,
)
from eispy2d.evoalglib.selection import Selection, BinaryTournament, Roullete
from eispy2d.evoalglib.crossover import Crossover, Binomial, SimulatedBinary
from eispy2d.evoalglib.mutation import Mutation, Polynomial, Gaussian
from eispy2d.evoalglib.boundary import BoundaryCondition, Reflection
from eispy2d.evoalglib.de import DifferentialEvolution
from eispy2d.evoalglib.pso import ParticleSwarmOptimization
from eispy2d.evoalglib.ga import GeneticAlgorithm

__all__ = [
    # Version
    '__version__', '__author__',
    
    # Core
    'Configuration', 'Error', 'MissingInputError', 'ExcessiveInputsError',
    'MissingAttributesError', 'WrongTypeInput', 'WrongValueInput', 'EmptyAttribute',
    
    # Data
    'InputData', 'Result', 'TestSet',
    
    # Result indicators
    'RESIDUAL_NORM_ERROR', 'RESIDUAL_PAD_ERROR', 'REL_PERMITTIVITY_PAD_ERROR',
    'REL_PERMITTIVITY_OBJECT_ERROR', 'REL_PERMITTIVITY_BACKGROUND_ERROR',
    'CONDUCTIVITY_AD_ERROR', 'CONDUCTIVITY_OBJECT_ERROR', 'CONDUCTIVITY_BACKGROUND_ERROR',
    'TOTALFIELD_MAGNITUDE_PAD', 'TOTALFIELD_PHASE_AD', 'TOTAL_VARIATION',
    'SHAPE_ERROR', 'POSITION_ERROR', 'EXECUTION_TIME', 'OBJECTIVE_FUNCTION',
    'NUMBER_EVALUATIONS', 'NUMBER_ITERATIONS', 'SSIM_ERROR', 'PATH',
    
    # Solvers base
    'ForwardSolver', 'InverseSolver', 'Deterministic', 'Stochastic', 'OutputMode',
    
    # Forward solvers
    'MoM_CG_FFT', 'Analytical', 'FFTProduct',
    
    # Inverse solvers
    'BornIterativeMethod', 'DistortedBornIterativeMethod', 'ContrastSourceInversion',
    'ExtendedContrastSourceInversion', 'MRContrastSourceInversion', 'ConjugatedGradientMethod',
    'FirstOrderBornApproximation', 'BackPropagation', 'LinearSamplingMethod',
    'OrthogonalitySamplingMethod', 'SubspaceBasedOptimizationMethod', 'MUSIC',
    'EvolutionaryAlgorithm',
    
    # Discretization
    'Discretization', 'Collocation', 'Richmond',
    
    # Experiments
    'Experiment', 'CaseStudy', 'Benchmark',
    
    # Drawing
    'square', 'circle', 'ellipse', 'triangle', 'polygon', 'star4', 'star5', 'star6',
    'ring', 'cross', 'line', 'rhombus', 'trapezoid', 'parallelogram', 'random',
    'wave', 'random_waves', 'random_gaussians',
    
    # Regularization
    'Regularization', 'Tikhonov', 'Landweber', 'ConjugatedGradient',
    'LeastSquares', 'SingularValueDecomposition', 'TIK_FIXED', 'TIK_MOZOROV', 'TIK_LCURVE',
    
    # Other utils
    'StopCriteria', 'compare1sample', 'compare2samples', 'compare_multiple',
    'confint', 'confintplot', 'normalitiyplot', 'homoscedasticityplot',
    'rcbd', 'factorial_analysis', 'dunnetttest',
    
    # Evoalglib
    'Representation', 'DiscretizationElementBased', 'CanonicalProblems',
    'ObjectiveFunction', 'WeightedSum', 'Rastrigin', 'Rosenbrock', 'Ackley',
    'Initialization', 'UniformRandomDistribution', 'BornApproximation',
    'Selection', 'BinaryTournament', 'Roullete',
    'Crossover', 'Binomial', 'Exponential', 'SimulatedBinary', 'Uniform',
    'Mutation', 'Polynomial', 'UniformMutation', 'Gaussian',
    'BoundaryCondition', 'Reflection', 'Clamping', 'Projection',
    'DifferentialEvolution', 'ParticleSwarmOptimization', 'GeneticAlgorithm',
]