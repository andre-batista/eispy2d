# eispy2d/solvers/__init__.py

"""
Solvers module for eispy2d.

Contains forward and inverse solver implementations:
- base: Abstract base classes for solvers
- forward: Forward problem solvers (MoM-CG-FFT, Analytical)
- inverse: Inverse problem solvers (BIM, DBIM, CSI, etc.)
"""

# Base classes
from eispy2d.solvers.base.forward import ForwardSolver
from eispy2d.solvers.base.inverse import InverseSolver
from eispy2d.solvers.base.deterministic import Deterministic
from eispy2d.solvers.base.stochastic import Stochastic, OutputMode

# Forward solvers
from eispy2d.solvers.forward.mom_cg_fft import MoM_CG_FFT
from eispy2d.solvers.forward.analytical import Analytical
from eispy2d.solvers.forward.fftproduct import FFTProduct

# Inverse solvers
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

__all__ = [
    # Base classes
    'ForwardSolver',
    'InverseSolver',
    'Deterministic',
    'Stochastic',
    'OutputMode',
    
    # Forward solvers
    'MoM_CG_FFT',
    'Analytical',
    'FFTProduct',
    
    # Inverse solvers
    'BornIterativeMethod',
    'DistortedBornIterativeMethod',
    'ContrastSourceInversion',
    'ExtendedContrastSourceInversion',
    'MRContrastSourceInversion',
    'ConjugatedGradientMethod',
    'FirstOrderBornApproximation',
    'BackPropagation',
    'LinearSamplingMethod',
    'OrthogonalitySamplingMethod',
    'SubspaceBasedOptimizationMethod',
    'MUSIC',
    'EvolutionaryAlgorithm',
]