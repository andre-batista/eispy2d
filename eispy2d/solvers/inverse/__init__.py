# eispy2d/solvers/inverse/__init__.py

"""
Inverse solvers for electromagnetic inverse scattering problems.

Comprehensive collection of inverse scattering algorithms:
- Deterministic methods: BIM, DBIM, CSI, ECSI, MRCSI, CGM, Born, Backprop
- Qualitative methods: LSM, OSM, SOM, MUSIC
- Stochastic methods: Evolutionary algorithms
"""

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