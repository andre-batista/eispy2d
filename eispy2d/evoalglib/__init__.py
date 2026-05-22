# eispy2d/evoalglib/__init__.py

"""
Evolutionary Algorithms Library for eispy2d.

Comprehensive framework for evolutionary computation applied to
electromagnetic inverse scattering problems.

Components:
- representation: Solution encoding schemes
- objectivefunction: Fitness evaluation functions
- initialization: Population initialization strategies
- selection: Parent selection operators
- crossover: Recombination operators
- mutation: Variation operators
- boundary: Constraint handling methods
- de: Differential Evolution
- pso: Particle Swarm Optimization
- ga: Genetic Algorithm
"""

from eispy2d.evoalglib.representation import (
    Representation,
    DiscretizationElementBased,
    CanonicalProblems,
)

from eispy2d.evoalglib.objectivefunction import (
    ObjectiveFunction,
    WeightedSum,
    Rastrigin,
    Rosenbrock,
    Ackley,
)

from eispy2d.evoalglib.initialization import (
    Initialization,
    UniformRandomDistribution,
    BornApproximation,
)

from eispy2d.evoalglib.selection import (
    Selection,
    BinaryTournament,
    Roullete,
)

from eispy2d.evoalglib.crossover import (
    Crossover,
    Binomial,
    SimulatedBinary,
)

from eispy2d.evoalglib.mutation import (
    Mutation,
    Polynomial,
    Gaussian,
)

from eispy2d.evoalglib.boundary import (
    Reflection,
)

from eispy2d.evoalglib.de import DifferentialEvolution
from eispy2d.evoalglib.pso import ParticleSwarmOptimization
from eispy2d.evoalglib.ga import GeneticAlgorithm

__all__ = [
    # Representation
    'Representation',
    'DiscretizationElementBased',
    'CanonicalProblems',
    
    # Objective functions
    'ObjectiveFunction',
    'WeightedSum',
    'Rastringin',
    'Rosenbrock',
    'Ackley',
    
    # Initialization
    'Initialization',
    'UniformRandomDistribution',
    'BornApproximation',
    
    # Selection
    'Selection',
    'BinaryTournament',
    'RouletteWheel',
    
    # Crossover
    'Crossover',
    'Binomial',
    'Exponential',
    'SimulatedBinary',
    'Uniform',
    
    # Mutation
    'Mutation',
    'Polynomial',
    'UniformMutation',
    'Gaussian',
    
    # Boundary handling
    'Boundary',
    'Reflection',
    'Clamping',
    'Projection',
    
    # Algorithms
    'DifferentialEvolution',
    'ParticleSwarmOptimization',
    'GeneticAlgorithm',
]