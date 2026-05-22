# eispy2d/experiments/__init__.py

"""
Experiments module for eispy2d.

Provides framework for systematic algorithm evaluation:
- base: Abstract base class for experiments
- casestudy: Single test case studies with analysis tools
- benchmark: Multi-test benchmarking with statistical comparison
"""

from eispy2d.experiments.experiment import Experiment
from eispy2d.experiments.casestudy import CaseStudy
from eispy2d.experiments.benchmark import Benchmark

__all__ = [
    'Experiment',
    'CaseStudy',
    'Benchmark',
]