# eispy2d/data/__init__.py

"""
Data module for eispy2d.

Contains data container classes:
- InputData: Problem input data and configuration
- Result: Reconstruction results and error metrics
- TestSet: Test generation for benchmarking
"""

from eispy2d.data.inputdata import (
    InputData,
    PERMITTIVITY,
    CONDUCTIVITY,
    BOTH_PROPERTIES,
    CONTRAST,
    TOTAL_FIELD,
)
from eispy2d.data.result import (
    Result,
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
    INDICATOR_SET,
    LABELS,
    TITLES,
)
from eispy2d.data.testset import TestSet

__all__ = [
    'InputData',
    'Result',
    'TestSet',
    'PERMITTIVITY',
    'CONDUCTIVITY',
    'BOTH_PROPERTIES',
    'CONTRAST',
    'TOTAL_FIELD',
    'RESIDUAL_NORM_ERROR',
    'RESIDUAL_PAD_ERROR',
    'REL_PERMITTIVITY_PAD_ERROR',
    'REL_PERMITTIVITY_OBJECT_ERROR',
    'REL_PERMITTIVITY_BACKGROUND_ERROR',
    'CONDUCTIVITY_AD_ERROR',
    'CONDUCTIVITY_OBJECT_ERROR',
    'CONDUCTIVITY_BACKGROUND_ERROR',
    'TOTALFIELD_MAGNITUDE_PAD',
    'TOTALFIELD_PHASE_AD',
    'TOTAL_VARIATION',
    'SHAPE_ERROR',
    'POSITION_ERROR',
    'EXECUTION_TIME',
    'OBJECTIVE_FUNCTION',
    'NUMBER_EVALUATIONS',
    'NUMBER_ITERATIONS',
    'SSIM_ERROR',
    'PATH',
    'INDICATOR_SET',
    'LABELS',
    'TITLES',
]