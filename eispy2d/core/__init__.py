# eispy2d/core/__init__.py

"""
Core module for eispy2d.

Contains fundamental classes and utilities used throughout the library:
- Configuration: Problem domain configuration
- Error: Custom exception classes
"""

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

__all__ = [
    'Configuration',
    'Error',
    'MissingInputError',
    'ExcessiveInputsError',
    'MissingAttributesError',
    'WrongTypeInput',
    'WrongValueInput',
    'EmptyAttribute',
]