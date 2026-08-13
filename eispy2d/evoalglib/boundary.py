import numpy as np
from abc import ABC, abstractmethod


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions in evolutionary algorithms.

    Boundary conditions handle how variables that go outside the allowed range
    are treated during evolution. This class defines the interface for all
    boundary handling strategies.

    Methods
    -------
    run(x)
        Apply boundary condition to the solution vector.
    """
    def __init__(self):
        pass
    @abstractmethod
    def run(self, x):
        """Apply boundary condition to the solution vector.

        Parameters
        ----------
        x : numpy.ndarray
            Solution vector (1D or 2D) to be processed.
        """
        pass
    @abstractmethod
    def __str__(self):
        return 'Boundary Condition: '


class Truncation(BoundaryCondition):
    """Truncation boundary condition.

    Clips values to the [0, 1] range. Values below 0 are set to 0,
    values above 1 are set to 1.

    Notes
    -----
    This is a simple boundary handling strategy that preserves the
    interval [0, 1] by clipping out-of-bounds values.
    """
    def __init__(self):
        super().__init__()
    def run(self, x):
        """Apply truncation to the solution vector.

        Parameters
        ----------
        x : numpy.ndarray
            Solution vector to be truncated.
        """
        super().run(x)
        x[x < 0] = 0
        x[x > 1] = 1
    def __str__(self):
        return super().__str__() + 'Truncation'


class Reflection(BoundaryCondition):
    """Reflection boundary condition.

    Reflects values that fall outside [0, 1] back into the range using
    a reflection strategy. Values are reflected multiple times if needed.

    Notes
    -----
    This boundary handling strategy preserves the interval [0, 1] by
    reflecting out-of-bounds values back into the valid range using
    a mirroring technique.
    """
    def __init__(self):
        super().__init__()
    def run(self, x):
        """Apply reflection to the solution vector.

        Parameters
        ----------
        x : numpy.ndarray
            Solution vector to be reflected.
        """
        super().run(x)
        i = np.logical_and(x < 0, np.mod(np.abs(x), 2) < 1)
        j = np.logical_and(x < 0, np.mod(np.abs(x), 2) >= 1)
        x[i] = np.ceil(x[i]) - x[i]
        x[j] = 1 - (np.ceil(x[j])-x[j])
        i = np.logical_and(x > 1, np.mod(x, 2) < 1)
        j = np.logical_and(x > 1, np.mod(x, 2) >= 1)
        x[i] = x[i] - np.floor(x[i])
        x[j] = 1 - (x[j]-np.floor(x[j]))
    def __str__(self):
        return super().__str__() + 'Reflection'