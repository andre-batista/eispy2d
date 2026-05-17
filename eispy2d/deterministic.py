r"""Deterministic Inverse Solver Base Class.

This module provides the base class for deterministic inverse scattering methods
in the eispy2d library. The Deterministic class serves as an abstract base class
that defines the common interface and behavior for all deterministic inverse
solvers used in electromagnetic inverse scattering problems.

Deterministic methods are characterized by their reproducible, non-random
approach to solving inverse problems. They typically employ iterative algorithms
that converge to a solution through mathematical optimization techniques,
gradient-based methods, or other systematic approaches.

Classes
-------
Deterministic
    Base class for all deterministic inverse scattering solvers

Notes
-----
This class is designed to be inherited by specific deterministic algorithms
such as:
- Born Iterative Method (BIM)
- Distorted Born Iterative Method (DBIM)
- Contrast Source Inversion (CSI)
- Conjugate Gradient Method (CGM)
- Other deterministic reconstruction methods

The class provides a consistent interface for solver initialization,
execution, data persistence, and copying, ensuring uniform behavior
across all deterministic solvers in the library.

Examples
--------
This class is not intended to be used directly, but rather as a base
for specific implementations:

>>> # Example of a derived class
>>> class MyDeterministicSolver(Deterministic):
...     def __init__(self, **kwargs):
...         super().__init__(alias='my_solver')
...         # Additional initialization
...     
...     def solve(self, inputdata, discretization, **kwargs):
...         # Implement specific algorithm
...         result = super().solve(inputdata, discretization, **kwargs)
...         # Algorithm-specific processing
...         return result
"""

import sys

import eispy2d.inverse as inv

class Deterministic(inv.InverseSolver):
    """Base class for deterministic inverse scattering solvers.

    This class provides the foundation for all deterministic inverse scattering
    methods in the eispy2d library. It inherits from InverseSolver and
    implements the common interface required by all deterministic algorithms.

    Deterministic solvers are characterized by their reproducible behavior -
    given the same input data and parameters, they will always produce the
    same output. This is in contrast to stochastic methods that may produce
    different results on different runs due to random components.

    The class serves as an abstract base that defines the standard workflow
    for deterministic inverse solvers:
    1. Initialize solver with configuration parameters
    2. Solve the inverse problem using deterministic algorithm
    3. Save/load solver state for persistence
    4. Copy solver instances for parallel or comparative studies

    Parameters
    ----------
    alias : str, default=''
        Unique identifier for the solver instance. Used for file naming
        when saving/loading solver state and for identification in
        multi-solver comparisons.
    parallelization : bool, default=False
        Whether to enable parallel processing capabilities. When True,
        the solver may utilize multiple CPU cores or parallel algorithms
        where available.

    Attributes
    ----------
    alias : str
        Solver identifier string
    parallelization : bool
        Parallelization flag
    name : str
        Human-readable name of the solver (inherited from InverseSolver)

    Methods
    -------
    solve(inputdata, discretization, print_info=True, print_file=sys.stdout)
        Solve the inverse scattering problem
    save(file_path='')
        Save solver state to file
    importdata(file_name, file_path='')
        Load solver state from file
    copy(new=None)
        Create copy of solver instance

    Notes
    -----
    This class is designed to be inherited by specific deterministic
    algorithms. The base implementation provides standard functionality
    while derived classes implement algorithm-specific behavior.

    All methods in this class call the parent InverseSolver methods,
    ensuring consistent behavior across the solver hierarchy. Derived
    classes should override these methods to add algorithm-specific
    functionality while maintaining the standard interface.

    Examples
    --------
    This class is not used directly, but serves as base for specific solvers:

    >>> # Example derived class structure
    >>> class MyDeterministicMethod(Deterministic):
    ...     def __init__(self, my_param, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.my_param = my_param
    ...         self.name = 'My Deterministic Method'
    ...     
    ...     def solve(self, inputdata, discretization, **kwargs):
    ...         result = super().solve(inputdata, discretization, **kwargs)
    ...         # Add algorithm-specific processing
    ...         return result

    See Also
    --------
    inverse.InverseSolver : Parent class providing basic solver interface
    """
    def __init__(self, alias='', parallelization=False):
        """Initialize the deterministic inverse solver.

        Parameters
        ----------
        alias : str, default=''
            Unique identifier for the solver instance. This string is used
            for file naming when saving/loading solver state and for
            identification in multi-solver studies. Should be descriptive
            and unique within the application context.
        parallelization : bool, default=False
            Flag to enable parallel processing capabilities. When True,
            the solver may utilize multiple CPU cores or parallel algorithms
            where available in the specific implementation.

        Notes
        -----
        This method calls the parent InverseSolver initialization to
        establish the basic solver framework. Derived classes should
        call this method via super() and then add their specific
        initialization requirements.

        The parallelization flag is stored for use by derived classes
        but does not activate any parallel processing by itself. Each
        specific algorithm implementation decides how to utilize this flag.

        Examples
        --------
        >>> # Basic initialization
        >>> solver = Deterministic(alias='test_solver')
        >>> print(solver.alias)
        'test_solver'

        >>> # With parallelization enabled
        >>> solver = Deterministic(alias='parallel_solver', parallelization=True)
        >>> print(solver.parallelization)
        True
        """
        super().__init__(alias=alias, parallelization=parallelization)
    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        """Solve the electromagnetic inverse scattering problem.

        This method provides the standard interface for solving inverse
        scattering problems using deterministic methods. The base implementation
        calls the parent InverseSolver method to handle common setup and
        validation tasks.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing:
            - scattered_field: Measured scattered field data
            - configuration: Problem configuration (frequency, background, etc.)
            - resolution: Desired reconstruction resolution
            - indicators: Performance metrics to compute
        discretization : Discretization
            Discretization object containing:
            - elements: Spatial discretization grid
            - GS: Scattered field Green's function matrix
            - GD: Domain Green's function matrix (if needed)
            - Methods for field interpolation and imaging
        print_info : bool, default=True
            Whether to print algorithm progress and iteration information
            during the solving process. Useful for monitoring convergence
            and debugging.
        print_file : file-like object, default=sys.stdout
            Output stream for printing information. Can be redirected to
            files or other output streams as needed.

        Returns
        -------
        Result
            Result object containing reconstruction results and performance
            metrics. The specific contents depend on the algorithm
            implementation but typically include:
            - Reconstructed electromagnetic properties
            - Convergence history
            - Performance metrics
            - Algorithm-specific information

        Notes
        -----
        This base implementation performs common setup tasks and validation
        that are required by all deterministic solvers. Derived classes
        should override this method to implement their specific algorithm
        while calling the parent method for initialization:

        ```python
        def solve(self, inputdata, discretization, **kwargs):
            result = super().solve(inputdata, discretization, **kwargs)
            # Algorithm-specific implementation
            return result
        ```

        The method ensures consistent behavior across all deterministic
        solvers in terms of input validation, result formatting, and
        error handling.

        Examples
        --------
        >>> # Basic usage (in derived class)
        >>> result = solver.solve(input_data, discretization)
        >>> print(f"Reconstruction completed: {result.success}")

        >>> # With custom output stream
        >>> with open('solver_log.txt', 'w') as f:
        ...     result = solver.solve(input_data, discretization, print_file=f)

        >>> # Silent execution
        >>> result = solver.solve(input_data, discretization, print_info=False)
        """
        return super().solve(inputdata, discretization, print_info=print_info,
                             print_file=print_file)
    def save(self, file_path=''):
        """Save solver state to file for later restoration.

        This method serializes the current solver state including all
        configuration parameters, algorithm settings, and internal state
        to a file. The saved state can later be restored using the
        `importdata` method.

        Parameters
        ----------
        file_path : str, default=''
            Base path for saving the solver state. The actual filename
            will be constructed by appending the solver's alias to this
            base path. If empty, saves to current directory.

        Returns
        -------
        dict
            Dictionary containing the serialized solver state data.
            This includes all necessary information to fully restore
            the solver configuration and state.

        Notes
        -----
        The base implementation saves common solver attributes. Derived
        classes should override this method to include algorithm-specific
        state information:

        ```python
        def save(self, file_path=''):
            data = super().save(file_path=file_path)
            data['my_specific_param'] = self.my_specific_param
            # Save additional algorithm-specific data
            return data
        ```

        The method uses the solver's alias to construct the filename,
        ensuring each solver instance can be saved independently.

        Examples
        --------
        >>> # Save to current directory
        >>> data = solver.save()
        >>> print(f"Saved solver data with {len(data)} parameters")

        >>> # Save to specific directory
        >>> data = solver.save(file_path='/path/to/save/directory/')
        >>> print(f"Solver state saved to {file_path + solver.alias}")

        See Also
        --------
        importdata : Load solver state from file
        """
        return super().save(file_path=file_path)
    def importdata(self, file_name, file_path=''):
        """Load solver state from previously saved file.

        This method deserializes solver state from a file that was
        previously created using the `save` method. It restores all
        configuration parameters, algorithm settings, and internal state
        to recreate the exact solver configuration.

        Parameters
        ----------
        file_name : str
            Name of the file containing the saved solver state. This
            should match the filename created by the `save` method.
        file_path : str, default=''
            Directory path where the save file is located. If empty,
            looks in the current directory.

        Returns
        -------
        dict
            Dictionary containing the loaded solver state data. This
            provides access to all restored parameters and settings.

        Notes
        -----
        The base implementation loads common solver attributes. Derived
        classes should override this method to restore algorithm-specific
        state information:

        ```python
        def importdata(self, file_name, file_path=''):
            data = super().importdata(file_name, file_path=file_path)
            self.my_specific_param = data['my_specific_param']
            # Restore additional algorithm-specific data
            return data
        ```

        The method completely replaces the current solver state with
        the loaded configuration, so any existing settings will be lost.

        Examples
        --------
        >>> # Load from current directory
        >>> data = solver.importdata('solver_state.pkl')
        >>> print(f"Loaded {len(data)} parameters")

        >>> # Load from specific directory
        >>> data = solver.importdata('state.pkl', file_path='/path/to/files/')
        >>> print(f"Restored solver configuration from {file_path}")

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        pickle.UnpicklingError
            If the file is corrupted or incompatible

        See Also
        --------
        save : Save solver state to file
        """
        return super().importdata(file_name, file_path=file_path)
    def copy(self, new=None):
        """Create a copy of the solver instance.

        This method creates a deep copy of the current solver instance,
        including all configuration parameters and settings. The copy
        can be used independently without affecting the original solver.

        Parameters
        ----------
        new : Deterministic, optional
            If provided, the current solver's configuration will be
            copied into this existing instance. If None, a new instance
            will be created and returned.

        Returns
        -------
        Deterministic or None
            If `new` is None, returns a new Deterministic instance with
            the same configuration. If `new` is provided, returns None
            and the configuration is copied into the `new` instance.

        Notes
        -----
        This method creates independent copies that can be used for
        parallel processing, parameter studies, or comparative analysis
        without interference between instances.

        The base implementation copies common solver attributes. Derived
        classes should override this method to handle algorithm-specific
        parameters:

        ```python
        def copy(self, new=None):
            if new is None:
                return MyDeterministicSolver(
                    my_param=self.my_param,
                    alias=self.alias,
                    parallelization=self.parallelization
                )
            else:
                super().copy(new)
                new.my_param = self.my_param
        ```

        Examples
        --------
        >>> # Create independent copy
        >>> solver_copy = solver.copy()
        >>> print(f"Original: {solver.alias}, Copy: {solver_copy.alias}")

        >>> # Copy configuration into existing instance
        >>> target_solver = Deterministic()
        >>> solver.copy(target_solver)
        >>> print(f"Target now has alias: {target_solver.alias}")

        >>> # Use for parameter studies
        >>> solvers = [solver.copy() for _ in range(5)]
        >>> # Modify each copy independently for different parameters

        See Also
        --------
        __init__ : Initialize new solver instance
        """
        if new is None:
            return Deterministic(self.alias, self.parallelization)
        else:
            super().copy(new)
        
