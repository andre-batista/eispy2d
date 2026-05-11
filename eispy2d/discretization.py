r"""Discretization Abstract Base Class for Electromagnetic Inverse Scattering.

This module provides the abstract base class for spatial discretization methods
used in electromagnetic inverse scattering problems. The Discretization class
defines the interface that all discretization schemes must implement to work
with the eispy2d library.

Discretization methods are responsible for:
- Converting continuous electromagnetic fields to discrete representations
- Defining spatial grids and basis functions
- Computing residuals for data and state equations
- Solving linear systems arising from discretization
- Converting between different field representations
- Interpolating solutions to different resolutions

The abstract nature of this class ensures that different discretization methods
(such as method of moments, finite differences, finite elements, etc.) can be
used interchangeably within the same inverse scattering algorithms.

Classes
-------
Discretization
    Abstract base class for all discretization methods

Constants
---------
NAME : str
    Dictionary key for discretization name in saved data
ALIAS : str
    Dictionary key for discretization alias in saved data  
CONFIGURATION : str
    Dictionary key for configuration object in saved data

Notes
-----
This class is designed to be inherited by specific discretization implementations
such as:
- Method of Moments (MoM)
- Finite Difference Methods
- Finite Element Methods
- Collocation Methods
- Other spatial discretization schemes

All methods are abstract and must be implemented by derived classes to provide
the specific mathematical operations required for each discretization approach.

Examples
--------
This class is not intended to be used directly, but rather as a base for
specific implementations:

>>> # Example of a derived class structure
>>> class MyDiscretization(Discretization):
...     def __init__(self, configuration, **kwargs):
...         super().__init__(configuration, name='My Discretization', **kwargs)
...         # Initialize discretization-specific parameters
...     
...     def solve(self, scattered_field, **kwargs):
...         # Implement specific discretization method
...         return solution
...     
...     # Implement other abstract methods...
"""

import error
import configuration as cfg
from abc import ABC, abstractmethod

NAME = 'name'
ALIAS = 'alias'
CONFIGURATION = 'configuration'

class Discretization(ABC):
    """Abstract base class for spatial discretization methods.

    This class defines the interface that all discretization methods must
    implement for electromagnetic inverse scattering problems. It provides
    the foundation for converting continuous field problems into discrete
    algebraic systems that can be solved numerically.

    The discretization handles the spatial aspect of the inverse scattering
    problem, defining how continuous electromagnetic fields are represented
    on discrete grids and how the governing equations are discretized.

    Key responsibilities include:
    - Spatial grid generation and management
    - Field representation using basis functions
    - Residual computation for data and state equations
    - Linear system solution for field reconstruction
    - Resolution conversion and field interpolation
    - Green's function matrix computation and storage

    Parameters
    ----------
    configuration : Configuration, optional
        Problem configuration object containing electromagnetic parameters,
        geometry definitions, and solver settings. If None, must be set
        before solving.
    name : str, optional
        Human-readable name for the discretization method
    alias : str, default=''
        Short identifier for the discretization, used in file naming
        and method identification
    import_filename : str, optional
        If provided, loads discretization state from this file
    import_filepath : str, default=''
        Directory path for import file

    Attributes
    ----------
    configuration : Configuration
        Problem configuration object
    name : str
        Discretization method name
    alias : str
        Method identifier string

    Methods
    -------
    residual_data(scattered_field, contrast=None, total_field=None, current=None)
        Compute residual for data equation
    residual_state(incident_field, contrast=None, total_field=None, current=None)
        Compute residual for state equation
    solve(scattered_field=None, incident_field=None, contrast=None, total_field=None, current=None, linear_solver=None)
        Solve linear system for field reconstruction
    scattered_field(contrast=None, total_field=None, current=None)
        Compute scattered field from given parameters
    contrast_image(coefficients, resolution)
        Convert coefficients to contrast image at given resolution
    total_image(coefficients, resolution)
        Convert coefficients to total field image at given resolution
    copy(new=None)
        Create copy of discretization instance
    save(file_path='')
        Save discretization state to file
    importdata(file_name, file_path='')
        Load discretization state from file

    Notes
    -----
    This is an abstract base class - all methods marked with @abstractmethod
    must be implemented by derived classes. The class provides parameter
    validation and error checking for the abstract methods.

    Different discretization methods will implement these abstract methods
    according to their specific mathematical formulations:
    - Method of Moments uses basis functions and testing functions
    - Finite Differences use grid-based approximations
    - Finite Elements use element-based basis functions
    - Collocation uses point-wise matching

    Examples
    --------
    This class cannot be instantiated directly. Use derived classes:

    >>> # Example derived class usage
    >>> from collocation import Collocation
    >>> discretization = Collocation(configuration, name='Pulse Collocation')
    >>> 
    >>> # Solve linear inverse problem
    >>> contrast = discretization.solve(
    ...     scattered_field=measured_data,
    ...     incident_field=incident_data,
    ...     linear_solver=regularization_method
    ... )
    >>>
    >>> # Convert to image format
    >>> image = discretization.contrast_image(contrast, resolution)

    See Also
    --------
    collocation.Collocation : Collocation method implementation
    """
    def __init__(self, configuration=None, name=None, alias='',
                 import_filename=None, import_filepath=''):
        """Initialize the discretization method.

        Parameters
        ----------
        configuration : Configuration, optional
            Problem configuration object containing electromagnetic parameters,
            geometry, frequency, and other problem-specific settings. If None,
            must be set before solving any problems.
        name : str, optional
            Human-readable name for the discretization method. Used for
            identification and display purposes.
        alias : str, default=''
            Short identifier string for the discretization method. Used in
            file naming and method identification within algorithms.
        import_filename : str, optional
            If provided, loads the discretization state from this file
            instead of initializing with the other parameters.
        import_filepath : str, default=''
            Directory path where the import file is located.

        Notes
        -----
        If `import_filename` is provided, the method loads its state from
        the specified file and ignores other initialization parameters.
        Otherwise, it initializes with the provided parameters.

        The configuration object is copied to avoid unintended modifications
        to the original configuration during discretization operations.

        Examples
        --------
        >>> # Initialize with configuration
        >>> config = Configuration(frequency=1e9, background_epsilon=1.0)
        >>> discretization = MyDiscretization(
        ...     configuration=config,
        ...     name='My Method',
        ...     alias='mymeth'
        ... )

        >>> # Initialize from saved state
        >>> discretization = MyDiscretization(
        ...     import_filename='saved_state.pkl',
        ...     import_filepath='/path/to/files/'
        ... )
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            if configuration is not None:
                self.configuration = configuration.copy()
            else:
                self.configuration = None
            self.name = name
            self.alias = alias
    @abstractmethod
    def residual_data(self, scattered_field, contrast=None, total_field=None,
                      current=None):
        """Compute residual for the data equation.

        This method computes the residual (error) between measured scattered
        field data and the scattered field predicted by the current estimate
        of the scatterer properties. The residual is used in optimization
        algorithms to minimize the data misfit.

        The data equation relates the scattered field to the scatterer
        properties and internal fields:
        .. math::
            E^s = G^s J

        where :math:`E^s` is the scattered field, :math:`G^s` is the scattered
        field Green's function, and :math:`J` is the contrast source.

        Parameters
        ----------
        scattered_field : array_like
            Measured scattered field data at receiver locations
        contrast : array_like, optional
            Contrast function values at discretization points. Required
            if `total_field` is provided.
        total_field : array_like, optional
            Total electric field at discretization points. Required
            if `contrast` is provided.
        current : array_like, optional
            Contrast source (current) at discretization points. Alternative
            to providing contrast and total_field separately.

        Returns
        -------
        array_like
            Data residual vector representing the difference between
            measured and computed scattered field

        Raises
        ------
        MissingInputError
            If required parameter combinations are not provided
        Error
            If no valid parameter combination is given

        Notes
        -----
        The method requires either:
        - Both `contrast` and `total_field` parameters, or
        - The `current` parameter alone

        The residual is computed as:
        residual = measured_scattered_field - predicted_scattered_field

        A small residual indicates good agreement between the model and data.

        Examples
        --------
        >>> # Using contrast and total field
        >>> residual = discretization.residual_data(
        ...     scattered_field=measured_data,
        ...     contrast=current_contrast,
        ...     total_field=current_total_field
        ... )

        >>> # Using current directly
        >>> residual = discretization.residual_data(
        ...     scattered_field=measured_data,
        ...     current=current_source
        ... )
        """
        if contrast is not None and total_field is None:
            raise error.MissingInputError('Discretization.residual_data',
                                          'total_field')
        elif contrast is None and total_field is not None:
            raise error.MissingInputError('Discretization.residual_data',
                                          'contrast')
        elif contrast is None and total_field is None and current is None:
            raise error.Error('Discretization.residual_data: either '
                              + 'contrast-total_field or current must be given!')
    @abstractmethod   
    def residual_state(self, incident_field, contrast=None, total_field=None,
                       current=None):
        """Compute residual for the state equation.

        This method computes the residual (error) in the state equation,
        which relates the total field, contrast function, and current source.
        The state equation ensures consistency between the electromagnetic
        field quantities within the scattering domain.

        The state equation can be written as:
        .. math::
            J = \\chi E^t

        where :math:`J` is the contrast source, :math:`\\chi` is the contrast
        function, and :math:`E^t` is the total electric field.

        Parameters
        ----------
        incident_field : array_like
            Incident electric field at discretization points
        contrast : array_like, optional
            Contrast function values at discretization points. Required
            if `total_field` or `current` is provided.
        total_field : array_like, optional
            Total electric field at discretization points. Used with
            `contrast` to compute current source.
        current : array_like, optional
            Contrast source (current) at discretization points. Used with
            `contrast` to validate state equation.

        Returns
        -------
        array_like
            State residual vector representing the inconsistency in the
            state equation

        Raises
        ------
        MissingInputError
            If required parameter combinations are not provided
        Error
            If no valid parameter combination is given

        Notes
        -----
        The method requires:
        - `contrast` parameter along with either `total_field` or `current`

        The residual measures how well the state equation is satisfied:
        - If using total_field: residual = contrast * total_field - current
        - If using current: residual = contrast * (incident + scattered) - current

        A small residual indicates good consistency between field quantities.

        Examples
        --------
        >>> # Using contrast and total field
        >>> residual = discretization.residual_state(
        ...     incident_field=incident_data,
        ...     contrast=current_contrast,
        ...     total_field=current_total_field
        ... )

        >>> # Using contrast and current
        >>> residual = discretization.residual_state(
        ...     incident_field=incident_data,
        ...     contrast=current_contrast,
        ...     current=current_source
        ... )
        """
        if total_field is not None and contrast is None:
            raise error.MissingInputError('Discretization.residual_state',
                                          'contrast')
        elif current is not None and contrast is None:
            raise error.MissingInputError('Discretization.residual_state',
                                          'contrast')
        elif contrast is None and total_field is None and current is None:
            raise error.Error('Discretization.residual_state: either '
                              + 'contrast-total_field or contrast-current must be'
                              + ' given!')
    @abstractmethod
    def solve(self, scattered_field=None, incident_field=None, contrast=None,
              total_field=None, current=None, linear_solver=None):
        """Solve the linear inverse scattering problem.

        This method solves the discretized linear inverse scattering problem
        to reconstruct the contrast function from measured scattered field data.
        The solution involves solving a linear system that may be ill-conditioned,
        requiring regularization techniques.

        The linear problem typically has the form:
        .. math::
            G \\chi = E^s

        where :math:`G` is the system matrix, :math:`\\chi` is the contrast
        function to be reconstructed, and :math:`E^s` is the scattered field data.

        Parameters
        ----------
        scattered_field : array_like, optional
            Measured scattered field data at receiver locations
        incident_field : array_like, optional
            Incident electric field at discretization points
        contrast : array_like, optional
            Initial or known contrast function values
        total_field : array_like, optional
            Total electric field at discretization points
        current : array_like, optional
            Contrast source (current) at discretization points
        linear_solver : LinearSolver, optional
            Regularization/linear solver method to use for solving the
            potentially ill-conditioned linear system

        Returns
        -------
        array_like
            Reconstructed contrast function coefficients at discretization points

        Notes
        -----
        The specific parameters required depend on the discretization method
        and the formulation of the linear system. Common approaches include:
        - Born approximation: uses incident field as total field
        - Moment method: requires Green's function matrices
        - Finite differences: uses grid-based operators

        The linear solver handles regularization to deal with ill-conditioning
        that is common in inverse scattering problems.

        Examples
        --------
        >>> # Basic Born approximation solution
        >>> contrast = discretization.solve(
        ...     scattered_field=measured_data,
        ...     incident_field=incident_data,
        ...     linear_solver=tikhonov_solver
        ... )

        >>> # Advanced solution with known total field
        >>> contrast = discretization.solve(
        ...     scattered_field=measured_data,
        ...     total_field=computed_total_field,
        ...     linear_solver=truncated_svd_solver
        ... )
        """
        pass
    @abstractmethod
    def scattered_field(self, contrast=None, total_field=None, current=None):
        """Compute scattered field from scatterer properties.

        This method computes the scattered electric field at receiver locations
        given the scatterer properties (contrast function, total field, or
        current source). This is the forward scattering operation.

        The scattered field is computed using:
        .. math::
            E^s = G^s J

        where :math:`G^s` is the scattered field Green's function matrix and
        :math:`J` is the contrast source.

        Parameters
        ----------
        contrast : array_like, optional
            Contrast function values at discretization points. Required
            if `total_field` is provided.
        total_field : array_like, optional
            Total electric field at discretization points. Required
            if `contrast` is provided.
        current : array_like, optional
            Contrast source (current) at discretization points. Alternative
            to providing contrast and total_field separately.

        Returns
        -------
        array_like
            Scattered electric field at receiver locations

        Raises
        ------
        MissingInputError
            If required parameter combinations are not provided

        Notes
        -----
        The method requires either:
        - Both `contrast` and `total_field` parameters, or
        - The `current` parameter alone

        If contrast and total_field are provided, the current source is
        computed as J = χE^t, then used to calculate the scattered field.

        Examples
        --------
        >>> # Using contrast and total field
        >>> scattered = discretization.scattered_field(
        ...     contrast=contrast_values,
        ...     total_field=total_field_values
        ... )

        >>> # Using current directly
        >>> scattered = discretization.scattered_field(
        ...     current=current_source
        ... )
        """
        if contrast is not None and total_field is None:
            raise error.MissingInputError('Discretization.scattered_field',
                                          'total_field')
        elif total_field is not None and contrast is None:
            raise error.MissingInputError('Discretization.scattered_field',
                                          'contrast')
        elif total_field is None and contrast is None and current is None:
            raise error.MissingInputError('Discretization.scattered_field',
                                          'contrast')
    @abstractmethod
    def contrast_image(self, coefficients, resolution):
        """Convert contrast coefficients to image format.

        This method converts the discrete contrast function coefficients
        from the discretization basis to a regular image grid at the
        specified resolution. This is useful for visualization and
        analysis of reconstruction results.

        Parameters
        ----------
        coefficients : array_like
            Contrast function coefficients at discretization points,
            typically obtained from solving the inverse problem
        resolution : array_like or tuple
            Desired image resolution as (nx, ny) where nx and ny are
            the number of pixels in x and y directions respectively

        Returns
        -------
        array_like
            2D array representing the contrast function image with
            shape matching the specified resolution

        Notes
        -----
        The conversion process typically involves:
        1. Interpolation from discretization points to image grid
        2. Proper handling of boundary conditions
        3. Scaling and normalization if needed

        The image format allows for easy visualization and comparison
        with reference solutions or other reconstruction methods.

        Examples
        --------
        >>> # Convert to 64x64 image
        >>> image = discretization.contrast_image(
        ...     coefficients=solution_coeffs,
        ...     resolution=(64, 64)
        ... )
        >>> 
        >>> # Display the reconstruction
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(image.real, cmap='jet')
        >>> plt.title('Reconstructed Contrast Function')
        """
        pass
    @abstractmethod
    def total_image(self, coefficients, resolution):
        """Convert total field coefficients to image format.

        This method converts the discrete total field coefficients
        from the discretization basis to a regular image grid at the
        specified resolution. This is useful for visualization and
        analysis of field distributions.

        Parameters
        ----------
        coefficients : array_like
            Total field coefficients at discretization points,
            typically obtained from solving the forward problem
        resolution : array_like or tuple
            Desired image resolution as (nx, ny) where nx and ny are
            the number of pixels in x and y directions respectively

        Returns
        -------
        array_like
            2D array representing the total field image with
            shape matching the specified resolution

        Notes
        -----
        The conversion process typically involves:
        1. Interpolation from discretization points to image grid
        2. Proper handling of boundary conditions
        3. Phase and amplitude representation
        4. Scaling and normalization if needed

        The image format allows for visualization of field patterns
        and understanding of scattering behavior.

        Examples
        --------
        >>> # Convert to 128x128 image
        >>> field_image = discretization.total_image(
        ...     coefficients=field_coeffs,
        ...     resolution=(128, 128)
        ... )
        >>> 
        >>> # Display field magnitude
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(np.abs(field_image), cmap='viridis')
        >>> plt.title('Total Field Magnitude')
        """
        pass
    def copy(self, new=None):
        """Create a copy of the discretization instance.

        This method creates a copy of the current discretization instance
        with the same configuration and settings. Since this is an abstract
        base class, the method cannot create a new instance directly.

        Parameters
        ----------
        new : Discretization, optional
            If provided, the current discretization's configuration will be
            copied into this existing instance. If None, raises an error
            since abstract classes cannot be instantiated.

        Returns
        -------
        None
            This method only supports copying into an existing instance
            due to the abstract nature of the base class.

        Raises
        ------
        TypeError
            If new is None, since abstract classes cannot be instantiated

        Notes
        -----
        This method is designed to be overridden by derived classes to
        provide proper copying functionality:

        ```python
        def copy(self, new=None):
            if new is None:
                return MyDiscretization(
                    configuration=self.configuration,
                    name=self.name,
                    alias=self.alias
                )
            else:
                new.name = self.name
                new.configuration = self.configuration
                new.alias = self.alias
        ```

        Examples
        --------
        >>> # Copy into existing instance (derived class)
        >>> target = MyDiscretization()
        >>> source.copy(target)
        >>> print(target.name)  # Same as source.name
        """
        if new is None:
            raise TypeError("Cannot instantiate abstract class Discretization. "
                          "Use a concrete derived class instead.")
        else:
            new.name = self.name
            new.configuration = self.configuration
            new.alias = self.alias
    @abstractmethod
    def save(self, file_path=''):
        """Save discretization state to file.

        This method saves the discretization configuration and state to a
        file for later restoration. The base implementation provides the
        common data structure, but derived classes should extend this to
        include method-specific parameters.

        Parameters
        ----------
        file_path : str, default=''
            Base path for saving the discretization state. The specific
            filename will be determined by the derived class implementation.

        Returns
        -------
        dict
            Dictionary containing the discretization state data including:
            - name: Discretization method name
            - alias: Method identifier
            - configuration: Problem configuration object

        Notes
        -----
        Derived classes should override this method to include additional
        method-specific data:

        ```python
        def save(self, file_path=''):
            data = super().save(file_path=file_path)
            data['my_specific_param'] = self.my_specific_param
            # Save to file using file_path
            return data
        ```

        The returned dictionary contains all necessary information to
        restore the discretization state using `importdata`.

        Examples
        --------
        >>> # Save discretization state
        >>> data = discretization.save(file_path='/path/to/save/')
        >>> print(f"Saved {len(data)} parameters")
        """
        return {NAME: self.name,
                ALIAS: self.alias,
                CONFIGURATION: self.configuration}
    @abstractmethod
    def importdata(self, file_name, file_path=''):
        """Load discretization state from file.

        This method loads discretization configuration and state from a
        previously saved file. The base implementation handles common
        parameters, but derived classes should extend this to restore
        method-specific data.

        Parameters
        ----------
        file_name : str
            Name of the file containing the saved discretization state
        file_path : str, default=''
            Directory path where the save file is located

        Returns
        -------
        dict
            Dictionary containing the loaded discretization state data

        Notes
        -----
        Derived classes should override this method to restore additional
        method-specific data:

        ```python
        def importdata(self, file_name, file_path=''):
            data = super().importdata(file_name, file_path=file_path)
            self.my_specific_param = data['my_specific_param']
            return data
        ```

        The method completely replaces the current discretization state
        with the loaded configuration.

        Examples
        --------
        >>> # Load discretization state
        >>> data = discretization.importdata(
        ...     file_name='discretization_state.pkl',
        ...     file_path='/path/to/files/'
        ... )
        >>> print(f"Loaded discretization: {discretization.name}")

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        KeyError
            If required keys are missing from the saved data
        """
        data = cfg.import_dict(file_name, file_path)
        self.name = data[NAME]
        self.alias = data[ALIAS]
        self.configuration = data[CONFIGURATION]
        return data
    @abstractmethod
    def __str__(self):
        """Return string representation of the discretization.

        This method provides a string representation of the discretization
        method for display and debugging purposes. The base implementation
        provides a generic identifier.

        Returns
        -------
        str
            String representation of the discretization method

        Notes
        -----
        Derived classes should override this method to provide more
        specific information about the discretization:

        ```python
        def __str__(self):
            return f"{self.name}: {self.alias} - {self.configuration}"
        ```

        Examples
        --------
        >>> print(discretization)
        Discretization: MyMethod - Config(freq=1GHz, nx=64, ny=64)
        """
        return "Discretization: "
