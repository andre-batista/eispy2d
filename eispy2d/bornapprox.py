"""First-Order Born Approximation for Electromagnetic Inverse Scattering.

This module implements the First-Order Born Approximation (FOBA) method for
solving electromagnetic inverse scattering problems. The Born approximation
is a linearization technique that assumes the total field inside the scattering
object is approximately equal to the incident field.

This is a non-iterative method that provides a fast solution to the inverse
scattering problem by solving a single linear system. While it's limited to
weak scatterers, it serves as an excellent starting point for more complex
iterative methods and provides good results for low-contrast objects.

The method solves the linearized scattering equation:
    E_s = G_S * \\chi * E_i

where E_s is the scattered field, G_S is the scattering Green's function,
\\chi is the contrast function, and E_i is the incident field.

Classes
-------
FirstOrderBornApproximation : dtm.Deterministic
    Main class implementing the First-Order Born Approximation method.

Constants
---------
REGULARIZATION : str
    Key for regularization method in serialization.
FORWARD : str
    Key for forward solver in serialization.

References
----------
.. [1] Chew, Weng Cho. "Waves and fields in inhomogeneous media."
   IEEE Press, 1995.
.. [2] Colton, David, and Rainer Kress. "Inverse acoustic and electromagnetic
   scattering theory." Springer Science & Business Media, 2012.
"""

# Standard libraries
import sys
import pickle
import time as tm
from numpy import pi

# Developed libraries
import eispy2d.deterministic as dtm
import eispy2d.mom_cg_fft as mom
import eispy2d.configuration as cfg
import eispy2d.result as rst

REGULARIZATION = 'regularization'
FORWARD = 'forward'

class FirstOrderBornApproximation(dtm.Deterministic):
    """First-Order Born Approximation for Electromagnetic Inverse Scattering.

    This class implements the First-Order Born Approximation (FOBA), a
    linearization technique for solving electromagnetic inverse scattering
    problems. The method assumes that the total field inside the scattering
    object is approximately equal to the incident field, making it suitable
    for weak scatterers.

    The Born approximation converts the nonlinear inverse scattering problem
    into a linear one by replacing the unknown total field with the known
    incident field in the scattering integral equation. This results in a
    single linear system that can be solved directly.

    Mathematical Formulation:
    The method solves the linearized equation:
        E_s = G_S * \\chi * E_i
    
    where:
    - E_s: scattered field (measured data)
    - G_S: scattering Green's function matrix
    - \\chi: contrast function (unknown to be reconstructed)
    - E_i: incident field (known)

    Parameters
    ----------
    regularization : Regularization
        Regularization method for solving the linear inverse problem.
        Required to handle the ill-conditioned nature of the inverse problem.
    forward : ForwardSolver, default: mom.MoM_CG_FFT()
        Forward solver for computing incident fields and Green's functions.
    alias : str, default: 'ba'
        Alias name for the method instance.
    import_filename : str, optional
        Filename to import previously saved method state.
    import_filepath : str, default: ''
        Path to the file containing saved method state.

    Attributes
    ----------
    name : str
        Name of the method ('First-Order Born Approximation').
    regularization : Regularization
        Regularization method for the linear inverse problem.
    forward : ForwardSolver
        Forward solver instance for field computation.

    Methods
    -------
    solve(inputdata, discretization, print_info=True, print_file=sys.stdout)
        Solve the linearized inverse scattering problem.
    save(file_path='')
        Save the method state to file.
    importdata(file_name, file_path='')
        Import method state from file.
    copy(new=None)
        Create a copy of the method instance.

    Notes
    -----
    **Validity Range**: The Born approximation is valid when:
    - The contrast is small (|\chi| << 1)
    - The scatterer is electrically small
    - The refractive index variation is gradual

    **Advantages**:
    - Fast computation (single linear solve)
    - No iteration required
    - Good initial guess for iterative methods
    - Stable and well-understood

    **Limitations**:
    - Limited to weak scatterers
    - Poor performance for high-contrast objects
    - No nonlinear coupling effects

    References
    ----------
    .. [1] Chew, Weng Cho. "Waves and fields in inhomogeneous media."
       IEEE Press, 1995.
    .. [2] Colton, David, and Rainer Kress. "Inverse acoustic and electromagnetic
       scattering theory." Springer Science & Business Media, 2012.
    """

    def __init__(self, regularization, forward=mom.MoM_CG_FFT(), alias='ba',
                 import_filename=None, import_filepath=''):
        """Initialize the First-Order Born Approximation method.

        Creates a new FOBA instance with the specified regularization method
        and forward solver configuration.

        Parameters
        ----------
        regularization : Regularization
            Regularization method for solving the ill-conditioned linear
            inverse problem. Common choices include Tikhonov regularization,
            truncated SVD, or iterative methods like CGLS.
        forward : ForwardSolver, default: mom.MoM_CG_FFT()
            Forward solver implementation for computing incident fields and
            Green's function matrices. Must implement the ForwardSolver interface.
        alias : str, default: 'ba'
            Short alias name for this method instance, used for identification
            and file naming purposes.
        import_filename : str, optional
            Filename to import previously saved method state. If provided,
            other parameters are ignored and the state is loaded from file.
        import_filepath : str, default: ''
            Path to the file containing saved method state.

        Notes
        -----
        The regularization parameter is crucial for the Born approximation
        because the inverse scattering problem is inherently ill-conditioned.
        The choice of regularization method and its parameters significantly
        affects the reconstruction quality.

        The forward solver is used to compute the incident field, which serves
        as the approximation for the total field in the Born approximation.
        Different forward solvers may be used depending on the problem geometry
        and computational requirements.

        Examples
        --------
        >>> import regularization as reg
        >>> import mom_cg_fft as mom
        >>> 
        >>> # Create regularization method
        >>> tikhonov = reg.Tikhonov(regularization_parameter=1e-3)
        >>> 
        >>> # Create forward solver
        >>> forward_solver = mom.MoM_CG_FFT()
        >>> 
        >>> # Initialize Born approximation
        >>> born = FirstOrderBornApproximation(tikhonov, forward_solver)
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=False)
            self.name = 'First-Order Born Approximation'
            self.regularization = regularization
            self.forward = forward

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        """Solve the inverse scattering problem using First-Order Born Approximation.

        Implements the Born approximation to reconstruct material properties
        from scattered field measurements. The method solves a single linearized
        system by approximating the total field with the incident field.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing scattered field measurements,
            configuration parameters, and problem setup information.
        discretization : Discretization
            Discretization object containing domain information, Green's
            function matrices, and numerical methods for field computation.
        print_info : bool, default: True
            Whether to print algorithm progress and results information.
        print_file : file-like object, default: sys.stdout
            File object to write progress information to.

        Returns
        -------
        result : Result
            Result object containing the reconstructed material properties,
            fields, error metrics, and algorithm performance information.

        Notes
        -----
        **Algorithm Steps**:
        
        1. **Incident Field Computation**: Calculate the incident field E_i
           in the discretization domain using the forward solver.
        
        2. **Linear System Solution**: Solve the linearized equation:
           
           .. math::
               \\chi = \\mathcal{R}[G_S^H E_s]
           
           where :math:`\\mathcal{R}` is the regularization operator,
           :math:`G_S^H` is the adjoint scattering Green's function,
           and :math:`E_s` is the scattered field.
        
        3. **Field Reconstruction**: Compute the reconstructed scattered field
           using the estimated contrast:
           
           .. math::
               E_s^{rec} = G_S \\chi E_i
        
        4. **Parameter Conversion**: Convert the contrast function to physical
           parameters (relative permittivity, conductivity) based on the
           problem configuration.
        
        5. **Error Computation**: Calculate various error metrics and
           performance indicators.

        **Computational Complexity**: O(N²) where N is the number of
        discretization points, dominated by the linear system solution.

        **Memory Requirements**: Depends on the regularization method and
        Green's function matrix storage.

        The method automatically handles both dielectric and conducting
        materials, converting between contrast and physical parameters
        as needed based on the problem configuration.

        Examples
        --------
        >>> # Solve inverse problem
        >>> result = born.solve(input_data, discretization)
        >>> 
        >>> # Access reconstructed properties
        >>> rel_permittivity = result.rel_permittivity
        >>> conductivity = result.conductivity
        >>> 
        >>> # Check reconstruction quality
        >>> rel_error = result.rel_permittivity_error
        """
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)

        incident_field = self.forward.incident_field(discretization.elements,
                                                     inputdata.configuration)
        
        tic = tm.time()
        contrast = discretization.solve(scattered_field=inputdata.scattered_field,
                                        total_field=incident_field,
                                        linear_solver=self.regularization)
        execution_time = tm.time()-tic


        scattered_field = discretization.scattered_field(contrast=contrast,
                                                         total_field=incident_field)
        total_field = self.forward.incident_field(inputdata.resolution,
                                                  inputdata.configuration)
        contrast = discretization.contrast_image(contrast,
                                                 inputdata.resolution)

        result.update_error(inputdata, scattered_field=scattered_field,
                            total_field=total_field, contrast=contrast)
        result.scattered_field = scattered_field
        result.total_field = incident_field

        if not inputdata.configuration.good_conductor:
            result.rel_permittivity = cfg.get_relative_permittivity(
                contrast, inputdata.configuration.epsilon_rb
            )
        if not inputdata.configuration.perfect_dielectric:
            result.conductivity = cfg.get_conductivity(
                contrast, 2*pi*inputdata.configuration.f,
                inputdata.configuration.epsilon_rb,
                inputdata.configuration.sigma_b
            )
        if rst.EXECUTION_TIME in inputdata.indicators:
            result.execution_time = execution_time
        if rst.NUMBER_ITERATIONS in inputdata.indicators:
            result.number_iterations = 1

        return result

    def save(self, file_path=''):
        """Save the First-Order Born Approximation method state to file.

        Saves the complete method configuration including regularization
        method and forward solver settings to a file for later retrieval.

        Parameters
        ----------
        file_path : str, default: ''
            Path where to save the method state file. The file will be
            saved with the method's alias as the filename.

        Notes
        -----
        The method saves all necessary information to recreate the FOBA
        instance, including:
        - Regularization method and its parameters
        - Forward solver configuration
        - Base class attributes (alias, etc.)

        The file is saved using pickle serialization format, which preserves
        the complete object state including any custom parameters or
        configurations.

        Examples
        --------
        >>> # Save method to current directory
        >>> born.save()
        >>> 
        >>> # Save to specific path
        >>> born.save('/path/to/save/')
        """
        data = super().save(file_path=file_path)
        data[REGULARIZATION] = self.regularization
        data[FORWARD] = self.forward
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import First-Order Born Approximation method state from file.

        Loads a previously saved FOBA configuration from file, restoring
        all method parameters and settings.

        Parameters
        ----------
        file_name : str
            Name of the file containing saved method state.
        file_path : str, default: ''
            Path to the file containing saved method state.

        Notes
        -----
        This method restores the complete FOBA configuration including:
        - Regularization method with all its parameters
        - Forward solver configuration and settings
        - Base class attributes and state

        The loaded configuration can be used to recreate the exact same
        method instance as was saved, ensuring reproducibility of results.

        Examples
        --------
        >>> # Import from current directory
        >>> born = FirstOrderBornApproximation(None)  # dummy init
        >>> born.importdata('ba')  # load from file
        >>> 
        >>> # Import from specific path
        >>> born.importdata('my_method', '/path/to/files/')
        """
        data = super().importdata(file_name, file_path=file_path)
        self.regularization = data[REGULARIZATION]
        self.forward = data[FORWARD]

    def copy(self, new=None):
        """Create a copy of the First-Order Born Approximation instance.

        Creates a new FOBA instance with the same configuration as the current
        one, or copies the current configuration to an existing instance.

        Parameters
        ----------
        new : FirstOrderBornApproximation, optional
            Existing FOBA instance to copy configuration to. If None,
            creates a new instance.

        Returns
        -------
        FirstOrderBornApproximation or None
            If new is None, returns a new FOBA instance with copied configuration.
            If new is provided, modifies it in place and returns None.

        Notes
        -----
        The copy includes all method components:
        - Regularization method (reference copy)
        - Forward solver (reference copy)
        - Base class attributes (alias, etc.)

        This method is useful for creating multiple instances with the same
        configuration, parameter studies, or backup purposes before
        modifying parameters.

        Examples
        --------
        >>> # Create a copy
        >>> born_copy = born.copy()
        >>> 
        >>> # Copy to existing instance
        >>> new_born = FirstOrderBornApproximation(some_regularization)
        >>> born.copy(new_born)  # copies configuration to new_born
        """
        if new is None:
            return FirstOrderBornApproximation(self.regularization, self.forward,
                                               self.alias)
        else:
            super().copy(new)
            self.regularization = new.regularization
            self.forward = new.forward

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        """Print algorithm title and configuration information.

        Prints the algorithm header including forward solver and regularization
        method information for progress tracking and debugging.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing problem configuration.
        discretization : Discretization
            Discretization object containing domain information.
        print_file : file-like object, default: sys.stdout
            File object to write information to.

        Notes
        -----
        This method is called automatically by the solve method when
        print_info=True. It provides useful information about the
        method configuration and problem setup.
        """
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)
        print(self.regularization, file=print_file)

    def __str__(self):
        """Return string representation of the First-Order Born Approximation.

        Creates a comprehensive string description of the FOBA instance
        including all its components and configuration.

        Returns
        -------
        str
            String representation including:
            - Base class information (name, alias, etc.)
            - Regularization method details
            - Forward solver configuration

        Notes
        -----
        This method provides a complete overview of the FOBA configuration,
        which is useful for debugging, logging, and understanding the
        method setup. The string includes detailed information about
        the regularization method and forward solver parameters.

        Examples
        --------
        >>> print(born)
        First-Order Born Approximation (ba)
        Regularization: Tikhonov (λ=1e-3)
        Forward Solver: MoM-CG-FFT
        ...
        """
        message = super().__str__()
        message += str(self.regularization)
        message += str(self.forward)
        return message

