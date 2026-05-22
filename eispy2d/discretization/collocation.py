"""
Collocation Method for Electromagnetic Inverse Scattering Discretization

This module implements the Collocation Method for discretizing electromagnetic
inverse scattering problems. The collocation method is a numerical technique
that uses trial functions and collocation points to approximate the solution
of integral equations in electromagnetic scattering problems.

The module provides the Collocation class for method configuration and several
optimized kernel functions for efficient computation of matrix operations
commonly used in electromagnetic inverse scattering algorithms.

Classes
-------
Collocation : Extends discretization.Discretization
    Main collocation method implementation for electromagnetic discretization

Functions
---------
kernel_GSE(GS, E) : Compute kernel matrix for scattered field Green's function
kernel_GSX(GS, X) : Compute kernel matrix for scattered field with contrast
kernel_GDX(GD, X) : Compute kernel matrix for domain Green's function with contrast
kernel_GDE(GD, E) : Compute kernel matrix for domain Green's function with field
lhs_XEi(X, Ei) : Compute left-hand side matrix for contrast and incident field

Constants
---------
TRIAL_FUNCTION : str
    Dictionary key for trial function type
ELEMENTS : str
    Dictionary key for discretization elements

Notes
-----
The collocation method approximates the solution using basis functions and
enforces the integral equation at specific collocation points. This approach
is particularly effective for electromagnetic scattering problems where the
Green's function matrices can be computed efficiently.

All kernel functions are optimized using Numba's just-in-time compilation
for improved performance in iterative reconstruction algorithms.

Examples
--------
>>> # Create collocation discretization
>>> collocation = Collocation(configuration=config, 
...                          trial='pulse', 
...                          elements=(64, 64))

>>> # Use kernel functions for matrix operations
>>> K_GSE = kernel_GSE(GS_matrix, E_field)
>>> K_GSX = kernel_GSX(GS_matrix, contrast)
"""

from eispy2d.core import error
from eispy2d.discretization import discretization as dct

import pickle
import numpy as np

from numba import jit

TRIAL_FUNCTION = 'trial'
ELEMENTS = 'elements'

class Collocation(dct.Discretization):
    """
    Collocation method for electromagnetic inverse scattering discretization.
    
    This class implements the collocation method for discretizing electromagnetic
    inverse scattering problems. The method uses trial functions and collocation
    points to approximate the solution of integral equations, providing an
    efficient framework for electromagnetic reconstruction algorithms.
    
    The collocation method is particularly well-suited for problems where the
    Green's function can be computed analytically or semi-analytically, making
    it computationally efficient for iterative reconstruction methods.
    
    Parameters
    ----------
    configuration : configuration.Configuration, optional
        Problem configuration object containing geometry and material properties
    trial : str, optional
        Type of trial function to use (e.g., 'pulse', 'linear', 'cubic')
    elements : int or tuple of int, optional
        Number of discretization elements. If int, creates square grid (N×N).
        If tuple, creates rectangular grid (NY×NX)
    name : str, optional
        Custom name for the discretization method
    alias : str, default='clc'
        Short alias for the method used in file operations
    import_filename : str, optional
        Filename to import configuration from
    import_filepath : str, default=''
        Path to import file
    
    Attributes
    ----------
    trial : str
        Type of trial function used
    elements : tuple of int
        Grid dimensions (NY, NX)
    name : str
        Descriptive name of the discretization method
    alias : str
        Short alias for the method
    
    Methods
    -------
    copy(new=None)
        Create a copy of the collocation instance
    save(file_path='')
        Save configuration to file
    importdata(file_name, file_path='')
        Import configuration from file
    
    Raises
    ------
    error.MissingInputError
        If required elements parameter is not provided
    
    Notes
    -----
    The collocation method discretizes the electromagnetic integral equation
    by choosing specific collocation points where the equation is enforced
    exactly. This approach transforms the continuous integral equation into
    a system of linear equations that can be solved numerically.
    
    Trial functions commonly used include:
    - **pulse**: Piecewise constant functions
    - **linear**: Piecewise linear functions
    - **cubic**: Piecewise cubic functions
    
    Examples
    --------
    >>> # Create square grid collocation
    >>> collocation = Collocation(configuration=config, 
    ...                          trial='pulse', 
    ...                          elements=64)
    
    >>> # Create rectangular grid collocation
    >>> collocation = Collocation(configuration=config, 
    ...                          trial='linear', 
    ...                          elements=(32, 64))
    
    >>> # Import from saved configuration
    >>> collocation = Collocation(import_filename='collocation_config.pkl')
    
    >>> # Print method information
    >>> print(collocation)
    Collocation Method (64x64), trial function: pulse
    """
    def __init__(self, configuration=None, trial=None, elements=None,
                 name=None, alias='clc', import_filename=None,
                 import_filepath=''):
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(configuration=configuration, name=name,
                             alias=alias)
            self.trial = trial
            if elements is None:
                raise error.MissingInputError('Collocation.__init__',
                                              'elements')
            elif type(elements) is int:
                self.elements = (elements, elements)
            else:
                self.elements = (elements[0], elements[1])
            self.name = ('Collocation Method (%dx' % self.elements[0] + '%d), '
                         % self.elements[1] + 'trial function: ' + self.trial)
    def copy(self, new=None):
        """
        Create a copy of the collocation instance.
        
        Creates either a new independent instance or copies the configuration
        to an existing instance.
        
        Parameters
        ----------
        new : Collocation or None, default=None
            If None, creates a new independent instance
            If provided, copies configuration to this instance
            
        Returns
        -------
        Collocation or None
            New instance if new=None, otherwise None
            
        Examples
        --------
        >>> # Create independent copy
        >>> collocation_copy = collocation.copy()
        
        >>> # Copy configuration to existing instance
        >>> new_collocation = Collocation(configuration=config, 
        ...                              trial='pulse', 
        ...                              elements=(32, 32))
        >>> collocation.copy(new_collocation)
        """
        if new is None:
            return Collocation(self.configuration, self.trial, self.elements,
                               self.name)
        else:
            super().copy(new)
            new.trial = self.trial
            new.elements = self.elements
    def __str__(self):
        """
        Return string representation of the collocation method.
        
        Creates a formatted string containing the method configuration
        including discretization details and alias.
        
        Returns
        -------
        str
            Formatted string representation of the collocation method
            
        Examples
        --------
        >>> print(collocation)
        Collocation Method (64x64), trial function: pulse
        Alias: clc
        """
        message = super().__str__()
        message += 'Discretization:' + self.name + '\n'
        message += 'Alias: ' + self.alias + '\n'
        return message
    def save(self, file_path=''):
        """
        Save the collocation configuration to a file.
        
        Saves the complete collocation method configuration including
        trial function type and discretization elements using pickle serialization.
        
        Parameters
        ----------
        file_path : str, default=''
            Path where the configuration file will be saved
            
        Examples
        --------
        >>> collocation.save('/path/to/save/')
        >>> collocation.save()  # Save in current directory
        """
        data = super().save(file_path=file_path)
        data[TRIAL_FUNCTION] = self.trial
        data[ELEMENTS] = self.elements
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)
    def importdata(self, file_name, file_path=''):
        """
        Import collocation configuration from a saved file.
        
        Loads a previously saved collocation configuration including
        trial function type and discretization elements.
        
        Parameters
        ----------
        file_name : str
            Name of the file to import from
        file_path : str, default=''
            Path to the file location
            
        Examples
        --------
        >>> collocation = Collocation()
        >>> collocation.importdata('collocation_config.pkl', '/path/to/files/')
        """
        data = super().importdata(file_name, file_path=file_path)
        self.trial = data[TRIAL_FUNCTION]
        self.elements = data[ELEMENTS]

def kernel_GSE(GS, E):
    """
    Compute kernel matrix for scattered field Green's function.
    
    Computes the kernel matrix K for the scattered field using the Green's
    function matrix GS and electric field E. This function is optimized
    for electromagnetic inverse scattering computations.
    
    Parameters
    ----------
    GS : numpy.ndarray
        Green's function matrix for scattered field with shape (NM, N)
        where NM is the number of measurement points and N is the number
        of discretization elements
    E : numpy.ndarray
        Electric field matrix with shape (N, NS) where N is the number
        of discretization elements and NS is the number of sources
        
    Returns
    -------
    numpy.ndarray
        Kernel matrix K with shape (NM*NS, N) containing the computed
        kernel values for electromagnetic scattering
        
    Notes
    -----
    The kernel matrix is computed as:
    K[m*NS + s, n] = GS[m, n] * E[n, s]
    
    This function uses Numba JIT compilation for improved performance
    in iterative electromagnetic reconstruction algorithms.
    
    Examples
    --------
    >>> GS = np.random.complex128((128, 64*64))  # 128 measurements, 64x64 grid
    >>> E = np.random.complex128((64*64, 16))    # 64x64 grid, 16 sources
    >>> K = kernel_GSE(GS, E)
    >>> print(K.shape)  # (128*16, 64*64) = (2048, 4096)
    """
    N, NS = E.shape
    NM = GS.shape[0]
    return _kernel_GSE(GS, E, NM, NS, N)


@jit(nopython=True)
def _kernel_GSE(GS, E, NM, NS, N):
    """
    Optimized kernel computation for scattered field Green's function.
    
    Internal JIT-compiled function for efficient computation of the
    kernel matrix used in electromagnetic inverse scattering problems.
    
    Parameters
    ----------
    GS : numpy.ndarray
        Green's function matrix for scattered field
    E : numpy.ndarray
        Electric field matrix
    NM : int
        Number of measurement points
    NS : int
        Number of sources
    N : int
        Number of discretization elements
        
    Returns
    -------
    numpy.ndarray
        Computed kernel matrix
        
    Notes
    -----
    This function is compiled with Numba for performance optimization
    and should not be called directly. Use kernel_GSE() instead.
    """
    K = 1j*np.ones((NM*NS, N))
    row = 0
    for m in range(NM):
        for s in range(NS):
            K[row, :] = GS[m, :].flatten()*E[:, s].flatten()
            row += 1
    return K


def kernel_GSX(GS, X):
    """
    Compute kernel matrix for scattered field Green's function with contrast.
    
    Computes the kernel matrix for the scattered field using the Green's
    function matrix GS and contrast function X. This function handles
    different input formats for the contrast function.
    
    Parameters
    ----------
    GS : numpy.ndarray
        Green's function matrix for scattered field with shape (NM, N)
        where NM is the number of measurement points and N is the number
        of discretization elements
    X : numpy.ndarray
        Contrast function that can be:
        - 1D array of length N (contrast values)
        - 2D array with total elements N (reshaped to 1D)
        - 2D diagonal matrix with shape (N, N) (diagonal extracted)
        
    Returns
    -------
    numpy.ndarray
        Kernel matrix K with shape (NM, N) containing the computed
        kernel values for electromagnetic scattering with contrast
        
    Notes
    -----
    The kernel matrix is computed as:
    K[m, n] = GS[m, n] * X[n]
    
    This function automatically handles different contrast formats:
    - Vector contrast: Direct multiplication
    - Matrix contrast: Diagonal elements used
    - Reshaped contrast: Automatically flattened
    
    Examples
    --------
    >>> GS = np.random.complex128((128, 64*64))  # 128 measurements, 64x64 grid
    >>> X = np.random.complex128((64*64,))       # 1D contrast
    >>> K = kernel_GSX(GS, X)
    >>> print(K.shape)  # (128, 64*64) = (128, 4096)
    
    >>> # Using 2D contrast matrix
    >>> X_2d = np.random.complex128((64*64, 64*64))
    >>> K = kernel_GSX(GS, X_2d)  # Uses diagonal elements
    """
    NM, N = GS.shape
    if X.ndim == 1:
        return _kernel_GSX(GS, X, NM, N)
    elif X.ndim == 2 and np.prod(X.shape) == N:
        return _kernel_GSX(GS, X.flatten(), NM, N)
    elif X.ndim == 2 and X.shape[0] == N:
        return _kernel_GSX(GS, np.diagonal(X), NM, N)


@jit(nopython=True)
def _kernel_GSX(GS, X, NM, N):
    """
    Optimized kernel computation for scattered field with contrast.
    
    Internal JIT-compiled function for efficient computation of the
    kernel matrix involving Green's function and contrast function.
    
    Parameters
    ----------
    GS : numpy.ndarray
        Green's function matrix for scattered field
    X : numpy.ndarray
        Contrast function (1D array)
    NM : int
        Number of measurement points
    N : int
        Number of discretization elements
        
    Returns
    -------
    numpy.ndarray
        Computed kernel matrix
        
    Notes
    -----
    This function is compiled with Numba for performance optimization
    and should not be called directly. Use kernel_GSX() instead.
    """
    K = 1j*np.ones((NM, N))
    for m in range(NM):
        K[m, :] = GS[m, :].flatten()*X
    return K


def kernel_GDX(GD, X):
    """
    Compute kernel matrix for domain Green's function with contrast.
    
    Computes the kernel matrix for the domain interaction using the
    Green's function matrix GD and contrast function X. This creates
    the modified identity matrix (I - GD*X) used in electromagnetic
    inverse scattering formulations.
    
    Parameters
    ----------
    GD : numpy.ndarray
        Green's function matrix for domain interaction with shape (N, N)
        where N is the number of discretization elements
    X : numpy.ndarray
        Contrast function that can be:
        - 1D array of length N (contrast values)
        - 2D array with total elements N (reshaped to 1D)
        - 2D diagonal matrix with shape (N, N) (diagonal extracted)
        
    Returns
    -------
    numpy.ndarray
        Kernel matrix K with shape (N, N) representing the modified
        identity matrix (I - GD*X) for electromagnetic scattering
        
    Notes
    -----
    The kernel matrix is computed as:
    K[n, m] = -GD[n, m] * X[n]  for n ≠ m
    K[n, n] = 1 - GD[n, n] * X[n]
    
    This represents the discretized form of the Lippmann-Schwinger equation
    operator (I - GD*X) commonly used in electromagnetic inverse scattering.
    
    Examples
    --------
    >>> GD = np.random.complex128((4096, 4096))  # 64x64 grid
    >>> X = np.random.complex128((4096,))        # 1D contrast
    >>> K = kernel_GDX(GD, X)
    >>> print(K.shape)  # (4096, 4096)
    
    >>> # Verify identity structure
    >>> I = np.eye(4096)
    >>> expected = I - GD * X[:, np.newaxis]
    >>> np.allclose(K, expected)  # Should be True
    """
    N = GD.shape[0]
    if X.ndim == 1:
        return _kernel_GDX(GD, X, N)
    elif X.ndim == 2 and np.prod(X.shape) == N:
        return _kernel_GDX(GD, X.flatten(), N)
    elif X.ndim == 2 and X.shape[0] == N:
        return _kernel_GDX(GD, np.diagonal(X), N)


@jit(nopython=True)
def _kernel_GDX(GD, X, N):
    """
    Optimized kernel computation for domain Green's function with contrast.
    
    Internal JIT-compiled function for efficient computation of the
    modified identity matrix (I - GD*X) used in electromagnetic
    inverse scattering problems.
    
    Parameters
    ----------
    GD : numpy.ndarray
        Green's function matrix for domain interaction
    X : numpy.ndarray
        Contrast function (1D array)
    N : int
        Number of discretization elements
        
    Returns
    -------
    numpy.ndarray
        Computed kernel matrix representing (I - GD*X)
        
    Notes
    -----
    This function is compiled with Numba for performance optimization
    and should not be called directly. Use kernel_GDX() instead.
    """
    K = 1j*np.ones((N, N))
    for n in range(N):
        K[n, :] = - GD[n, :].flatten()*X
        K[n, n] += 1
    return K


def kernel_GDE(GD, E):
    """
    Compute kernel matrix for domain Green's function with electric field.
    
    Computes the kernel matrix for the domain interaction using the
    Green's function matrix GD and electric field E. This creates
    a 3D kernel matrix used in electromagnetic inverse scattering
    computations involving field interactions.
    
    Parameters
    ----------
    GD : numpy.ndarray
        Green's function matrix for domain interaction with shape (N, N)
        where N is the number of discretization elements
    E : numpy.ndarray
        Electric field matrix with shape (N, NS) where N is the number
        of discretization elements and NS is the number of sources
        
    Returns
    -------
    numpy.ndarray
        Kernel matrix K with shape (N, N, NS) containing the computed
        kernel values for electromagnetic field interactions
        
    Notes
    -----
    The kernel matrix is computed as:
    K[n, m, s] = GD[n, m] * E[m, s]
    
    This 3D kernel matrix represents the interaction between the domain
    Green's function and the electric field for all source configurations
    simultaneously, which is useful for multi-source electromagnetic
    inverse scattering problems.
    
    Examples
    --------
    >>> GD = np.random.complex128((4096, 4096))  # 64x64 grid
    >>> E = np.random.complex128((4096, 16))     # 64x64 grid, 16 sources
    >>> K = kernel_GDE(GD, E)
    >>> print(K.shape)  # (4096, 4096, 16)
    
    >>> # Access kernel for specific source
    >>> K_source_0 = K[:, :, 0]  # Kernel for first source
    >>> print(K_source_0.shape)  # (4096, 4096)
    """
    N, NS = E.shape
    return _kernel_GDE(GD, E, N, NS)


@jit(nopython=True)
def _kernel_GDE(GD, E, N, NS):
    """
    Optimized kernel computation for domain Green's function with electric field.
    
    Internal JIT-compiled function for efficient computation of the
    3D kernel matrix involving domain Green's function and electric field.
    
    Parameters
    ----------
    GD : numpy.ndarray
        Green's function matrix for domain interaction
    E : numpy.ndarray
        Electric field matrix
    N : int
        Number of discretization elements
    NS : int
        Number of sources
        
    Returns
    -------
    numpy.ndarray
        Computed 3D kernel matrix
        
    Notes
    -----
    This function is compiled with Numba for performance optimization
    and should not be called directly. Use kernel_GDE() instead.
    """
    K = 1j*np.ones((N, N, NS))
    row = 0
    for s in range(NS):
        for n in range(N):
            K[n, :, s] = GD[n, :].flatten()*E[:, s].flatten()
            row += 1
    return K


def lhs_XEi(X, Ei):
    """
    Compute left-hand side matrix for contrast and incident field.
    
    Computes the left-hand side matrix by multiplying the contrast
    function X with the incident electric field Ei. This operation
    is commonly used in electromagnetic inverse scattering formulations
    for computing the source term in the Lippmann-Schwinger equation.
    
    Parameters
    ----------
    X : numpy.ndarray
        Contrast function with shape (N,) where N is the number
        of discretization elements
    Ei : numpy.ndarray
        Incident electric field matrix with shape (N, NS) where
        N is the number of discretization elements and NS is the
        number of sources
        
    Returns
    -------
    numpy.ndarray
        Left-hand side matrix with shape (N, NS) containing the
        element-wise product X * Ei for each source
        
    Notes
    -----
    The computation is performed as:
    lhs[n, s] = X[n] * Ei[n, s]
    
    This represents the source term in the electromagnetic inverse
    scattering equation: X * Ei, where X is the contrast function
    and Ei is the incident field.
    
    Examples
    --------
    >>> X = np.random.complex128((4096,))      # 64x64 grid contrast
    >>> Ei = np.random.complex128((4096, 16))  # 64x64 grid, 16 sources
    >>> lhs = lhs_XEi(X, Ei)
    >>> print(lhs.shape)  # (4096, 16)
    
    >>> # Verify computation for first source
    >>> expected = X * Ei[:, 0]
    >>> np.allclose(lhs[:, 0], expected)  # Should be True
    """
    N, NS = Ei.shape
    return _lhs_XEi(X, Ei, N, NS)


@jit(nopython=True)
def _lhs_XEi(X, Ei, N, NS):
    """
    Optimized computation for contrast and incident field multiplication.
    
    Internal JIT-compiled function for efficient computation of the
    element-wise product between contrast function and incident field.
    
    Parameters
    ----------
    X : numpy.ndarray
        Contrast function (1D array)
    Ei : numpy.ndarray
        Incident electric field matrix
    N : int
        Number of discretization elements
    NS : int
        Number of sources
        
    Returns
    -------
    numpy.ndarray
        Computed left-hand side matrix
        
    Notes
    -----
    This function is compiled with Numba for performance optimization
    and should not be called directly. Use lhs_XEi() instead.
    """
    lhs = 1j*np.ones((N, NS))
    for s in range(NS):
        lhs[:, s] = X*Ei[:, s]
    return lhs
