r"""Results storage and visualization module for electromagnetic inverse scattering.

This module provides comprehensive functionality for storing, analyzing, and visualizing
results from electromagnetic inverse scattering algorithms. It contains the main
:class:`Result` class for storing reconstruction results and error metrics, along with
plotting utilities for visualization of reconstructed images and convergence curves.

The module supports various error metrics for quantitative evaluation of reconstruction
quality, including residual norm errors, percentage average deviation (PAD) errors,
and shape/position errors. It also provides sophisticated plotting capabilities for
displaying contrast maps, field distributions, and convergence behavior.

Key Features
------------
- Comprehensive result storage with multiple error metrics
- Advanced plotting capabilities for maps and convergence curves
- Support for both permittivity and conductivity reconstructions
- Flexible visualization options with customizable layouts
- Statistical analysis tools for reconstruction quality assessment
- Pickle-based serialization for result persistence

Classes
-------
Result
    Main class for storing and managing reconstruction results from inverse scattering methods.

Functions
---------
add_image
    Add image plots to matplotlib axes with standardized formatting.
add_plot
    Add line plots to matplotlib axes for convergence visualization.
add_box
    Add box plots for statistical analysis of reconstruction metrics.
get_figure
    Create figure layouts for different plot configurations.
indicator_label
    Get formatted labels for error indicators.
check_indicator
    Validate error indicator names.
compute_*
    Various functions for computing specific error metrics.

Constants
---------
Error Indicator Constants
    RESIDUAL_NORM_ERROR, RESIDUAL_PAD_ERROR, REL_PERMITTIVITY_PAD_ERROR, etc.
    String constants for accessing different error metrics.

Plot Label Constants
    XLABEL_STANDARD, YLABEL_STANDARD, COLORBAR_REL_PERMITTIVITY, etc.
    Pre-defined labels for consistent plot formatting.

Image Type Constants
    PERMITTIVITY, CONDUCTIVITY, CONTRAST, TOTAL_FIELD, BOTH_PROPERTIES
    Constants for specifying different visualization types.

Notes
-----
The module integrates closely with the electromagnetic inverse scattering framework,
providing standardized result storage and visualization capabilities. It supports
both single-frequency and multi-frequency reconstruction results.

Examples
--------
>>> import result as rst
>>> 
>>> # Create result object
>>> result = rst.Result(name='test_reconstruction')
>>> 
>>> # Store reconstruction data
>>> result.rel_permittivity = reconstructed_permittivity
>>> result.conductivity = reconstructed_conductivity
>>> 
>>> # Plot results
>>> result.plot_map(image=rst.BOTH_PROPERTIES, show=True)
>>> 
>>> # Plot convergence
>>> result.plot_convergence(indicators=[rst.RESIDUAL_NORM_ERROR], show=True)
"""

from math import pi
import pickle
import copy as cp
import numpy as np
from scipy.stats import linregress
from skimage import measure
from skimage.metrics import structural_similarity
from statsmodels.graphics.boxplots import violinplot
import matplotlib.pyplot as plt

from eispy2d.core import error
from eispy2d.core import configuration as cfg


# Strings for easier implementation of plots
XLABEL_STANDARD = r'x [$\lambda_b$]'
YLABEL_STANDARD = r'y [$\lambda_b$]'
XLABEL_UNDEFINED = r'x [$L_x$]'
YLABEL_UNDEFINED = r'y [$L_y$]'
COLORBAR_REL_PERMITTIVITY = r'$\epsilon_r$'
COLORBAR_CONDUCTIVITY = r'$\sigma$ [S/m]'
TITLE_REL_PERMITTIVITY = 'Relative Permittivity'
TITLE_CONDUCTIVITY = 'Conductivity'
TITLE_RECOVERED_REL_PERMITTIVITY = ('Recovered '
                                         + TITLE_REL_PERMITTIVITY)
TITLE_RECOVERED_CONDUCTIVITY = 'Recovered ' + TITLE_CONDUCTIVITY
TITLE_ORIGINAL_REL_PERMITTIVITY = ('Original '
                                        + TITLE_REL_PERMITTIVITY)
TITLE_ORIGINAL_CONDUCTIVITY = 'Original ' + TITLE_CONDUCTIVITY
LABEL_ZETA_RN = r'$\zeta_{RN} [V/m]$'
LABEL_ZETA_RPAD = r'$\zeta_{RPAD}$ [\%/sample]'
LABEL_ZETA_EPAD = r'$\zeta_{\epsilon PAD}$ [\%/pixel]'
LABEL_ZETA_EBE = r'$\zeta_{\epsilon BE}$ [\%/pixel]'
LABEL_ZETA_EOE = r'$\zeta_{\epsilon OE}$ [\%/pixel]'
LABEL_ZETA_SAD = r'$\zeta_{\sigma AD}$ [S/pixel]'
LABEL_ZETA_SBE = r'$\zeta_{\sigma BE}$ [S/pixel]'
LABEL_ZETA_SOE = r'$\zeta_{\sigma OE}$ [S/pixel]'
LABEL_ZETA_TV = r'$\zeta_{TV}$'
LABEL_ZETA_P = r'$\zeta_{P}$ [\%]'
LABEL_ZETA_S = r'$\zeta_{S}$ [\%]'
LABEL_ZETA_TFMPAD = r'$\zeta_{TFMPAD}$ [\%/pixel]'
LABEL_ZETA_TFPPAD = r'$\zeta_{TFPPAD}$ [\%/rad]'
LABEL_SSIM = r'SSIM'
LABEL_EXECUTION_TIME = r'$t_{exe}$ [sec]'
LABEL_OBJECTIVE_FUNCTION = r'$f(\\chi, E_z^s)$'
LABEL_NUMBER_EVALUATIONS = 'Evaluations'
LABEL_NUMBER_ITERATIONS = 'Iterations'
LABEL_PATH = 'Path of Optimum Solution'

IMAGE_SIZE_SINGLE = (6., 5.)
IMAGE_SIZE_1x2 = (9., 4.) # 9 x 5
IMAGE_SIZE_2X2 = (9., 9.)

# Constant string for easier access of dictionary fields
NAME = 'name'
CONFIGURATION = 'configuration'
INPUT_FILENAME = 'input_filename'
INPUT_FILEPATH = 'input_filepath'
METHOD_NAME = 'method_name'
TOTAL_FIELD = 'total_field'
SCATTERED_FIELD = 'scattered_field'
REL_PERMITTIVITY = 'rel_permittivity'
CONDUCTIVITY = 'conductivity'
EXECUTION_TIME = 'execution_time'
NUMBER_EVALUATIONS = 'number_evaluations'
NUMBER_ITERATIONS = 'number_iterations'
OBJECTIVE_FUNCTION = 'objective_function'
RESIDUAL_NORM_ERROR = 'zeta_rn'
RESIDUAL_PAD_ERROR = 'zeta_rpad'
REL_PERMITTIVITY_PAD_ERROR = 'zeta_epad'
CONDUCTIVITY_AD_ERROR = 'zeta_sad'
TOTAL_VARIATION = 'zeta_tv'
POSITION_ERROR = 'zeta_p'
SHAPE_ERROR = 'zeta_s'
SSIM_ERROR = 'ssim'
REL_PERMITTIVITY_BACKGROUND_ERROR = 'zeta_ebe'
REL_PERMITTIVITY_OBJECT_ERROR = 'zeta_eoe'
CONDUCTIVITY_BACKGROUND_ERROR = 'zeta_sbe'
CONDUCTIVITY_OBJECT_ERROR = 'zeta_soe'
TOTALFIELD_MAGNITUDE_PAD = 'zeta_tfmpad'
TOTALFIELD_PHASE_AD = 'zeta_tfpad'
PERMITTIVITY = 'epsilon_r'
CONDUCTIVITY = 'sigma'
BOTH_PROPERTIES = 'both'
CONTRAST = 'contrast'
TOTAL_FIELD = 'total field'
PATH = 'path'

INDICATOR_SET = [RESIDUAL_NORM_ERROR, RESIDUAL_PAD_ERROR,
                 REL_PERMITTIVITY_PAD_ERROR, REL_PERMITTIVITY_BACKGROUND_ERROR,
                 REL_PERMITTIVITY_OBJECT_ERROR, CONDUCTIVITY_AD_ERROR,
                 CONDUCTIVITY_OBJECT_ERROR, CONDUCTIVITY_BACKGROUND_ERROR,
                 TOTALFIELD_MAGNITUDE_PAD, TOTALFIELD_PHASE_AD,
                 TOTAL_VARIATION, SHAPE_ERROR, POSITION_ERROR, EXECUTION_TIME,
                 OBJECTIVE_FUNCTION, NUMBER_EVALUATIONS, NUMBER_ITERATIONS,
                 SSIM_ERROR, PATH]

LABELS = {RESIDUAL_NORM_ERROR: r'$\zeta_{RN}$ (V/m)',
          RESIDUAL_PAD_ERROR: r'$\zeta_{RPAD}$ [%/sample]',
          REL_PERMITTIVITY_PAD_ERROR: r'$\zeta_{\epsilon PAD}$ [%/pixel]',
          REL_PERMITTIVITY_BACKGROUND_ERROR: r'$\zeta_{\epsilon BE}$ [%/pixel]',
          REL_PERMITTIVITY_OBJECT_ERROR: r'$\zeta_{\epsilon OE}$ [%/pixel]',
          CONDUCTIVITY_AD_ERROR: r'$\zeta_{\sigma AD}$ [S/m]',
          CONDUCTIVITY_OBJECT_ERROR: r'$\zeta_{\sigma OE}$ [S/m]',
          CONDUCTIVITY_BACKGROUND_ERROR: r'$\zeta_{\sigma BE}$ [S/m]',
          TOTALFIELD_MAGNITUDE_PAD: r'$\zeta_{TFMPAD}$ [%/pixel]',
          TOTALFIELD_PHASE_AD: r'$\zeta_{TFPAD}$ [rad/pixel]',
          TOTAL_VARIATION: r'$\zeta_{tv}$',
          SHAPE_ERROR: r'$\zeta_{S}$ [%]',
          POSITION_ERROR: r'$\zeta_{P}$ [%]',
          EXECUTION_TIME: 'Execution Time [sec]',
          OBJECTIVE_FUNCTION: 'Objective Function',
          NUMBER_EVALUATIONS: 'Evaluations',
          NUMBER_ITERATIONS: 'Iterations',
          SSIM_ERROR: 'SSIM',
          PATH: 'Path of Optimum Solution'}

TITLES = {RESIDUAL_NORM_ERROR: 'Residual Norm',
          RESIDUAL_PAD_ERROR: 'Residual PAD',
          REL_PERMITTIVITY_PAD_ERROR: 'Rel. Per. PAD',
          REL_PERMITTIVITY_BACKGROUND_ERROR: 'Background Rel. Per. PAD',
          REL_PERMITTIVITY_OBJECT_ERROR: 'Object Rel. Per. PAD',
          CONDUCTIVITY_AD_ERROR: 'Conductivity AD',
          CONDUCTIVITY_OBJECT_ERROR: 'Object Con. AD',
          CONDUCTIVITY_BACKGROUND_ERROR: 'Background Con. AD',
          TOTALFIELD_MAGNITUDE_PAD: 'To. Field Mag. PAD',
          TOTALFIELD_PHASE_AD: 'To. Field Phase AD',
          TOTAL_VARIATION: 'Total Variation',
          SHAPE_ERROR: 'Shape error',
          POSITION_ERROR: 'Position error',
          EXECUTION_TIME: 'Execution Time',
          OBJECTIVE_FUNCTION: 'Ob. Func. Evaluation',
          NUMBER_EVALUATIONS: 'Evaluations',
          NUMBER_ITERATIONS: 'Iterations',
          SSIM_ERROR: 'Structural Similarity',
          PATH: 'Path of Optimum Solution'}


class Result:
    r"""Storage and analysis class for electromagnetic inverse scattering results.

    This class provides comprehensive storage and analysis capabilities for results
    from electromagnetic inverse scattering algorithms. It stores reconstructed
    fields, material properties, and various error metrics, while providing
    sophisticated plotting and analysis tools.

    The class supports multiple error indicators for quantitative evaluation of
    reconstruction quality, including residual norm errors, percentage average
    deviation (PAD) errors, and shape/position analysis. It also provides
    extensive plotting capabilities for visualizing reconstructed images and
    convergence behavior.

    Attributes
    ----------
    name : str
        Unique identifier for the stored result, typically combining method,
        input data, and configuration names.
    method_name : str
        Name of the inverse scattering method that generated this result.
    configuration : :class:`configuration.Configuration`
        Problem configuration object containing frequency, geometry, and other
        parameters used in the reconstruction.
    total_field : :class:`numpy.ndarray`
        Reconstructed total electric field distribution in the investigation
        domain. Shape: (N_pixels, N_sources). Units: [V/m].
    scattered_field : :class:`numpy.ndarray`
        Computed scattered electric field at measurement points. 
        Shape: (N_measurements, N_sources). Units: [V/m].
    rel_permittivity : :class:`numpy.ndarray`
        Reconstructed relative permittivity distribution. 
        Shape: (N_x, N_y). Dimensionless.
    conductivity : :class:`numpy.ndarray`
        Reconstructed conductivity distribution. 
        Shape: (N_x, N_y). Units: [S/m].
    execution_time : float
        Total execution time for the reconstruction algorithm. Units: [sec].
    number_evaluations : int
        Number of objective function evaluations (relevant for stochastic methods).
    number_iterations : int
        Number of iterations performed by the reconstruction algorithm.
    objective_function : list
        History of objective function values throughout the iterative process.

    Error Metrics
    -------------
    zeta_rn : list
        Residual norm error: :math:`\\zeta_{RN} = ||\\mathbf{E}^s_{meas} - \\mathbf{E}^s_{comp}||_2`
    zeta_rpad : list
        Residual percentage average deviation: :math:`\\zeta_{RPAD} = \\frac{100}{N_m} \\sum_{i=1}^{N_m} \\frac{|E^s_{meas,i} - E^s_{comp,i}|}{|E^s_{meas,i}|}`
    zeta_epad : list
        Permittivity percentage average deviation: :math:`\\zeta_{\\epsilon PAD} = \\frac{100}{N_p} \\sum_{i=1}^{N_p} \\frac{|\\epsilon_{r,true,i} - \\epsilon_{r,rec,i}|}{\\epsilon_{r,true,i}}`
    zeta_ebe : list
        Background permittivity error for pixels outside the object region.
    zeta_eoe : list
        Object permittivity error for pixels inside the object region.
    zeta_sad : list
        Conductivity average deviation: :math:`\\zeta_{\\sigma AD} = \\frac{1}{N_p} \\sum_{i=1}^{N_p} |\\sigma_{true,i} - \\sigma_{rec,i}|`
    zeta_sbe : list
        Background conductivity error for pixels outside the object region.
    zeta_soe : list
        Object conductivity error for pixels inside the object region.
    zeta_tv : list
        Total variation: :math:`\\zeta_{TV} = \\sum_{i,j} |\\nabla \\chi_{i,j}|`
    zeta_p : list
        Position error: percentage difference in object centroid location.
    zeta_s : list
        Shape error: percentage difference in object area/shape.
    zeta_tfmpad : list
        Total field magnitude percentage average deviation.
    zeta_tfpad : list
        Total field phase average deviation.

    Examples
    --------
    >>> import result as rst
    >>> import numpy as np
    >>> 
    >>> # Create result object
    >>> result = rst.Result(name='csi_reconstruction')
    >>> 
    >>> # Store reconstruction data
    >>> result.rel_permittivity = np.random.rand(64, 64) * 2 + 1
    >>> result.conductivity = np.random.rand(64, 64) * 0.1
    >>> 
    >>> # Update with error metrics
    >>> result.zeta_rn.append(1e-3)
    >>> result.zeta_epad.append(15.5)
    >>> 
    >>> # Plot results
    >>> result.plot_map(image=rst.BOTH_PROPERTIES, show=True)
    >>> result.plot_convergence(show=True)
    >>> 
    >>> # Save results
    >>> result.save(file_path='/path/to/results/')

    Notes
    -----
    The class automatically computes various error metrics when the
    :meth:`update_error` method is called with appropriate input data
    and ground truth information. Error metrics are stored as lists
    to track evolution during iterative reconstruction processes.
    """
    def __init__(self, name=None, method_name=None,
                 configuration=None, scattered_field=None,
                 total_field=None, rel_permittivity=None,
                 conductivity=None, execution_time=None,
                 number_evaluations=None, objective_function=None,
                 number_iterations=None, import_filename=None,
                 import_filepath='', path=None):
        r"""Initialize a Result object for storing reconstruction results.

        Creates a new Result object to store electromagnetic inverse scattering
        reconstruction results. The object can be initialized with reconstruction
        data directly or loaded from a previously saved file.

        Parameters
        ----------
        name : str, optional
            Unique identifier for this result. Required if not loading from file.
            Typically combines method name, input data, and configuration info.
        method_name : str, optional
            Name of the reconstruction method that generated this result.
            Examples: 'CSI', 'DBIM', 'Born', 'Gauss-Newton'.
        configuration : :class:`configuration.Configuration`, optional
            Problem configuration object containing frequency, geometry, and
            material parameters used in the reconstruction.
        scattered_field : :class:`numpy.ndarray`, optional
            Computed scattered field at measurement points.
            Shape: (N_measurements, N_sources). Units: [V/m].
        total_field : :class:`numpy.ndarray`, optional
            Reconstructed total field in the investigation domain.
            Shape: (N_pixels, N_sources). Units: [V/m].
        rel_permittivity : :class:`numpy.ndarray`, optional
            Reconstructed relative permittivity map.
            Shape: (N_x, N_y). Dimensionless.
        conductivity : :class:`numpy.ndarray`, optional
            Reconstructed conductivity map.
            Shape: (N_x, N_y). Units: [S/m].
        execution_time : float, optional
            Total execution time for the reconstruction. Units: [sec].
        number_evaluations : int, optional
            Number of objective function evaluations (for stochastic methods).
        number_iterations : int, optional
            Number of iterations performed by the reconstruction algorithm.
        objective_function : list or float, optional
            Objective function value(s) recorded during reconstruction.
        import_filename : str, optional
            Name of file containing previously saved Result object.
            If provided, all other parameters are ignored.
        import_filepath : str, optional
            Directory path containing the import file. Default is current directory.

        Raises
        ------
        error.MissingInputError
            If name is None and import_filename is None.
        FileNotFoundError
            If import_filename is specified but file cannot be found.

        Examples
        --------
        >>> # Create new result object
        >>> result = Result(name='csi_test', method_name='CSI')
        >>> 
        >>> # Create with reconstruction data
        >>> result = Result(
        ...     name='dbim_reconstruction',
        ...     method_name='DBIM',
        ...     rel_permittivity=epsilon_r_map,
        ...     conductivity=sigma_map,
        ...     execution_time=45.2
        ... )
        >>> 
        >>> # Load from saved file
        >>> result = Result(import_filename='saved_result.pkl')

        Notes
        -----
        If import_filename is provided, the object is initialized from the
        saved file and all other parameters are ignored. Otherwise, a new
        object is created with the provided parameters.

        All error metric lists (zeta_*) are initialized as empty lists
        and can be populated using the :meth:`update_error` method.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            if name is None:
                raise error.MissingInputError('Results.__init__()', 'name')
            self.name = name
            self.method_name = method_name
            self.configuration = configuration
            self.total_field = total_field
            self.scattered_field = scattered_field
            self.rel_permittivity = rel_permittivity
            self.conductivity = conductivity
            self.execution_time = execution_time
            self.number_evaluations = number_evaluations
            self.number_iterations = number_iterations
            self.zeta_rn, self.zeta_rpad = list(), list()
            self.zeta_epad, self.zeta_sad = list(), list()
            self.zeta_tv, self.zeta_p, self.zeta_s = list(), list(), list()
            self.zeta_ebe, self.zeta_sbe = list(), list()
            self.zeta_eoe, self.zeta_soe = list(), list()
            self.zeta_tfmpad, self.zeta_tfpad = list(), list()
            self.ssim = list()
            if objective_function is None:
                self.objective_function = list()
            else:
                self.objective_function = objective_function
            if path is None:
                self.path = list()
            else:
                self.path = path

    def save(self, file_path=''):
        r"""Save the Result object to a pickle file.

        Serializes the complete Result object including all reconstruction data
        and error metrics to a pickle file for later loading and analysis.

        Parameters
        ----------
        file_path : str, optional
            Directory path where the file will be saved. The file is saved
            with the object's name as the filename. Default is current directory.

        Notes
        -----
        The saved file contains all reconstruction results, error metrics,
        and configuration information. The file can be loaded later using
        the :meth:`importdata` method or by initializing a new Result object
        with the import_filename parameter.

        Examples
        --------
        >>> result = Result(name='my_reconstruction')
        >>> result.rel_permittivity = epsilon_r_map
        >>> result.save(file_path='/path/to/results/')
        >>> # Creates file: /path/to/results/my_reconstruction
        """
        data = {
            NAME: self.name,
            CONFIGURATION: self.configuration,
            METHOD_NAME: self.method_name,
            TOTAL_FIELD: self.total_field,
            SCATTERED_FIELD: self.scattered_field,
            REL_PERMITTIVITY: self.rel_permittivity,
            CONDUCTIVITY: self.conductivity,
            EXECUTION_TIME: self.execution_time,
            NUMBER_EVALUATIONS: self.number_evaluations,
            NUMBER_ITERATIONS: self.number_iterations,
            OBJECTIVE_FUNCTION: self.objective_function,
            RESIDUAL_NORM_ERROR: self.zeta_rn,
            RESIDUAL_PAD_ERROR: self.zeta_rpad,
            REL_PERMITTIVITY_PAD_ERROR: self.zeta_epad,
            REL_PERMITTIVITY_BACKGROUND_ERROR: self.zeta_ebe,
            REL_PERMITTIVITY_OBJECT_ERROR: self.zeta_eoe,
            CONDUCTIVITY_AD_ERROR: self.zeta_sad,
            CONDUCTIVITY_BACKGROUND_ERROR: self.zeta_sbe,
            CONDUCTIVITY_OBJECT_ERROR: self.zeta_soe,
            TOTAL_VARIATION: self.zeta_tv,
            SHAPE_ERROR: self.zeta_s,
            POSITION_ERROR: self.zeta_p,
            TOTALFIELD_MAGNITUDE_PAD: self.zeta_tfmpad,
            TOTALFIELD_PHASE_AD: self.zeta_tfpad,
            PATH: self.path,
            SSIM_ERROR: self.ssim
        }

        with open(file_path + self.name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        r"""Load Result object data from a saved pickle file.

        Restores a previously saved Result object by loading all reconstruction
        data, error metrics, and configuration information from a pickle file.

        Parameters
        ----------
        file_name : str
            Name of the pickle file containing the saved Result object.
        file_path : str, optional
            Directory path containing the file. Default is current directory.

        Raises
        ------
        FileNotFoundError
            If the specified file cannot be found.
        pickle.UnpicklingError
            If the file cannot be unpickled or contains invalid data.

        Examples
        --------
        >>> result = Result(name='empty')
        >>> result.importdata('saved_result.pkl', '/path/to/results/')
        >>> print(result.name)  # Will show the name from saved file
        >>> print(result.rel_permittivity.shape)  # Access loaded data

        Notes
        -----
        This method completely overwrites the current object state with
        data from the saved file. Any existing data in the object is lost.
        """
        with open(file_path + file_name, 'rb') as datafile:
            data = pickle.load(datafile)
        self.name = data[NAME]
        self.configuration = data[CONFIGURATION]
        self.method_name = data[METHOD_NAME]
        self.total_field = data[TOTAL_FIELD]
        self.scattered_field = data[SCATTERED_FIELD]
        self.rel_permittivity = data[REL_PERMITTIVITY]
        self.conductivity = data[CONDUCTIVITY]
        self.execution_time = data[EXECUTION_TIME]
        self.number_evaluations = data[NUMBER_EVALUATIONS]
        self.number_iterations = data[NUMBER_ITERATIONS]
        self.objective_function = data[OBJECTIVE_FUNCTION]
        self.zeta_rn = data[RESIDUAL_NORM_ERROR]
        self.zeta_rpad = data[RESIDUAL_PAD_ERROR]
        self.zeta_epad = data[REL_PERMITTIVITY_PAD_ERROR]
        self.zeta_ebe = data[REL_PERMITTIVITY_BACKGROUND_ERROR]
        self.zeta_eoe = data[REL_PERMITTIVITY_OBJECT_ERROR]
        self.zeta_sad = data[CONDUCTIVITY_AD_ERROR]
        self.zeta_sbe = data[CONDUCTIVITY_BACKGROUND_ERROR]
        self.zeta_soe = data[CONDUCTIVITY_OBJECT_ERROR]
        self.zeta_tv = data[TOTAL_VARIATION]
        self.zeta_tfmpad = data[TOTALFIELD_MAGNITUDE_PAD]
        self.zeta_tfpad = data[TOTALFIELD_PHASE_AD]
        self.zeta_p = data[POSITION_ERROR]
        self.zeta_s = data[SHAPE_ERROR]
        self.ssim = data[SSIM_ERROR]
        self.path = data[PATH]

    def plot_map(self, axis=None, image=CONTRAST, groundtruth=None, title=None,
                 show=False, save=False, file_path='', file_format='eps',
                 fontsize=10, file_name=None, source=None, interpolation=None):
        r"""Plot reconstructed maps and field distributions.

        Creates visualizations of reconstructed material properties and field
        distributions with optional ground truth comparison. Supports various
        image types including permittivity, conductivity, contrast, and total field.

        Parameters
        ----------
        axis : :class:`matplotlib.axes.Axes` or :class:`numpy.ndarray`, optional
            Pre-existing axes for plotting. If None, new figure is created.
            For multiple plots, provide array of axes objects.
        image : str, optional
            Type of image to plot. Options: 'contrast', 'epsilon_r', 'sigma',
            'both', 'total field'. Default is 'contrast'.
        groundtruth : :class:`inputdata.InputData`, optional
            Ground truth data for comparison plotting. If provided, both
            ground truth and reconstructed images are displayed.
        title : str, list, or bool, optional
            Plot title(s). If list, separate titles for each subplot.
            If False, no titles are shown. Default is None (automatic titles).
        show : bool, optional
            If True, display the plot window. Default is False.
        save : bool, optional
            If True, save the figure to file. Default is False.
        file_path : str, optional
            Directory path for saving the figure. Default is current directory.
        file_format : str, optional
            File format for saving ('eps', 'png', 'pdf', etc.). Default is 'eps'.
        fontsize : int, optional
            Font size for labels and titles. Default is 10.
        file_name : str, optional
            Custom filename for saving. If None, uses object name. Default is None.
        source : int, list, or None, optional
            Source indices for total field plotting. If None, plots all sources.
            For int, plots single source. For list, plots multiple sources.
        interpolation : str, optional
            Interpolation method for imshow ('nearest', 'bilinear', etc.).
            Default is None (matplotlib default).

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            Figure object (only if show=False and save=False).
        ax : :class:`numpy.ndarray`
            Array of axes objects (only if show=False and save=False).

        Raises
        ------
        error.MissingAttributesError
            If required data (e.g., total_field) is missing for the requested plot.
        error.WrongValueInput
            If source index is out of range or axis array has wrong size.
        error.WrongTypeInput
            If source parameter has invalid type.

        Examples
        --------
        >>> # Plot contrast map
        >>> result.plot_map(image='contrast', show=True)
        >>> 
        >>> # Plot both properties with ground truth
        >>> result.plot_map(image='both', groundtruth=input_data,
        ...                  title=['Ground Truth', 'Reconstructed'], show=True)
        >>> 
        >>> # Plot total field for specific source
        >>> result.plot_map(image='total field', source=0, show=True)
        >>> 
        >>> # Save high-resolution figure
        >>> result.plot_map(image='epsilon_r', save=True, file_format='png',
        ...                  file_path='/path/to/figures/', fontsize=14)

        Notes
        -----
        - Spatial coordinates are normalized by background wavelength λ_b
        - Color bars are automatically added with appropriate units
        - Ground truth comparison creates side-by-side plots
        - Total field plotting supports multiple sources
        - Images are displayed with 'lower' origin (bottom-left corner)
        """
        xlabel, ylabel = r'x [$\lambda_b$]', r'y [$\lambda_b$]'
        xmin, xmax = cfg.get_bounds(self.configuration.Lx)
        ymin, ymax = cfg.get_bounds(self.configuration.Ly)
        extent = [xmin/self.configuration.lambda_b,
                  xmax/self.configuration.lambda_b,
                  ymin/self.configuration.lambda_b,
                  ymax/self.configuration.lambda_b]
        clb_epsilon_r = r'$\epsilon_r$'
        clb_sigma = r'$\sigma$ [S/m]'
        clb_contrast = r'$|\\chi|$'
        clb_total = r'$|E_z|$ [V/m]'

        if image == TOTAL_FIELD:
            if self.total_field is None:
                raise error.MissingAttributesError('Result', 'et')
            elif source is None:
                source = range(self.configuration)
            elif type(source) is int:
                if source >= self.configuration.NS:
                    raise error.WrongValueInput('Result.plot_map', 'source',
                                                '0 to %d', str(source))
                source = [source]
            elif type(source) is list:
                if any([s >= self.configuration.NS for s in
                        range(self.configuration.NS)]):
                    raise error.WrongValueInput('Result.plot_map', 'source',
                                                '0 to %d', str(source))
            else:
                raise error.WrongTypeInput('Result.plot_map', 'source',
                                           'None, int or int-list',
                                           str(type(source)))

        if groundtruth is not None:
            if image == BOTH_PROPERTIES:
                nfig = 4
            elif image == TOTAL_FIELD:
                nfig = 2*len(source)
            else:
                nfig = 2
        else:
            if image == BOTH_PROPERTIES:
                nfig = 2
            elif image == TOTAL_FIELD:
                nfig = len(source)
            else:
                nfig = 1

        if axis is None:
            fig, ax, _ = get_figure(nfig)
        else:
            if type(axis) is np.ndarray and axis.size != nfig:
                raise error.WrongValueInput('Result.plot_map', 'axis',
                                            '%d-numpy.ndarray' % nfig,
                                            '%d-numpy.ndarray' % axis.size)
            elif isinstance(axis, plt.Axes) and nfig != 1:
                raise error.WrongValueInput('Result.plot_map', 'axis',
                                            '%d-numpy.ndarray' % nfig,
                                            'matplotlib.axes.Axes')
            fig = plt.gcf()
            if type(axis) is not np.ndarray:
                ax = [axis]
            else:
                ax = axis

        if title == False:
            figure_title = ''
        elif type(title) is list:
            figure_title = title[0]
        
        ifig = 0
        if groundtruth is not None:
            if title is None or title is True:
                figure_title = 'Ground-Truth'
            if image == BOTH_PROPERTIES:
                groundtruth.draw(image=BOTH_PROPERTIES,
                                 axis=ax[:2],
                                 title=figure_title,
                                 show=False,
                                 save=False,
                                 fontsize=fontsize,)
                ifig = 2
            elif image != TOTAL_FIELD:
                groundtruth.draw(image=image,
                                 axis=ax[0],
                                 title=figure_title,
                                 show=False,
                                 save=False,
                                 fontsize=fontsize)
                ifig = 1
            elif image == TOTAL_FIELD:
                groundtruth.plot_total_field(axis=ax[:len(source)],
                                             source=source,
                                             figure_title=figure_title,
                                             fontsize=fontsize)
                ifig = len(source)

        if title is None or title == True:
            figure_title = 'Recovered'
        elif type(title) is str:
            figure_title = title
        elif type(title) is list:
            figure_title = title[1]

        if image == PERMITTIVITY:
            add_image(ax[ifig], self.rel_permittivity, figure_title,
                      clb_epsilon_r, bounds=extent, xlabel=xlabel,
                      ylabel=ylabel, fontsize=fontsize,
                      interpolation=interpolation)

        elif image == CONDUCTIVITY:
            add_image(ax[ifig], self.conductivity, figure_title, clb_sigma,
                      bounds=extent, xlabel=xlabel, ylabel=ylabel,
                      fontsize=fontsize, interpolation=interpolation)
            
        elif image == CONTRAST:
            X = cfg.get_contrast_map(epsilon_r=self.rel_permittivity,
                                     sigma=self.conductivity,
                                     configuration=self.configuration)
            add_image(ax[ifig], np.abs(X), figure_title, clb_contrast,
                      bounds=extent, xlabel=xlabel, ylabel=ylabel,
                      fontsize=fontsize, interpolation=interpolation)
        elif image == TOTAL_FIELD:
            for s in source:
                E = np.abs(
                    self.total_field[:, s].reshape(self.rel_permittivity.shape)
                )
                add_image(ax[ifig], E, figure_title, clb_total,
                          bounds=extent, xlabel=xlabel, ylabel=ylabel,
                          fontsize=fontsize, interpolation=interpolation)
                ifig += 1
        
        else:
            if title is None or title == True:
                figure_title = 'Recovered Rel. Per.'
            add_image(ax[ifig], self.rel_permittivity, figure_title,
                      clb_epsilon_r, bounds=extent, xlabel=xlabel,
                      ylabel=ylabel, fontsize=fontsize,
                      interpolation=interpolation)
            if title is None or title == True:
                figure_title = 'Recovered Con.'
            add_image(ax[ifig+1], self.conductivity, figure_title, clb_sigma,
                      bounds=extent, xlabel=xlabel, ylabel=ylabel,
                      fontsize=fontsize, interpolation=interpolation)

        if save:
            plt.tight_layout()
            if file_name is None:
                plt.savefig(file_path + self.name + '.' + file_format,
                            format=file_format)
            else:
                plt.savefig(file_path + file_name + '.' + file_format,
                            format=file_format)
        if show:
            plt.tight_layout()
            plt.show()
        if save:
            plt.close()
        elif not show and axis is None:
            return fig, ax

    def update_error(self, inputdata, scattered_field=None, total_field=None,
                     rel_permittivity=None, conductivity=None,
                     contrast=None, objective_function=None, optimum=None):
        r"""Compute and update error metrics for reconstruction quality assessment.

        Calculates various error indicators based on the specified input data
        indicators and updates the corresponding error metric lists. This method
        is typically called during iterative reconstruction to track convergence
        and quality metrics.

        Parameters
        ----------
        inputdata : :class:`inputdata.InputData`
            Input data object containing ground truth information and indicator
            specifications. The `indicators` attribute determines which error
            metrics will be computed.
        scattered_field : :class:`numpy.ndarray`, optional
            Computed scattered field for comparison with measured data.
            Shape: (N_measurements, N_sources). Units: [V/m].
            Required for residual norm and PAD error calculations.
        total_field : :class:`numpy.ndarray`, optional
            Reconstructed total field distribution.
            Shape: (N_pixels, N_sources). Units: [V/m].
            Required for total field error calculations.
        rel_permittivity : :class:`numpy.ndarray`, optional
            Reconstructed relative permittivity map.
            Shape: (N_x, N_y). Dimensionless.
            Required for permittivity error calculations.
        conductivity : :class:`numpy.ndarray`, optional
            Reconstructed conductivity map.
            Shape: (N_x, N_y). Units: [S/m].
            Required for conductivity error calculations.
        contrast : :class:`numpy.ndarray`, optional
            Reconstructed contrast function.
            Shape: (N_x, N_y). Complex-valued.
            Alternative to providing separate permittivity and conductivity.
        objective_function : float, optional
            Current objective function value for tracking convergence.

        Raises
        ------
        error.MissingInputError
            If required input for a specified indicator is missing.
        error.MissingAttributesError
            If required ground truth data is missing from inputdata.
        error.Error
            If array shapes are incompatible between ground truth and reconstruction.

        Notes
        -----
        The method computes the following error metrics based on inputdata.indicators:

        **Field-Based Errors:**
        - RESIDUAL_NORM_ERROR: :math:`\\zeta_{RN} = ||\\mathbf{E}^s_{meas} - \\mathbf{E}^s_{comp}||_2`
        - RESIDUAL_PAD_ERROR: :math:`\\zeta_{RPAD} = \\frac{100}{N_m} \\sum_{i=1}^{N_m} \\frac{|E^s_{meas,i} - E^s_{comp,i}|}{|E^s_{meas,i}|}`

        **Permittivity Errors:**
        - REL_PERMITTIVITY_PAD_ERROR: Percentage average deviation of permittivity
        - REL_PERMITTIVITY_BACKGROUND_ERROR: Background region permittivity error
        - REL_PERMITTIVITY_OBJECT_ERROR: Object region permittivity error
        - SSIM_ERROR: Structural Similarity Index Measure between true and reconstructed permittivity

        **Conductivity Errors:**
        - CONDUCTIVITY_AD_ERROR: Average deviation of conductivity
        - CONDUCTIVITY_BACKGROUND_ERROR: Background region conductivity error
        - CONDUCTIVITY_OBJECT_ERROR: Object region conductivity error

        **Shape and Position Errors:**
        - SHAPE_ERROR: Percentage difference in reconstructed object shape
        - POSITION_ERROR: Percentage difference in reconstructed object position

        **Regularization Metrics:**
        - TOTAL_VARIATION: Total variation of the reconstructed contrast

        **Total Field Errors:**
        - TOTALFIELD_MAGNITUDE_PAD: Magnitude percentage average deviation
        - TOTALFIELD_PHASE_AD: Phase average deviation

        Examples
        --------
        >>> import result as rst
        >>> 
        >>> # During iterative reconstruction
        >>> for iteration in range(max_iterations):
        ...     # ... perform reconstruction step ...
        ...     
        ...     # Update error metrics
        ...     result.update_error(
        ...         inputdata=input_data,
        ...         scattered_field=computed_scattered_field,
        ...         rel_permittivity=reconstructed_permittivity,
        ...         conductivity=reconstructed_conductivity,
        ...         objective_function=current_obj_value
        ...     )
        ...     
        ...     # Check convergence
        ...     if len(result.zeta_rn) > 1:
        ...         if result.zeta_rn[-1] < convergence_threshold:
        ...             break

        >>> # Access error history
        >>> print(f"Final residual norm: {result.zeta_rn[-1]:.3e}")
        >>> print(f"Final permittivity PAD: {result.zeta_epad[-1]:.2f}%")
        """
        if RESIDUAL_NORM_ERROR in inputdata.indicators:
            if scattered_field is None:
                raise error.MissingInputError('Result.update_error',
                                              'scattered_field')
            elif inputdata.scattered_field is None:
                raise error.MissingAttributesError('InputData', 'es')
            else:
                self.zeta_rn.append(compute_zeta_rn(inputdata.scattered_field,
                                                    scattered_field))

        if RESIDUAL_PAD_ERROR in inputdata.indicators:
            if scattered_field is None:
                raise error.MissingInputError('Result.update_error',
                                              'scattered_field')
            elif inputdata.scattered_field is None:
                raise error.MissingAttributesError('InputData', 'es')
            else:
                self.zeta_rpad.append(
                    compute_zeta_rpad(inputdata.scattered_field,
                                      scattered_field)
                )

        if REL_PERMITTIVITY_PAD_ERROR in inputdata.indicators:
            if rel_permittivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'contrast')
            elif inputdata.rel_permittivity is None:
                raise error.MissingAttributesError('InputData', 'epsilon_r')
            if rel_permittivity is None:
                epsilon_r = cfg.get_relative_permittivity(
                    contrast, self.configuration.epsilon_rb
                )
            else:
                epsilon_r = rel_permittivity
            if epsilon_r.shape != inputdata.rel_permittivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.rel_permittivity'"
                                  + " and 'epsilon_r' must have the same "
                                  + "shape.")
            self.zeta_epad.append(compute_zeta_epad(inputdata.rel_permittivity,
                                                    epsilon_r))
        
        if REL_PERMITTIVITY_OBJECT_ERROR in inputdata.indicators:
            if rel_permittivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'contrast')
            elif inputdata.rel_permittivity is None:
                raise error.MissingAttributesError('InputData', 'epsilon_r')
            if rel_permittivity is None:
                epsilon_r = cfg.get_relative_permittivity(
                    contrast, self.configuration.epsilon_rb
                )
            else:
                epsilon_r = rel_permittivity
            if epsilon_r.shape != inputdata.rel_permittivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.rel_permittivity'"
                                  + " and 'epsilon_r' must have the same "
                                  + "shape.")
            epsilon_rb = self.configuration.epsilon_rb
            self.zeta_eoe.append(compute_zeta_eoe(inputdata.rel_permittivity,
                                                  epsilon_r, epsilon_rb))

        if REL_PERMITTIVITY_BACKGROUND_ERROR in inputdata.indicators:
            if rel_permittivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'contrast')
            elif inputdata.rel_permittivity is None:
                raise error.MissingAttributesError('InputData', 'epsilon_r')
            if rel_permittivity is None:
                epsilon_r = cfg.get_relative_permittivity(
                    contrast, self.configuration.epsilon_rb
                )
            else:
                epsilon_r = rel_permittivity
            if epsilon_r.shape != inputdata.rel_permittivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.rel_permittivity'"
                                  + " and 'epsilon_r' must have the same "
                                  + "shape.")
            epsilon_rb = self.configuration.epsilon_rb
            self.zeta_ebe.append(compute_zeta_ebe(inputdata.rel_permittivity,
                                                  epsilon_r, epsilon_rb))
        
        if CONDUCTIVITY_AD_ERROR in inputdata.indicators:
            if conductivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'conductivity or contrast')
            elif inputdata.conductivity is None:
                raise error.MissingAttributesError('InputData', 'sigma')
            if conductivity is None:
                omega = 2*pi*self.configuration.f
                epsilon_rb = self.configuration.epsilon_rb
                sigma_b = self.configuration.sigma_b
                sigma = cfg.get_conductivity(contrast, omega, epsilon_rb,
                                             sigma_b)
            else:
                sigma = conductivity
            if sigma.shape != inputdata.conductivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.conductivity'"
                                  + " and 'sigma' must have the same "
                                  + "shape.")
            self.zeta_sad.append(compute_zeta_sad(inputdata.conductivity,
                                                  sigma))

        if CONDUCTIVITY_OBJECT_ERROR in inputdata.indicators:
            if conductivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'conductivity or contrast')
            elif inputdata.conductivity is None:
                raise error.MissingAttributesError('InputData', 'sigma')
            if conductivity is None:
                omega = 2*pi*self.configuration.f
                epsilon_rb = self.configuration.epsilon_rb
                sigma_b = self.configuration.sigma_b
                sigma = cfg.get_conductivity(contrast, omega, epsilon_rb,
                                             sigma_b)
            else:
                sigma = conductivity
            if sigma.shape != inputdata.conductivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.conductivity'"
                                  + " and 'sigma' must have the same "
                                  + "shape.")
            sigma_b = self.configuration.sigma_b
            self.zeta_soe.append(compute_zeta_soe(inputdata.conductivity,
                                                  sigma, sigma_b))
        
        if CONDUCTIVITY_BACKGROUND_ERROR in inputdata.indicators:
            if conductivity is None and contrast is None:
                raise error.MissingInputError('Result.update_error',
                                              'conductivity or contrast')
            elif inputdata.conductivity is None:
                raise error.MissingAttributesError('InputData', 'sigma')
            if conductivity is None:
                omega = 2*pi*self.configuration.f
                epsilon_rb = self.configuration.epsilon_rb
                sigma_b = self.configuration.sigma_b
                sigma = cfg.get_conductivity(contrast, omega, epsilon_rb,
                                             sigma_b)
            else:
                sigma = conductivity
            if sigma.shape != inputdata.conductivity.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.conductivity'"
                                  + " and 'sigma' must have the same "
                                  + "shape.")
            sigma_b = self.configuration.sigma_b
            self.zeta_sbe.append(compute_zeta_sbe(inputdata.conductivity,
                                                  sigma, sigma_b))

        if SHAPE_ERROR in inputdata.indicators:
            if (conductivity is None and rel_permittivity is None
                    and contrast is None):
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'conductivity or contrast')
            elif (inputdata.rel_permittivity is None
                    and inputdata.conductivity is None):
                raise error.MissingAttributesError('InputData',
                                                   'epsilon_r or sigma')
            Xo = cfg.get_contrast_map(epsilon_r=inputdata.rel_permittivity,
                                      sigma=inputdata.conductivity,
                                      configuration=self.configuration)
            if contrast is None:
                Xr = cfg.get_contrast_map(epsilon_r=rel_permittivity,
                                          sigma=conductivity,
                                          configuration=self.configuration)
            else:
                Xr = contrast
            self.zeta_s.append(compute_zeta_s(Xo, Xr))

        if POSITION_ERROR in inputdata.indicators:
            if (conductivity is None and rel_permittivity is None
                    and contrast is None):
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'conductivity or contrast')
            elif (inputdata.rel_permittivity is None
                    and inputdata.conductivity is None):
                raise error.MissingAttributesError('InputData',
                                                   'epsilon_r or sigma')
            Xo = cfg.get_contrast_map(epsilon_r=inputdata.rel_permittivity,
                                      sigma=inputdata.conductivity,
                                      configuration=self.configuration)
            if contrast is None:
                Xr = cfg.get_contrast_map(epsilon_r=rel_permittivity,
                                          sigma=conductivity,
                                          configuration=self.configuration)
            else:
                Xr = contrast
            self.zeta_p.append(compute_zeta_p(Xo, Xr))

        if TOTAL_VARIATION in inputdata.indicators:
            if (conductivity is None and rel_permittivity is None
                    and contrast is None):
                raise error.MissingInputError('Result.update_error',
                                              'rel_permittivity or '
                                              + 'conductivity or contrast')
            if contrast is None:
                X = cfg.get_contrast_map(epsilon_r=rel_permittivity,
                                          sigma=conductivity,
                                          configuration=self.configuration)
            else:
                X = contrast
            x, y = cfg.get_coordinates_ddomain(
                configuration=self.configuration, resolution=X.shape
            )
            self.zeta_tv.append(compute_zeta_tv(X, x, y))

        if TOTALFIELD_MAGNITUDE_PAD in inputdata.indicators:
            if total_field is None:
                raise error.MissingInputError('Result.update_error',
                                              'total_field')
            elif inputdata.total_field is None:
                raise error.MissingAttributesError('InputData', 'et')
            elif inputdata.total_field.shape != total_field.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.total_field' and"
                                  + " 'total_field' must have the same shape.")
            self.zeta_tfmpad.append(compute_zeta_tfmpad(inputdata.total_field,
                                                        total_field))

        if TOTALFIELD_PHASE_AD in inputdata.indicators:
            if total_field is None:
                raise error.MissingInputError('Result.update_error',
                                              'total_field')
            elif inputdata.total_field is None:
                raise error.MissingAttributesError('InputData', 'et')
            elif inputdata.total_field.shape != total_field.shape:
                raise error.Error("Result.update_error: "
                                  + "'inputdata.total_field' and"
                                  + " 'total_field' must have the same shape.")
            self.zeta_tfpad.append(compute_zeta_tfpad(inputdata.total_field,
                                                      total_field))

        if OBJECTIVE_FUNCTION in inputdata.indicators:
            if objective_function is None:
                raise error.MissingInputError('Result.update_error',
                                              'objective_function')
            if (type(objective_function) is list
                or isinstance(objective_function, np.ndarray)):
                self.objective_function = objective_function.copy()
            else:
                self.objective_function.append(objective_function)
        
        if SSIM_ERROR in inputdata.indicators:
            Xo = cfg.get_contrast_map(epsilon_r=inputdata.rel_permittivity,
                                      sigma=inputdata.conductivity,
                                      configuration=self.configuration)
            if contrast is None:
                Xr = cfg.get_contrast_map(epsilon_r=rel_permittivity,
                                          sigma=conductivity,
                                          configuration=self.configuration)
            else:
                Xr = contrast
            self.ssim.append(compute_ssim(Xo, Xr))

        if PATH in inputdata.indicators and optimum is not None:
            self.path.append(optimum)

    def last_error_message(self, pre_message=None):
        r"""Generate a formatted summary of the latest error metrics.

        Creates a comprehensive text summary of the most recent error indicator
        values for display or logging purposes. This method is typically used
        to provide concise status updates during iterative reconstruction processes.

        Parameters
        ----------
        pre_message : str, optional
            Custom prefix message to prepend to the error summary.
            If None, uses default prefix "Indicators:".

        Returns
        -------
        str
            Formatted string containing the latest values of all available
            error indicators. Format: "Indicators: Residual norm: 1.23e-4,
            Residual PAD: 12.34%, Rel. Per. PAD: 8.56%, ..."

        Notes
        -----
        Only error indicators that have been computed (non-empty lists) are
        included in the summary. The formatting is optimized for readability
        with appropriate precision for each metric type:
        
        - Residual norm errors: scientific notation with 3 decimal places
        - Percentage errors: fixed point with 2 decimal places + % symbol
        - Regularization metrics: fixed point with 2 decimal places

        Examples
        --------
        >>> result = Result(name='test')
        >>> result.zeta_rn = [1e-3, 5e-4, 1e-4]
        >>> result.zeta_epad = [20.5, 15.2, 12.8]
        >>> print(result.last_error_message())
        Indicators: Residual norm: 1.000e-04, Rel. Per. PAD: 12.80%,
        
        >>> # With custom prefix
        >>> print(result.last_error_message("Final results:"))
        Final results: Residual norm: 1.000e-04, Rel. Per. PAD: 12.80%,
        """
        if pre_message is not None:
            message = pre_message
        else:
            message = 'Indicators:'

        if self.zeta_rn is not None and len(self.zeta_rn) != 0:
            message += ' Residual norm: %.3e,' % self.zeta_rn[-1]

        if self.zeta_rpad is not None and len(self.zeta_rpad) != 0:
            message += ' Residual PAD: %.2f%%,' % self.zeta_rpad[-1]

        if self.zeta_epad is not None and len(self.zeta_epad) != 0:
            message += ' Rel. Per. PAD: %.2f%%,' % self.zeta_epad[-1]

        if self.zeta_eoe is not None and len(self.zeta_eoe) != 0:
            message += ' Rel. Per. Ob.: %.2f%%,' % self.zeta_eoe[-1]

        if self.zeta_ebe is not None and len(self.zeta_ebe) != 0:
            message += ' Rel. Per. Back.: %.2f%%,' % self.zeta_ebe[-1]

        if self.zeta_sad is not None and len(self.zeta_sad) != 0:
            message += ' Con. AD: %.3e,' % self.zeta_sad[-1]

        if self.zeta_soe is not None and len(self.zeta_soe) != 0:
            message += ' Con. Ob.: %.3e,' % self.zeta_soe[-1]

        if self.zeta_sbe is not None and len(self.zeta_sbe) != 0:
            message += ' Con. Back.: %.3e,' % self.zeta_sbe[-1]

        if self.zeta_s is not None and len(self.zeta_s) != 0:
            message += ' Shape: %.2f,' % self.zeta_s[-1]

        if self.zeta_p is not None and len(self.zeta_p) != 0:
            message += ' Position: %.2f,' % self.zeta_p[-1]

        if self.zeta_tv is not None and len(self.zeta_tv) != 0:
            message += ' Total Variation: %.2f,' % self.zeta_tv[-1]

        if self.zeta_tfmpad is not None and len(self.zeta_tfmpad) != 0:
            message += ' To. Field Mag. PAD: %.2f%%,' % self.zeta_tfmpad[-1]

        if self.zeta_tfpad is not None and len(self.zeta_tfpad) != 0:
            message += ' To. Field Phase AD: %.2f%%,' % self.zeta_tfpad[-1]
        
        if self.objective_function is not None and len(self.objective_function) != 0:
            message += ' Ob. Func.: %.3e,' % self.objective_function[-1]

        if self.path is not None and len(self.path) != 0:
            message += ' Optimum solution: ' + str(self.path[-1]) + ','

        if self.ssim is not None and len(self.ssim) != 0:
            message += ' SSIM: %.3f,' % self.ssim[-1]

        return message

    def valid_indicators(self):
        r"""Identify which error indicators have been computed and contain data.

        Examines all error metric attributes to determine which indicators have
        been calculated and contain valid data. This method is used internally
        for plotting and analysis functions to determine which metrics are
        available for display.

        Returns
        -------
        list of str
            List of indicator names (constants) that have been computed and
            contain at least one data point. Possible indicators include:
            
            - RESIDUAL_NORM_ERROR: Residual norm error
            - RESIDUAL_PAD_ERROR: Residual percentage average deviation
            - REL_PERMITTIVITY_PAD_ERROR: Permittivity PAD error
            - REL_PERMITTIVITY_OBJECT_ERROR: Object permittivity error
            - REL_PERMITTIVITY_BACKGROUND_ERROR: Background permittivity error
            - CONDUCTIVITY_AD_ERROR: Conductivity average deviation
            - CONDUCTIVITY_OBJECT_ERROR: Object conductivity error
            - CONDUCTIVITY_BACKGROUND_ERROR: Background conductivity error
            - SHAPE_ERROR: Shape reconstruction error
            - POSITION_ERROR: Position reconstruction error
            - TOTAL_VARIATION: Total variation regularization metric
            - TOTALFIELD_MAGNITUDE_PAD: Total field magnitude PAD
            - TOTALFIELD_PHASE_AD: Total field phase average deviation
            - OBJECTIVE_FUNCTION: Objective function values

        Notes
        -----
        This method checks both that the attribute exists (is not None) and
        that it contains data (length > 0). Empty lists are considered invalid
        indicators and are excluded from the results.

        Examples
        --------
        >>> result = Result(name='test')
        >>> result.zeta_rn = [1e-3, 5e-4]
        >>> result.zeta_epad = [20.5, 15.2]
        >>> result.zeta_sad = []  # Empty list
        >>> 
        >>> indicators = result.valid_indicators()
        >>> print(indicators)
        ['zeta_rpad', 'zeta_epad']  # Only non-empty indicators
        >>> 
        >>> # Use with plotting
        >>> result.plot_convergence(indicators=result.valid_indicators())
        """
        indicators = []
        if self.zeta_rn is not None and len(self.zeta_rn) != 0:
            indicators.append(RESIDUAL_PAD_ERROR)
        if self.zeta_rpad is not None and len(self.zeta_rpad) != 0:
            indicators.append(RESIDUAL_NORM_ERROR)
        if self.zeta_epad is not None and len(self.zeta_epad) != 0:
            indicators.append(REL_PERMITTIVITY_PAD_ERROR)
        if self.zeta_eoe is not None and len(self.zeta_eoe) != 0:
            indicators.append(REL_PERMITTIVITY_OBJECT_ERROR)
        if self.zeta_ebe is not None and len(self.zeta_ebe) != 0:
            indicators.append(REL_PERMITTIVITY_BACKGROUND_ERROR)
        if self.zeta_sad is not None and len(self.zeta_sad) != 0:
            indicators.append(CONDUCTIVITY_AD_ERROR)
        if self.zeta_soe is not None and len(self.zeta_soe) != 0:
            indicators.append(CONDUCTIVITY_OBJECT_ERROR)
        if self.zeta_sbe is not None and len(self.zeta_sbe) != 0:
            indicators.append(CONDUCTIVITY_BACKGROUND_ERROR)
        if self.zeta_s is not None and len(self.zeta_s) != 0:
            indicators.append(SHAPE_ERROR)
        if self.zeta_p is not None and len(self.zeta_p) != 0:
            indicators.append(POSITION_ERROR)
        if self.zeta_tv is not None and len(self.zeta_tv) != 0:
            indicators.append(TOTAL_VARIATION)
        if self.zeta_tfmpad is not None and len(self.zeta_tfmpad) != 0:
            indicators.append(TOTALFIELD_MAGNITUDE_PAD)
        if self.zeta_tfpad is not None and len(self.zeta_tfpad) != 0:
            indicators.append(TOTALFIELD_PHASE_AD)
        if (self.objective_function is not None
                and len(self.objective_function) != 0):
            indicators.append(OBJECTIVE_FUNCTION)
        if (self.path is not None and len(self.path) != 0):
            indicators.append(PATH)
        if self.ssim is not None and len(self.ssim) != 0:
            indicators.append(SSIM_ERROR)
        return indicators

    def plot_convergence(self, axis=None, indicators=None, show=False,
                         file_name=None, file_path='', file_format='eps',
                         fontsize=10, title=None, style='--*', yscale=None,
                         markersize=None):
        r"""Plot convergence curves for error indicators over iterations.

        Creates line plots showing the evolution of error metrics during the
        iterative reconstruction process. Supports multiple indicators in
        separate subplots with customizable styling and formatting.

        Parameters
        ----------
        axis : :class:`matplotlib.axes.Axes` or :class:`numpy.ndarray`, optional
            Pre-existing axes for plotting. If None, new figure is created.
            For multiple indicators, provide array of axes objects.
        indicators : str, list of str, or None, optional
            Error indicators to plot. If None, plots all valid indicators.
            If string, plots single indicator. If list, plots multiple indicators.
            Valid indicators are returned by :meth:`valid_indicators`.
        show : bool, optional
            If True, display the plot window. Default is False.
        file_name : str, optional
            Filename for saving the figure (without extension). If None,
            figure is not saved. Default is None.
        file_path : str, optional
            Directory path for saving the figure. Default is current directory.
        file_format : str, optional
            File format for saving ('eps', 'png', 'pdf', etc.). Default is 'eps'.
        fontsize : int, optional
            Font size for labels and titles. Default is 10.
        title : str, list of str, or None, optional
            Plot title(s). If None, uses standard titles. If string, uses
            same title for all plots. If list, uses separate titles for each plot.
        style : str, optional
            Line style for plots (matplotlib format). Default is '--*'.
        yscale : str, list of str, or None, optional
            Y-axis scale ('linear', 'log', 'symlog', etc.). If None, uses
            linear scale. If string, uses same scale for all plots. If list,
            uses separate scales for each plot.
        markersize : float, optional
            Size of markers in the plot. Default is None (matplotlib default).

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            Figure object (only if axis is None and show=False and file_name=None).
        ax : :class:`numpy.ndarray`
            Array of axes objects (only if axis is None and show=False and file_name=None).

        Raises
        ------
        error.WrongValueInput
            If provided axis array size doesn't match number of indicators.

        Examples
        --------
        >>> # Plot all available indicators
        >>> result.plot_convergence(show=True)
        >>> 
        >>> # Plot specific indicators with logarithmic scale
        >>> result.plot_convergence(
        ...     indicators=['zeta_rn', 'zeta_epad'],
        ...     yscale='log',
        ...     show=True
        ... )
        >>> 
        >>> # Save convergence plot
        >>> result.plot_convergence(
        ...     file_name='convergence_plot',
        ...     file_path='/path/to/figures/',
        ...     file_format='png'
        ... )
        >>> 
        >>> # Custom styling
        >>> result.plot_convergence(
        ...     style='-o',
        ...     markersize=6,
        ...     fontsize=12,
        ...     title='Algorithm Convergence',
        ...     show=True
        ... )

        Notes
        -----
        - X-axis represents iteration numbers starting from 1
        - Each indicator is plotted in a separate subplot
        - Grid is enabled for all plots to improve readability
        - Automatic titles are generated based on indicator types
        - Supports both linear and logarithmic y-axis scaling
        """
        if indicators is None:
            indicators = self.valid_indicators()
        elif type(indicators) is str:
            indicators = [indicators]
        nplots = len(indicators)
        if axis is None:
            fig, axis, _ = get_figure(nplots)
            given_axis = False
        else:
            if nplots > 1 and axis.size != nplots:
                raise error.WrongValueInput('Result.plot_convergence', 'axis',
                                            '%dd-ndarray' %nplots,
                                            '%dd' % axis.size)
            fig = plt.gcf()
            given_axis = True

        for n in range(nplots):
            y = getattr(self, indicators[n])
            x = np.arange(len(y))+1
            if title is None:
                figtitle = TITLES[indicators[n]]
            elif type(title) is str:
                figtitle = title
            elif type(title) is list:
                figtitle = title[n]
            else:
                figtitle = None
            if yscale is None:
                figyscale = None
            elif type(yscale) is str:
                figyscale = yscale
            elif type(yscale) is list:
                figyscale = yscale[n]
            add_plot(axis[n], y, x=x, title=figtitle,
                     ylabel=indicator_label(indicators[n]), style=style,
                     yscale=figyscale, fontsize=fontsize,
                     markersize=markersize)

        plt.tight_layout()

        if file_name is not None:
            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
        if show:
            plt.show()
        if file_name is not None:
            plt.close()
        if not given_axis:
            return fig, axis

    def final_value(self, indicator):
        r"""Get the final (most recent) value of a specific error indicator.

        Retrieves the last computed value from the specified error indicator's
        history. This method is useful for accessing the final reconstruction
        quality metrics after algorithm completion.

        Parameters
        ----------
        indicator : str
            Name of the error indicator to retrieve. Must be a valid indicator
            constant such as 'zeta_rn', 'zeta_epad', 'zeta_sad', etc.
            Use :meth:`valid_indicators` to see available indicators.

        Returns
        -------
        float
            The final (most recent) value of the specified indicator.
            For list-type indicators, returns the last element.
            For scalar indicators, returns the scalar value.

        Raises
        ------
        error.WrongTypeInput
            If indicator is not a string.
        error.WrongValueInput
            If indicator is not a valid indicator name.

        Examples
        --------
        >>> result = Result(name='test')
        >>> result.zeta_rn = [1e-2, 5e-3, 1e-3]
        >>> result.zeta_epad = [25.0, 15.0, 8.5]
        >>> 
        >>> # Get final residual norm error
        >>> final_rn = result.final_value('zeta_rn')
        >>> print(f"Final residual norm: {final_rn:.3e}")
        Final residual norm: 1.000e-03
        >>> 
        >>> # Get final permittivity error
        >>> final_epad = result.final_value('zeta_epad')
        >>> print(f"Final permittivity PAD: {final_epad:.1f}%")
        Final permittivity PAD: 8.5%
        >>> 
        >>> # Error handling
        >>> try:
        ...     result.final_value('invalid_indicator')
        ... except error.WrongValueInput as e:
        ...     print(f"Error: {e}")

        Notes
        -----
        This method assumes that the indicator has been computed and contains
        at least one value. If the indicator list is empty, IndexError will
        be raised when trying to access the last element.
        """
        if type(indicator) is not str:
            raise error.WrongTypeInput('Result.final_value', 'indicator',
                                       'str', str(type(indicator)))
        elif not check_indicator(indicator):
            raise error.WrongValueInput('Result.plot', 'indicator',
                                        INDICATOR_SET, indicator)
        
        output = getattr(self, indicator)
        if type(output) is list or type(output) is np.ndarray:
            return output[-1]
        else:
            return output

    def copy(self, new=None):
        r"""Create a deep copy of the Result object or copy data from another Result.

        Creates a complete deep copy of the current Result object with all
        reconstruction data and error metrics, or copies data from another
        Result object into the current one. This method is useful for creating
        backups or comparing different reconstruction results.

        Parameters
        ----------
        new : :class:`Result`, optional
            If provided, copies data from this Result object into the current
            object, overwriting existing data. If None, creates a new Result
            object as a copy of the current one.

        Returns
        -------
        :class:`Result` or None
            If new is None, returns a new Result object containing a deep copy
            of all data. If new is provided, returns None and modifies the
            current object in-place.

        Examples
        --------
        >>> # Create a backup copy
        >>> result = Result(name='original')
        >>> result.zeta_rn = [1e-2, 5e-3, 1e-3]
        >>> result.rel_permittivity = np.random.rand(64, 64)
        >>> 
        >>> backup = result.copy()
        >>> print(backup.name)  # 'original'
        >>> print(len(backup.zeta_rn))  # 3
        >>> 
        >>> # Copy data from another result
        >>> result2 = Result(name='experiment2')
        >>> result2.zeta_rn = [2e-2, 1e-2, 5e-3]
        >>> 
        >>> result.copy(result2)  # Copies result2 data into result
        >>> print(result.name)  # 'experiment2'
        >>> print(result.zeta_rn)  # [2e-2, 1e-2, 5e-3]

        Notes
        -----
        - Deep copy is performed on all array data and error metric lists
        - Configuration objects are also copied to avoid reference sharing
        - When copying from another Result (new parameter), all existing
          data in the current object is overwritten
        - The copy includes all reconstruction data, error metrics, and
          metadata such as execution time and iteration counts
        """
        if new is None:
            new = Result(
                name=self.name, method_name=self.method_name,
                configuration=self.configuration,
                scattered_field=cp.deepcopy(self.scattered_field),
                total_field=cp.deepcopy(self.total_field),
                rel_permittivity=cp.deepcopy(self.rel_permittivity),
                conductivity=cp.deepcopy(self.conductivity),
                execution_time=self.execution_time,
                number_evaluations=self.number_evaluations,
                number_iterations=self.number_iterations
            )
            new.zeta_rn = cp.deepcopy(self.zeta_rn)
            new.zeta_rpad = cp.deepcopy(self.zeta_rpad)
            new.zeta_epad = cp.deepcopy(self.zeta_epad)
            new.zeta_sad = cp.deepcopy(self.zeta_sad)
            new.zeta_tv = cp.deepcopy(self.zeta_tv)
            new.zeta_p = cp.deepcopy(self.zeta_p)
            new.zeta_s = cp.deepcopy(self.zeta_s)
            new.zeta_ebe = cp.deepcopy(self.zeta_ebe)
            new.zeta_sbe = cp.deepcopy(self.zeta_sbe)
            new.zeta_eoe = cp.deepcopy(self.zeta_eoe)
            new.zeta_soe = cp.deepcopy(self.zeta_soe)
            new.zeta_tfmpad = cp.deepcopy(self.zeta_tfmpad)
            new.zeta_tfpad = cp.deepcopy(self.zeta_tfpad)
            new.objective_function = cp.deepcopy(self.objective_function)
            new.path = cp.deepcopy(self.path)
            new.ssim = cp.deepcopy(self.ssim)
            return new
        else:
            self.name = new.name
            self.method_name = new.method_name
            self.configuration = new.configuration.copy()
            self.scattered_field = np.copy(new.scattered_field)
            self.total_field = np.copy(new.total_field)
            self.rel_permittivity = np.copy(new.rel_permittivity)
            self.conductivity = np.copy(new.conductivity)
            self.execution_time = new.execution_time
            self.number_evaluations = new.number_evaluations
            self.objective_function = cp.deepcopy(new.objective_function)
            self.number_iterations = new.number_iterations
            self.zeta_rn = cp.deepcopy(new.zeta_rn)
            self.zeta_rpad = cp.deepcopy(new.zeta_rpad)
            self.zeta_epad = cp.deepcopy(new.zeta_epad)
            self.zeta_ebe = cp.deepcopy(new.zeta_ebe)
            self.zeta_eoe = cp.deepcopy(new.zeta_eoe)
            self.zeta_sad = cp.deepcopy(new.zeta_sad)
            self.zeta_sbe = cp.deepcopy(new.zeta_sbe)
            self.zeta_soe = cp.deepcopy(new.zeta_soe)
            self.zeta_s = cp.deepcopy(new.zeta_s)
            self.zeta_p = cp.deepcopy(new.zeta_p)
            self.zeta_tfmpad = cp.deepcopy(new.zeta_tfmpad)
            self.zeta_tfpad = cp.deepcopy(new.zeta_tfpad)
            self.zeta_tv = cp.deepcopy(new.zeta_tv)
            self.path = cp.deepcopy(new.path)
            self.ssim = cp.deepcopy(new.ssim)

    def __str__(self):
        r"""Return a comprehensive string representation of the Result object.

        Generates a detailed text summary of the Result object including all
        reconstruction data, error metrics, and algorithm performance statistics.
        This method provides a human-readable overview of the complete result.

        Returns
        -------
        str
            Multi-line string containing:
            - Result name and configuration
            - Field data dimensions and resolution
            - Material property map resolutions
            - Execution time and performance metrics
            - Complete error metric histories with formatting
            - Algorithm iteration and evaluation counts

        Examples
        --------
        >>> result = Result(name='csi_experiment')
        >>> result.rel_permittivity = np.random.rand(64, 64)
        >>> result.zeta_rn = [1e-2, 5e-3, 1e-3]
        >>> result.execution_time = 45.2
        >>> 
        >>> print(result)
        Results name: csi_experiment
        Configuration: test_config
        Relative Permit. map resolution: 64x64
        Residual norm error: [1.00e-02, 5.00e-03, 1.00e-03]
        Execution time: 45.20 [sec]

        Notes
        -----
        - Error metrics are formatted with appropriate precision
        - Long error histories (>30 values) are truncated with ellipsis
        - Field dimensions show both measurement and source sample counts
        - Material property maps show spatial resolution
        - Performance metrics include execution time and iteration counts
        """
        message = 'Results name: ' + self.name
        message += '\nConfiguration: ' + self.configuration.name
        if self.scattered_field is not None:
            message = (message + '\nScattered field - measurement samples: %d'
                       % self.scattered_field.shape[0]
                       + '\nScattered field - source samples: %d'
                       % self.scattered_field.shape[1])
        if self.total_field is not None:
            message = (message + '\nTotal field - measurement samples: %d'
                       % self.total_field.shape[0]
                       + '\nTotal field - source samples: %d'
                       % self.total_field.shape[1])
        if self.rel_permittivity is not None:
            if self.rel_permittivity.ndim == 1:
                message += ('\nSolution: ' + str(self.rel_permittivity))
            elif self.rel_permittivity.ndim == 2:
                message += ('\nRelative Permit. map resolution: %dx'
                            % self.rel_permittivity.shape[0] + '%d'
                            % self.rel_permittivity.shape[1])
        if self.conductivity is not None:
            message += ('\nConductivity map resolution: %dx'
                        % self.conductivity.shape[0]
                        + '%d' % self.conductivity.shape[1])
        if self.execution_time is not None:
            print('Execution time: %.2f [sec]' % self.execution_time)
        if len(self.zeta_rn) > 0:
            if len(self.zeta_rn) == 1:
                info = '%.3e' % self.zeta_rn[0]
            elif len(self.zeta_rn) > 30:
                info = '%.3e' % self.zeta_rn[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_rn) + ']')
            message = message + '\nResidual norm error: ' + info
        if len(self.zeta_rpad) > 0:
            if len(self.zeta_rpad) == 1:
                info = '%.2f%%' % self.zeta_rpad[0]
            elif len(self.zeta_rpad) > 30:
                info = '%.2f%%' % self.zeta_rpad[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_rpad) + ']')
            message = message + '\nPercent. Aver. Devi. of Residuals: ' + info
        if len(self.zeta_epad) > 0:
            if len(self.zeta_epad) == 1:
                info = '%.2f%%' % self.zeta_epad[0]
            if len(self.zeta_epad) > 30:
                info = '%.2f%%' % self.zeta_epad[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_epad) + ']')
            message = (message + '\nPercent. Aver. Devi. of Rel. Permittivity:'
                       + ' ' + info)
        if len(self.zeta_sad) > 0:
            if len(self.zeta_sad) == 1:
                info = '%.3e' % self.zeta_sad[0]
            elif len(self.zeta_sad) > 30:
                info = '%.3e' % self.zeta_sad[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_sad) + ']')
            message = (message + '\nAver. Devi. of Conductivity: '
                       + info)
        if len(self.zeta_tv) > 0:
            if len(self.zeta_tv) == 1:
                info = '%.3e' % self.zeta_tv[0]
            elif len(self.zeta_tv) > 30:
                info = '%.3e' % self.zeta_tv[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_tv) + ']')
            message = message + '\nTotal Variation: ' + info
        if len(self.zeta_ebe) > 0:
            if len(self.zeta_ebe) == 1:
                info = '%.2f%%' % self.zeta_ebe[0]
            elif len(self.zeta_ebe) > 30:
                info = '%.2f%%' % self.zeta_ebe[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_ebe) + ']')
            message = message + '\nBackground Rel. Permit. error: ' + info
        if len(self.zeta_sbe) > 0:
            if len(self.zeta_sbe) == 1:
                info = '%.3e' % self.zeta_sbe[0]
            elif len(self.zeta_sbe) > 30:
                info = '%.3e' % self.zeta_sbe[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_sbe) + ']')
            message = message + '\nBackground Conductivity error: ' + info
        if len(self.zeta_eoe) > 0:
            if len(self.zeta_eoe) == 1:
                info = '%.2f%%' % self.zeta_eoe[0]
            elif len(self.zeta_eoe) > 30:
                info = '%.2f%%' % self.zeta_eoe[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_eoe) + ']')
            message = message + '\nObject Rel. Permit. error: ' + info
        if len(self.zeta_soe) > 0:
            if len(self.zeta_soe) == 1:
                info = '%.3e' % self.zeta_soe[0]
            elif len(self.zeta_soe) > 30:
                info = '%.3e' % self.zeta_soe[-1]
            else:
                info = '[' + str(', '.join('{:.3e}'.format(i)
                                           for i in self.zeta_soe) + ']')
            message = message + '\nObject Conduc. error: ' + info
        if len(self.zeta_tfmpad) > 0:
            if len(self.zeta_tfmpad) == 1:
                info = '%.2f%%' % self.zeta_tfmpad[0]
            elif len(self.zeta_tfmpad) > 30:
                info = '%.2f%%' % self.zeta_tfmpad[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_tfmpad) + ']')
            message = (message + '\nTotal Field Mag. Per. Aver. Devi. error: '
                       + info)
        if len(self.zeta_tfpad) > 0:
            if len(self.zeta_tfpad) == 1:
                info = '%.2f%%' % self.zeta_tfpad[0]
            elif len(self.zeta_tfpad) > 30:
                info = '%.2f%%' % self.zeta_tfpad[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_tfpad) + ']')
            message = (message + '\nTotal Field Phase Aver. Devi. error:'
                       + ' ' + info)
        if len(self.zeta_p) > 0:
            if len(self.zeta_p) == 1:
                info = '%.2f%%' % self.zeta_p[0]
            elif len(self.zeta_p) > 30:
                info = '%.2f%%' % self.zeta_p[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_p) + ']')
            message += ('\nPosition error: ' + info)
        if len(self.zeta_s) > 0:
            if len(self.zeta_s) == 1:
                info = '%.2f%%' % self.zeta_s[0]
            elif len(self.zeta_s) > 30:
                info = '%.2f%%' % self.zeta_s[-1]
            else:
                info = '[' + str(', '.join('{:.2f}%'.format(i)
                                           for i in self.zeta_s) + ']')
            message += ('\nShape error: ' + info)
        if type(self.objective_function) is float:
            message += ('\nObjective function evaluation: %.3e'
                        % self.objective_function)
        elif len(self.objective_function) > 0:
            if len(self.objective_function) == 1:
                info = '%.3e' % self.objective_function[0]
            if len(self.objective_function) > 30:
                info = '%.3e' % self.objective_function[-1]
            else:
                info = '[' + str(', '.join('{:.2e}'.format(i)
                                           for i in self.objective_function)
                                 + ']')
            message += '\nObjective function:' + ' ' + info
        if self.number_iterations is not None:
            message += '\nNumber of iterations: %d' % self.number_iterations
        if self.number_evaluations is not None:
            message += '\nNumber of evaluations: %d' % self.number_evaluations
        if self.path is not None and len(self.path) > 0:
            message += '\nOptimum solution: ' + str(self.path[-1])
        if self.ssim is not None and len(self.ssim) > 0:
            if len(self.ssim) == 1:
                info = '%.3f' % self.ssim[0]
            elif len(self.ssim) > 30:
                info = '%.3f' % self.ssim[-1]
            else:
                info = '[' + str(', '.join('{:.3f}'.format(i)
                                           for i in self.ssim) + ']')
            message += '\nSSIM: ' + info

        return message


def add_image(axes, image, title, colorbar_name, bounds=(-1., 1., -1., 1.),
              origin='lower', xlabel=XLABEL_STANDARD, ylabel=YLABEL_STANDARD,
              aspect='equal', interpolation=None, fontsize=10):
    r"""Add standardized image plot to matplotlib axes.

    Creates a standardized image plot with proper scaling, colorbar, and labels
    for electromagnetic inverse scattering visualizations. Handles both real
    and complex-valued images appropriately.

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The axes object where the image will be plotted.
    image : :class:`numpy.ndarray`
        2D array containing the image data. If complex-valued, the magnitude
        will be displayed.
    title : str
        Title to be displayed above the image.
    colorbar_name : str
        Label for the colorbar indicating the physical quantity and units.
    bounds : 4-tuple of float, optional
        Spatial bounds for the image: (xmin, xmax, ymin, ymax).
        Default is (-1., 1., -1., 1.).
    origin : {'lower', 'upper'}, optional
        Origin of the y-axis. 'lower' places origin at bottom-left.
        Default is 'lower'.
    xlabel : str, optional
        Label for the x-axis. Default is standard wavelength-normalized label.
    ylabel : str, optional
        Label for the y-axis. Default is standard wavelength-normalized label.
    aspect : str or float, optional
        Aspect ratio of the image. Default is 'equal'.
    interpolation : str, optional
        Interpolation method for image display ('nearest', 'bilinear', etc.).
        Default is None (matplotlib default).
    fontsize : int, optional
        Font size for all text elements. Default is 10.

    Notes
    -----
    - Complex images are automatically converted to magnitude
    - Colorbar is positioned with standard fraction and padding
    - All text elements use consistent font sizing
    - Spatial coordinates are typically normalized by background wavelength

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> 
    >>> fig, ax = plt.subplots()
    >>> image_data = np.random.rand(64, 64) * 2 + 1
    >>> add_image(ax, image_data, 'Relative Permittivity', r'$\\epsilon_r$',
    ...           bounds=(-2, 2, -2, 2), fontsize=12)
    >>> plt.show()
    """
    if image.dtype == complex:
        im = axes.imshow(np.abs(image),
                         extent=[bounds[0], bounds[1],
                                 bounds[2], bounds[3]],
                         origin=origin, aspect=aspect,
                         interpolation=interpolation)
    else:
        im = axes.imshow(image,
                         extent=[bounds[0], bounds[1],
                                 bounds[2], bounds[3]],
                         origin=origin, aspect=aspect,
                         interpolation=interpolation)
    axes.set_xlabel(xlabel, fontsize=fontsize)
    axes.set_ylabel(ylabel, fontsize=fontsize)
    axes.set_title(title, fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize)
    cbar = plt.colorbar(ax=axes, mappable=im, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_name, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)


def add_plot(axes, data, x=None, title=None, xlabel='Iterations', ylabel=None,
             style='--*', xticks=None, legend=None, legend_fontsize=None,
             yscale=None, fontsize=10, color=None, markersize=None):
    r"""Add standardized line plot to matplotlib axes.

    Creates a standardized line plot with proper formatting for convergence
    curves and other time-series data in electromagnetic inverse scattering
    analysis.

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The axes object where the plot will be created.
    data : :class:`numpy.ndarray`
        1D array containing the y-data to be plotted.
    x : :class:`numpy.ndarray`, optional
        1D array containing the x-data. If None, uses range(len(data)).
    title : str, optional
        Title to be displayed above the plot.
    xlabel : str, optional
        Label for the x-axis. Default is 'Iterations'.
    ylabel : str, optional
        Label for the y-axis. If None, no y-label is set.
    style : str, optional
        Line style specification (e.g., '--*', '-o', ':^'). Default is '--*'.
    xticks : array-like, optional
        Custom x-axis tick positions. If None, uses matplotlib defaults.
    legend : str, optional
        Legend label for the plot line. If None, no legend is added.
    legend_fontsize : int, optional
        Font size for legend text. If None, uses default.
    yscale : {'linear', 'log', 'symlog', 'logit'}, optional
        Scale for the y-axis. Default is None (linear).
    fontsize : int, optional
        Font size for labels and title. Default is 10.
    color : str, optional
        Color specification for the plot line. Default is None (automatic).
    markersize : int, optional
        Size of markers in the plot. Default is None (matplotlib default).

    Notes
    -----
    - Automatically generates x-data if not provided
    - Supports various matplotlib scales including logarithmic
    - Consistent font sizing across all text elements
    - Flexible styling options for different visualization needs

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> 
    >>> fig, ax = plt.subplots()
    >>> error_data = np.logspace(-1, -4, 50)  # Decreasing error
    >>> add_plot(ax, error_data, title='Convergence', ylabel='Error',
    ...          yscale='log', fontsize=12)
    >>> plt.show()
    """
    if x is None:
        if type(data) is list:
            length = len(data)
        else:
            length = data.size
        x = np.arange(1, length+1)

    axes.plot(x, data, style, color=color, markersize=markersize)
    axes.set_xlabel(xlabel, fontsize=fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize)
    if xticks is not None:
        axes.set_xticks(xticks)
    if ylabel is not None:
        axes.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        axes.set_title(title, fontsize=fontsize)
    if legend is not None:
        if legend_fontsize is not None:
            axes.legend(legend, fontsize=legend_fontsize)
        else:
            axes.legend(legend)
    if yscale is not None:
        axes.set_yscale(yscale)
    axes.grid(True)


def add_box(data, axis=None, meanline=False, labels=None, xlabel=None,
            ylabel=None, color='b', legend=None, title=None, notch=False,
            legend_fontsize=None, fontsize=10, positions=None, yscale=None,
            widths=.5):
    """Improved boxplot routine.

    This routine does not show any plot. It only draws the graphic.

    Parameters
    ----------
        data : list of :class:`numpy.ndarray`
            A list of 1-d arrays meaning the samples.

        axis : :class:`matplotlib.Axes.axes`, default: None
            A specified axis for plotting the graphics. If none is
            provided, then one will be created and returned.

        meanline : bool, default: False
            Draws a line through linear regression of the means among
            the samples.

        labels : list of str, default: None
            Names of the samples.

        xlabel : str, default: None

        ylabel : list of str, default: None

        color : str, default: 'b'
            Color of boxes. Check some `here <https://matplotlib.org/
            3.1.1/gallery/color/named_colors.html>`_

        legend : str, default: None
            Label for meanline.

        title : str, default: None
            A possible title to the plot.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> y3 = np.random.normal(loc=6., size=10)
    >>> boxplot([y1, y2, y3], title='Samples',
                labels=['Sample 1', 'Sample 2', 'Sample 3'],
                xlabel='Samples', ylabel='Unit', color='tab:blue',
                meanline=True, legend='Progression')
    >>> plt.show()
    """
    if (meanline is not None and type(meanline) is not bool
            and meanline != 'regression' and meanline != 'pointwise'):
        raise error.WrongValueInput('result.add_box', 'meanline',
                                    "None, bool, 'regression', 'pointwise'",
                                    str(meanline))
    if axis is None:
        fig, axis = plt.subplots()

    if type(data) is np.ndarray:
        mydata = data.tolist()
    else:
        mydata = data

    if positions is None:
        try:
            _ = len(data[0])
            positions = np.arange(1, len(data)+1)
        except:
            positions = None

    bplot = axis.boxplot(mydata, patch_artist=True, labels=labels,
                         positions=positions, notch=notch, widths=widths)
    for i in range(len(bplot['boxes'])):
        bplot['boxes'][i].set_facecolor(color)

    if meanline == True or meanline == 'regression':
        M = len(mydata)
        x = np.array([positions[0]-.5, positions[-1]+.5])
        means = np.zeros(M)
        for m in range(M):
            means[m] = np.mean(mydata[m])
        a, b = linregress(positions, means)[:2]
        if legend is not None:
            axis.plot(x, a*x + b, '--', color=color, label=legend)
            if legend_fontsize is not None:
                axis.legend(fontsize=legend_fontsize)
            else:
                axis.legend()
        else:
            axis.plot(x, a*x + b, '--', color=color)
    elif meanline == 'pointwise':
        means = np.zeros(len(mydata))
        for m in range(len(mydata)):
            means[m] = np.mean(mydata[m])
        if legend is not None:
            axis.plot(positions, means, '--', color=color, label=legend)
            if legend_fontsize is not None:
                axis.legend(fontsize=legend_fontsize)
            else:
                axis.legend()
        else:
            axis.plot(positions, means, '--', color=color)

    axis.grid(True)
    axis.tick_params(axis='both', which='major', labelsize=fontsize)
    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        axis.set_title(title, fontsize=fontsize)
    if yscale is not None:
        axis.set_yscale(yscale)

    return axis


def add_violin(data, axis=None, meanline=False, labels=None, xlabel=None,
               ylabel=None, color='b', legend=None, title=None,
               legend_fontsize=None, fontsize=10, positions=None, yscale=None):
    r"""Create violin plots for data distribution visualization.

    Creates violin plots to visualize the distribution of data samples,
    with optional mean line overlays and customizable formatting. Violin
    plots show the probability density of data at different values.

    Parameters
    ----------
    data : list of :class:`numpy.ndarray`
        List of 1D arrays containing the data samples for each violin.
        Each array represents one distribution to be plotted.
    axis : :class:`matplotlib.axes.Axes`, optional
        Pre-existing axes for plotting. If None, creates new figure and axes.
    meanline : bool or str, optional
        Controls mean line overlay. Options:
        - False: No mean line
        - True or 'regression': Linear regression through means
        - 'pointwise': Point-to-point line through means
        Default is False.
    labels : list of str, optional
        Labels for each violin (x-axis tick labels). If None, uses default numbering.
    xlabel : str, optional
        Label for the x-axis. Default is None.
    ylabel : str, optional
        Label for the y-axis. Default is None.
    color : str, optional
        Color for the violin faces. Default is 'b' (blue).
        See matplotlib color specification for options.
    legend : str, optional
        Legend label for the mean line. Only used if meanline is True.
        Default is None.
    title : str, optional
        Title for the plot. Default is None.
    legend_fontsize : float, optional
        Font size for legend text. Default is None (uses matplotlib default).
    fontsize : int, optional
        Font size for labels and title. Default is 10.
    positions : array-like, optional
        Positions for the violins along the x-axis. If None, uses
        sequential positions starting from 1.
    yscale : str, optional
        Y-axis scale ('linear', 'log', 'symlog', etc.). Default is None.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        The axes object containing the violin plot.

    Raises
    ------
    error.WrongValueInput
        If meanline parameter has an invalid value.

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> 
    >>> # Create sample data
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> y3 = np.random.normal(loc=6., size=10)
    >>> 
    >>> # Create violin plot
    >>> ax = add_violin([y1, y2, y3], title='Error Distributions',
    ...                 labels=['Method 1', 'Method 2', 'Method 3'],
    ...                 xlabel='Methods', ylabel='Error Value',
    ...                 color='tab:blue', meanline=True, legend='Mean Trend')
    >>> plt.show()
    >>> 
    >>> # Violin plot with custom positions
    >>> ax = add_violin([y1, y2, y3], positions=[1, 3, 5],
    ...                 meanline='pointwise', yscale='log')
    >>> plt.show()

    Notes
    -----
    - Violin plots show the full distribution shape, not just summary statistics
    - Mean line helps visualize trends across different data groups
    - Useful for comparing reconstruction error distributions across methods
    - Color can be specified using matplotlib color names or hex codes
    """
    if (meanline is not None and type(meanline) is not bool
            and meanline != 'regression' and meanline != 'pointwise'):
        raise error.WrongValueInput('result.add_violin', 'meanline',
                                    "None, bool, 'regression', 'pointwise'",
                                    str(meanline))

    plot_opts = {'violin_fc': color,
                 'violin_ec': 'w',
                 'violin_alpha': .2}

    if axis is None:
        fig, axis = plt.subplots()

    if type(data) is np.ndarray:
        mydata = data.tolist()
    else:
        mydata = data

    if positions is None:
        positions = np.arange(1, len(data)+1)
    
    violinplot(mydata, ax=axis, labels=labels, positions=positions,
               plot_opts=plot_opts)

    if meanline == True or meanline == 'regression':
        M = len(mydata)
        x = np.array([positions[0]-.5, positions[-1]+.5])
        means = np.zeros(M)
        for m in range(M):
            means[m] = np.mean(mydata[m])
        a, b = linregress(positions, means)[:2]
        if legend is not None:
            axis.plot(x, a*x + b, '--', color=color, label=legend)
            if legend_fontsize is not None:
                axis.legend(fontsize=legend_fontsize)
            else:
                axis.legend()
        else:
            axis.plot(x, a*x + b, '--', color=color)
    elif meanline == 'pointwise':
        means = np.zeros(len(mydata))
        for m in range(len(mydata)):
            means[m] = np.mean(mydata[m])
        if legend is not None:
            axis.plot(positions, means, '--', color=color, label=legend)
            if legend_fontsize is not None:
                axis.legend(fontsize=legend_fontsize)
            else:
                axis.legend()
        else:
            axis.plot(positions, means, '--', color=color)
    axis.tick_params(axis='both', which='major', labelsize=fontsize)
    axis.grid(True)
    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        axis.set_title(title, fontsize=fontsize)
    if yscale is not None:
        axis.set_yscale(yscale)

    return axis


def get_figure(nsubplots=1, number_lines=1):
    r"""Create a matplotlib figure with optimized layout and sizing.

    ... (restante já está correto)

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        The created figure object.
    axes : :class:`numpy.ndarray`
        Flattened array of axes objects (1D). For nsubplots=1, returns array with one element.
    legend_fontsize : float
        Recommended font size for legends.
    """
    # Compute number of rows and columns
    nrows = round(np.sqrt(nsubplots))
    ncols = int(np.ceil(nsubplots/nrows))
    legend_fontsize = get_legend_fontsize(number_lines, nrows)

    width, height = 6.4*ncols, 4.8*nrows

    # Figure creation
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(width, height))

    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    if nsubplots < axes.size:
        for i in range(nsubplots, axes.size):
            axes[i].set_visible(False)

    return fig, axes, legend_fontsize


def get_legend_fontsize(number_lines, nrows):
    r"""Calculate optimal legend font size based on plot layout.

    Determines the appropriate font size for legends based on the number
    of legend entries and the number of subplot rows. This helps maintain
    readability as plot complexity increases.

    Parameters
    ----------
    number_lines : int
        Number of lines/entries expected in the legend.
    nrows : int
        Number of rows in the subplot grid.

    Returns
    -------
    float or None
        Recommended font size for legends. Returns None if the standard
        matplotlib font size is appropriate (no adjustment needed).

    Notes
    -----
    The function uses empirically determined scaling factors to reduce
    font size when there are many legend entries or many subplot rows.
    Different scaling factors are used for different numbers of rows:
    
    - For nrows ≤ 2: More gentle scaling (factor 0.55)
    - For nrows > 2: More aggressive scaling (factor 0.65)

    The maximum recommended number of lines decreases with more rows:
    - 1 row: up to 15 lines
    - 2 rows: up to 13 lines  
    - 3 rows: up to 8 lines
    - 4 rows: up to 5 lines
    - etc.

    Examples
    --------
    >>> # For single row with many lines
    >>> fontsize = get_legend_fontsize(number_lines=20, nrows=1)
    >>> print(f"Recommended font size: {fontsize}")
    >>> 
    >>> # For multiple rows with few lines
    >>> fontsize = get_legend_fontsize(number_lines=3, nrows=4)
    >>> print(f"Recommended font size: {fontsize}")  # None (use default)
    """
    max_lines = np.array([0, 15, 13, 8, 5, 3, 2, 1, 1, 1, 1, 1])
    if number_lines > max_lines[nrows] and nrows > 2:
        legend_fontsize = 10-(number_lines-max_lines[nrows])*.65
    elif number_lines > max_lines[nrows] and nrows < 3:
        legend_fontsize = 10-(number_lines-max_lines[nrows])*.55
    else:
        legend_fontsize = None
    return legend_fontsize


def compute_zeta_rn(es_o, es_a):
    r"""Compute the residual norm error between measured and computed scattered fields.

    Calculates the L2 norm of the difference between the original (measured)
    and approximated (computed) scattered field data. This metric quantifies
    the data fidelity of the reconstruction.

    Parameters
    ----------
    es_o : :class:`numpy.ndarray`
        Original (measured) scattered field matrix.
        Shape: (N_measurements, N_sources). Units: [V/m].
    es_a : :class:`numpy.ndarray`
        Approximated (computed) scattered field matrix.
        Shape: (N_measurements, N_sources). Units: [V/m].

    Returns
    -------
    float
        Residual norm error. Units: [V/m].

    Notes
    -----
    The error is computed using the L2 norm with trapezoidal integration:

    .. math:: 
        \\zeta_{RN} = \\sqrt{\\iint_S |E^s - E^{s,\\delta}|^2 d\\theta d\\phi}

    where :math:`E^s` is the measured scattered field, :math:`E^{s,\\delta}` is
    the computed scattered field, and the integration is over the measurement
    surface with angles θ and φ.

    Examples
    --------
    >>> measured_field = np.random.complex128((64, 8))
    >>> computed_field = measured_field + 0.1 * np.random.complex128((64, 8))
    >>> error = compute_zeta_rn(measured_field, computed_field)
    >>> print(f"Residual norm error: {error:.3e} V/m")
    """
    NM, NS = es_o.shape
    theta = cfg.get_angles(NM)
    phi = cfg.get_angles(NS)
    y = (es_o-es_a)*np.conj(es_o-es_a)
    return np.real(np.sqrt(np.trapezoid(np.trapezoid(y, x=phi), x=theta)))


def compute_rre(es_o, es_a):
    r"""Compute the Relative Residual Error (RRE) between scattered fields.

    Calculates the relative residual error as defined in the literature [1]_
    for electromagnetic inverse scattering problems.

    Parameters
    ----------
    es_o : numpy.ndarray
        Original (measured) scattered field matrix.
        Shape: (N_measurements, N_sources). Units: [V/m].
    es_a : numpy.ndarray
        Approximated (computed) scattered field matrix.
        Shape: (N_measurements, N_sources). Units: [V/m].

    Returns
    -------
    float
        Relative residual error as a percentage. Formula:

        .. math::
            \text{RRE} = 100 \times \frac{\left\|\mathbf{E}^s_{\mathrm{meas}} - \mathbf{E}^s_{\mathrm{comp}}\right\|_2}{\left\|\mathbf{E}^s_{\mathrm{meas}}\right\|_2}

    References
    ----------
    .. [1] Lavarello, Roberto, and Michael Oelze. "A study on the
           reconstruction of moderate contrast targets using the
           distorted Born iterative method." IEEE transactions on
           ultrasonics, ferroelectrics, and frequency control 55.1
           (2008): 112-124.

    Examples
    --------
    >>> measured_field = np.random.complex128((64, 8))
    >>> computed_field = measured_field + 0.05 * measured_field
    >>> rre = compute_rre(measured_field, computed_field)
    >>> print(f"Relative residual error: {rre:.2f}%")
    >>> 
    >>> # Perfect reconstruction
    >>> rre_perfect = compute_rre(measured_field, measured_field)
    >>> print(f"Perfect reconstruction RRE: {rre_perfect:.2f}%")  # Should be 0%
    """
    return (100*compute_zeta_rn(es_o, es_a)
            / compute_zeta_rn(es_o, np.zeros(es_o.shape, dtype=complex)))


def compute_zeta_rpad(es_o, es_r):
    r"""Compute the residual percentage average deviation.

    Calculates the percentage average deviation between original (measured)
    and reconstructed (computed) scattered field data. This metric provides
    a relative measure of reconstruction accuracy normalized by the data magnitude.

    Parameters
    ----------
    es_o : :class:`numpy.ndarray`
        Original (measured) scattered field matrix.
        Shape: (N_measurements, N_sources). Units: [V/m].
    es_r : :class:`numpy.ndarray`
        Reconstructed (computed) scattered field matrix.
        Shape: (N_measurements, N_sources). Units: [V/m].

    Returns
    -------
    float
        Residual percentage average deviation. Units: [%/sample].

    Notes
    -----
    The error is computed as:

    .. math:: 
        \\zeta_{RPAD} = \\frac{100}{N} \\sum_{i=1}^{N} \\frac{|y_i - y_i^\\delta|}{|y_i|}

    where :math:`y_i` represents the real and imaginary parts of the measured
    scattered field, :math:`y_i^\\delta` represents the computed values, and
    :math:`N` is the total number of real-valued samples.

    Examples
    --------
    >>> measured_field = np.random.complex128((64, 8)) + 1j
    >>> computed_field = measured_field * 0.95  # 5% error
    >>> error = compute_zeta_rpad(measured_field, computed_field)
    >>> print(f"Residual PAD error: {error:.2f}%/sample")
    """
    y = np.hstack((np.real(es_o.flatten()), np.imag(es_o.flatten())))
    yd = np.hstack((np.real(es_r.flatten()), np.imag(es_r.flatten())))
    return np.mean(np.abs((y-yd)/y))*100


def compute_zeta_epad(epsilon_ro, epsilon_rr):
    r"""Compute the percentage average deviation of relative permittivity.

    Calculates the percentage average deviation between the original (true)
    and reconstructed relative permittivity maps. This metric quantifies
    the pixel-wise accuracy of permittivity reconstruction.

    Parameters
    ----------
    epsilon_ro : numpy.ndarray
        Original (true) relative permittivity map.
        Shape: (N_x, N_y). Dimensionless.
    epsilon_rr : numpy.ndarray
        Reconstructed relative permittivity map.
        Shape: (N_x, N_y). Dimensionless.

    Returns
    -------
    float
        Percentage average deviation of relative permittivity. Units: [%/pixel].

    Notes
    -----
    The error is computed as:

    .. math::
        \zeta_{\epsilon \mathrm{PAD}} = \frac{100}{N_p} \sum_{i=1}^{N_p} 
        \frac{|\epsilon_{r,r,i}^{\text{rec}} - \epsilon_{r,o,i}^{\text{true}}|}{|\epsilon_{r,o,i}^{\text{true}}|}

    where :math:`\epsilon_{r,\mathrm{o},i}` is the original permittivity at pixel :math:`i`,
    :math:`\epsilon_{r,\mathrm{r},i}` is the reconstructed permittivity, and :math:`N_p`
    is the total number of pixels.
    

    Examples
    --------
    >>> true_eps = np.ones((64, 64)) * 2.0  # Background permittivity
    >>> true_eps[20:40, 20:40] = 4.0  # Object with higher permittivity
    >>> reconstructed_eps = true_eps + 0.1 * np.random.randn(64, 64)
    >>> error = compute_zeta_epad(true_eps, reconstructed_eps)
    >>> print(f"Permittivity PAD error: {error:.2f}%/pixel")
    """
    y = epsilon_ro.flatten()
    yd = epsilon_rr.flatten()
    return np.mean(np.abs((y-yd)/y))*100


def compute_zeta_sad(sigma_o, sigma_r):
    r"""Compute the average deviation of conductivity maps.

    Calculates the pixel-wise average deviation between the original
    and reconstructed conductivity maps. This metric provides a global
    measure of conductivity reconstruction accuracy.

    Parameters
    ----------
    sigma_o : :class:`numpy.ndarray`
        Original (ground truth) conductivity map.
        Shape: (N_x, N_y). Units: [S/m].
    sigma_r : :class:`numpy.ndarray`
        Reconstructed conductivity map.
        Shape: (N_x, N_y). Units: [S/m].

    Returns
    -------
    float
        Average deviation of conductivity values.
        Units: [S/m].

    Notes
    -----
    The error is computed as:

    .. math::
        \\zeta_{\\sigma AD} = \\frac{1}{N_p} \\sum_{i=1}^{N_p} |\\sigma_{o,i} - \\sigma_{r,i}|

    where :math:`\\sigma_{o,i}` and :math:`\\sigma_{r,i}` are the original
    and reconstructed conductivity values at pixel :math:`i`, and
    :math:`N_p` is the total number of pixels.

    This metric provides an absolute measure of conductivity reconstruction
    error that is useful for comparing different reconstruction methods.

    Examples
    --------
    >>> # Create sample conductivity maps
    >>> sigma_true = np.zeros((64, 64))  # Background: σ = 0 S/m
    >>> sigma_true[20:40, 20:40] = 0.1  # Object: σ = 0.1 S/m
    >>> 
    >>> # Simulated reconstruction with errors
    >>> sigma_recon = sigma_true + 0.01 * np.random.randn(64, 64)
    >>> 
    >>> # Compute average deviation
    >>> avg_dev = compute_zeta_sad(sigma_true, sigma_recon)
    >>> print(f"Conductivity average deviation: {avg_dev:.4f} S/m")
    """
    y = sigma_o.flatten()
    yd = sigma_r.flatten()
    return np.mean(np.abs((y-yd)))


def compute_zeta_tv(chi, x, y):
    r"""Compute the total variation of a contrast map.

    Calculates the total variation (TV) regularization functional commonly
    used in electromagnetic inverse scattering to promote smooth solutions
    and suppress artifacts. The TV functional measures the variation of
    the contrast function across the spatial domain.

    Parameters
    ----------
    chi : :class:`numpy.ndarray`
        Complex-valued contrast map representing the scattering properties.
        Shape: (N_x, N_y). Units: [dimensionless].
    x : :class:`numpy.ndarray`
        Meshgrid array of x-coordinates corresponding to the contrast map.
        Shape: (N_x, N_y). Units: [m].
    y : :class:`numpy.ndarray`
        Meshgrid array of y-coordinates corresponding to the contrast map.
        Shape: (N_x, N_y). Units: [m].

    Returns
    -------
    float
        Total variation value. Units: [dimensionless].

    Notes
    -----
    The total variation is computed using a modified formulation:

    .. math::
        TV = \\int \\int \\frac{|\\nabla \\chi|^2}{|\\nabla \\chi|^2 + 1} \\, dx \\, dy

    where :math:`\\nabla \\chi` is the spatial gradient of the contrast function.
    This formulation provides better numerical stability compared to the
    standard TV functional.

    References
    ----------
    .. [1] Lobel, P., et al. "A new regularization scheme for
       inverse scattering." Inverse Problems 13.2 (1997): 403.

    Examples
    --------
    >>> # Create a simple contrast map
    >>> nx, ny = 64, 64
    >>> x = np.linspace(-0.1, 0.1, nx)
    >>> y = np.linspace(-0.1, 0.1, ny)
    >>> X, Y = np.meshgrid(x, y)
    >>> 
    >>> # Piecewise constant contrast (low TV)
    >>> chi_smooth = np.ones_like(X, dtype=complex)
    >>> chi_smooth[20:40, 20:40] = 2.0 + 0.1j
    >>> tv_smooth = compute_zeta_tv(chi_smooth, X, Y)
    >>> 
    >>> # Noisy contrast (high TV)
    >>> chi_noisy = chi_smooth + 0.1 * np.random.randn(*X.shape)
    >>> tv_noisy = compute_zeta_tv(chi_noisy, X, Y)
    >>> 
    >>> print(f"Smooth TV: {tv_smooth:.2f}, Noisy TV: {tv_noisy:.2f}")
    """
    grad_chi = np.gradient(chi, y[:, 0], x[0, :])
    X = np.sqrt(np.abs(grad_chi[1])**2 + np.abs(grad_chi[0])**2)
    return np.trapezoid(np.trapezoid(X**2/(X**2+1), x=x[0, :]), x=y[:, 0])


def compute_zeta_ebe(epsilon_ro, epsilon_rr, epsilon_rb):
    r"""Compute the background relative permittivity estimation error.

    Calculates the estimation error for background regions in the relative
    permittivity reconstruction. This metric quantifies false-positive-like
    errors where the background is incorrectly reconstructed as having
    different permittivity values.

    Parameters
    ----------
    epsilon_ro : :class:`numpy.ndarray`
        Original (ground truth) relative permittivity map.
        Shape: (N_x, N_y). Dimensionless.
    epsilon_rr : :class:`numpy.ndarray`
        Recovered (reconstructed) relative permittivity map.
        Shape: (N_x, N_y). Dimensionless.
    epsilon_rb : float
        Background relative permittivity value used to identify
        background regions. Dimensionless.

    Returns
    -------
    float
        Background relative permittivity estimation error as a percentage.
        Units: [%].

    Notes
    -----
    The error is computed only for pixels where the original permittivity
    equals the background value:

    .. math::
        \\zeta_{\\epsilon BE} = \\frac{100}{N_{bg}} \\sum_{i \\in \\text{background}} \\frac{|\\epsilon_{r,o,i} - \\epsilon_{r,r,i}|}{|\\epsilon_{r,o,i}|}

    where :math:`N_{bg}` is the number of background pixels and the sum is
    over pixels where :math:`\\epsilon_{r,o,i} = \\epsilon_{rb}`.

    This metric is analogous to the false-positive rate in classification,
    measuring how well the background regions are preserved in the reconstruction.

    Examples
    --------
    >>> # Create ground truth with background and object
    >>> epsilon_true = np.ones((64, 64)) * 1.0  # Background: εᵣ = 1.0
    >>> epsilon_true[20:40, 20:40] = 3.0  # Object: εᵣ = 3.0
    >>> 
    >>> # Simulated reconstruction with background errors
    >>> epsilon_recon = epsilon_true.copy()
    >>> epsilon_recon[0:20, 0:20] = 1.1  # Background error
    >>> 
    >>> # Compute background error
    >>> bg_error = compute_zeta_ebe(epsilon_true, epsilon_recon, 1.0)
    >>> print(f"Background estimation error: {bg_error:.2f}%")
    """
    background = np.zeros(epsilon_ro.shape, dtype=bool)
    background[epsilon_ro == epsilon_rb] = True
    y = epsilon_ro[background]
    yd = epsilon_rr[background]
    return np.mean(np.abs(y-yd)/y)*100


def compute_zeta_sbe(sigma_o, sigma_r, sigma_b):
    r"""Compute the background conductivity estimation error.

    Calculates the estimation error for background regions in the conductivity
    reconstruction. This metric quantifies false-positive-like errors where
    the background is incorrectly reconstructed as having different
    conductivity values.

    Parameters
    ----------
    sigma_o : :class:`numpy.ndarray`
        Original (ground truth) conductivity map.
        Shape: (N_x, N_y). Units: [S/m].
    sigma_r : :class:`numpy.ndarray`
        Recovered (reconstructed) conductivity map.
        Shape: (N_x, N_y). Units: [S/m].
    sigma_b : float
        Background conductivity value used to identify background regions.
        Units: [S/m].

    Returns
    -------
    float
        Background conductivity estimation error.
        Units: [S/m].

    Notes
    -----
    The error is computed only for pixels where the original conductivity
    equals the background value:

    .. math::
        \\zeta_{\\sigma BE} = \\frac{1}{N_{bg}} \\sum_{i \\in \\text{background}} |\\sigma_{o,i} - \\sigma_{r,i}|

    where :math:`N_{bg}` is the number of background pixels and the sum is
    over pixels where :math:`\\sigma_{o,i} = \\sigma_b`.

    This metric is analogous to the false-positive rate in classification,
    measuring how well the background conductivity is preserved in the
    reconstruction.

    Examples
    --------
    >>> # Create ground truth with background and object
    >>> sigma_true = np.zeros((64, 64))  # Background: σ = 0 S/m
    >>> sigma_true[20:40, 20:40] = 0.1  # Object: σ = 0.1 S/m
    >>> 
    >>> # Simulated reconstruction with background errors
    >>> sigma_recon = sigma_true.copy()
    >>> sigma_recon[0:20, 0:20] = 0.01  # Background error
    >>> 
    >>> # Compute background error
    >>> bg_error = compute_zeta_sbe(sigma_true, sigma_recon, 0.0)
    >>> print(f"Background conductivity error: {bg_error:.3f} S/m")
    """
    background = np.zeros(sigma_o.shape, dtype=bool)
    background[sigma_o == sigma_b] = True
    y = sigma_o[background]
    yd = sigma_r[background]
    return np.mean(np.abs(y-yd))


def compute_zeta_eoe(epsilon_ro, epsilon_rr, epsilon_rb):
    r"""Compute the object relative permittivity estimation error.

    Calculates the estimation error for object regions in the relative
    permittivity reconstruction. This metric quantifies false-negative-like
    errors where object regions are incorrectly reconstructed with wrong
    permittivity values.

    Parameters
    ----------
    epsilon_ro : :class:`numpy.ndarray`
        Original (ground truth) relative permittivity map.
        Shape: (N_x, N_y). Dimensionless.
    epsilon_rr : :class:`numpy.ndarray`
        Recovered (reconstructed) relative permittivity map.
        Shape: (N_x, N_y). Dimensionless.
    epsilon_rb : float
        Background relative permittivity value used to identify
        object regions (pixels with values different from background).
        Dimensionless.

    Returns
    -------
    float
        Object relative permittivity estimation error as a percentage.
        Units: [%].

    Notes
    -----
    The error is computed only for pixels where the original permittivity
    differs from the background value:

    .. math::
        \\zeta_{\\epsilon OE} = \\frac{100}{N_{obj}} \\sum_{i \\in \\text{object}} \\frac{|\\epsilon_{r,o,i} - \\epsilon_{r,r,i}|}{|\\epsilon_{r,o,i}|}

    where :math:`N_{obj}` is the number of object pixels and the sum is
    over pixels where :math:`\\epsilon_{r,o,i} \\neq \\epsilon_{rb}`.

    This metric is analogous to the false-negative rate in classification,
    measuring how accurately the object regions are reconstructed.

    Examples
    --------
    >>> # Create ground truth with background and object
    >>> epsilon_true = np.ones((64, 64)) * 1.0  # Background: εᵣ = 1.0
    >>> epsilon_true[20:40, 20:40] = 3.0  # Object: εᵣ = 3.0
    >>> 
    >>> # Simulated reconstruction with object errors
    >>> epsilon_recon = epsilon_true.copy()
    >>> epsilon_recon[20:40, 20:40] = 2.5  # Object reconstruction error
    >>> 
    >>> # Compute object error
    >>> obj_error = compute_zeta_eoe(epsilon_true, epsilon_recon, 1.0)
    >>> print(f"Object estimation error: {obj_error:.2f}%")
    """
    not_background = np.zeros(epsilon_ro.shape, dtype=bool)
    not_background[epsilon_ro != epsilon_rb] = True
    y = epsilon_ro[not_background]
    yd = epsilon_rr[not_background]
    return np.mean(np.abs(y-yd)/y)*100


def compute_zeta_soe(sigma_o, sigma_r, sigma_b):
    r"""Compute the object conductivity estimation error.

    Calculates the estimation error for object regions in the conductivity
    reconstruction. This metric quantifies false-negative-like errors where
    object regions are incorrectly reconstructed with wrong conductivity
    values.

    Parameters
    ----------
    sigma_o : :class:`numpy.ndarray`
        Original (ground truth) conductivity map.
        Shape: (N_x, N_y). Units: [S/m].
    sigma_r : :class:`numpy.ndarray`
        Recovered (reconstructed) conductivity map.
        Shape: (N_x, N_y). Units: [S/m].
    sigma_b : float
        Background conductivity value used to identify object regions
        (pixels with values different from background). Units: [S/m].

    Returns
    -------
    float
        Object conductivity estimation error.
        Units: [S/m].

    Notes
    -----
    The error is computed only for pixels where the original conductivity
    differs from the background value:

    .. math::
        \\zeta_{\\sigma OE} = \\frac{1}{N_{obj}} \\sum_{i \\in \\text{object}} |\\sigma_{o,i} - \\sigma_{r,i}|

    where :math:`N_{obj}` is the number of object pixels and the sum is
    over pixels where :math:`\\sigma_{o,i} \\neq \\sigma_b`.

    This metric is analogous to the false-negative rate in classification,
    measuring how accurately the object conductivity is reconstructed.

    Examples
    --------
    >>> # Create ground truth with background and object
    >>> sigma_true = np.zeros((64, 64))  # Background: σ = 0 S/m
    >>> sigma_true[20:40, 20:40] = 0.1  # Object: σ = 0.1 S/m
    >>> 
    >>> # Simulated reconstruction with object errors
    >>> sigma_recon = sigma_true.copy()
    >>> sigma_recon[20:40, 20:40] = 0.08  # Object reconstruction error
    >>> 
    >>> # Compute object error
    >>> obj_error = compute_zeta_soe(sigma_true, sigma_recon, 0.0)
    >>> print(f"Object conductivity error: {obj_error:.3f} S/m")
    """
    not_background = np.zeros(sigma_o.shape, dtype=bool)
    not_background[sigma_o != sigma_b] = True
    y = sigma_o[not_background]
    yp = sigma_r[not_background]
    return np.mean(np.abs(y-yp))


def compute_zeta_tfmpad(et_o, et_r):
    r"""Compute the total field magnitude percentage average deviation.

    Calculates the percentage average deviation between the magnitudes of
    the original and reconstructed total electric field distributions.
    This metric quantifies how accurately the field magnitude is recovered
    throughout the investigation domain.

    Parameters
    ----------
    et_o : :class:`numpy.ndarray`
        Original (ground truth) total electric field.
        Shape: (N_pixels, N_sources). Units: [V/m].
    et_r : :class:`numpy.ndarray`
        Reconstructed total electric field.
        Shape: (N_pixels, N_sources). Units: [V/m].

    Returns
    -------
    float
        Total field magnitude percentage average deviation.
        Units: [%].

    Notes
    -----
    The error is computed as:

    .. math::
        \\zeta_{TF MAG PAD} = \\frac{100}{N} \\sum_{i=1}^{N} \\frac{||E_{t,o,i}| - |E_{t,r,i}||}{|E_{t,o,i}|}

    where :math:`|E_{t,o,i}|` and :math:`|E_{t,r,i}|` are the magnitudes of
    the original and reconstructed total field at pixel :math:`i`, and
    :math:`N` is the total number of field samples.

    This metric is particularly useful for evaluating the accuracy of
    field-based reconstruction methods where the total field distribution
    is of primary interest.

    Examples
    --------
    >>> # Create sample total field data
    >>> et_true = np.random.complex128((1024, 8))  # 1024 pixels, 8 sources
    >>> et_true *= np.exp(1j * np.random.uniform(0, 2*np.pi, et_true.shape))
    >>> 
    >>> # Simulated reconstruction with magnitude errors
    >>> et_recon = et_true * (1 + 0.05 * np.random.randn(*et_true.shape))
    >>> 
    >>> # Compute magnitude error
    >>> mag_error = compute_zeta_tfmpad(et_true, et_recon)
    >>> print(f"Total field magnitude PAD: {mag_error:.2f}%")
    """
    y = np.abs(et_o.flatten())
    yd = np.abs(et_r.flatten())
    return np.mean(np.abs((y-yd)/y))*100


def compute_zeta_tfpad(et_o, et_r):
    r"""Compute the total field phase average deviation.

    Calculates the average deviation between the phases of the original
    and reconstructed total electric field distributions. This metric
    quantifies how accurately the field phase is recovered throughout
    the investigation domain.

    Parameters
    ----------
    et_o : :class:`numpy.ndarray`
        Original (ground truth) total electric field.
        Shape: (N_pixels, N_sources). Units: [V/m].
    et_r : :class:`numpy.ndarray`
        Reconstructed total electric field.
        Shape: (N_pixels, N_sources). Units: [V/m].

    Returns
    -------
    float
        Total field phase average deviation.
        Units: [rad].

    Notes
    -----
    The error is computed as:

    .. math::
        \\zeta_{TF PHASE AD} = \\frac{1}{N} \\sum_{i=1}^{N} |\\arg(E_{t,o,i}) - \\arg(E_{t,r,i})|

    where :math:`\\arg(E_{t,o,i})` and :math:`\\arg(E_{t,r,i})` are the phases
    of the original and reconstructed total field at pixel :math:`i`, and
    :math:`N` is the total number of field samples.

    Phase information is crucial for many inverse scattering applications,
    particularly those involving interferometric or holographic techniques.

    Examples
    --------
    >>> # Create sample total field data
    >>> et_true = np.random.complex128((1024, 8))  # 1024 pixels, 8 sources
    >>> et_true *= np.exp(1j * np.random.uniform(0, 2*np.pi, et_true.shape))
    >>> 
    >>> # Simulated reconstruction with phase errors
    >>> phase_error = 0.1 * np.random.randn(*et_true.shape)
    >>> et_recon = np.abs(et_true) * np.exp(1j * (np.angle(et_true) + phase_error))
    >>> 
    >>> # Compute phase error
    >>> phase_dev = compute_zeta_tfpad(et_true, et_recon)
    >>> print(f"Total field phase AD: {phase_dev:.3f} rad")
    """
    y = np.angle(et_o.flatten())
    yd = np.angle(et_r.flatten())
    return np.mean(np.abs(y-yd))


def compute_zeta_p(chi_o, chi_r):
    r"""Compute the position error between original and reconstructed objects.

    Calculates the percentage position error by comparing the centroids of
    the original and reconstructed scattering objects. This metric quantifies
    how accurately the object location is recovered in the reconstruction.

    Parameters
    ----------
    chi_o : :class:`numpy.ndarray`
        Original (ground truth) contrast map.
        Shape: (N_x, N_y). Complex-valued, dimensionless.
    chi_r : :class:`numpy.ndarray`
        Reconstructed contrast map.
        Shape: (N_x, N_y). Complex-valued, dimensionless.

    Returns
    -------
    float
        Position error as a percentage of the domain size.
        Units: [%].

    Notes
    -----
    The position error is computed as:

    .. math::
        \\zeta_p = 100 \\times \\sqrt{(x_{co} - x_{cr})^2 + (y_{co} - y_{cr})^2}

    where :math:`(x_{co}, y_{co})` and :math:`(x_{cr}, y_{cr})` are the
    centroids of the original and reconstructed objects, respectively.

    The algorithm:
    1. Identifies object regions using thresholding
    2. Computes weighted centroids of the objects
    3. Calculates Euclidean distance between centroids
    4. Normalizes by domain size and converts to percentage

    If no object is detected in the reconstruction, returns 100% error.

    Examples
    --------
    >>> # Create original contrast with centered object
    >>> chi_true = np.zeros((64, 64), dtype=complex)
    >>> chi_true[28:36, 28:36] = 2.0 + 0.5j  # Centered object
    >>> 
    >>> # Create reconstructed contrast with shifted object
    >>> chi_recon = np.zeros((64, 64), dtype=complex)
    >>> chi_recon[30:38, 30:38] = 1.8 + 0.4j  # Slightly shifted
    >>> 
    >>> # Compute position error
    >>> pos_error = compute_zeta_p(chi_true, chi_recon)
    >>> print(f"Position error: {pos_error:.2f}%")
    """
    Xo, Xr = np.abs(chi_o), np.abs(chi_r)
    threshold = (np.amin(np.abs(Xr))
                 + .5*(np.amax(np.abs(Xr))-np.amin(np.abs(Xr))))

    masko = np.zeros(Xo.shape, dtype=bool)
    maskr = np.zeros(Xr.shape, dtype=bool)

    masko[Xo > 0.] = True
    maskr[Xr >= threshold] = True

    xo, yo = np.meshgrid(np.linspace(0, 1, Xo.shape[1]),
                         np.linspace(0, 1, Xo.shape[0]))

    xr, yr = np.meshgrid(np.linspace(0, 1, Xr.shape[1]),
                         np.linspace(0, 1, Xr.shape[0]))

    if not np.any(maskr) or np.any(np.isnan(maskr)):
        return 100.

    xco = np.sum(masko*xo)/np.sum(masko)
    yco = np.sum(masko*yo)/np.sum(masko)
    xcr = np.sum(maskr*xr)/np.sum(maskr)
    ycr = np.sum(maskr*yr)/np.sum(maskr)

    return np.sqrt((xco-xcr)**2 + (yco-ycr)**2)*100


def compute_zeta_s(chi_o, chi_r):
    r"""Compute the shape error between original and reconstructed objects.

    Calculates the percentage shape error by comparing the areas of the
    original and reconstructed scattering objects. This metric quantifies
    how accurately the object shape and size are recovered in the reconstruction.

    Parameters
    ----------
    chi_o : :class:`numpy.ndarray`
        Original (ground truth) contrast map.
        Shape: (N_x, N_y). Complex-valued, dimensionless.
    chi_r : :class:`numpy.ndarray`
        Reconstructed contrast map.
        Shape: (N_x, N_y). Complex-valued, dimensionless.

    Returns
    -------
    float
        Shape error as a percentage of the original object area.
        Units: [%].

    Notes
    -----
    The shape error is computed as:

    .. math::
        \\zeta_s = \\frac{100 \\times |A_{diff}|}{A_{original}}

    where :math:`A_{diff}` is the area of the symmetric difference between
    the original and reconstructed object regions, and :math:`A_{original}`
    is the area of the original object.

    The algorithm:
    1. Identifies object regions using thresholding
    2. Finds contours of both objects
    3. Normalizes spatial scales and centers objects
    4. Computes symmetric difference using XOR operation
    5. Calculates area ratio as percentage

    If no object is detected in the reconstruction, returns 100% error.

    Examples
    --------
    >>> # Create original contrast with square object
    >>> chi_true = np.zeros((64, 64), dtype=complex)
    >>> chi_true[20:40, 20:40] = 2.0 + 0.5j  # Square object
    >>> 
    >>> # Create reconstructed contrast with similar but different shape
    >>> chi_recon = np.zeros((64, 64), dtype=complex)
    >>> chi_recon[22:38, 22:38] = 1.8 + 0.4j  # Smaller square
    >>> 
    >>> # Compute shape error
    >>> shape_error = compute_zeta_s(chi_true, chi_recon)
    >>> print(f"Shape error: {shape_error:.2f}%")
    """
    Xo, Xr = np.abs(chi_o), np.abs(chi_r)
    threshold = (np.amin(np.abs(Xr))
                 + .5*(np.amax(np.abs(Xr))-np.amin(np.abs(Xr))))

    co = measure.find_contours(Xo, .0, fully_connected='high')
    cr = measure.find_contours(Xr, threshold)

    # Converting scale
    for i in range(len(cr)):
        cr[i][:, 1] = Xo.shape[1]*cr[i][:, 1]/Xr.shape[1]
        cr[i][:, 0] = Xo.shape[0]*cr[i][:, 0]/Xr.shape[0]

    masko = np.zeros(Xo.shape, dtype=bool)
    maskr = np.zeros(Xr.shape, dtype=bool)

    masko[Xo > 0] = True
    maskr[Xr >= threshold] = True

    xo, yo = np.meshgrid(np.arange(0, Xo.shape[1]), np.arange(0, Xo.shape[0]))
    xr, yr = np.meshgrid(np.linspace(0, Xo.shape[1]-1, Xr.shape[1]),
                         np.linspace(0, Xo.shape[0]-1, Xr.shape[0]))

    if np.sum(maskr*Xr) == 0:
        return 100.

    xco = np.sum(masko*Xo*xo)/np.sum(masko*Xo)
    yco = np.sum(masko*Xo*yo)/np.sum(masko*Xo)
    xcr = np.sum(maskr*Xr*xr)/np.sum(maskr*Xr)
    ycr = np.sum(maskr*Xr*yr)/np.sum(maskr*Xr)

    # Centralization
    for i in range(len(co)):
        co[i][:, 0] = co[i][:, 0]-yco+Xo.shape[0]/2
        co[i][:, 1] = co[i][:, 1]-xco+Xo.shape[1]/2

    # Centralization
    for i in range(len(cr)):
        cr[i][:, 0] = cr[i][:, 0]-ycr+Xo.shape[0]/2
        cr[i][:, 1] = cr[i][:, 1]-xcr+Xo.shape[1]/2

    masko = np.zeros(Xo.shape, dtype=bool)
    counter = np.zeros(Xo.shape)
    for i in range(len(co)):
        maskt = measure.grid_points_in_poly(Xo.shape, co[i])
        counter[maskt] += 1
    masko[np.mod(counter, 2) == 1] = True

    maskr = np.zeros(Xo.shape, dtype=bool)
    counter = np.zeros(Xo.shape)
    for i in range(len(cr)):
        maskt = measure.grid_points_in_poly(Xo.shape, cr[i])
        counter[maskt] += 1
    maskr[np.mod(counter, 2) == 1] = True
    
    # Xor operation
    diff = np.logical_xor(masko, maskr)

    # Area of the difference
    area_diff = np.sum(diff)/np.sum(masko)*100

    return area_diff


def compute_ssim(chi_o, chi_r):
    r"""Compute the Structural Similarity Index (SSIM) between two contrast maps.

    Calculates the Structural Similarity Index (SSIM) to assess the
    similarity between the original and reconstructed contrast maps.
    SSIM is a perceptual metric that considers changes in structural
    information, luminance, and contrast.

    Parameters
    ----------

    chi_o : :class:`numpy.ndarray`
        Original (ground truth) contrast map.
        Shape: (N_x, N_y). Complex-valued, dimensionless.
    chi_r : :class:`numpy.ndarray`
        Reconstructed contrast map.
        Shape: (N_x, N_y). Complex-valued, dimensionless.
    
    Returns
    -------

    float
        Structural Similarity Index (SSIM) value between the two contrast maps.
        Ranges from -1 to 1, where 1 indicates perfect similarity.

    Notes
    -----

    SSIM is computed using the `structural_similarity` function from
    `skimage.metrics`, which evaluates local patterns of pixel
    intensities that have been normalized for luminance and contrast.
    The SSIM value provides a more comprehensive assessment of image
    quality compared to traditional metrics like Mean Squared Error (MSE).

    Examples
    --------

    >>> # Create sample contrast maps
    >>> chi_true = np.zeros((64, 64), dtype=complex)
    >>> chi_true[20:40, 20:40] = 2.0 + 0.5j  # Original object
    >>> chi_recon = np.zeros((64, 64), dtype=complex)
    >>> chi_recon[22:38, 22:38] = 1.8 + 0.4j  # Reconstructed object
    >>> # Compute SSIM
    >>> ssim_value = compute_ssim(chi_true, chi_recon)
    >>> print(f"SSIM: {ssim_value:.4f}")
    """
    chi_o = np.abs(chi_o)
    chi_r = np.abs(chi_r)
    range_o = chi_o.max() - chi_o.min()
    range_r = chi_r.max() - chi_r.min()
    if range_o > range_r:
        data_range = range_o
    else:
        data_range = range_r
    ssim = structural_similarity(chi_o, chi_r, data_range=data_range)
    return ssim


def check_indicator(indicator):
    r"""Validate whether the given indicator name(s) are valid.

    Checks if the provided indicator name or list of indicator names
    correspond to valid error indicators supported by the Result class.
    This function is used internally for input validation.

    Parameters
    ----------
    indicator : str or list of str
        Error indicator name(s) to validate. Can be a single string
        or a list of strings representing indicator names.

    Returns
    -------
    bool
        True if all provided indicator names are valid, False otherwise.

    Notes
    -----
    Valid indicator names are defined in the INDICATOR_SET constant and
    include metrics such as:
    - 'zeta_rn': Residual norm error
    - 'zeta_rpad': Residual percentage average deviation
    - 'zeta_epad': Permittivity percentage average deviation
    - 'zeta_sad': Conductivity average deviation
    - 'zeta_tv': Total variation
    - And many others...

    Examples
    --------
    >>> # Check single indicator
    >>> is_valid = check_indicator('zeta_rn')
    >>> print(is_valid)  # True
    >>> 
    >>> # Check invalid indicator
    >>> is_valid = check_indicator('invalid_indicator')
    >>> print(is_valid)  # False
    >>> 
    >>> # Check multiple indicators
    >>> is_valid = check_indicator(['zeta_rn', 'zeta_epad'])
    >>> print(is_valid)  # True
    >>> 
    >>> # Check with one invalid indicator
    >>> is_valid = check_indicator(['zeta_rn', 'invalid_indicator'])
    >>> print(is_valid)  # False
    """
    if type(indicator) is str:
        return any(indicator == n for n in INDICATOR_SET)
    else:
        return all(any(m == n for n in INDICATOR_SET) for m in indicator)


def indicator_label(indicator):
    r"""Get the display label for an error indicator.

    Retrieves the human-readable label associated with a specific error
    indicator. These labels are used for plot axes, legends, and other
    display purposes.

    Parameters
    ----------
    indicator : str
        Name of the error indicator. Must be a valid indicator name
        as defined in INDICATOR_SET.

    Returns
    -------
    str
        Human-readable label for the indicator, typically including
        units and mathematical notation where appropriate.

    Raises
    ------
    error.WrongValueInput
        If the indicator name is not valid.

    Examples
    --------
    >>> # Get label for residual norm error
    >>> label = indicator_label('zeta_rn')
    >>> print(label)  # "Residual Norm Error [V/m]"
    >>> 
    >>> # Get label for permittivity error
    >>> label = indicator_label('zeta_epad')
    >>> print(label)  # "Permittivity PAD Error [%]"
    >>> 
    >>> # Error for invalid indicator
    >>> try:
    ...     label = indicator_label('invalid_indicator')
    ... except error.WrongValueInput as e:
    ...     print(f"Error: {e}")

    Notes
    -----
    The labels are stored in the LABELS dictionary and are designed
    to be suitable for use in plots, tables, and other display contexts.
    They typically include units and use mathematical notation where
    appropriate.
    """
    if not check_indicator(indicator):
        raise error.WrongValueInput('indicator_label', 'indicator',
                                    INDICATOR_SET, indicator)
    return LABELS[indicator]
    