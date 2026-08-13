# eispy2d/utils/__init__.py

"""
Utilities module for eispy2d.

Collection of helper functions and classes:
- draw: Geometric shape drawing for test generation
- regularization: Regularization methods for ill-posed problems
- stopcriteria: Convergence criteria for iterative algorithms
- outputmode: Output aggregation for stochastic methods
- statistics: Statistical tests and analysis tools
- visualization: Plotting utilities for results
"""

from eispy2d.utils.draw import (
    square,
    circle,
    ellipse,
    triangle,
    polygon,
    star4,
    star5,
    star6,
    ring,
    cross,
    line,
    rhombus,
    trapezoid,
    parallelogram,
    random,
    wave,
    random_waves,
    random_gaussians,
)

from eispy2d.solvers.inverse.regularization import (
    Regularization,
    Tikhonov,
    Landweber,
    ConjugatedGradient,
    LeastSquares,
    SingularValueDecomposition,
    TIK_FIXED,
    TIK_MOZOROV,
    TIK_LCURVE,
)

from eispy2d.utils.stopcriteria import StopCriteria


from eispy2d.utils.statisticsutils import (
    compare1sample,
    compare2samples,
    compare_multiple,
    confint,
    confintplot,
    homoscedasticityplot,
    rcbd,
    factorial_analysis,
    dunnetttest,
)


__all__ = [
    # Drawing
    'square',
    'circle',
    'ellipse',
    'triangle',
    'polygon',
    'star4',
    'star5',
    'star6',
    'ring',
    'cross',
    'line',
    'rhombus',
    'trapezoid',
    'parallelogram',
    'random',
    'wave',
    'random_waves',
    'random_gaussians',
    
    # Regularization
    'Regularization',
    'Tikhonov',
    'Landweber',
    'ConjugatedGradient',
    'LeastSquares',
    'SingularValueDecomposition',
    'TIK_FIXED',
    'TIK_MOZOROV',
    'TIK_LCURVE',
    
    # Other utilities
    'StopCriteria',
    'OutputMode',
    
    # Statistics
    'compare1sample',
    'compare2samples',
    'compare_multiple',
    'confint',
    'confintplot',
    'normalityplot',
    'homoscedasticityplot',
    'rcbd',
    'factorial_analysis',
    'dunnetttest',
    
    # Visualization
    'add_image',
    'add_plot',
    'add_box',
    'add_violin',
    'get_figure',
    'indicator_label',
]