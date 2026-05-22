# eispy2d/solvers/forward/__init__.py

"""
Forward solvers for electromagnetic scattering problems.

Implements methods for computing scattered fields from known material
distributions:
- MoM_CG_FFT: Method of Moments with Conjugate Gradient FFT
- Analytical: Analytical solution for cylindrical scatterers
- FFTProduct: FFT-based convolution for Green's function
"""

from eispy2d.solvers.forward.mom_cg_fft import MoM_CG_FFT
from eispy2d.solvers.forward.analytical import Analytical
from eispy2d.solvers.forward.fftproduct import FFTProduct

__all__ = [
    'MoM_CG_FFT',
    'Analytical',
    'FFTProduct',
]