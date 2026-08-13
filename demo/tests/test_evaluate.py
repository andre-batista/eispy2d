"""
Unit tests for eispy2d.api.evaluate function.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eispy2d.api import api
from eispy2d.core.configuration import Configuration
from eispy2d.core.inputdata import InputData
from eispy2d.core.result import Result
from eispy2d.core.error import MissingInputError
from eispy2d.discretization import richmond as ric
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.solvers.inverse import bim
from eispy2d.solvers.inverse import backprop
from eispy2d.solvers.inverse.regularization import Regularization as reg
from eispy2d.utils import stopcriteria as stp


def test_evaluate(scattered_field, incident_field, GS, GD, recover_resolution):
    """Test algorithm that returns zero contrast."""
    contrast = np.zeros(recover_resolution, dtype=complex)
    recon_scattered_field = scattered_field.copy()
    return recon_scattered_field, contrast


def alg(scattered_field, incident_field, GS, GD, resolution):
    """Simple algorithm using BIM-like approach."""
    import numpy as np
    from scipy.sparse import spdiags
    from numpy.linalg import inv
    
    chi = np.zeros(resolution, dtype=complex)
    N = resolution[0] * resolution[1]
    C = spdiags(chi.reshape(-1), 0, N, N)
    I = np.eye(N, dtype=complex)
    L = inv(I - GD @ C)
    recon_scattered_field = GS @ C @ L @ incident_field
    return recon_scattered_field, chi


class TestEvaluateAPI(unittest.TestCase):
    """Test cases for evaluate function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Default parameters for evaluation
        self.params = {
            "wavelength": 1.0,
            "image_size": [0.8, 0.8],
            "number_measurements": 10,
            "number_sources": 10,
            "observation_radius": 1.0,
            "background_permittivity": 1.0,
            "resolution": (32, 32),
            "noise_level": 1.0,
            "disp": False,
            "shape": "triangle"
        }
        
        # Simple test algorithm
        def test_algorithm(scattered_field, incident_field, GS, GD, resolution):
            """Simple test algorithm that returns zero contrast."""
            chi = np.zeros(resolution, dtype=complex)
            recon_scattered = scattered_field.copy()
            return recon_scattered, chi
        
        self.test_algorithm = test_algorithm
    
    def test_evaluate_returns_result(self):
        """Test that evaluate returns a Result object."""
        result = api.evaluate(self.test_algorithm, params=self.params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate returns Result object")
    
    def test_evaluate_with_default_params(self):
        """Test evaluate with default parameters."""
        result = api.evaluate(self.test_algorithm)
        self.assertIsInstance(result, Result)
        self.assertEqual(result.configuration.NM, 10)
        self.assertEqual(result.configuration.NS, 10)
        print("✓ evaluate works with default parameters")
    
    def test_evaluate_with_triangle_shape(self):
        """Test evaluate with triangle shape."""
        params = self.params.copy()
        params["shape"] = "triangle"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with triangle shape")
    
    def test_evaluate_with_circle_shape(self):
        """Test evaluate with circle shape."""
        params = self.params.copy()
        params["shape"] = "circle"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with circle shape")
    
    def test_evaluate_with_square_shape(self):
        """Test evaluate with square shape."""
        params = self.params.copy()
        params["shape"] = "square"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with square shape")
    
    def test_evaluate_with_star_shape(self):
        """Test evaluate with star shape."""
        params = self.params.copy()
        params["shape"] = "star5"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with star shape")
    
    def test_evaluate_with_random_shape(self):
        """Test evaluate with random shape."""
        params = self.params.copy()
        params["shape"] = "random"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with random shape")
    
    def test_evaluate_with_gaussian_shape(self):
        """Test evaluate with random_gaussians shape."""
        params = self.params.copy()
        params["shape"] = "random_gaussians"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with random gaussians shape")
    
    def test_evaluate_with_cross_shape(self):
        """Test evaluate with cross shape."""
        params = self.params.copy()
        params["shape"] = "cross"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with cross shape")
    
    def test_evaluate_with_ellipse_shape(self):
        """Test evaluate with ellipse shape."""
        params = self.params.copy()
        params["shape"] = "ellipse"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with ellipse shape")
    
    def test_evaluate_with_parallelogram_shape(self):
        """Test evaluate with parallelogram shape."""
        params = self.params.copy()
        params["shape"] = "parallelogram"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with parallelogram shape")
    
    def test_evaluate_with_rhombus_shape(self):
        """Test evaluate with rhombus shape."""
        params = self.params.copy()
        params["shape"] = "rhombus"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with rhombus shape")
    
    def test_evaluate_with_trapezoid_shape(self):
        """Test evaluate with trapezoid shape."""
        params = self.params.copy()
        params["shape"] = "trapezoid"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with trapezoid shape")
    
    def test_evaluate_with_ring_shape(self):
        """Test evaluate with ring shape."""
        params = self.params.copy()
        params["shape"] = "ring"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with ring shape")
    
    def test_evaluate_with_polygon_shape(self):
        """Test evaluate with polygon shape."""
        params = self.params.copy()
        params["shape"] = "polygon"
        result = api.evaluate(self.test_algorithm, params=params)
        self.assertIsInstance(result, Result)
        print("✓ evaluate works with polygon shape")
    
    def test_evaluate_invalid_shape(self):
        """Test evaluate with invalid shape raises error."""
        params = self.params.copy()
        params["shape"] = "invalid_shape"
        
        # O erro deve ser ValueError, não MissingAttributesError
        with self.assertRaises(ValueError):
            api.evaluate(self.test_algorithm, params=params)
        print("✓ evaluate raises error for invalid shape")
    
    def test_test_evaluate_algorithm(self):
        """Test the test_evaluate algorithm."""
        params = self.params.copy()
        result = api.evaluate(test_evaluate, params=params)
        self.assertIsInstance(result, Result)
        print("✓ test_evaluate algorithm works")
    
    def test_alg_algorithm(self):
        """Test the alg algorithm."""
        params = self.params.copy()
        result = api.evaluate(alg, params=params)
        self.assertIsInstance(result, Result)
        print("✓ alg algorithm works")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Running Evaluate API Tests")
    print("=" * 60 + "\n")
    unittest.main(argv=[''], exit=False, verbosity=2)