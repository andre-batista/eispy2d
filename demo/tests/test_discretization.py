# tests/test_discretization.py
"""
Unit tests for eispy2d.discretization module.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eispy2d.core.configuration import Configuration
from eispy2d.discretization.discretization import Discretization


class TestDiscretizationBase(unittest.TestCase):
    """Test base Discretization abstract class."""
    
    def test_discretization_abstract(self):
        """Test that Discretization cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            Discretization()
        print("✓ Discretization abstract test passed")
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            name='disc_test',
            wavelength=1.0,
            number_measurements=16,
            number_sources=8,
            image_size=[2.0, 2.0],
            background_permittivity=1.0
        )


class TestRichmond(unittest.TestCase):
    """Test Richmond discretization (concrete implementation)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            name='richmond_test',
            wavelength=1.0,
            number_measurements=16,
            number_sources=8,
            image_size=[2.0, 2.0],
            background_permittivity=1.0
        )
    
    def test_richmond_import(self):
        """Test Richmond import."""
        try:
            from eispy2d.discretization.richmond import Richmond
            
            richmond = Richmond(
                configuration=self.config,
                elements=(16, 16),
                state=True,
                alias='test_ric'
            )
            
            self.assertEqual(richmond.alias, 'test_ric')
            self.assertEqual(richmond.elements, (16, 16))
            print("✓ Richmond import test passed")
        except ImportError as e:
            self.skipTest(f"Richmond not available: {e}")
    
    def test_richmond_with_int_elements(self):
        """Test Richmond with integer elements."""
        try:
            from eispy2d.discretization.richmond import Richmond
            
            richmond = Richmond(
                configuration=self.config,
                elements=16,
                state=True
            )
            
            self.assertEqual(richmond.elements, (16, 16))
            print("✓ Richmond with int elements test passed")
        except ImportError as e:
            self.skipTest(f"Richmond not available: {e}")
    
    def test_richmond_copy(self):
        """Test Richmond copy method."""
        try:
            from eispy2d.discretization.richmond import Richmond
            
            original = Richmond(
                configuration=self.config,
                elements=(16, 16),
                state=True,
                alias='original'
            )
            
            # Create a copy
            copy = original.copy()
            
            # Check that copy has same elements and configuration
            self.assertEqual(copy.elements, original.elements)
            self.assertEqual(copy.configuration.name, original.configuration.name)
            
            # Verify it's a different object (deep copy)
            self.assertIsNot(copy, original)
            
            print("✓ Richmond copy test passed")
        except ImportError as e:
            self.skipTest(f"Richmond not available: {e}")
    
    def test_richmond_string(self):
        """Test Richmond __str__ method."""
        try:
            from eispy2d.discretization.richmond import Richmond
            
            richmond = Richmond(
                configuration=self.config,
                elements=(16, 16),
                state=True,
                alias='ric'
            )
            
            str_repr = str(richmond)
            self.assertIn('16x16', str_repr)
            print("✓ Richmond string test passed")
        except ImportError as e:
            self.skipTest(f"Richmond not available: {e}")
    
    def test_richmond_green_functions(self):
        """Test Richmond Green's function matrices."""
        try:
            from eispy2d.discretization.richmond import Richmond
            
            richmond = Richmond(
                configuration=self.config,
                elements=(16, 16),
                state=True
            )
            
            self.assertIsNotNone(richmond.GS)
            self.assertIsNotNone(richmond.GD)
            self.assertEqual(richmond.GS.shape[0], self.config.NM)
            self.assertEqual(richmond.GS.shape[1], 16*16)
            self.assertEqual(richmond.GD.shape[0], 16*16)
            self.assertEqual(richmond.GD.shape[1], 16*16)
            print("✓ Richmond Green's functions test passed")
        except ImportError as e:
            self.skipTest(f"Richmond not available: {e}")
    
    def test_richmond_without_state(self):
        """Test Richmond without state Green's function."""
        try:
            from eispy2d.discretization.richmond import Richmond
            
            richmond = Richmond(
                configuration=self.config,
                elements=(16, 16),
                state=False
            )
            
            self.assertIsNotNone(richmond.GS)
            self.assertIsNone(richmond.GD)
            print("✓ Richmond without state test passed")
        except ImportError as e:
            self.skipTest(f"Richmond not available: {e}")
    
    def test_richmond_scattered_field_computation(self):
        """Test scattered field computation."""
        try:
            from eispy2d.discretization.richmond import Richmond
            
            richmond = Richmond(
                configuration=self.config,
                elements=(16, 16),
                state=False
            )
            
            # Create dummy contrast and total field
            contrast = np.random.randn(16*16) + 1j * np.random.randn(16*16)
            total_field = np.random.randn(16*16, 8) + 1j * np.random.randn(16*16, 8)
            
            scattered = richmond.scattered_field(contrast=contrast, total_field=total_field)
            self.assertEqual(scattered.shape[0], self.config.NM)
            self.assertEqual(scattered.shape[1], 8)
            print("✓ Richmond scattered field test passed")
        except ImportError as e:
            self.skipTest(f"Richmond not available: {e}")
    
    def test_richmond_contrast_image_conversion(self):
        """Test contrast to image conversion."""
        try:
            from eispy2d.discretization.richmond import Richmond
            
            richmond = Richmond(
                configuration=self.config,
                elements=(8, 8),
                state=False
            )
            
            # Create dummy coefficients
            coefficients = np.random.randn(64) + 1j * np.random.randn(64)
            
            # Convert to higher resolution image
            image = richmond.contrast_image(coefficients, (32, 32))
            self.assertEqual(image.shape, (32, 32))
            print("✓ Richmond contrast image conversion test passed")
        except ImportError as e:
            self.skipTest(f"Richmond not available: {e}")


class TestKernelFunctions(unittest.TestCase):
    """Test kernel functions from collocation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.NM = 16  # measurements
        self.N = 64   # domain points
        self.NS = 8   # sources
        
        self.GS = np.random.randn(self.NM, self.N) + 1j * np.random.randn(self.NM, self.N)
        self.GD = np.random.randn(self.N, self.N) + 1j * np.random.randn(self.N, self.N)
        self.E = np.random.randn(self.N, self.NS) + 1j * np.random.randn(self.N, self.NS)
        self.X = np.random.randn(self.N) + 1j * np.random.randn(self.N)
    
    def test_kernel_gse_import(self):
        """Test kernel_GSE import."""
        try:
            from eispy2d.discretization.collocation import kernel_GSE
            
            K = kernel_GSE(self.GS, self.E)
            expected_shape = (self.NM * self.NS, self.N)
            self.assertEqual(K.shape, expected_shape)
            print("✓ kernel_GSE test passed")
        except ImportError as e:
            self.skipTest(f"Kernel functions not available: {e}")
    
    def test_kernel_gsx_import(self):
        """Test kernel_GSX import."""
        try:
            from eispy2d.discretization.collocation import kernel_GSX
            
            K = kernel_GSX(self.GS, self.X)
            self.assertEqual(K.shape, (self.NM, self.N))
            print("✓ kernel_GSX test passed")
        except ImportError as e:
            self.skipTest(f"Kernel functions not available: {e}")
    
    def test_kernel_gdx_import(self):
        """Test kernel_GDX import - check structure, not exact values."""
        try:
            from eispy2d.discretization.collocation import kernel_GDX
            
            K = kernel_GDX(self.GD, self.X)
            self.assertEqual(K.shape, (self.N, self.N))
            
            # Just verify the diagonal is modified (1 - GD[n,n]*X[n])
            I = np.eye(self.N)
            for n in range(min(5, self.N)):
                expected_diag = 1 - self.GD[n, n] * self.X[n]
                self.assertAlmostEqual(K[n, n], expected_diag, places=5)
            
            print("✓ kernel_GDX test passed")
        except ImportError as e:
            self.skipTest(f"Kernel functions not available: {e}")
    
    def test_kernel_gde_import(self):
        """Test kernel_GDE import."""
        try:
            from eispy2d.discretization.collocation import kernel_GDE
            
            K = kernel_GDE(self.GD, self.E)
            self.assertEqual(K.shape, (self.N, self.N, self.NS))
            print("✓ kernel_GDE test passed")
        except ImportError as e:
            self.skipTest(f"Kernel functions not available: {e}")
    
    def test_lhs_xei_import(self):
        """Test lhs_XEi import."""
        try:
            from eispy2d.discretization.collocation import lhs_XEi
            
            lhs = lhs_XEi(self.X, self.E)
            self.assertEqual(lhs.shape, (self.N, self.NS))
            
            # Verify computation for first source
            for s in range(min(3, self.NS)):
                np.testing.assert_array_almost_equal(lhs[:, s], self.X * self.E[:, s], decimal=5)
            
            print("✓ lhs_XEi test passed")
        except ImportError as e:
            self.skipTest(f"Kernel functions not available: {e}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Discretization Tests")
    print("="*60 + "\n")
    unittest.main()