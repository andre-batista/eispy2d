# tests/test_discretization.py
"""
Unit tests for eispy2d.discretization module.
"""

import sys
import os
sys.path.insert(1, '../../../eispy2d/')

import unittest
import numpy as np


from eispy2d.core.configuration import Configuration
from eispy2d.discretization.discretization import Discretization


class TestDiscretizationBase(unittest.TestCase):
    """Test base Discretization abstract class."""
    
    def test_discretization_abstract(self):
        """Test that Discretization cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            Discretization()
    
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


class TestCollocation(unittest.TestCase):
    """Test Collocation discretization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            name='collocation_test',
            wavelength=1.0,
            number_measurements=16,
            number_sources=8,
            image_size=[2.0, 2.0],
            background_permittivity=1.0
        )
    
    def test_collocation_import(self):
        """Test Collocation import."""
        try:
            from eispy2d.discretization.collocation import Collocation
            
            collocation = Collocation(
                configuration=self.config,
                trial='pulse',
                elements=(16, 16),
                alias='test_clc'
            )
            
            self.assertEqual(collocation.alias, 'test_clc')
            self.assertEqual(collocation.elements, (16, 16))
            self.assertEqual(collocation.trial, 'pulse')
        except ImportError:
            self.skipTest("Collocation not available")
    
    def test_collocation_with_int_elements(self):
        """Test Collocation with integer elements."""
        try:
            from eispy2d.discretization.collocation import Collocation
            
            collocation = Collocation(
                configuration=self.config,
                trial='pulse',
                elements=16
            )
            
            self.assertEqual(collocation.elements, (16, 16))
        except ImportError:
            self.skipTest("Collocation not available")
    
    def test_collocation_copy(self):
        """Test Collocation copy method."""
        try:
            from eispy2d.discretization.collocation import Collocation
            
            original = Collocation(
                configuration=self.config,
                trial='pulse',
                elements=(16, 16),
                alias='original'
            )
            
            copy = original.copy()
            self.assertEqual(copy.alias, 'original')
            self.assertEqual(copy.elements, original.elements)
            self.assertEqual(copy.trial, original.trial)
        except ImportError:
            self.skipTest("Collocation not available")
    
    def test_collocation_string(self):
        """Test Collocation __str__ method."""
        try:
            from eispy2d.discretization.collocation import Collocation
            
            collocation = Collocation(
                configuration=self.config,
                trial='pulse',
                elements=(16, 16),
                alias='clc'
            )
            
            str_repr = str(collocation)
            self.assertIn('16x16', str_repr)
            self.assertIn('pulse', str_repr)
        except ImportError:
            self.skipTest("Collocation not available")


class TestRichmond(unittest.TestCase):
    """Test Richmond discretization."""
    
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
        except ImportError:
            self.skipTest("Richmond not available")
    
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
        except ImportError:
            self.skipTest("Richmond not available")
    
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
        except ImportError:
            self.skipTest("Richmond not available")
    
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
        except ImportError:
            self.skipTest("Richmond not available")
    
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
        except ImportError:
            self.skipTest("Richmond not available")


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
        except ImportError:
            self.skipTest("Kernel functions not available")
    
    def test_kernel_gsx_import(self):
        """Test kernel_GSX import."""
        try:
            from eispy2d.discretization.collocation import kernel_GSX
            
            K = kernel_GSX(self.GS, self.X)
            self.assertEqual(K.shape, (self.NM, self.N))
        except ImportError:
            self.skipTest("Kernel functions not available")
    
    def test_kernel_gdx_import(self):
        """Test kernel_GDX import."""
        try:
            from eispy2d.discretization.collocation import kernel_GDX
            
            K = kernel_GDX(self.GD, self.X)
            self.assertEqual(K.shape, (self.N, self.N))
            
            # Check diagonal modification
            I = np.eye(self.N)
            expected = I - self.GD * self.X[:, np.newaxis]
            np.testing.assert_array_almost_equal(K, expected, decimal=5)
        except ImportError:
            self.skipTest("Kernel functions not available")
    
    def test_kernel_gde_import(self):
        """Test kernel_GDE import."""
        try:
            from eispy2d.discretization.collocation import kernel_GDE
            
            K = kernel_GDE(self.GD, self.E)
            self.assertEqual(K.shape, (self.N, self.N, self.NS))
        except ImportError:
            self.skipTest("Kernel functions not available")
    
    def test_lhs_xei_import(self):
        """Test lhs_XEi import."""
        try:
            from eispy2d.discretization.collocation import lhs_XEi
            
            lhs = lhs_XEi(self.X, self.E)
            self.assertEqual(lhs.shape, (self.N, self.NS))
            
            # Verify computation
            for s in range(self.NS):
                np.testing.assert_array_almost_equal(lhs[:, s], self.X * self.E[:, s])
        except ImportError:
            self.skipTest("Kernel functions not available")


if __name__ == '__main__':
    unittest.main()