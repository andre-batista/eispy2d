# tests/test_configuration.py
"""
Unit tests for eispy2d.core.configuration module.
"""

import sys
import os
sys.path.insert(1, '../../../eispy2d/')

import unittest
import numpy as np


from eispy2d.core.configuration import (
    Configuration, 
    degrees_of_freedom, 
    get_angles,
    get_bounds,
    get_coordinates_ddomain,
    get_coordinates_sdomain,
    get_relative_permittivity,
    get_conductivity,
    get_contrast_map,
    compute_wavelength,
    compute_frequency,
    compute_wavenumber,
)
from eispy2d.core.error import MissingInputError, WrongTypeInput


class TestConfiguration(unittest.TestCase):
    """Test cases for Configuration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            name='test_config',
            wavelength=1.0,
            number_measurements=32,
            number_sources=32,
            image_size=[4.0, 4.0],
            observation_radius=6.0,
            background_permittivity=1.0,
            background_conductivity=0.0,
            magnitude=1.0,
            perfect_dielectric=True,
            good_conductor=False
        )
    
    def test_configuration_creation(self):
        """Test that Configuration object creates correctly."""
        self.assertEqual(self.config.name, 'test_config')
        self.assertEqual(self.config.lambda_b, 1.0)
        self.assertEqual(self.config.NM, 32)
        self.assertEqual(self.config.NS, 32)
        self.assertEqual(self.config.Lx, 4.0)
        self.assertEqual(self.config.Ly, 4.0)
        self.assertEqual(self.config.Ro, 6.0)
        self.assertEqual(self.config.epsilon_rb, 1.0)
        self.assertEqual(self.config.sigma_b, 0.0)
        self.assertEqual(self.config.E0, 1.0)
        self.assertTrue(self.config.perfect_dielectric)
        self.assertFalse(self.config.good_conductor)
    
    def test_configuration_with_frequency(self):
        """Test configuration using frequency instead of wavelength."""
        config = Configuration(
            name='freq_test',
            frequency=300e6,  # 300 MHz
            image_size=[1.0, 1.0],
            background_permittivity=1.0
        )
        # In free space, lambda = c/f = 3e8/3e8 = 1.0 m
        self.assertAlmostEqual(config.lambda_b, 1.0, places=2)
        self.assertAlmostEqual(config.kb, 2 * np.pi, places=2)
    
    def test_configuration_missing_name(self):
        """Test that missing name raises error."""
        with self.assertRaises(MissingInputError):
            Configuration(name=None, wavelength=1.0)
    
    def test_configuration_missing_wavelength_and_frequency(self):
        """Test that missing both wavelength and frequency raises error."""
        with self.assertRaises(Exception):  # Should raise some error
            Configuration(name='test', image_size=[1.0, 1.0])
    
    def test_copy_method(self):
        """Test configuration copy method."""
        config_copy = self.config.copy()
        self.assertEqual(config_copy.name, self.config.name)
        self.assertEqual(config_copy.lambda_b, self.config.lambda_b)
        
        # Modify copy shouldn't affect original
        config_copy.name = 'modified'
        self.assertNotEqual(self.config.name, config_copy.name)
    
    def test_string_representation(self):
        """Test __str__ method."""
        str_repr = str(self.config)
        self.assertIn('test_config', str_repr)
        self.assertIn('Configuration', str_repr)
    
    def test_save_and_import(self):
        """Test save and import functionality."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'test_config')
            self.config.save(file_path=file_path + '_')
            
            # Create new config and import
            new_config = Configuration(name='temp')
            new_config.importdata('test_config', file_path=tmpdir + '/')
            
            self.assertEqual(new_config.name, self.config.name)
            self.assertEqual(new_config.lambda_b, self.config.lambda_b)


class TestConfigurationProperties(unittest.TestCase):
    """Test configuration properties and calculations."""
    
    def test_get_angles(self):
        """Test angle generation."""
        angles = get_angles(8)
        self.assertEqual(len(angles), 8)
        self.assertAlmostEqual(angles[0], 0.0)
        self.assertAlmostEqual(angles[-1], 2 * np.pi - angles[1], places=5)
    
    def test_get_angles_single(self):
        """Test angle generation with single sample."""
        angles = get_angles(1)
        self.assertEqual(len(angles), 1)
        self.assertAlmostEqual(angles[0], 0.0)
    
    def test_get_bounds(self):
        """Test bound calculation."""
        bounds = get_bounds(2.0)
        self.assertEqual(bounds, (-1.0, 1.0))
        
        bounds = get_bounds(3.0)
        self.assertEqual(bounds, (-1.5, 1.5))
    
    def test_get_coordinates_ddomain_with_configuration(self):
        """Test D-domain coordinate generation with configuration."""
        config = Configuration(name='test', wavelength=1.0, image_size=[2.0, 2.0])
        x, y = get_coordinates_ddomain(configuration=config, resolution=(10, 10))
        
        self.assertEqual(x.shape, (10, 10))
        self.assertEqual(y.shape, (10, 10))
        self.assertAlmostEqual(x[0, 0], -0.9, places=1)
        self.assertAlmostEqual(y[0, 0], -0.9, places=1)
    
    def test_get_coordinates_ddomain_with_dx_dy(self):
        """Test D-domain coordinate generation with dx, dy."""
        x, y = get_coordinates_ddomain(
            dx=0.1, dy=0.1, 
            xmin=-1.0, xmax=1.0, 
            ymin=-1.0, ymax=1.0
        )
        self.assertEqual(x.shape, (20, 20))
        self.assertEqual(y.shape, (20, 20))
    
    def test_get_coordinates_sdomain(self):
        """Test S-domain coordinate generation."""
        x, y = get_coordinates_sdomain(radius=1.0, n_samples=8)
        
        self.assertEqual(len(x), 8)
        self.assertEqual(len(y), 8)
        
        # Check that points lie on circle
        r = np.sqrt(x**2 + y**2)
        self.assertTrue(np.allclose(r, 1.0))
    
    def test_degrees_of_freedom(self):
        """Test degrees of freedom calculation."""
        dof = degrees_of_freedom(
            object_radius=0.5,
            wavelength=1.0,
            epsilon_r=2.0
        )
        self.assertIsInstance(dof, int)
        self.assertGreater(dof, 0)
        
        # Test with frequency instead of wavelength
        dof2 = degrees_of_freedom(
            object_radius=0.5,
            frequency=300e6,
            epsilon_r=1.0
        )
        self.assertIsInstance(dof2, int)
    
    def test_compute_wavelength(self):
        """Test wavelength computation."""
        wavelength = compute_wavelength(frequency=300e6, epsilon_r=1.0)
        self.assertAlmostEqual(wavelength, 1.0, places=2)
        
        wavelength_medium = compute_wavelength(frequency=300e6, epsilon_r=4.0)
        self.assertAlmostEqual(wavelength_medium, 0.5, places=2)
    
    def test_compute_frequency(self):
        """Test frequency computation."""
        frequency = compute_frequency(wavelength=1.0, epsilon_r=1.0)
        self.assertAlmostEqual(frequency, 299792458.0, places=0)  # c = 299792458
    
    def test_compute_wavenumber(self):
        """Test wavenumber computation."""
        k = compute_wavenumber(frequency=300e6, epsilon_r=1.0)
        self.assertAlmostEqual(k, 2 * np.pi, places=2)
    
    def test_get_relative_permittivity(self):
        """Test contrast to permittivity conversion."""
        eps_r = get_relative_permittivity(chi=0.25, epsilon_rb=1.0)
        self.assertEqual(eps_r, 1.25)
        
        eps_r = get_relative_permittivity(chi=-0.5, epsilon_rb=4.0)
        self.assertEqual(eps_r, 2.0)
    
    def test_get_conductivity(self):
        """Test contrast to conductivity conversion."""
        import numpy as np
        omega = 2 * np.pi * 300e6
        
        sigma = get_conductivity(chi=0.1j, omega=omega, epsilon_rb=1.0, sigma_b=0.0)
        self.assertGreater(sigma, 0)
    
    def test_get_contrast_map(self):
        """Test contrast map computation."""
        import numpy as np
        
        eps_r_map = np.ones((10, 10)) * 2.0
        sigma_map = np.zeros((10, 10))
        
        config = Configuration(name='test', wavelength=1.0, image_size=[1.0, 1.0])
        contrast = get_contrast_map(
            epsilon_r=eps_r_map,
            sigma=sigma_map,
            configuration=config
        )
        
        self.assertEqual(contrast.shape, (10, 10))
        self.assertTrue(np.all(contrast == 1.0))  # eps_r/eps_rb - 1 = 2/1 - 1 = 1


class TestPhysicalConstants(unittest.TestCase):
    """Test physical constant calculations."""
    
    def test_wavelength_relation(self):
        """Test wavelength-frequency relation."""
        wavelength = compute_wavelength(frequency=1e9, epsilon_r=1.0)
        frequency = compute_frequency(wavelength=wavelength, epsilon_r=1.0)
        self.assertAlmostEqual(frequency, 1e9, places=0)
    
    def test_permittivity_contrast_relation(self):
        """Test permittivity and contrast relation."""
        epsilon_rb = 2.0
        epsilon_r = 3.0
        chi = (epsilon_r - epsilon_rb) / epsilon_rb
        
        recovered_eps = get_relative_permittivity(chi, epsilon_rb)
        self.assertAlmostEqual(recovered_eps, epsilon_r)
    
    def test_conductivity_contrast_relation(self):
        """Test conductivity and contrast relation."""
        import numpy as np
        
        epsilon_rb = 1.0
        sigma_b = 0.0
        omega = 2 * np.pi * 1e9
        
        # For perfect conductor, chi = -1 (since epsilon_r=1, sigma large)
        sigma = get_conductivity(chi=-1j, omega=omega, epsilon_rb=epsilon_rb, sigma_b=sigma_b)
        self.assertGreater(sigma, 0)


if __name__ == '__main__':
    unittest.main()