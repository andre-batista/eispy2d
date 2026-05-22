# tests/test_inputdata.py
"""
Unit tests for eispy2d.data.inputdata module.
"""

import sys
import os
sys.path.insert(1, '../../../eispy2d/')


import unittest
import numpy as np

from eispy2d.core.configuration import Configuration
from eispy2d.data.inputdata import InputData
from eispy2d.data.result import (
    RESIDUAL_NORM_ERROR,
    REL_PERMITTIVITY_PAD_ERROR,
    INDICATOR_SET
)
from eispy2d.core.error import MissingInputError, MissingAttributesError


class TestInputData(unittest.TestCase):
    """Test cases for InputData class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            name='test_config',
            wavelength=1.0,
            number_measurements=16,
            number_sources=16,
            image_size=[2.0, 2.0],
            observation_radius=3.0,
            background_permittivity=1.0
        )
        
        # Create sample data
        self.resolution = (32, 32)
        self.epsilon_r = np.ones(self.resolution) * 2.0
        self.epsilon_r[14:18, 14:18] = 3.0  # Object in center
        
        self.input_data = InputData(
            name='test_input',
            configuration=self.config,
            resolution=self.resolution,
            rel_permittivity=self.epsilon_r,
            noise=1.0
        )
    
    def test_inputdata_creation(self):
        """Test InputData object creation."""
        self.assertEqual(self.input_data.name, 'test_input')
        self.assertEqual(self.input_data.resolution, (32, 32))
        self.assertEqual(self.input_data.noise, 1.0)
        self.assertIsNotNone(self.input_data.rel_permittivity)
        self.assertIsNone(self.input_data.scattered_field)
    
    def test_inputdata_without_rel_permittivity(self):
        """Test InputData creation without permittivity."""
        sigma = np.zeros(self.resolution)
        sigma[14:18, 14:18] = 0.1
        
        input_data = InputData(
            name='conductivity_test',
            configuration=self.config,
            resolution=self.resolution,
            conductivity=sigma
        )
        
        self.assertIsNotNone(input_data.conductivity)
        self.assertIsNone(input_data.rel_permittivity)
    
    def test_inputdata_missing_name(self):
        """Test that missing name raises error."""
        with self.assertRaises(MissingInputError):
            InputData(name=None, configuration=self.config)
    
    def test_inputdata_missing_configuration(self):
        """Test that missing configuration raises error."""
        with self.assertRaises(MissingInputError):
            InputData(name='test', configuration=None, resolution=(32, 32))
    
    def test_copy_method(self):
        """Test InputData copy method."""
        input_copy = self.input_data.copy()
        
        self.assertEqual(input_copy.name, self.input_data.name)
        self.assertEqual(input_copy.resolution, self.input_data.resolution)
        np.testing.assert_array_equal(input_copy.rel_permittivity, self.input_data.rel_permittivity)
        
        # Modify copy shouldn't affect original
        input_copy.name = 'modified'
        input_copy.rel_permittivity[0, 0] = 5.0
        self.assertNotEqual(self.input_data.name, input_copy.name)
        self.assertNotEqual(self.input_data.rel_permittivity[0, 0], input_copy.rel_permittivity[0, 0])
    
    def test_compute_dnl(self):
        """Test DNL (Degree of Non-Linearity) computation."""
        self.input_data.compute_dnl()
        self.assertIsNotNone(self.input_data.dnl)
        self.assertGreater(self.input_data.dnl, 0)
    
    def test_indicators_default(self):
        """Test default indicators."""
        self.assertEqual(len(self.input_data.indicators), len(INDICATOR_SET))
        self.assertIn(RESIDUAL_NORM_ERROR, self.input_data.indicators)
        self.assertIn(REL_PERMITTIVITY_PAD_ERROR, self.input_data.indicators)
    
    def test_indicators_custom(self):
        """Test custom indicators."""
        input_data = InputData(
            name='custom_indicators',
            configuration=self.config,
            resolution=self.resolution,
            indicators=[RESIDUAL_NORM_ERROR]
        )
        
        self.assertEqual(len(input_data.indicators), 1)
        self.assertIn(RESIDUAL_NORM_ERROR, input_data.indicators)
    
    def test_string_representation(self):
        """Test __str__ method."""
        str_repr = str(self.input_data)
        self.assertIn('test_input', str_repr)
        self.assertIn('test_config', str_repr)
        self.assertIn('32x32', str_repr)
    
    def test_save_and_import(self):
        """Test save and import functionality."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'test_input')
            self.input_data.save(file_path=file_path + '_')
            
            # Create new input data and import
            new_input = InputData(name='temp', configuration=self.config)
            new_input.importdata('test_input', file_path=tmpdir + '/')
            
            self.assertEqual(new_input.name, self.input_data.name)
            self.assertEqual(new_input.resolution, self.input_data.resolution)
            np.testing.assert_array_equal(
                new_input.rel_permittivity, 
                self.input_data.rel_permittivity
            )


class TestInputDataFieldData(unittest.TestCase):
    """Test InputData with field data."""
    
    def setUp(self):
        """Set up test fixtures with field data."""
        self.config = Configuration(
            name='test_config',
            wavelength=1.0,
            number_measurements=16,
            number_sources=8,
            image_size=[2.0, 2.0]
        )
        
        # Create synthetic scattered field
        self.scattered_field = np.random.randn(16, 8) + 1j * np.random.randn(16, 8)
        
        self.input_data = InputData(
            name='field_test',
            configuration=self.config,
            resolution=(32, 32),
            scattered_field=self.scattered_field
        )
    
    def test_scattered_field_storage(self):
        """Test scattered field storage."""
        np.testing.assert_array_equal(self.input_data.scattered_field, self.scattered_field)
    
    def test_scattered_field_shape(self):
        """Test scattered field shape validation."""
        self.assertEqual(self.input_data.scattered_field.shape[0], self.config.NM)
        self.assertEqual(self.input_data.scattered_field.shape[1], self.config.NS)
    
    def test_total_field_storage(self):
        """Test total field storage."""
        total_field = np.random.randn(1024, 8) + 1j * np.random.randn(1024, 8)
        self.input_data.total_field = total_field
        np.testing.assert_array_equal(self.input_data.total_field, total_field)
    
    def test_total_field_shape(self):
        """Test total field shape validation."""
        total_field = np.random.randn(32*32, 8) + 1j * np.random.randn(32*32, 8)
        self.input_data.total_field = total_field
        self.assertEqual(self.input_data.total_field.shape[0], 32*32)
        self.assertEqual(self.input_data.total_field.shape[1], self.config.NS)


if __name__ == '__main__':
    unittest.main()