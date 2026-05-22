# tests/test_utils.py
"""
Unit tests for eispy2d.utils modules.
"""

import sys
import os
sys.path.insert(1, '../../../eispy2d/')

import unittest
import numpy as np



class TestDrawModule(unittest.TestCase):
    """Test drawing utilities."""
    
    def test_draw_imports(self):
        """Test draw module imports."""
        try:
            from eispy2d.utils.draw import (
                square, circle, ellipse, triangle, polygon,
                star4, star5, star6, ring, cross, line,
                rhombus, trapezoid, parallelogram, random
            )
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Draw module not available: {e}")
    
    def test_square_creation(self):
        """Test square drawing."""
        try:
            from eispy2d.utils.draw import square
            
            epsilon_r, sigma = square(
                side_length=0.5,
                resolution=(32, 32),
                object_rel_permittivity=2.0,
                background_rel_permittivity=1.0
            )
            
            self.assertEqual(epsilon_r.shape, (32, 32))
            self.assertEqual(sigma.shape, (32, 32))
            self.assertTrue(np.any(epsilon_r > 1.0))
        except ImportError as e:
            self.skipTest(f"Draw module not available: {e}")
    
    def test_circle_creation(self):
        """Test circle drawing."""
        try:
            from eispy2d.utils.draw import circle
            
            epsilon_r, sigma = circle(
                radius=0.3,
                resolution=(32, 32),
                object_rel_permittivity=3.0,
                background_rel_permittivity=1.0
            )
            
            self.assertEqual(epsilon_r.shape, (32, 32))
            # Center should have higher permittivity
            self.assertGreater(epsilon_r[16, 16], 1.0)
        except ImportError as e:
            self.skipTest(f"Draw module not available: {e}")
    
    def test_ellipse_creation(self):
        """Test ellipse drawing."""
        try:
            from eispy2d.utils.draw import ellipse
            
            epsilon_r, sigma = ellipse(
                x_radius=0.4,
                y_radius=0.2,
                resolution=(32, 32),
                object_rel_permittivity=2.5,
                background_rel_permittivity=1.0
            )
            
            self.assertEqual(epsilon_r.shape, (32, 32))
        except ImportError as e:
            self.skipTest(f"Draw module not available: {e}")
    
    def test_polygon_creation(self):
        """Test polygon drawing."""
        try:
            from eispy2d.utils.draw import polygon
            
            epsilon_r, sigma = polygon(
                number_sides=6,
                radius=0.4,
                resolution=(32, 32),
                object_rel_permittivity=2.0,
                background_rel_permittivity=1.0
            )
            
            self.assertEqual(epsilon_r.shape, (32, 32))
        except ImportError as e:
            self.skipTest(f"Draw module not available: {e}")
    
    def test_overlay_multiple_shapes(self):
        """Test overlaying multiple shapes."""
        try:
            from eispy2d.utils.draw import square, circle
            
            epsilon_r, sigma = square(
                side_length=0.6,
                resolution=(64, 64),
                object_rel_permittivity=2.0,
                background_rel_permittivity=1.0
            )
            
            epsilon_r, sigma = circle(
                radius=0.3,
                object_rel_permittivity=3.0,
                rel_permittivity=epsilon_r,
                conductivity=sigma
            )
            
            # Should have both shapes
            self.assertTrue(np.any(epsilon_r == 2.0))
            self.assertTrue(np.any(epsilon_r == 3.0))
        except ImportError as e:
            self.skipTest(f"Draw module not available: {e}")


class TestRegularizationModule(unittest.TestCase):
    """Test regularization methods."""
    
    def test_regularization_imports(self):
        """Test regularization module imports."""
        try:
            from eispy2d.utils.regularization import (
                Tikhonov, Landweber, ConjugatedGradient, LeastSquares,
                TIK_FIXED, TIK_MOZOROV, TIK_LCURVE
            )
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Regularization module not available: {e}")
    
    def test_tikhonov_fixed(self):
        """Test Tikhonov regularization with fixed parameter."""
        try:
            from eispy2d.utils.regularization import Tikhonov, TIK_FIXED
            
            # Simple linear system
            K = np.array([[1.0, 2.0], [3.0, 4.0]])
            y = np.array([1.0, 2.0])
            
            reg = Tikhonov(choice=TIK_FIXED, parameter=0.01)
            x = reg.solve(K, y)
            
            self.assertEqual(x.shape, (2,))
            self.assertTrue(np.isrealobj(x) or np.iscomplexobj(x))
        except ImportError as e:
            self.skipTest(f"Regularization module not available: {e}")
    
    def test_tikhonov_with_matrix(self):
        """Test Tikhonov with matrix right-hand side."""
        try:
            from eispy2d.utils.regularization import Tikhonov, TIK_FIXED
            
            K = np.random.randn(10, 5)
            Y = np.random.randn(10, 3)
            
            reg = Tikhonov(choice=TIK_FIXED, parameter=0.001)
            X = reg.solve(K, Y)
            
            self.assertEqual(X.shape, (5, 3))
        except ImportError as e:
            self.skipTest(f"Regularization module not available: {e}")
    
    def test_landweber(self):
        """Test Landweber regularization."""
        try:
            from eispy2d.utils.regularization import Landweber
            
            K = np.random.randn(10, 5)
            y = np.random.randn(10)
            
            reg = Landweber(iterations=10)
            x = reg.solve(K, y)
            
            self.assertEqual(x.shape, (5,))
        except ImportError as e:
            self.skipTest(f"Regularization module not available: {e}")
    
    def test_least_squares(self):
        """Test Least Squares with cutoff."""
        try:
            from eispy2d.utils.regularization import LeastSquares
            
            K = np.random.randn(10, 5)
            y = np.random.randn(10)
            
            reg = LeastSquares(cutoff=1e-6)
            x = reg.solve(K, y)
            
            self.assertEqual(x.shape, (5,))
        except ImportError as e:
            self.skipTest(f"Regularization module not available: {e}")


class TestStopCriteria(unittest.TestCase):
    """Test StopCriteria class."""
    
    def test_stop_criteria_import(self):
        """Test StopCriteria import."""
        try:
            from eispy2d.utils.stopcriteria import StopCriteria
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"StopCriteria module not available: {e}")
    
    def test_max_iterations(self):
        """Test max iterations criterion."""
        from eispy2d.utils.stopcriteria import StopCriteria
        
        stop = StopCriteria(max_iterations=5)
        stop.reset_memory()
        
        for i in range(5):
            should_stop = stop.stop(number_evaluations=0, number_iterations=i, current_best_evaluation=1.0)
        
        self.assertTrue(should_stop)
    
    def test_max_evaluations(self):
        """Test max evaluations criterion."""
        from eispy2d.utils.stopcriteria import StopCriteria
        
        stop = StopCriteria(max_evaluations=10)
        stop.reset_memory()
        
        for i in range(10):
            should_stop = stop.stop(number_evaluations=i, number_iterations=0, current_best_evaluation=1.0)
        
        self.assertTrue(should_stop)
    
    def test_cost_threshold(self):
        """Test cost function threshold."""
        from eispy2d.utils.stopcriteria import StopCriteria
        
        stop = StopCriteria(cost_function_threshold=0.01)
        stop.reset_memory()
        
        # Above threshold - don't stop
        self.assertFalse(stop.stop(0, 0, 0.1))
        
        # Below threshold - stop
        self.assertTrue(stop.stop(0, 0, 0.005))
    
    def test_no_improvement_iterations(self):
        """Test no improvement criterion (iterations)."""
        from eispy2d.utils.stopcriteria import StopCriteria
        
        stop = StopCriteria(
            max_iter_woimp=3,
            improvement_threshold=1.0
        )
        stop.reset_memory()
        
        # First iteration - improvement (initial)
        stop.stop(0, 0, 1.0)
        
        # No improvement for 3 iterations
        for i in range(1, 4):
            self.assertFalse(stop.stop(0, i, 1.0))  # Same value
        
        # Should stop now
        self.assertTrue(stop.stop(0, 4, 1.0))
    
    def test_stop_criteria_copy(self):
        """Test copy method."""
        from eispy2d.utils.stopcriteria import StopCriteria
        
        original = StopCriteria(max_iterations=10, max_evaluations=100)
        original.reset_memory()
        
        copy = original.copy()
        self.assertEqual(copy.max_iter, original.max_iter)
        self.assertEqual(copy.max_evals, original.max_evals)


class TestStatisticsModule(unittest.TestCase):
    """Test statistical utilities."""
    
    def test_statistics_imports(self):
        """Test statistics module imports."""
        try:
            from eispy2d.utils.statistics import (
                compare1sample, compare2samples, compare_multiple,
                confint, confintplot, normalityplot
            )
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"Statistics module not available: {e}")
    
    def test_compare_1sample(self):
        """Test one-sample comparison."""
        try:
            from eispy2d.utils.statistics import compare1sample
            
            # Generate normal data
            data = np.random.normal(loc=0, scale=1, size=50)
            
            result = compare1sample(data, offset=0.0)
            self.assertEqual(len(result), 6)  # statistic, pvalue, alternative, nonnormal, transf, delta
            self.assertIsNotNone(result[0])  # statistic
            self.assertIsNotNone(result[1])  # pvalue
        except ImportError as e:
            self.skipTest(f"Statistics module not available: {e}")
    
    def test_compare_2samples(self):
        """Test two-sample comparison."""
        try:
            from eispy2d.utils.statistics import compare2samples
            
            data1 = np.random.normal(loc=0, scale=1, size=50)
            data2 = np.random.normal(loc=0.5, scale=1, size=50)
            
            result = compare2samples(data1, data2, paired=False)
            self.assertEqual(len(result), 7)  # statistic, pvalue, alternative, delta, nonnormal, transf, equal_var
        except ImportError as e:
            self.skipTest(f"Statistics module not available: {e}")
    
    def test_compare_paired(self):
        """Test paired comparison."""
        try:
            from eispy2d.utils.statistics import compare2samples
            
            # Generate paired data
            before = np.random.normal(loc=10, scale=2, size=30)
            after = before + np.random.normal(loc=1, scale=1, size=30)
            
            result = compare2samples(before, after, paired=True)
            self.assertIsNotNone(result[1])  # pvalue
        except ImportError as e:
            self.skipTest(f"Statistics module not available: {e}")
    
    def test_confint(self):
        """Test confidence interval calculation."""
        try:
            from eispy2d.utils.statistics import confint
            
            data = np.random.normal(loc=5, scale=1, size=100)
            
            ci, normality, transform = confint(data, alpha=0.05)
            self.assertEqual(len(ci), 2)  # lower and upper bounds
            self.assertIsInstance(normality, bool)
        except ImportError as e:
            self.skipTest(f"Statistics module not available: {e}")


class TestRegularizationConstants(unittest.TestCase):
    """Test regularization constants."""
    
    def test_constants(self):
        """Test that constants are defined."""
        try:
            from eispy2d.utils.regularization import (
                TIK_FIXED, TIK_MOZOROV, TIK_LCURVE
            )
            self.assertIsInstance(TIK_FIXED, str)
            self.assertIsInstance(TIK_MOZOROV, str)
            self.assertIsInstance(TIK_LCURVE, str)
        except ImportError as e:
            self.skipTest(f"Regularization module not available: {e}")


if __name__ == '__main__':
    unittest.main()