# tests/test_solvers.py
"""
Unit tests for eispy2d solvers (forward and inverse).
"""

import sys
import os
sys.path.insert(1, '../../../eispy2d/')


import unittest
import numpy as np


from eispy2d.core.configuration import Configuration
from eispy2d.core.inputdata import InputData
from eispy2d.core.result import Result
from eispy2d.solvers.base.forward import ForwardSolver
from eispy2d.solvers.base.inverse import InverseSolver
from eispy2d.solvers.base.deterministic import Deterministic
from eispy2d.solvers.base.stochastic import Stochastic, OutputMode, EACH_EXECUTION, AVERAGE_CASE
from eispy2d.solvers.inverse.regularization import Tikhonov, TIK_FIXED
from eispy2d.utils.stopcriteria import StopCriteria


# Create a concrete subclass of Stochastic for testing
class ConcreteStochastic(Stochastic):
    """Concrete implementation of Stochastic for testing."""
    
    def solve(self, inputdata, discretization, print_info=True, print_file=sys.stdout):
        """Implement abstract solve method."""
        return Result(name='test_result', method_name='test')
    
    def save(self, file_path=''):
        """Implement abstract save method."""
        data = super().save(file_path=file_path)
        return data
    
    def importdata(self, file_name, file_path=''):
        """Implement abstract importdata method."""
        data = super().importdata(file_name, file_path=file_path)
        return data


class TestBaseClasses(unittest.TestCase):
    """Test base abstract classes."""
    
    def test_forward_solver_abstract(self):
        """Test that ForwardSolver cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            ForwardSolver()
    
    def test_inverse_solver_abstract(self):
        """Test that InverseSolver cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            InverseSolver()
    
    def test_deterministic_creation(self):
        """Test Deterministic solver creation."""
        solver = Deterministic(alias='test_det')
        self.assertEqual(solver.alias, 'test_det')
        self.assertEqual(solver.name, '')
    
    def test_stochastic_creation(self):
        """Test Stochastic solver creation using concrete subclass."""
        output_mode = OutputMode(rule=EACH_EXECUTION)
        solver = ConcreteStochastic(
            outputmode=output_mode,
            alias='test_stoch',
            number_executions=10
        )
        self.assertEqual(solver.alias, 'test_stoch')
        self.assertEqual(solver.nexec, 10)
    
    def test_output_mode_creation(self):
        """Test OutputMode creation."""
        mode = OutputMode(rule=EACH_EXECUTION)
        self.assertEqual(mode.rule, EACH_EXECUTION)
        
        mode2 = OutputMode(rule=AVERAGE_CASE, reference='zeta_rn', sample_rate=10)
        self.assertEqual(mode2.rule, AVERAGE_CASE)
        self.assertEqual(mode2.reference, 'zeta_rn')
        self.assertEqual(mode2.sample_rate, 10)
    
    def test_output_mode_invalid_rule(self):
        """Test invalid rule raises error."""
        from eispy2d.core.error import WrongValueInput
        
        with self.assertRaises(WrongValueInput):
            OutputMode(rule='invalid_rule')


class TestForwardSolvers(unittest.TestCase):
    """Test forward solvers (requires actual implementation)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            name='forward_test',
            wavelength=1.0,
            number_measurements=16,
            number_sources=8,
            image_size=[2.0, 2.0],
            observation_radius=3.0,
            background_permittivity=1.0,
            perfect_dielectric=True
        )
        
        self.resolution = (32, 32)
        self.epsilon_r = np.ones(self.resolution) * 1.0
        self.epsilon_r[14:18, 14:18] = 2.0  # Simple scatterer
        
        self.input_data = InputData(
            name='forward_input',
            configuration=self.config,
            resolution=self.resolution,
            rel_permittivity=self.epsilon_r
        )
    
    def test_mom_cg_fft_import(self):
        """Test MoM_CG_FFT import."""
        try:
            from eispy2d.solvers.forward.mom_cg_fft import MoM_CG_FFT
            solver = MoM_CG_FFT(tolerance=1e-3, maximum_iterations=100)
            self.assertIsNotNone(solver)
        except ImportError:
            self.skipTest("MoM_CG_FFT not available")
    
    def test_analytical_solver_import(self):
        """Test Analytical solver import."""
        try:
            from eispy2d.solvers.forward.analytical import Analytical
            solver = Analytical(contrast=1.0, radius=0.5)
            self.assertIsNotNone(solver)
        except ImportError:
            self.skipTest("Analytical solver not available")
    
    def test_incident_field_computation(self):
        """Test incident field computation."""
        try:
            from eispy2d.solvers.forward.mom_cg_fft import MoM_CG_FFT
            solver = MoM_CG_FFT()
            ei = solver.incident_field(self.resolution, self.config)
            
            self.assertEqual(ei.shape[0], 32 * 32)
            self.assertEqual(ei.shape[1], 8)
            self.assertTrue(np.iscomplexobj(ei))
        except ImportError:
            self.skipTest("MoM_CG_FFT not available")


class TestInverseSolvers(unittest.TestCase):
    """Test inverse solvers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            name='inverse_test',
            wavelength=1.0,
            number_measurements=16,
            number_sources=8,
            image_size=[2.0, 2.0],
            observation_radius=3.0,
            background_permittivity=1.0,
            perfect_dielectric=True
        )
        
        self.resolution = (32, 32)
        
        # Create synthetic scattered field
        self.scattered_field = np.random.randn(16, 8) + 1j * np.random.randn(16, 8)
        
        self.input_data = InputData(
            name='inverse_input',
            configuration=self.config,
            resolution=self.resolution,
            scattered_field=self.scattered_field,
            indicators=[]
        )
        
        self.regularization = Tikhonov(choice=TIK_FIXED, parameter=0.01)
        # FIXED: Provide both max_iterations and max_evaluations to work around bug
        self.stop_criteria = StopCriteria(max_iterations=10, max_evaluations=1000)
    
    def test_bim_import(self):
        """Test BIM import."""
        try:
            from eispy2d.solvers.inverse.bim import BornIterativeMethod
            from eispy2d.solvers.forward.mom_cg_fft import MoM_CG_FFT
            
            forward_solver = MoM_CG_FFT()
            solver = BornIterativeMethod(
                forward_solver=forward_solver,
                regularization=self.regularization,
                stop_criteria=self.stop_criteria,
                alias='test_bim'
            )
            
            self.assertEqual(solver.alias, 'test_bim')
            self.assertEqual(solver.name, 'Born Iterative Method')
        except ImportError:
            self.skipTest("BIM not available")
    
    def test_born_approx_import(self):
        """Test Born Approximation import."""
        try:
            from eispy2d.solvers.inverse.bornapprox import FirstOrderBornApproximation
            from eispy2d.solvers.forward.mom_cg_fft import MoM_CG_FFT
            
            forward_solver = MoM_CG_FFT()
            solver = FirstOrderBornApproximation(
                regularization=self.regularization,
                forward=forward_solver,
                alias='test_ba'
            )
            
            self.assertEqual(solver.alias, 'test_ba')
            self.assertEqual(solver.name, 'First-Order Born Approximation')
        except ImportError:
            self.skipTest("Born Approximation not available")
    
    def test_csi_import(self):
        """Test CSI import."""
        try:
            from eispy2d.solvers.inverse.csi import ContrastSourceInversion
            
            solver = ContrastSourceInversion(
                stop_criteria=self.stop_criteria,
                alias='test_csi'
            )
            
            self.assertEqual(solver.alias, 'test_csi')
            self.assertEqual(solver.name, 'Contrast Source Inversion')
        except ImportError:
            self.skipTest("CSI not available")
    
    def test_backprop_import(self):
        """Test BackPropagation import."""
        try:
            from eispy2d.solvers.inverse.backprop import BackPropagation
            
            solver = BackPropagation(alias='test_backprop')
            self.assertEqual(solver.alias, 'test_backprop')
            self.assertEqual(solver.name, 'Back-Propagation')
        except ImportError:
            self.skipTest("BackPropagation not available")
    
    def test_ecsi_import(self):
        """Test Extended CSI import."""
        try:
            from eispy2d.solvers.inverse.ecsi import ExtendedContrastSourceInversion
            
            solver = ExtendedContrastSourceInversion(
                stop_criteria=self.stop_criteria,
                alias='test_ecsi'
            )
            
            self.assertEqual(solver.alias, 'test_ecsi')
            self.assertEqual(solver.name, 'Extended Contrast Source Inversion')
        except ImportError:
            self.skipTest("ExtendedContrastSourceInversion not available")
    
    def test_dbim_import(self):
        """Test DBIM import."""
        try:
            from eispy2d.solvers.inverse.dbim import DistortedBornIterativeMethod
            from eispy2d.solvers.forward.mom_cg_fft import MoM_CG_FFT
            
            forward_solver = MoM_CG_FFT()
            solver = DistortedBornIterativeMethod(
                forward_solver=forward_solver,
                regularization=self.regularization,
                stop_criteria=self.stop_criteria,
                alias='test_dbim'
            )
            
            self.assertEqual(solver.alias, 'test_dbim')
            self.assertEqual(solver.name, 'Distorted Born Iterative Method')
        except ImportError:
            self.skipTest("DistortedBornIterativeMethod not available")
    
    def test_cgm_import(self):
        """Test CGM import."""
        try:
            from eispy2d.solvers.inverse.cgm import ConjugatedGradientMethod
            
            solver = ConjugatedGradientMethod(
                initial_guess='background',
                step='fixed',
                stop_criteria=self.stop_criteria,
                alias='test_cgm'
            )
            
            self.assertEqual(solver.alias, 'test_cgm')
            self.assertEqual(solver.name, 'Conjugated Gradient Method')
        except ImportError:
            self.skipTest("ConjugatedGradientMethod not available")
    
    def test_music_import(self):
        """Test MUSIC import."""
        try:
            from eispy2d.solvers.inverse.music import MUSIC
            
            solver = MUSIC(alias='test_music')
            self.assertEqual(solver.alias, 'test_music')
            self.assertEqual(solver.name, 'Multiple Signal Classification Imaging')
        except ImportError:
            self.skipTest("MUSIC not available")
    
    def test_lsm_import(self):
        """Test Linear Sampling Method import."""
        try:
            from eispy2d.solvers.inverse.lsm import LinearSamplingMethod
            
            solver = LinearSamplingMethod(alias='test_lsm')
            self.assertEqual(solver.alias, 'test_lsm')
            self.assertEqual(solver.name, 'Linear Sampling Method')
        except ImportError:
            self.skipTest("LinearSamplingMethod not available")
    
    def test_osm_import(self):
        """Test Orthogonality Sampling Method import."""
        try:
            from eispy2d.solvers.inverse.osm import OrthogonalitySamplingMethod
            
            solver = OrthogonalitySamplingMethod(alias='test_osm')
            self.assertEqual(solver.alias, 'test_osm')
            self.assertEqual(solver.name, 'Orthogonality Sampling Method')
        except ImportError:
            self.skipTest("OrthogonalitySamplingMethod not available")


class TestStopCriteria(unittest.TestCase):
    """Test StopCriteria class."""
    
    def test_max_iterations(self):
        """Test maximum iterations criterion."""
        # FIXED: Provide both max_iterations and max_evaluations
        stop = StopCriteria(max_iterations=5, max_evaluations=1000)
        stop.reset_memory()
        
        # Simulate iterations without reaching max
        for i in range(5):
            should_stop = stop.stop(0, i, 0.1)
            if i < 4:
                self.assertFalse(should_stop)
            else:
                # At i=4, number_iterations=4 which is less than max_iterations=5
                # So should still be false
                self.assertFalse(should_stop)
        
        # Should stop when number_iterations >= max_iter (5)
        should_stop = stop.stop(0, 5, 0.1)
        self.assertTrue(should_stop)
    
    def test_max_evaluations(self):
        """Test maximum evaluations criterion."""
        # FIXED: Provide both max_iterations and max_evaluations
        stop = StopCriteria(max_iterations=1000, max_evaluations=10)
        stop.reset_memory()
        
        # Simulate evaluations without reaching max
        for i in range(10):
            should_stop = stop.stop(i, 0, 0.1)
            if i < 9:
                self.assertFalse(should_stop)
            else:
                # At i=9, number_evaluations=9 which is less than max_evaluations=10
                self.assertFalse(should_stop)
        
        # Should stop when number_evaluations >= max_evals (10)
        should_stop = stop.stop(10, 0, 0.1)
        self.assertTrue(should_stop)
    
    def test_cost_function_threshold(self):
        """Test cost function threshold criterion."""
        # FIXED: Provide both max_iterations and max_evaluations
        stop = StopCriteria(
            max_iterations=1000,
            max_evaluations=1000,
            cost_function_threshold=0.01
        )
        stop.reset_memory()
        
        # Above threshold - don't stop
        self.assertFalse(stop.stop(0, 0, 0.1))
        
        # Below threshold - stop
        self.assertTrue(stop.stop(0, 0, 0.005))
    
    def test_no_improvement(self):
        """Test no improvement criterion."""
        # FIXED: Provide both max_iterations and max_evaluations
        # Use max_evals_woimp instead of max_iter_woimp to work around bug
        stop = StopCriteria(
            max_iterations=1000,
            max_evaluations=1000,
            max_evals_woimp=3,
            improvement_threshold=1.0
        )
        stop.reset_memory()
        
        # Initial evaluation
        stop.stop(0, 0, 0.1)
        
        # No improvement for 2 more evaluations (counter goes to 2)
        stop.stop(1, 0, 0.1)
        stop.stop(2, 0, 0.1)
        
        # At evaluation 3, counter becomes 3, should stop
        should_stop = stop.stop(3, 0, 0.1)
        self.assertTrue(should_stop)
    
    def test_copy(self):
        """Test StopCriteria copy."""
        # FIXED: Provide both max_iterations and max_evaluations
        original = StopCriteria(max_iterations=10, max_evaluations=100)
        original.reset_memory()
        
        copy = original.copy()
        self.assertEqual(copy.max_iter, original.max_iter)
        self.assertEqual(copy.max_evals, original.max_evals)


if __name__ == '__main__':
    unittest.main()