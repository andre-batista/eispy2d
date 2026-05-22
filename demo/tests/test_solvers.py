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
from eispy2d.data.inputdata import InputData
from eispy2d.data.result import Result
from eispy2d.solvers.base.forward import ForwardSolver
from eispy2d.solvers.base.inverse import InverseSolver
from eispy2d.solvers.base.deterministic import Deterministic
from eispy2d.solvers.base.stochastic import Stochastic, OutputMode, EACH_EXECUTION, AVERAGE_CASE
from eispy2d.regularization.regularization import Tikhonov, TIK_FIXED
from eispy2d.utils.stopcriteria import StopCriteria


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
        """Test Stochastic solver creation."""
        output_mode = OutputMode(rule=EACH_EXECUTION)
        solver = Stochastic(
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
        self.stop_criteria = StopCriteria(max_iterations=10)
    
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


class TestStopCriteria(unittest.TestCase):
    """Test StopCriteria class."""
    
    def test_max_iterations(self):
        """Test maximum iterations criterion."""
        stop = StopCriteria(max_iterations=5)
        stop.reset_memory()
        
        for i in range(4):
            self.assertFalse(stop.stop(0, i, 0.1))
        self.assertTrue(stop.stop(0, 5, 0.1))
    
    def test_max_evaluations(self):
        """Test maximum evaluations criterion."""
        stop = StopCriteria(max_evaluations=10)
        stop.reset_memory()
        
        for i in range(9):
            self.assertFalse(stop.stop(i, 0, 0.1))
        self.assertTrue(stop.stop(10, 0, 0.1))
    
    def test_cost_function_threshold(self):
        """Test cost function threshold criterion."""
        stop = StopCriteria(cost_function_threshold=0.01)
        stop.reset_memory()
        
        self.assertFalse(stop.stop(0, 0, 0.1))
        self.assertTrue(stop.stop(0, 0, 0.005))
    
    def test_no_improvement(self):
        """Test no improvement criterion."""
        stop = StopCriteria(
            max_iter_woimp=3,
            improvement_threshold=1.0
        )
        stop.reset_memory()
        
        # No improvement for 3 iterations
        for i in range(3):
            stop.stop(0, i, 0.1)
        self.assertTrue(stop.stop(0, 3, 0.1))
    
    def test_copy(self):
        """Test StopCriteria copy."""
        original = StopCriteria(max_iterations=10, max_evaluations=100)
        original.reset_memory()
        
        copy = original.copy()
        self.assertEqual(copy.max_iter, original.max_iter)
        self.assertEqual(copy.max_evals, original.max_evals)


if __name__ == '__main__':
    unittest.main()