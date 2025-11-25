import unittest
import numpy as np
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.pk_submodel import PKSubmodel


class TestPKSubmodel(unittest.TestCase):
    """Test cases for PKSubmodel class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.pk_model = PKSubmodel()
        self.default_params = self.pk_model.parameters

    def test_initialization(self):
        """Test that model initializes with correct parameters"""
        # Test default parameters
        self.assertIsInstance(self.pk_model.parameters, dict)
        self.assertIn('V_plasma', self.pk_model.parameters)
        self.assertIn('CL', self.pk_model.parameters)

        # Test custom parameters - now should work with partial parameter sets
        custom_params = {'V_plasma': 3.0, 'CL': 3.0}
        custom_model = PKSubmodel(custom_params)

        # Custom parameters should be set
        self.assertEqual(custom_model.parameters['V_plasma'], 3.0)
        self.assertEqual(custom_model.parameters['CL'], 3.0)

        # Other parameters should fall back to defaults
        self.assertEqual(custom_model.parameters['V_tight'], 8.11)  # Default value
        self.assertEqual(custom_model.parameters['k_a'], 0.131)  # Default value

    def test_parameter_validation(self):
        """Test that parameter merging works correctly"""
        # Test that incomplete parameters are merged with defaults
        incomplete_params = {'V_plasma': 2.6, 'CL': 2.47}  # Missing many required params

        # This should NOT raise an error anymore - it should merge with defaults
        custom_model = PKSubmodel(incomplete_params)

        # Custom parameters should be set
        self.assertEqual(custom_model.parameters['V_plasma'], 2.6)
        self.assertEqual(custom_model.parameters['CL'], 2.47)

        # Missing parameters should be filled with defaults
        self.assertEqual(custom_model.parameters['V_tight'], 8.11)
        self.assertEqual(custom_model.parameters['k_a'], 0.131)

    def test_initial_conditions(self):
        """Test initial conditions generation"""
        # Test with zero dose
        ic_zero = self.pk_model.get_initial_conditions(dose=0.0)
        expected_zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        np.testing.assert_array_equal(ic_zero, expected_zero)

        # Test with non-zero dose
        ic_dose = self.pk_model.get_initial_conditions(dose=100.0)
        expected_dose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
        np.testing.assert_array_equal(ic_dose, expected_dose)

    def test_distribution_terms(self):
        """Test distribution term calculations"""
        # Create test concentrations (all 1.0 for simplicity)
        test_concentrations = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]

        dist_terms = self.pk_model.distribution_terms(test_concentrations)

        # Check that all expected terms are present
        expected_terms = [
            'plasma_to_leaky', 'plasma_to_tight', 'plasma_to_spleen',
            'lymph_to_plasma', 'leaky_to_lymph', 'tight_to_lymph',
            'spleen_to_node', 'node_to_lymph'
        ]

        for term in expected_terms:
            self.assertIn(term, dist_terms)
            self.assertIsInstance(dist_terms[term], float)

        # Test specific term calculations
        p = self.default_params
        expected_plasma_to_leaky = p['L_leaky'] * (1 - p['sigma_leaky']) * 1.0
        self.assertAlmostEqual(dist_terms['plasma_to_leaky'], expected_plasma_to_leaky)

    def test_clearance_terms(self):
        """Test clearance term calculations"""
        # Test with zero concentration
        CL_linear_zero, CL_nonlinear_zero = self.pk_model.clearance_terms(0.0)
        self.assertEqual(CL_linear_zero, 0.0)
        self.assertEqual(CL_nonlinear_zero, 0.0)

        # Test with non-zero concentration
        test_concentration = 1.0
        CL_linear, CL_nonlinear = self.pk_model.clearance_terms(test_concentration)

        p = self.default_params
        expected_CL_linear = p['CL'] * test_concentration
        expected_CL_nonlinear = (p['V_max'] * test_concentration) / (p['K_m'] + test_concentration)

        self.assertAlmostEqual(CL_linear, expected_CL_linear)
        self.assertAlmostEqual(CL_nonlinear, expected_CL_nonlinear)



    def test_derivatives_structure(self):
        """Test that derivatives function returns correct structure"""
        # Test with zero initial conditions
        initial_conditions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        derivatives = self.pk_model.derivatives(initial_conditions, 0.0)

        # Should return 7 derivatives (one for each state variable)
        self.assertEqual(len(derivatives), 7)
        self.assertIsInstance(derivatives, list)

        # All derivatives should be floats
        for deriv in derivatives:
            self.assertIsInstance(deriv, float)

    def test_derivatives_with_dose(self):
        """Test derivatives with non-zero injection site amount"""
        # With drug at injection site, should see positive derivatives in some compartments
        initial_conditions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]  # 100 nM at injection site
        derivatives = self.pk_model.derivatives(initial_conditions, 0.0)

        # Injection site derivative should be negative (depletion)
        self.assertLess(derivatives[6], 0.0)

    def test_mass_balance(self):
        """Test mass balance in the system (excluding clearance)"""
        # Create a simplified model with no clearance to test mass balance
        no_clearance_params = self.default_params.copy()
        no_clearance_params['CL'] = 0.0
        no_clearance_params['V_max'] = 0.0

        test_model = PKSubmodel(no_clearance_params)

        # Test concentrations
        test_concentrations = [1.0, 0.5, 0.5, 0.2, 0.1, 0.3, 0.0]
        derivatives = test_model.derivatives(test_concentrations, 0.0)

        # Calculate total mass flow (should be approximately zero without clearance)
        total_flow = 0.0
        volumes = [
            test_model.parameters['V_plasma'],
            test_model.parameters['V_leaky'],
            test_model.parameters['V_tight'],
            test_model.parameters['V_spleen'],
            test_model.parameters['V_node'],
            test_model.parameters['V_lymph']
        ]

        for i, (vol, deriv) in enumerate(zip(volumes, derivatives[:-1])):  # Exclude injection site
            total_flow += vol * deriv

        # Total flow should be very close to zero (within numerical precision)
        self.assertAlmostEqual(total_flow, 0.0, places=10)

    def test_simulation_output(self):
        """Test that simulation runs and returns expected output"""
        # Simple short simulation
        initial_conditions = self.pk_model.get_initial_conditions(dose=100.0)
        t_span = np.linspace(0, 1, 10)  # 1 day, 10 points

        solution = self.pk_model.simulate(initial_conditions, t_span)

        # Check output shape
        self.assertEqual(solution.shape, (len(t_span), len(initial_conditions)))

        # Check that injection site decreases
        injection_site_values = solution[:, 6]
        self.assertTrue(np.all(np.diff(injection_site_values) <= 0))  # Should be non-increasing

    def test_parameter_sensitivity(self):
        """Test that model responds to parameter changes"""
        # Test clearance sensitivity
        high_clearance_params = self.default_params.copy()
        high_clearance_params['CL'] = 10.0  # Much higher clearance

        high_cl_model = PKSubmodel(high_clearance_params)

        # Run simulations with both models
        initial_conditions = self.pk_model.get_initial_conditions(dose=100.0)
        t_span = np.linspace(0, 7, 50)  # 1 week

        sol_default = self.pk_model.simulate(initial_conditions, t_span)
        sol_high_cl = high_cl_model.simulate(initial_conditions, t_span)

        # With higher clearance, plasma concentrations should be lower
        plasma_default = sol_default[:, 0]
        plasma_high_cl = sol_high_cl[:, 0]

        # Check that high clearance gives lower concentrations (after initial time)
        self.assertTrue(np.all(plasma_high_cl[10:] < plasma_default[10:]))


class TestPKSubmodelIntegration(unittest.TestCase):
    """Integration tests for PK submodel"""

    def setUp(self):
        self.pk_model = PKSubmodel()

    def test_extended_simulation(self):
        """Test longer simulation to check stability"""
        initial_conditions = self.pk_model.get_initial_conditions(dose=320.0)  # 48 mg equivalent
        t_span = np.linspace(0, 56, 2000)  # 8 weeks

        # Optional: Debug the absorption
        # solution = self.pk_model.debug_absorption(initial_conditions, t_span)

        solution = self.pk_model.simulate(initial_conditions, t_span)

        # Check that all concentrations remain non-negative
        self.assertTrue(np.all(solution >= 0))

        # Check that injection site goes to nearly zero
        final_injection = solution[-1, 6]
        initial_injection = initial_conditions[6]

        # With k_a = 0.131 day^-1, we expect about 0.065% remaining after 56 days
        # Let's check that at least 99.9% has been absorbed
        percent_remaining = (final_injection / initial_injection) * 100
        self.assertLess(percent_remaining, 0.1)  # Less than 0.1% remaining

        # Also verify that injection site is decreasing monotonically
        injection_site_values = solution[:, 6]
        differences = np.diff(injection_site_values)
        # Allow very small numerical increases due to solver tolerance
        self.assertTrue(np.all(differences <= 1e-10))

    def test_multiple_dose_scenario(self):
        """Test scenario with simulated multiple doses"""
        # Simulate weekly dosing for 4 weeks
        initial_conditions = self.pk_model.get_initial_conditions(dose=0.0)
        t_span = np.linspace(0, 28, 1000)

        # For this test, we'll just verify the model can handle the time span
        solution = self.pk_model.simulate(initial_conditions, t_span)

        # Should complete without errors and return correct shape
        self.assertEqual(solution.shape, (len(t_span), len(initial_conditions)))


def run_performance_test():
    """Performance test (not a unit test) to check simulation speed"""
    import time

    pk_model = PKSubmodel()
    initial_conditions = pk_model.get_initial_conditions(dose=320.0)

    # Time a moderately complex simulation
    t_span = np.linspace(0, 28, 1000)

    start_time = time.time()
    solution = pk_model.simulate(initial_conditions, t_span)
    end_time = time.time()

    print(f"Performance test: {end_time - start_time:.3f} seconds for 28-day simulation")
    print(f"Model dimensions: {solution.shape}")

    return end_time - start_time


if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2)

    # Run performance test separately
    print("\n" + "=" * 50)
    print("PERFORMANCE TEST")
    print("=" * 50)
    run_performance_test()