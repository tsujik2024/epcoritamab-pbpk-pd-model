import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.tcell_activation import TCellActivation


class TestTCellActivation(unittest.TestCase):
    """Test cases for TCellActivation class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.activation_model = TCellActivation()
        self.default_params = self.activation_model.parameters

    def test_initialization(self):
        """Test that model initializes with correct parameters"""
        self.assertIsInstance(self.activation_model.parameters, dict)

        # Check key parameters are present
        self.assertIn('sim_slope', self.default_params)
        self.assertIn('sim_slope_tumor', self.default_params)
        self.assertIn('Trimer_Threshold', self.default_params)
        self.assertIn('expand_factor', self.default_params)
        self.assertIn('kout_ATC', self.default_params)
        self.assertIn('TAD', self.default_params)
        self.assertIn('Tp', self.default_params)

        # Test custom parameters
        custom_params = {'sim_slope': 0.01, 'TAD': 5.0}
        custom_model = TCellActivation(parameters=custom_params)
        self.assertEqual(custom_model.parameters['sim_slope'], 0.01)
        self.assertEqual(custom_model.parameters['TAD'], 5.0)

    def test_initial_conditions(self):
        """Test initial conditions generation"""
        TC_counts = [1e9, 2e9, 1e8, 5e8]
        BC_counts = [2e8, 1e9, 5e7, 3e8]
        tumor_cell_count = 1e7

        initial_conditions = self.activation_model.get_initial_conditions(
            TC_counts, BC_counts, tumor_cell_count
        )

        # Should have 9 state variables
        self.assertEqual(len(initial_conditions), 9)

        # All initial conditions should be zero (no activated T-cells initially)
        for state in initial_conditions:
            self.assertEqual(state, 0.0)

    def test_activation_delay(self):
        """Test that activation doesn't occur before TAD"""
        p = self.default_params

        # Test before TAD
        t_before_TAD = p['TAD'] - 1.0
        trimers_per_BC = [10.0, 5.0, 2.0, 1.0]  # Some trimers present
        trimers_per_tumor = 50.0  # Above threshold

        rates = self.activation_model.calculate_activation_rates(
            t_before_TAD, trimers_per_BC, trimers_per_tumor,
            [1e9, 1e9, 1e8, 1e8], [1e8, 1e8, 1e7, 1e7], 1e6,
            [0, 0, 0, 0], 0.0, [0, 0, 0, 0]
        )

        # No activation should occur before TAD
        for comp in range(4):
            self.assertEqual(rates['activation_BC'][comp], 0.0)
        self.assertEqual(rates['activation_tumor'], 0.0)

    def test_RELU_function(self):
        """Test the stepwise RELU function for tumor activation"""
        p = self.default_params

        # Test below threshold
        trimers_below_threshold = p['Trimer_Threshold'] - 1.0
        rates_below = self.activation_model.calculate_activation_rates(
            p['TAD'] + 1.0, [0, 0, 0, 0], trimers_below_threshold,
            [0, 0, 1e8, 0], [0, 0, 0, 0], 1e6,
            [0, 0, 0, 0], 0.0, [0, 0, 0, 0]
        )
        self.assertEqual(rates_below['RELU'], 0.01)  # 1% activation

        # Test above threshold
        trimers_above_threshold = p['Trimer_Threshold'] + 1.0
        rates_above = self.activation_model.calculate_activation_rates(
            p['TAD'] + 1.0, [0, 0, 0, 0], trimers_above_threshold,
            [0, 0, 1e8, 0], [0, 0, 0, 0], 1e6,
            [0, 0, 0, 0], 0.0, [0, 0, 0, 0]
        )
        self.assertEqual(rates_above['RELU'], 1.0)  # Full activation

    def test_derivatives_structure(self):
        """Test that derivatives function returns correct structure"""
        TC_counts = [1e9, 2e9, 1e8, 5e8]
        BC_counts = [2e8, 1e9, 5e7, 3e8]
        tumor_cell_count = 1e7

        initial_conditions = self.activation_model.get_initial_conditions(
            TC_counts, BC_counts, tumor_cell_count
        )

        # Mock trafficking rates (same as lymphocyte model)
        trafficking_rates = {
            'blood_to_spleen': 50.0,
            'spleen_to_node': 1.67,
            'node_to_lymph': 1672.0,
            'lymph_to_blood': 2.63
        }

        trimers_per_BC = [1.0, 2.0, 5.0, 1.0]
        trimers_per_tumor = 30.0  # Above threshold

        derivatives = self.activation_model.derivatives(
            initial_conditions, self.default_params['TAD'] + 1.0,
            trimers_per_BC, trimers_per_tumor, TC_counts, BC_counts,
            tumor_cell_count, trafficking_rates
        )

        # Should return 9 derivatives (one for each state variable)
        self.assertEqual(len(derivatives), 9)

        # All derivatives should be floats
        for deriv in derivatives:
            self.assertIsInstance(deriv, float)

    def test_clonal_expansion(self):
        """Test that activated T-cells undergo clonal expansion"""
        p = self.default_params

        # Start with some virtual ATCs
        vATC_BC = [1000.0, 0, 0, 0]  # 1000 ATCs in blood
        pATC = [0, 0, 0, 0]  # No proliferating ATCs initially

        rates = self.activation_model.calculate_activation_rates(
            p['TAD'] + 1.0, [0, 0, 0, 0], 0.0,
            [1e9, 0, 0, 0], [0, 0, 0, 0], 0.0,
            vATC_BC, 0.0, pATC
        )

        # Should see expansion from existing ATCs
        self.assertGreater(rates['expansion'][0], 0.0)
        expected_expansion = p['expand_factor'] * vATC_BC[0]
        self.assertAlmostEqual(rates['expansion'][0], expected_expansion)


class TestTCellActivationIntegration(unittest.TestCase):
    """Integration tests for T-cell activation model"""

    def setUp(self):
        self.activation_model = TCellActivation()

    def test_extended_simulation(self):
        """Test longer simulation to check stability"""
        TC_counts = [1e9, 2e9, 1e8, 5e8]
        BC_counts = [2e8, 1e9, 5e7, 3e8]
        tumor_cell_count = 1e7

        initial_conditions = self.activation_model.get_initial_conditions(
            TC_counts, BC_counts, tumor_cell_count
        )

        # Simulate with constant trimer levels - start after TAD to ensure activation
        p = self.activation_model.parameters
        t_span = np.linspace(p['TAD'] + 0.1, 21, 100)  # Start after TAD
        trimers_per_BC = [2.0, 5.0, 10.0, 1.0]  # Constant trimers
        trimers_per_tumor = 50.0  # Above threshold

        # Mock trafficking rates
        trafficking_rates = {
            'blood_to_spleen': 50.0,
            'spleen_to_node': 1.67,
            'node_to_lymph': 1672.0,
            'lymph_to_blood': 2.63
        }

        solution = self.activation_model.simulate(
            initial_conditions, t_span, trimers_per_BC, trimers_per_tumor,
            TC_counts, BC_counts, tumor_cell_count, trafficking_rates
        )

        # Check that all cell counts remain non-negative
        self.assertTrue(np.all(solution >= 0))

        # Check that activation occurs (ATC counts increase from zero)
        total_ATC_blood = solution[:, 0] + solution[:, 4]  # vATC_BC + pATC in blood
        self.assertGreater(total_ATC_blood[-1], total_ATC_blood[0])

        # Check that tumor ATCs are created
        vATC_tumor = solution[:, 8]
        self.assertGreater(vATC_tumor[-1], 0.0)

    def test_activation_metrics(self):
        """Test calculation of activation metrics"""
        # Create mock solution
        t_span = np.linspace(0, 7, 10)
        solution = np.zeros((10, 9))

        # Set some ATC values
        solution[:, 0] = np.linspace(0, 1000, 10)  # vATC_BC in blood
        solution[:, 4] = np.linspace(0, 500, 10)  # pATC in blood
        solution[:, 8] = np.linspace(0, 200, 10)  # vATC_tumor

        TC_counts = [1e9, 2e9, 1e8, 5e8]

        metrics = self.activation_model.calculate_activation_metrics(solution, TC_counts)

        # Check metrics structure
        self.assertIn('total_ATC', metrics)
        self.assertIn('activation_fraction', metrics)

        # Check calculations
        total_ATC_blood = solution[:, 0] + solution[:, 4]
        np.testing.assert_array_equal(metrics['total_ATC'][:, 0], total_ATC_blood)

        activation_fraction_blood = total_ATC_blood / TC_counts[0]
        np.testing.assert_array_almost_equal(metrics['activation_fraction'][:, 0],
                                             activation_fraction_blood)

    def test_rates_dictionary_structure(self):
        """Test that the rates dictionary has all required keys"""
        p = self.activation_model.parameters

        # Test after TAD and Tp
        t_after_Tp = p['Tp'] + 1.0
        trimers_per_BC = [10.0, 5.0, 2.0, 1.0]
        trimers_per_tumor = 50.0

        rates = self.activation_model.calculate_activation_rates(
            t_after_Tp, trimers_per_BC, trimers_per_tumor,
            [1e9, 1e9, 1e8, 1e8], [1e8, 1e8, 1e7, 1e7], 1e6,
            [1000, 0, 0, 0], 500.0, [0, 0, 0, 0]
        )

        # Check that all required keys are present
        required_keys = [
            'activation_BC', 'activation_tumor', 'expansion',
            'death_vATC', 'death_pATC', 'death_vATC_tumor', 'RELU'
        ]

        for key in required_keys:
            self.assertIn(key, rates)

        # Check that death_vATC_tumor is calculated
        self.assertIsInstance(rates['death_vATC_tumor'], float)

        # With vATC_tumor = 500 and kout_ATC = 0.05, death should be 25
        expected_death = p['kout_ATC'] * 500.0
        self.assertAlmostEqual(rates['death_vATC_tumor'], expected_death)


def run_activation_performance_test():
    """Performance test for T-cell activation model"""
    import time

    model = TCellActivation()

    TC_counts = [1e9, 2e9, 1e8, 5e8]
    BC_counts = [2e8, 1e9, 5e7, 3e8]
    tumor_cell_count = 1e7

    initial_conditions = model.get_initial_conditions(TC_counts, BC_counts, tumor_cell_count)
    t_span = np.linspace(0, 21, 100)
    trimers_per_BC = [2.0, 5.0, 10.0, 1.0]
    trimers_per_tumor = 50.0

    trafficking_rates = {
        'blood_to_spleen': 50.0,
        'spleen_to_node': 1.67,
        'node_to_lymph': 1672.0,
        'lymph_to_blood': 2.63
    }

    start_time = time.time()
    solution = model.simulate(
        initial_conditions, t_span, trimers_per_BC, trimers_per_tumor,
        TC_counts, BC_counts, tumor_cell_count, trafficking_rates
    )
    end_time = time.time()

    print(f"Performance test: {end_time - start_time:.3f} seconds for 21-day simulation")
    print(f"Model dimensions: {solution.shape}")

    return end_time - start_time


if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2)

    # Run performance test
    print("\n" + "=" * 50)
    print("T-CELL ACTIVATION PERFORMANCE TEST")
    print("=" * 50)
    run_activation_performance_test()