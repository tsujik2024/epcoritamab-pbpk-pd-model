"""T-cell activation submodel"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.integrate import odeint

from src.parameters.tcell_activation_params import get_tcell_activation_default_parameters


class TCellActivation:
    """T-cell activation submodel"""

    def __init__(self, parameters: Optional[Dict] = None):
        self.parameters = self._build_parameters(parameters)
        self._validate_parameters()

    def _build_parameters(self, user_params: Optional[Dict]) -> Dict:
        """Build complete parameter set"""
        default_params = get_tcell_activation_default_parameters()

        if user_params is None:
            return default_params

        # Merge user parameters with defaults
        complete_params = default_params.copy()
        complete_params.update(user_params)
        return complete_params

    def _validate_parameters(self):
        """Validate that all required parameters are present"""
        required_params = [
            'sim_slope', 'sim_slope_tumor', 'Trimer_Threshold',
            'expand_factor', 'kout_ATC', 'TAD', 'Tp', 'million_cells'
        ]

        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")

    def get_initial_conditions(self,
                               TC_counts: List[float],
                               BC_counts: List[float],
                               tumor_cell_count: float = 0.0) -> List[float]:
        """
        Get initial conditions for simulation

        Args:
            TC_counts: [TC_blood, TC_spleen, TC_node, TC_lymph] in cells
            BC_counts: [BC_blood, BC_spleen, BC_node, BC_lymph] in cells
            tumor_cell_count: Number of tumor cells in lymph node

        Returns:
            initial_conditions: State vector for activation model
        """
        # State variables for each compartment (blood, spleen, node, lymph):
        # 0-3: vATC_BC (virtual ATCs against B-cells)
        # 4-7: pATC (proliferating ATCs - clonally expanded)
        # 8: vATC_tumor (virtual ATCs against tumor cells - only in lymph node)

        initial_conditions = [0.0] * 9  # 9 total state variables

        return initial_conditions

    def calculate_activation_rates(self,
                                   t: float,
                                   trimers_per_BC: List[float],
                                   trimers_per_tumor: float,
                                   TC_counts: List[float],
                                   BC_counts: List[float],
                                   tumor_cell_count: float,
                                   vATC_BC: List[float],
                                   vATC_tumor: float,
                                   pATC: List[float]) -> Dict[str, List[float]]:
        """
        Calculate activation and expansion rates

        Args:
            t: Current time
            trimers_per_BC: Trimer molecules per B-cell for each compartment
            trimers_per_tumor: Trimer molecules per tumor cell in lymph node
            TC_counts, BC_counts: Cell counts per compartment
            tumor_cell_count: Tumor cells in lymph node
            vATC_BC, vATC_tumor, pATC: Current activated T-cell counts

        Returns:
            Dictionary of activation rates
        """
        p = self.parameters

        rates = {
            'activation_BC': [0.0] * 4,
            'activation_tumor': 0.0,
            'expansion': [0.0] * 4,
            'death_vATC': [0.0] * 4,
            'death_pATC': [0.0] * 4,
            'death_vATC_tumor': 0.0,  # ADD THIS MISSING KEY
            'RELU': 0.01  # Default to low activation
        }

        # Check activation delay
        if t < p['TAD']:
            # No activation before TAD
            return rates

        # ===== ACTIVATION AGAINST B-CELLS =====
        for comp in range(4):  # blood, spleen, node, lymph
            if BC_counts[comp] > 0 and trimers_per_BC[comp] > 0:
                # Convert sim_slope from 10^6 cells/day to cells/day
                sim_slope_cells = p['sim_slope'] * p['million_cells']
                rates['activation_BC'][comp] = sim_slope_cells * trimers_per_BC[comp]

        # ===== ACTIVATION AGAINST TUMOR CELLS =====
        # Only in lymph node compartment (compartment 2)
        if tumor_cell_count > 0 and trimers_per_tumor > 0:
            # Apply RELU step function
            if trimers_per_tumor > p['Trimer_Threshold']:
                rates['RELU'] = 1.0  # Full activation
            else:
                rates['RELU'] = 0.01  # 1% of full activation

            # Convert sim_slope_tumor from 10^6 cells/day to cells/day
            sim_slope_tumor_cells = p['sim_slope_tumor'] * p['million_cells']
            rates['activation_tumor'] = rates['RELU'] * sim_slope_tumor_cells * trimers_per_tumor

        # ===== CLONAL EXPANSION =====
        for comp in range(4):
            if vATC_BC[comp] > 0:
                rates['expansion'][comp] = p['expand_factor'] * vATC_BC[comp]

        # ===== CELL DEATH =====
        # Check proliferation duration
        if t >= p['Tp']:
            # Activated T-cells can die after Tp
            for comp in range(4):
                rates['death_vATC'][comp] = p['kout_ATC'] * vATC_BC[comp]
                rates['death_pATC'][comp] = p['kout_ATC'] * pATC[comp]
            rates['death_vATC_tumor'] = p['kout_ATC'] * vATC_tumor  # SET THIS VALUE
        else:
            # No death before Tp
            rates['death_vATC_tumor'] = 0.0  # SET THIS VALUE

        return rates

    def derivatives(self, y: List[float], t: float,
                    trimers_per_BC: List[float],
                    trimers_per_tumor: float,
                    TC_counts: List[float],
                    BC_counts: List[float],
                    tumor_cell_count: float,
                    trafficking_rates: Dict[str, List[float]]) -> List[float]:
        """
        Calculate derivatives for T-cell activation

        Args:
            y: State vector [vATC_BC_blood, vATC_BC_spleen, vATC_BC_node, vATC_BC_lymph,
                            pATC_blood, pATC_spleen, pATC_node, pATC_lymph,
                            vATC_tumor]
            t: Time
            trimers_per_BC: Trimer molecules per B-cell for each compartment
            trimers_per_tumor: Trimer molecules per tumor cell in lymph node
            TC_counts, BC_counts: Cell counts per compartment
            tumor_cell_count: Tumor cells in lymph node
            trafficking_rates: Dictionary with trafficking rates between compartments

        Returns:
            derivatives: List of derivatives for all state variables
        """
        p = self.parameters

        # Unpack state variables
        vATC_BC = y[0:4]  # Virtual ATCs against B-cells
        pATC = y[4:8]  # Proliferating ATCs (clonally expanded)
        vATC_tumor = y[8]  # Virtual ATCs against tumor cells (only in lymph node)

        # Calculate activation rates
        rates = self.calculate_activation_rates(
            t, trimers_per_BC, trimers_per_tumor,
            TC_counts, BC_counts, tumor_cell_count,
            vATC_BC, vATC_tumor, pATC
        )

        derivatives = []

        # ===== vATC_BC DERIVATIVES (compartments 0-3) =====
        for comp in range(4):
            # Basic terms: activation - death
            d_vATC_BC = (rates['activation_BC'][comp]
                         - rates['death_vATC'][comp])

            # Add trafficking (using same rates as normal T-cells)
            if comp == 0:  # blood
                d_vATC_BC += (trafficking_rates['lymph_to_blood'] * vATC_BC[3]  # from lymph
                              - trafficking_rates['blood_to_spleen'] * vATC_BC[0])  # to spleen
            elif comp == 1:  # spleen
                d_vATC_BC += (trafficking_rates['blood_to_spleen'] * vATC_BC[0]  # from blood
                              - trafficking_rates['spleen_to_node'] * vATC_BC[1])  # to node
            elif comp == 2:  # lymph node
                d_vATC_BC += (trafficking_rates['spleen_to_node'] * vATC_BC[1]  # from spleen
                              - trafficking_rates['node_to_lymph'] * vATC_BC[2])  # to lymph
            else:  # lymph
                d_vATC_BC += (trafficking_rates['node_to_lymph'] * vATC_BC[2]  # from node
                              - trafficking_rates['lymph_to_blood'] * vATC_BC[3])  # to blood

            derivatives.append(d_vATC_BC)

        # ===== pATC DERIVATIVES (compartments 4-7) =====
        for comp in range(4):
            # Basic terms: expansion - death
            d_pATC = (rates['expansion'][comp]
                      - rates['death_pATC'][comp])

            # Add trafficking
            if comp == 0:  # blood
                d_pATC += (trafficking_rates['lymph_to_blood'] * pATC[3]  # from lymph
                           - trafficking_rates['blood_to_spleen'] * pATC[0])  # to spleen
            elif comp == 1:  # spleen
                d_pATC += (trafficking_rates['blood_to_spleen'] * pATC[0]  # from blood
                           - trafficking_rates['spleen_to_node'] * pATC[1])  # to node
            elif comp == 2:  # lymph node
                d_pATC += (trafficking_rates['spleen_to_node'] * pATC[1]  # from spleen
                           - trafficking_rates['node_to_lymph'] * pATC[2])  # to lymph
            else:  # lymph
                d_pATC += (trafficking_rates['node_to_lymph'] * pATC[2]  # from node
                           - trafficking_rates['lymph_to_blood'] * pATC[3])  # to blood

            derivatives.append(d_pATC)

        # ===== vATC_tumor DERIVATIVE (compartment 8) =====
        # Tumor-specific ATCs don't traffic - they stay in lymph node
        d_vATC_tumor = (rates['activation_tumor']
                        - rates['death_vATC_tumor'])
        derivatives.append(d_vATC_tumor)

        return derivatives

    def simulate(self, initial_conditions: List[float], t_span: np.ndarray,
                 trimers_per_BC: List[float],
                 trimers_per_tumor: float,
                 TC_counts: List[float],
                 BC_counts: List[float],
                 tumor_cell_count: float,
                 trafficking_rates: Dict[str, List[float]],
                 rtol: float = 1e-6, atol: float = 1e-8) -> np.ndarray:
        """
        Simulate the T-cell activation submodel
        """

        def derivs(y, t):
            return self.derivatives(y, t, trimers_per_BC, trimers_per_tumor,
                                    TC_counts, BC_counts, tumor_cell_count,
                                    trafficking_rates)

        solution = odeint(derivs, initial_conditions, t_span, rtol=rtol, atol=atol)
        return solution

    def calculate_activation_metrics(self, solution: np.ndarray,
                                     TC_counts: List[float]) -> Dict[str, np.ndarray]:
        """
        Calculate activation metrics from simulation results

        Args:
            solution: Simulation results
            TC_counts: Total T-cell counts per compartment

        Returns:
            Dictionary of activation metrics
        """
        # Extract state variables
        vATC_BC = solution[:, 0:4]  # Virtual ATCs against B-cells
        pATC = solution[:, 4:8]  # Proliferating ATCs
        vATC_tumor = solution[:, 8]  # Virtual ATCs against tumor cells

        metrics = {}

        # Calculate total activated T-cells per compartment
        total_ATC_blood = vATC_BC[:, 0] + pATC[:, 0]
        total_ATC_spleen = vATC_BC[:, 1] + pATC[:, 1]
        total_ATC_node = vATC_BC[:, 2] + pATC[:, 2] + vATC_tumor
        total_ATC_lymph = vATC_BC[:, 3] + pATC[:, 3]

        metrics['total_ATC'] = np.column_stack([
            total_ATC_blood, total_ATC_spleen, total_ATC_node, total_ATC_lymph
        ])

        # Calculate activation fraction (activated / total T-cells)
        activation_fraction_blood = total_ATC_blood / (TC_counts[0] + 1e-10)
        activation_fraction_spleen = total_ATC_spleen / (TC_counts[1] + 1e-10)
        activation_fraction_node = total_ATC_node / (TC_counts[2] + 1e-10)
        activation_fraction_lymph = total_ATC_lymph / (TC_counts[3] + 1e-10)

        metrics['activation_fraction'] = np.column_stack([
            activation_fraction_blood, activation_fraction_spleen,
            activation_fraction_node, activation_fraction_lymph
        ])

        return metrics