"""Lymphocyte trafficking and turnover submodel"""

import numpy as np
from typing import List, Dict, Optional
from scipy.integrate import odeint

from src.parameters.lymphocyte_params import get_lymphocyte_default_parameters, calculate_derived_parameters


class LymphocyteTrafficking:
    """Lymphocyte trafficking and turnover submodel"""

    def __init__(self, parameters: Optional[Dict] = None):
        self.parameters = self._build_parameters(parameters)
        self._validate_parameters()

    def _build_parameters(self, user_params: Optional[Dict]) -> Dict:
        """Build complete parameter set"""
        default_params = get_lymphocyte_default_parameters()
        complete_params = calculate_derived_parameters(default_params)

        if user_params is not None:
            # Update with user parameters and recalculate derived ones
            default_params.update(user_params)
            complete_params = calculate_derived_parameters(default_params)

        return complete_params

    def _validate_parameters(self):
        """Validate that all required parameters are present"""
        required_params = [
            'V_blood', 'V_spleen_tissue',
            'TC_plasma_base', 'BC_plasma_base',
            'kin_TC', 'kin_BC', 'kin_TC_cells', 'kin_BC_cells',
            'kout_TC', 'kout_BC',
            'k_pt', 'k_tn', 'k_nl', 'k_lp',
            'k_decay', 'INJ_Scaler',
            'k_t', 'r',
            'TC_blood_base', 'BC_blood_base',
            'TC_spleen_base', 'BC_spleen_base',
            'TC_node_base', 'BC_node_base',
            'TC_lymph_base', 'BC_lymph_base'
        ]

        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")

    def get_initial_conditions(self, injection_time: float = 0.0) -> List[float]:
        """Get initial conditions for simulation"""
        p = self.parameters

        # State variables:
        # 0: TC_blood, 1: TC_spleen, 2: TC_node, 3: TC_lymph
        # 4: BC_blood, 5: BC_spleen, 6: BC_node, 7: BC_lymph
        # 8: AF_TC (adaptive feedback for T cells)
        # 9: AF_BC (adaptive feedback for B cells)
        # 10: INJ (injection effect compartment)

        initial_conditions = [
            p['TC_blood_base'], p['TC_spleen_base'], p['TC_node_base'], p['TC_lymph_base'],
            p['BC_blood_base'], p['BC_spleen_base'], p['BC_node_base'], p['BC_lymph_base'],
            1.0,  # AF_TC starts at 1.0 (no perturbation)
            1.0,  # AF_BC starts at 1.0 (no perturbation)
            0.0  # INJ starts at 0 (no injection effect)
        ]

        return initial_conditions

    def administer_injection(self, t: float, injection_times: List[float]) -> float:
        """Calculate injection effect at time t"""
        INJ_value = 0.0
        for inj_time in injection_times:
            if t >= inj_time:
                # Injection effect: set to 1 at injection time, then decay
                time_since_injection = t - inj_time
                INJ_value += np.exp(-self.parameters['k_decay'] * time_since_injection)
        return INJ_value

    def derivatives(self, y: List[float], t: float, injection_times: List[float]) -> List[float]:
        """
        Calculate derivatives for lymphocyte trafficking
        y = [TC_blood, TC_spleen, TC_node, TC_lymph,
             BC_blood, BC_spleen, BC_node, BC_lymph,
             AF_TC, AF_BC, INJ]
        """
        p = self.parameters

        # Unpack state variables
        (TC_blood, TC_spleen, TC_node, TC_lymph,
         BC_blood, BC_spleen, BC_node, BC_lymph,
         AF_TC, AF_BC, INJ) = y

        # Calculate current injection effect
        current_INJ = self.administer_injection(t, injection_times)

        # ===== T-CELL DIFFERENTIAL EQUATIONS =====

        # T-cell production (enters blood compartment) - use cells/day, not millions
        TC_production = p['kin_TC_cells'] * (AF_TC ** p['r'])

        # T-cell death (occurs in all compartments)
        TC_death_blood = p['kout_TC'] * TC_blood
        TC_death_spleen = p['kout_TC'] * TC_spleen
        TC_death_node = p['kout_TC'] * TC_node
        TC_death_lymph = p['kout_TC'] * TC_lymph

        # T-cell trafficking - INCREASED injection effect for more redistribution
        injection_effect = 1 + p['INJ_Scaler'] * current_INJ
        TC_blood_to_spleen = p['k_pt'] * injection_effect * TC_blood
        TC_spleen_to_node = p['k_tn'] * TC_spleen
        TC_node_to_lymph = p['k_nl'] * TC_node
        TC_lymph_to_blood = p['k_lp'] * TC_lymph

        # T-cell derivatives
        dTC_blood_dt = (TC_production - TC_death_blood
                        - TC_blood_to_spleen + TC_lymph_to_blood)

        dTC_spleen_dt = (-TC_death_spleen
                         + TC_blood_to_spleen - TC_spleen_to_node)

        dTC_node_dt = (-TC_death_node
                       + TC_spleen_to_node - TC_node_to_lymph)

        dTC_lymph_dt = (-TC_death_lymph
                        + TC_node_to_lymph - TC_lymph_to_blood)

        # ===== B-CELL DIFFERENTIAL EQUATIONS =====

        # B-cell production (enters blood compartment)
        BC_production = p['kin_BC_cells'] * (AF_BC ** p['r'])

        # B-cell death (occurs in all compartments)
        BC_death_blood = p['kout_BC'] * BC_blood
        BC_death_spleen = p['kout_BC'] * BC_spleen
        BC_death_node = p['kout_BC'] * BC_node
        BC_death_lymph = p['kout_BC'] * BC_lymph

        # B-cell trafficking
        BC_blood_to_spleen = p['k_pt'] * injection_effect * BC_blood
        BC_spleen_to_node = p['k_tn'] * BC_spleen
        BC_node_to_lymph = p['k_nl'] * BC_node
        BC_lymph_to_blood = p['k_lp'] * BC_lymph

        # B-cell derivatives
        dBC_blood_dt = (BC_production - BC_death_blood
                        - BC_blood_to_spleen + BC_lymph_to_blood)

        dBC_spleen_dt = (-BC_death_spleen
                         + BC_blood_to_spleen - BC_spleen_to_node)

        dBC_node_dt = (-BC_death_node
                       + BC_spleen_to_node - BC_node_to_lymph)

        dBC_lymph_dt = (-BC_death_lymph
                        + BC_node_to_lymph - BC_lymph_to_blood)

        # ===== ADAPTIVE FEEDBACK COMPARTMENTS =====

        # Convert total blood cells to concentration (cells/mm^3) to match TC_plasma_base units
        TC_blood_conc = TC_blood / (p['V_blood'] * p['mm3_to_L'])
        BC_blood_conc = BC_blood / (p['V_blood'] * p['mm3_to_L'])

        # Exact equations from the paper:
        # dAF_TC/dt = k_t * (TC_plasma_base / TC_plasma) - k_t * AF_TC
        # dAF_BC/dt = k_t * (BC_plasma_base / BC_plasma) - k_t * AF_BC
        dAF_TC_dt = (p['k_t'] * (p['TC_plasma_base'] / TC_blood_conc)
                     - p['k_t'] * AF_TC)

        dAF_BC_dt = (p['k_t'] * (p['BC_plasma_base'] / BC_blood_conc)
                     - p['k_t'] * AF_BC)

        # ===== INJECTION EFFECT COMPARTMENT =====
        # This tracks the injection effect state
        dINJ_dt = -p['k_decay'] * INJ + (1.0 if any(abs(t - inj_time) < 1e-6 for inj_time in injection_times) else 0.0)

        return [
            dTC_blood_dt, dTC_spleen_dt, dTC_node_dt, dTC_lymph_dt,
            dBC_blood_dt, dBC_spleen_dt, dBC_node_dt, dBC_lymph_dt,
            dAF_TC_dt, dAF_BC_dt, dINJ_dt
        ]

    def simulate(self, initial_conditions: List[float], t_span: np.ndarray,
                 injection_times: List[float] = None, rtol: float = 1e-6,
                 atol: float = 1e-8) -> np.ndarray:
        """
        Simulate the lymphocyte trafficking model

        Args:
            initial_conditions: Initial state vector
            t_span: Time points to simulate
            injection_times: List of times when injections occur
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver

        Returns:
            solution: Array of state variables over time
        """
        if injection_times is None:
            injection_times = []

        # Set INJ to 1.0 at injection times in initial conditions
        initial_with_injections = initial_conditions.copy()
        if t_span[0] in injection_times:
            initial_with_injections[10] = 1.0

        def derivs(y, t):
            return self.derivatives(y, t, injection_times)

        solution = odeint(derivs, initial_with_injections, t_span,
                          rtol=rtol, atol=atol)
        return solution

    def calculate_concentrations(self, cell_counts: np.ndarray) -> np.ndarray:
        """Convert cell counts to concentrations (cells/mm^3)"""
        p = self.parameters

        # Extract cell counts from solution
        TC_blood = cell_counts[:, 0]
        BC_blood = cell_counts[:, 4]

        # Convert to concentrations (cells/mm^3)
        # Total cells in blood / (Blood volume in L * conversion to mm^3) = cells/mm^3
        TC_blood_conc = TC_blood / (p['V_blood'] * p['mm3_to_L'])
        BC_blood_conc = BC_blood / (p['V_blood'] * p['mm3_to_L'])

        return TC_blood_conc, BC_blood_conc