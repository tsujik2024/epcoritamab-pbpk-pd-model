"""Epcoritamab-binding submodel"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.integrate import odeint

from src.parameters.binding_params import get_binding_default_parameters, calculate_binding_derived_parameters


class BindingSubmodel:
    """Epcoritamab binding to CD3 and CD20 submodel"""

    def __init__(self, parameters: Optional[Dict] = None, lymphocyte_params: Optional[Dict] = None):
        self.base_parameters = get_binding_default_parameters()
        self.parameters = self._build_parameters(parameters, lymphocyte_params)
        self._validate_parameters()

    def _build_parameters(self, user_params: Optional[Dict], lymphocyte_params: Optional[Dict]) -> Dict:
        """Build complete parameter set"""
        complete_params = self.base_parameters.copy()

        if user_params is not None:
            complete_params.update(user_params)

        if lymphocyte_params is not None:
            complete_params = calculate_binding_derived_parameters(complete_params, lymphocyte_params)

        return complete_params

    def _validate_parameters(self):
        """Validate that all required parameters are present"""
        required_params = [
            'kon_CD3', 'koff_CD3', 'kon_CD20', 'koff_CD20',
            'kdeg_CD3', 'kdeg_CD20', 'kint_CD3', 'kint_CD20',
            'RCD3', 'RCD20', 'RCD20_tumor',
            'nM_to_molecules', 'L_to_nL'
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
        """
        p = self.parameters

        initial_conditions = []

        # Initialize all compartments
        for comp in range(4):  # blood, spleen, node, lymph
            # Calculate initial free CD3 based on T-cell count and receptor expression
            TC_count = TC_counts[comp] if comp < len(TC_counts) else 0.0

            # FIXED: The conversion was wrong
            # Original: CD3_free_initial = (TC_count * p['RCD3']) / (p['nM_to_molecules'] * p['L_to_nL'])
            # Correct: concentration_nM = (molecules) / (Avogadro * volume_L * 1e-9)
            # molecules = TC_count * RCD3
            # So: concentration_nM = (TC_count * RCD3) / (6.022e23 * volume_L * 1e-9)
            # = (TC_count * RCD3) / (6.022e14 * volume_L)

            # Get compartment volume
            if comp == 0:  # blood
                volume_L = p.get('V_blood', 4.73)
            elif comp == 1:  # spleen
                volume_L = p.get('V_spleen_tissue', 0.221)
            elif comp == 2:  # node
                volume_L = p.get('V_node', 0.0082)
            else:  # lymph
                volume_L = p.get('V_lymph', 5.2)

            CD3_free_initial = (TC_count * p['RCD3']) / (p['nM_to_molecules'] * volume_L)
            initial_conditions.extend([CD3_free_initial, 0.0])  # CD3_free, CD3_Ab

            # Calculate initial free CD20
            BC_count = BC_counts[comp] if comp < len(BC_counts) else 0.0
            CD20_free_initial = (BC_count * p['RCD20']) / (p['nM_to_molecules'] * volume_L)
            initial_conditions.extend([CD20_free_initial, 0.0])  # CD20_free, CD20_Ab

            # Trimers start at zero
            initial_conditions.append(0.0)  # trimer

            # Tumor-related states (only in lymph node compartment)
            if comp == 2:  # lymph node compartment
                CD20_free_tumor_initial = (tumor_cell_count * p['RCD20_tumor']) / (p['nM_to_molecules'] * volume_L)
                initial_conditions.extend([CD20_free_tumor_initial, 0.0, 0.0])
            else:
                initial_conditions.extend([0.0, 0.0, 0.0])

        return initial_conditions

    def calculate_binding_terms(self, concentrations: List[float], Ab_concentrations: List[float]) -> Dict[str, float]:
        """
        Calculate binding and unbinding terms for a single compartment

        Args:
            concentrations: [CD3_free, CD3_Ab, CD20_free, CD20_Ab, trimer,
                           CD20_free_tumor, CD20_Ab_tumor, trimer_tumor] for one compartment
            Ab_concentrations: Free antibody concentration in the compartment

        Returns:
            Dictionary of binding terms
        """
        p = self.parameters
        CD3_free, CD3_Ab, CD20_free, CD20_Ab, trimer, CD20_free_tumor, CD20_Ab_tumor, trimer_tumor = concentrations

        terms = {}

        # CD3 binding to free antibody
        terms['CD3_binding'] = p['kon_CD3'] * Ab_concentrations * CD3_free
        terms['CD3_unbinding'] = p['koff_CD3'] * CD3_Ab

        # CD20 binding to free antibody
        terms['CD20_binding'] = p['kon_CD20'] * Ab_concentrations * CD20_free
        terms['CD20_unbinding'] = p['koff_CD20'] * CD20_Ab

        # Trimer formation (CD3-Ab + CD20 and CD20-Ab + CD3)
        terms['trimer_formation_CD3'] = p['kon_CD20'] * CD20_free * CD3_Ab
        terms['trimer_formation_CD20'] = p['kon_CD3'] * CD3_free * CD20_Ab
        terms['trimer_dissociation_CD20'] = p['koff_CD20'] * trimer
        terms['trimer_dissociation_CD3'] = p['koff_CD3'] * trimer

        # Tumor-specific binding (only CD20 on tumor cells)
        terms['CD20_tumor_binding'] = p['kon_CD20'] * Ab_concentrations * CD20_free_tumor
        terms['CD20_tumor_unbinding'] = p['koff_CD20'] * CD20_Ab_tumor
        terms['trimer_tumor_formation'] = p['kon_CD20'] * CD20_free_tumor * CD3_Ab
        terms['trimer_tumor_dissociation'] = p['koff_CD20'] * trimer_tumor

        return terms

    def derivatives(self, y: List[float], t: float,
                    Ab_concentrations: List[float],
                    TC_counts: List[float],
                    BC_counts: List[float],
                    tumor_cell_count: float,
                    kout_TC: float, kout_BC: float,
                    kill_BC_rate: float = 0.0) -> List[float]:
        """
        Calculate derivatives for binding submodel

        Args:
            y: State vector for all compartments
            t: Time
            Ab_concentrations: [Ab_blood, Ab_spleen, Ab_node, Ab_lymph] in nM
            TC_counts: [TC_blood, TC_spleen, TC_node, TC_lymph] in cells
            BC_counts: [BC_blood, BC_spleen, BC_node, BC_lymph] in cells
            tumor_cell_count: Tumor cells in lymph node
            kout_TC, kout_BC: Cell death rates
            kill_BC_rate: Rate of B-cell killing by activated T-cells

        Returns:
            derivatives: List of derivatives for all state variables
        """
        p = self.parameters
        derivatives = []

        # Process each compartment
        for comp in range(4):  # blood, spleen, node, lymph
            # Extract state variables for this compartment
            idx = comp * 8  # 8 states per compartment
            comp_states = y[idx:idx + 8]
            Ab_conc = Ab_concentrations[comp] if comp < len(Ab_concentrations) else 0.0

            # Calculate binding terms
            binding_terms = self.calculate_binding_terms(comp_states, Ab_conc)

            CD3_free, CD3_Ab, CD20_free, CD20_Ab, trimer, CD20_free_tumor, CD20_Ab_tumor, trimer_tumor = comp_states

            # Get cell counts for this compartment
            TC_count = TC_counts[comp] if comp < len(TC_counts) else 0.0
            BC_count = BC_counts[comp] if comp < len(BC_counts) else 0.0

            # ===== CD3_FREE =====
            dCD3_free = (- binding_terms['CD3_binding'] + binding_terms['CD3_unbinding']
                         - binding_terms['trimer_formation_CD20'] + binding_terms['trimer_dissociation_CD3']
                         + self._synthesis_CD3(TC_count) - self._degradation_CD3(CD3_free)
                         + self._production_CD3(TC_count) - self._death_CD3_free(CD3_free, kout_TC))

            # ===== CD3_Ab (dimer) =====
            dCD3_Ab = (binding_terms['CD3_binding'] - binding_terms['CD3_unbinding']
                       - binding_terms['trimer_formation_CD3'] + binding_terms['trimer_dissociation_CD20']
                       - self._internalization_CD3(CD3_Ab) - self._death_CD3_Ab(CD3_Ab, kout_TC)
                       + self._death_trimer(trimer, kout_BC) + self._kill_BC_trimer(trimer, kill_BC_rate))

            # ===== CD20_FREE =====
            dCD20_free = (- binding_terms['CD20_binding'] + binding_terms['CD20_unbinding']
                          - binding_terms['trimer_formation_CD3'] + binding_terms['trimer_dissociation_CD20']
                          + self._synthesis_CD20(BC_count) - self._degradation_CD20(CD20_free)
                          + self._production_CD20(BC_count) - self._death_CD20_free(CD20_free, kout_BC)
                          - self._kill_BC_free(CD20_free, kill_BC_rate))

            # ===== CD20_Ab (dimer) =====
            dCD20_Ab = (binding_terms['CD20_binding'] - binding_terms['CD20_unbinding']
                        - binding_terms['trimer_formation_CD20'] + binding_terms['trimer_dissociation_CD3']
                        - self._internalization_CD20(CD20_Ab) - self._death_CD20_Ab(CD20_Ab, kout_BC)
                        + self._death_trimer(trimer, kout_TC) - self._kill_BC_Ab(CD20_Ab, kill_BC_rate))

            # ===== TRIMER =====
            dTrimer = (binding_terms['trimer_formation_CD3'] + binding_terms['trimer_formation_CD20']
                       - binding_terms['trimer_dissociation_CD20'] - binding_terms['trimer_dissociation_CD3']
                       - self._death_trimer(trimer, kout_TC) - self._death_trimer(trimer, kout_BC)
                       - self._kill_BC_trimer(trimer, kill_BC_rate))

            # ===== Tumor-specific states (only in lymph node) =====
            if comp == 2:  # lymph node compartment
                # CD20_free_tumor
                dCD20_free_tumor = (- binding_terms['CD20_tumor_binding'] + binding_terms['CD20_tumor_unbinding']
                                    - binding_terms['trimer_tumor_formation'] + binding_terms[
                                        'trimer_tumor_dissociation']
                                    + self._synthesis_CD20_tumor(tumor_cell_count) - self._degradation_CD20(
                            CD20_free_tumor))

                # CD20_Ab_tumor
                dCD20_Ab_tumor = (binding_terms['CD20_tumor_binding'] - binding_terms['CD20_tumor_unbinding']
                                  - binding_terms['trimer_tumor_formation'] + binding_terms['trimer_tumor_dissociation']
                                  - self._internalization_CD20(CD20_Ab_tumor))

                # trimer_tumor
                dTrimer_tumor = (binding_terms['trimer_tumor_formation'] - binding_terms['trimer_tumor_dissociation'])
            else:
                dCD20_free_tumor, dCD20_Ab_tumor, dTrimer_tumor = 0.0, 0.0, 0.0

            derivatives.extend([dCD3_free, dCD3_Ab, dCD20_free, dCD20_Ab, dTrimer,
                                dCD20_free_tumor, dCD20_Ab_tumor, dTrimer_tumor])

        return derivatives

    # Helper methods for the various terms
    def _synthesis_CD3(self, TC_count: float) -> float:
        """Synthesis of CD3 due to natural turnover"""
        p = self.parameters
        return (p['kdeg_CD3'] * TC_count * p['RCD3']) / (p['nM_to_molecules'] * p['L_to_nL'])

    def _synthesis_CD20(self, BC_count: float) -> float:
        """Synthesis of CD20 due to natural turnover"""
        p = self.parameters
        return (p['kdeg_CD20'] * BC_count * p['RCD20']) / (p['nM_to_molecules'] * p['L_to_nL'])

    def _synthesis_CD20_tumor(self, tumor_count: float) -> float:
        """Synthesis of CD20 on tumor cells"""
        p = self.parameters
        return (p['kdeg_CD20'] * tumor_count * p['RCD20_tumor']) / (p['nM_to_molecules'] * p['L_to_nL'])

    def _degradation_CD3(self, CD3_free: float) -> float:
        """Degradation of free CD3"""
        return self.parameters['kdeg_CD3'] * CD3_free

    def _degradation_CD20(self, CD20_free: float) -> float:
        """Degradation of free CD20"""
        return self.parameters['kdeg_CD20'] * CD20_free

    def _production_CD3(self, TC_count: float) -> float:
        """Production of new CD3 from T-cell production"""
        # This would come from kin_TC, but simplified for now
        return 0.0

    def _production_CD20(self, BC_count: float) -> float:
        """Production of new CD20 from B-cell production"""
        # This would come from kin_BC, but simplified for now
        return 0.0

    def _internalization_CD3(self, CD3_Ab: float) -> float:
        """Internalization of CD3-Ab dimer"""
        return self.parameters['kint_CD3'] * CD3_Ab

    def _internalization_CD20(self, CD20_Ab: float) -> float:
        """Internalization of CD20-Ab dimer"""
        return self.parameters['kint_CD20'] * CD20_Ab

    def _death_CD3_free(self, CD3_free: float, kout_TC: float) -> float:
        """Loss of free CD3 due to T-cell death"""
        return kout_TC * CD3_free

    def _death_CD3_Ab(self, CD3_Ab: float, kout_TC: float) -> float:
        """Loss of CD3-Ab due to T-cell death"""
        return kout_TC * CD3_Ab

    def _death_CD20_free(self, CD20_free: float, kout_BC: float) -> float:
        """Loss of free CD20 due to B-cell death"""
        return kout_BC * CD20_free

    def _death_CD20_Ab(self, CD20_Ab: float, kout_BC: float) -> float:
        """Loss of CD20-Ab due to B-cell death"""
        return kout_BC * CD20_Ab

    def _death_trimer(self, trimer: float, kout_cell: float) -> float:
        """Loss of trimers due to cell death"""
        return kout_cell * trimer

    def _kill_BC_free(self, CD20_free: float, kill_rate: float) -> float:
        """Loss of free CD20 due to B-cell killing"""
        return kill_rate * CD20_free

    def _kill_BC_Ab(self, CD20_Ab: float, kill_rate: float) -> float:
        """Loss of CD20-Ab due to B-cell killing"""
        return kill_rate * CD20_Ab

    def _kill_BC_trimer(self, trimer: float, kill_rate: float) -> float:
        """Loss of trimers due to B-cell killing"""
        return kill_rate * trimer

    def simulate(self, initial_conditions: List[float], t_span: np.ndarray,
                 Ab_concentrations: List[float], TC_counts: List[float],
                 BC_counts: List[float], tumor_cell_count: float,
                 kout_TC: float, kout_BC: float, kill_BC_rate: float = 0.0,
                 rtol: float = 1e-6, atol: float = 1e-8) -> np.ndarray:
        """
        Simulate the binding submodel
        """

        def derivs(y, t):
            return self.derivatives(y, t, Ab_concentrations, TC_counts, BC_counts,
                                    tumor_cell_count, kout_TC, kout_BC, kill_BC_rate)

        solution = odeint(derivs, initial_conditions, t_span, rtol=rtol, atol=atol)
        return solution

    def calculate_trimer_counts(self, solution: np.ndarray, compartment_volumes: List[float]) -> np.ndarray:
        """
        Calculate total trimer counts (molecules) from concentrations

        Args:
            solution: Simulation results
            compartment_volumes: [V_blood, V_spleen, V_node, V_lymph] in L

        Returns:
            trimer_counts: Total trimer molecules per compartment over time
        """
        p = self.parameters

        # Extract trimer concentrations (states 4, 9, 14, 19 for blood, spleen, node, lymph)
        trimer_conc_blood = solution[:, 4]  # State 4 in blood compartment
        trimer_conc_spleen = solution[:, 9]  # State 9 in spleen compartment
        trimer_conc_node = solution[:, 14]  # State 14 in node compartment
        trimer_conc_lymph = solution[:, 19]  # State 19 in lymph compartment

        # Convert concentrations to molecule counts
        trimer_counts_blood = trimer_conc_blood * p['nM_to_molecules'] * compartment_volumes[0] * 1e-9
        trimer_counts_spleen = trimer_conc_spleen * p['nM_to_molecules'] * compartment_volumes[1] * 1e-9
        trimer_counts_node = trimer_conc_node * p['nM_to_molecules'] * compartment_volumes[2] * 1e-9
        trimer_counts_lymph = trimer_conc_lymph * p['nM_to_molecules'] * compartment_volumes[3] * 1e-9

        return np.column_stack([trimer_counts_blood, trimer_counts_spleen,
                                trimer_counts_node, trimer_counts_lymph])