import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional


class PKSubmodel:
    """Pharmacokinetic submodel for epcoritamab"""

    def __init__(self, parameters: Optional[Dict] = None):
        self.parameters = self._build_parameters(parameters)
        self._validate_parameters()

    def _default_parameters(self) -> Dict:
        """Return default physiological and PK parameters from Table S1"""
        return {
            # Volumes (L)
            'V_plasma': 2.6,
            'V_tight': 8.11,
            'V_leaky': 4.37,
            'V_spleen': 0.0433,
            'V_node': 0.0082,
            'V_lymph': 5.2,

            # Lymph flows (L/d)
            'L_tight': 0.957,
            'L_leaky': 1.94,
            'L_spleen': 0.304,
            'L': 2.9,

            # Reflection coefficients
            'sigma_tight': 0.95,
            'sigma_leaky': 0.85,
            'sigma_spleen': 0.82,
            'sigma_lymph': 0.2,

            # PK parameters
            'k_a': 0.131,  # day^-1
            'CL': 2.47,  # L/d
            'K_m': 0.0461,  # nM/L
            'V_max': 0.185,  # nM/d
        }

    def _build_parameters(self, user_params: Optional[Dict]) -> Dict:
        """Build complete parameter set by combining defaults with user parameters"""
        default_params = self._default_parameters()

        if user_params is None:
            return default_params

        # Create a copy of defaults and update with user parameters
        complete_params = default_params.copy()
        complete_params.update(user_params)

        return complete_params

    def _validate_parameters(self):
        """Validate that all required parameters are present"""
        required_params = ['V_plasma', 'V_tight', 'V_leaky', 'V_spleen', 'V_node', 'V_lymph',
                           'L_tight', 'L_leaky', 'L_spleen', 'L',
                           'sigma_tight', 'sigma_leaky', 'sigma_spleen', 'sigma_lymph',
                           'k_a', 'CL', 'K_m', 'V_max']

        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")

    def get_initial_conditions(self, dose: float = 0.0) -> List[float]:
        """Get initial conditions for simulation"""
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dose]  # All compartments empty + injection site

    def distribution_terms(self, concentrations: List[float]) -> Dict[str, float]:
        """Calculate all distribution flows between compartments"""
        C_plasma, C_leaky, C_tight, C_spleen, C_node, C_lymph, _ = concentrations
        p = self.parameters

        return {
            'plasma_to_leaky': p['L_leaky'] * (1 - p['sigma_leaky']) * C_plasma,
            'plasma_to_tight': p['L_tight'] * (1 - p['sigma_tight']) * C_plasma,
            'plasma_to_spleen': p['L_spleen'] * (1 - p['sigma_spleen']) * C_plasma,
            'lymph_to_plasma': p['L'] * C_lymph,
            'leaky_to_lymph': p['L_leaky'] * (1 - p['sigma_lymph']) * C_leaky,
            'tight_to_lymph': p['L_tight'] * (1 - p['sigma_lymph']) * C_tight,
            'spleen_to_node': p['L_spleen'] * (1 - p['sigma_lymph']) * C_spleen,
            'node_to_lymph': p['L_spleen'] * C_node
        }

    def clearance_terms(self, C_plasma: float) -> Tuple[float, float]:
        """Calculate linear and nonlinear clearance"""
        p = self.parameters
        CL_linear = p['CL'] * C_plasma
        CL_nonlinear = (p['V_max'] * C_plasma) / (p['K_m'] + C_plasma)
        return CL_linear, CL_nonlinear

    def calculate_absorption_half_life(self) -> float:
        """Calculate the half-life of absorption for debugging"""
        import math
        return math.log(2) / self.parameters['k_a']

    def print_absorption_info(self, dose: float):
        """Print absorption information for debugging"""
        half_life = self.calculate_absorption_half_life()
        print(f"Absorption rate (k_a): {self.parameters['k_a']:.3f} day^-1")
        print(f"Absorption half-life: {half_life:.2f} days")
        print(f"Time for 99% absorption: {half_life * 6.64:.2f} days")  # 6.64 half-lives for 99%
        print(f"Initial dose: {dose:.1f} nM")

    def verify_parameters(self):
        """Verify that parameters make biological sense"""
        print("\n=== PARAMETER VERIFICATION ===")

        # Absorption half-life
        k_a = self.parameters['k_a']
        half_life = np.log(2) / k_a
        print(f"Absorption rate: {k_a:.3f} day^-1")
        print(f"Absorption half-life: {half_life:.2f} days")

        # Check if this makes sense for SC administration
        if half_life > 3:
            print("⚠️  WARNING: Absorption half-life seems long for SC administration")
        else:
            print("✓ Absorption half-life seems reasonable")

        # Total clearance
        CL = self.parameters['CL']
        print(f"Clearance: {CL:.2f} L/day")

        # Expected terminal half-life (approximate)
        V_plasma = self.parameters['V_plasma']
        expected_half_life = np.log(2) * V_plasma / CL
        print(f"Expected terminal half-life: {expected_half_life:.2f} days")

    def derivatives(self, y: List[float], t: float) -> List[float]:
        """
        Calculate derivatives for the PK system
        y = [C_plasma, C_leaky, C_tight, C_spleen, C_node, C_lymph, X_injection_site]
        """
        C_plasma, C_leaky, C_tight, C_spleen, C_node, C_lymph, X_injection_site = y
        p = self.parameters

        # Distribution terms
        dist = self.distribution_terms(y)

        # Clearance terms
        CL_linear, CL_nonlinear = self.clearance_terms(C_plasma)

        # Absorption from SC injection site
        absorption = p['k_a'] * X_injection_site

        # Differential equations
        dC_plasma_dt = (- CL_linear - CL_nonlinear
                        - dist['plasma_to_leaky'] - dist['plasma_to_tight'] - dist['plasma_to_spleen']
                        + dist['lymph_to_plasma']) / p['V_plasma']

        dC_leaky_dt = (dist['plasma_to_leaky'] - dist['leaky_to_lymph']) / p['V_leaky']
        dC_tight_dt = (dist['plasma_to_tight'] - dist['tight_to_lymph']) / p['V_tight']
        dC_spleen_dt = (dist['plasma_to_spleen'] - dist['spleen_to_node']) / p['V_spleen']
        dC_node_dt = (dist['spleen_to_node'] - dist['node_to_lymph']) / p['V_node']
        dC_lymph_dt = (absorption + dist['leaky_to_lymph'] + dist['tight_to_lymph']
                       + dist['node_to_lymph'] - dist['lymph_to_plasma']) / p['V_lymph']

        # Injection site depletion
        dX_injection_dt = -p['k_a'] * X_injection_site

        return [dC_plasma_dt, dC_leaky_dt, dC_tight_dt, dC_spleen_dt,
                dC_node_dt, dC_lymph_dt, dX_injection_dt]

    def simulate(self, initial_conditions: List[float], t_span: np.ndarray,
                 rtol: float = 1e-6, atol: float = 1e-8) -> np.ndarray:
        """
        Simulate the PK model

        Args:
            initial_conditions: [C_plasma, C_leaky, C_tight, C_spleen, C_node, C_lymph, X_injection_site]
            t_span: Time points to simulate
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver

        Returns:
            solution: Array of state variables over time
        """
        solution = odeint(self.derivatives, initial_conditions, t_span,
                          rtol=rtol, atol=atol)
        return solution