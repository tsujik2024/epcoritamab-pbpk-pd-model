import numpy as np
from scipy.integrate import odeint

# Table S1
DEFAULTS = {
    # volumes (L)
    'V_plasma': 2.6, 'V_tight': 8.11, 'V_leaky': 4.37,
    'V_spleen': 0.0433, 'V_node': 0.0082, 'V_lymph': 5.2,
    # lymph flows (L/d)
    'L_tight': 0.957, 'L_leaky': 1.94, 'L_spleen': 0.304, 'L': 2.9,
    # reflection coefficients
    'sigma_tight': 0.95, 'sigma_leaky': 0.85, 'sigma_spleen': 0.82, 'sigma_lymph': 0.2,
    'k_a': 0.131,  # 1/d
    'CL': 2.47,  # L/d
    'K_m': 0.0461,  # nM
    'V_max': 0.185,  # really nmol/d, the table says nM/d
}


class PKSubmodel:
    """SC epcoritamab PBPK. State: [plasma, leaky, tight, spleen, node, lymph, depot]
    Concentrations in nM, depot in nmol."""

    def __init__(self, parameters=None):
        self.parameters = {**DEFAULTS, **(parameters or {})}

    def get_initial_conditions(self, dose=0.0):
        return [0.0] * 6 + [dose]

    def distribution_terms(self, y):
        p = self.parameters
        c_plasma, c_leaky, c_tight, c_spleen, c_node, c_lymph = y[:6]
        out = 1 - p['sigma_lymph']
        return {
            'plasma_to_leaky': p['L_leaky'] * (1 - p['sigma_leaky']) * c_plasma,
            'plasma_to_tight': p['L_tight'] * (1 - p['sigma_tight']) * c_plasma,
            'plasma_to_spleen': p['L_spleen'] * (1 - p['sigma_spleen']) * c_plasma,
            'lymph_to_plasma': p['L'] * c_lymph,
            'leaky_to_lymph': p['L_leaky'] * out * c_leaky,
            'tight_to_lymph': p['L_tight'] * out * c_tight,
            'spleen_to_node': p['L_spleen'] * out * c_spleen,
            'node_to_lymph': p['L_spleen'] * c_node,
        }

    def clearance_terms(self, c_plasma):
        p = self.parameters
        linear = p['CL'] * c_plasma
        saturable = p['V_max'] * c_plasma / (p['K_m'] + c_plasma)
        return linear, saturable

    def derivatives(self, y, t, bound=None):
        """bound: nmol/d of free drug taken up by binding in [plasma, spleen, node, lymph]"""
        p = self.parameters
        d = self.distribution_terms(y)
        lin, sat = self.clearance_terms(y[0])
        absorb = p['k_a'] * y[6]
        b = np.zeros(4) if bound is None else bound

        return [
            (-lin - sat - d['plasma_to_leaky'] - d['plasma_to_tight']
             - d['plasma_to_spleen'] + d['lymph_to_plasma'] - b[0]) / p['V_plasma'],
            (d['plasma_to_leaky'] - d['leaky_to_lymph']) / p['V_leaky'],
            (d['plasma_to_tight'] - d['tight_to_lymph']) / p['V_tight'],
            (d['plasma_to_spleen'] - d['spleen_to_node'] - b[1]) / p['V_spleen'],
            (d['spleen_to_node'] - d['node_to_lymph'] - b[2]) / p['V_node'],
            (absorb + d['leaky_to_lymph'] + d['tight_to_lymph']
             + d['node_to_lymph'] - d['lymph_to_plasma'] - b[3]) / p['V_lymph'],
            -absorb,
        ]

    def simulate(self, y0, t_span, rtol=1e-6, atol=1e-8):
        return odeint(self.derivatives, y0, t_span, rtol=rtol, atol=atol)
