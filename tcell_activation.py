"""T-cell activation submodel"""

import numpy as np
from scipy.integrate import odeint

from src.models.traffic import loop_flux
from src.parameters.tcell_activation_params import get_tcell_activation_default_parameters

# state: 0-3 vATC vs B-cells, 4-7 pATC (blood, spleen, node, lymph), 8 vATC vs tumor (node only)
NODE = 2


class TCellActivation:

    def __init__(self, parameters=None):
        self.parameters = {**get_tcell_activation_default_parameters(), **(parameters or {})}

    def get_initial_conditions(self):
        return [0.0] * 9

    def calculate_activation_rates(self, t, per_bc, per_tumor, bc_counts, tumor_cells,
                                   v_bc, v_tumor, p_atc):
        p = self.parameters
        rates = {
            'activation_BC': np.zeros(4),
            'activation_tumor': 0.0,
            'expansion': np.zeros(4),
            'death_v': np.zeros(4),
            'death_p': np.zeros(4),
            'death_v_tumor': 0.0,
            'RELU': 0.01,
        }
        if t < p['TAD']:
            return rates

        slope = p['sim_slope'] * p['million_cells']
        for i in range(4):
            if bc_counts[i] > 0 and per_bc[i] > 0:
                rates['activation_BC'][i] = slope * per_bc[i]
        rates['expansion'] = p['expand_factor'] * np.asarray(v_bc, dtype=float)
        rates['expansion'][NODE] += p['expand_factor'] * v_tumor

        if tumor_cells > 0 and per_tumor > 0:
            rates['RELU'] = 1.0 if per_tumor > p['Trimer_Threshold'] else 0.01
            slope_tumor = p['sim_slope_tumor'] * p['million_cells']
            rates['activation_tumor'] = rates['RELU'] * slope_tumor * per_tumor

        if t >= p['Tp']:
            rates['death_v'] = p['kout_ATC'] * np.asarray(v_bc, dtype=float)
            rates['death_p'] = p['kout_ATC'] * np.asarray(p_atc, dtype=float)
            rates['death_v_tumor'] = p['kout_ATC'] * v_tumor

        return rates

    def derivatives(self, y, t, per_bc, per_tumor, bc_counts, tumor_cells, k_traffic, boost=1.0,
                    t_gate=None):
        # t_gate lets the caller pin the TAD/Tp switches for a whole segment, so the solver never sees a jump
        v_bc, p_atc, v_tumor = y[0:4], y[4:8], y[8]
        r = self.calculate_activation_rates(t if t_gate is None else t_gate, per_bc, per_tumor, bc_counts, tumor_cells,
                                            v_bc, v_tumor, p_atc)
        d = np.zeros(9)
        d[0:4] = r['activation_BC'] - r['death_v'] + loop_flux(v_bc, k_traffic, boost)
        d[4:8] = r['expansion'] - r['death_p'] + loop_flux(p_atc, k_traffic, boost)
        d[8] = r['activation_tumor'] - r['death_v_tumor']  # tumor ATCs stay in the node
        return d.tolist()

    def simulate(self, y0, t_span, per_bc, per_tumor, bc_counts, tumor_cells, k_traffic,
                 rtol=1e-6, atol=1e-8):
        """per_bc / per_tumor: constants, or arrays sampled on t_span ((n, 4) and (n,))"""
        per_bc = np.asarray(per_bc, dtype=float)
        per_tumor = np.asarray(per_tumor, dtype=float)

        def f(y, t):
            bc = per_bc
            if per_bc.ndim == 2:
                bc = [np.interp(t, t_span, per_bc[:, i]) for i in range(4)]
            tum = np.interp(t, t_span, per_tumor) if per_tumor.ndim == 1 else per_tumor
            return self.derivatives(y, t, bc, float(tum), bc_counts, tumor_cells, k_traffic)

        return odeint(f, y0, t_span, rtol=rtol, atol=atol)

    def calculate_activation_metrics(self, solution, tc_counts):
        total = solution[:, 0:4] + solution[:, 4:8]
        total[:, NODE] += solution[:, 8]
        frac = total / (np.asarray(tc_counts, dtype=float) + 1e-10)
        return {'total_ATC': total, 'activation_fraction': frac}
