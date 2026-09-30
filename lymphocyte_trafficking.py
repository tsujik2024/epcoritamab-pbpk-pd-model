"""Lymphocyte trafficking and turnover submodel"""

import numpy as np
from scipy.integrate import odeint

from src.models.traffic import loop_flux
from src.parameters.lymphocyte_params import get_lymphocyte_default_parameters, calculate_derived_parameters

# state: 0-3 TC (blood, spleen, node, lymph), 4-7 BC, 8-9 AF_TC/AF_BC, 10 INJ
INJ = 10


class LymphocyteTrafficking:

    def __init__(self, parameters=None):
        params = get_lymphocyte_default_parameters()
        if parameters:
            params.update(parameters)
        self.parameters = calculate_derived_parameters(params)

    def rates(self):
        p = self.parameters
        return p['k_pt'], p['k_tn'], p['k_nl'], p['k_lp']

    def get_initial_conditions(self):
        p = self.parameters
        return [
            p['TC_blood_base'], p['TC_spleen_base'], p['TC_node_base'], p['TC_lymph_base'],
            p['BC_blood_base'], p['BC_spleen_base'], p['BC_node_base'], p['BC_lymph_base'],
            1.0, 1.0, 0.0,
        ]

    def derivatives(self, y, t, kill=None, patc_blood=0.0):
        """kill: B cells killed per day in each compartment. patc_blood: expanded ATCs in blood (count toward T-cell homeostasis)"""
        p = self.parameters
        boost = 1 + p['INJ_Scaler'] * y[INJ]
        k = self.rates()
        dy = np.zeros(11)

        for i, (kin, kout) in enumerate([(p['kin_TC_cells'], p['kout_TC']),
                                         (p['kin_BC_cells'], p['kout_BC'])]):
            o = 4 * i
            x = np.asarray(y[o:o + 4])
            prod = np.array([kin * y[8 + i] ** p['r'], 0, 0, 0])
            dy[o:o + 4] = prod - kout * x + loop_flux(x, k, boost)

        if kill is not None:
            dy[4:8] -= kill

        vol = p['V_blood'] * p['mm3_to_L']
        dy[8] = p['k_t'] * (p['TC_plasma_base'] / ((y[0] + patc_blood) / vol) - y[8])
        dy[9] = p['k_t'] * (p['BC_plasma_base'] / (y[4] / vol) - y[9])
        dy[INJ] = -p['k_decay'] * y[INJ]
        return dy.tolist()

    def simulate(self, y0, t_span, injection_times=None, rtol=1e-6, atol=1e-8):
        t_span = np.asarray(t_span, dtype=float)
        inj = list(injection_times or [])
        y = np.array(y0, dtype=float)
        cuts = [t for t in sorted(inj) if t_span[0] < t < t_span[-1]]
        if t_span[0] in inj:
            y[INJ] = 1.0

        # restart the solver at each dose, otherwise LSODA can step right over the INJ jump
        times = np.union1d(t_span, cuts)
        out = np.empty((len(times), len(y)))
        out[0] = y
        edges = [0] + [int(np.searchsorted(times, c)) for c in cuts] + [len(times) - 1]

        for lo, hi in zip(edges[:-1], edges[1:]):
            seg = odeint(self.derivatives, y, times[lo:hi + 1], rtol=rtol, atol=atol)
            out[lo + 1:hi + 1] = seg[1:]
            y = seg[-1].copy()
            if hi < len(times) - 1:
                y[INJ] = 1.0  # dose sets INJ to 1, doesn't stack
                out[hi] = y

        return out[np.searchsorted(times, t_span)]

    def calculate_concentrations(self, sol):
        """Blood T/B-cell concentrations in cells/mm^3"""
        p = self.parameters
        vol = p['V_blood'] * p['mm3_to_L']
        return sol[:, 0] / vol, sol[:, 4] / vol
