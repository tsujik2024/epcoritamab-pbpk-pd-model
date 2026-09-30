"""Epcoritamab binding to CD3 / CD20"""

import numpy as np
from scipy.integrate import odeint

from src.models.traffic import loop_flux
from src.parameters.binding_params import get_binding_default_parameters, calculate_binding_derived_parameters

# per compartment (blood, spleen, node, lymph):
# CD3, CD3-Ab, CD20, CD20-Ab, trimer, tumor CD20, tumor CD20-Ab, tumor trimer
N_STATES = 8
NODE = 2


class BindingSubmodel:

    def __init__(self, parameters=None, lymphocyte_params=None):
        p = get_binding_default_parameters()
        if parameters:
            p.update(parameters)
        if lymphocyte_params:
            p = calculate_binding_derived_parameters(p, lymphocyte_params)
        self.parameters = p

    def volumes(self):
        p = self.parameters
        return np.array([p['V_blood'], p['V_spleen_tissue'], p['V_node'], p['V_lymph']])

    def _conc(self, cells, receptors, vol):
        """cells * receptors/cell -> nM in a compartment of `vol` liters"""
        return cells * receptors / (self.parameters['nM_to_molecules'] * vol)

    def get_initial_conditions(self, tc_counts, bc_counts, tumor_cells=0.0):
        p = self.parameters
        vols = self.volumes()
        y = np.zeros((4, N_STATES))
        for c in range(4):
            y[c, 0] = self._conc(tc_counts[c], p['RCD3'], vols[c])
            y[c, 2] = self._conc(bc_counts[c], p['RCD20'], vols[c])
        y[NODE, 5] = self._conc(tumor_cells, p['RCD20_tumor'], vols[NODE])
        return y.ravel().tolist()

    def calculate_binding_terms(self, x, ab):
        p = self.parameters
        cd3, cd3_ab, cd20, cd20_ab, tri, t20, t20_ab, t_tri = x
        return {
            'CD3_binding': p['kon_CD3'] * ab * cd3,
            'CD3_unbinding': p['koff_CD3'] * cd3_ab,
            'CD20_binding': p['kon_CD20'] * ab * cd20,
            'CD20_unbinding': p['koff_CD20'] * cd20_ab,
            'trimer_formation_CD3': p['kon_CD20'] * cd20 * cd3_ab,
            'trimer_formation_CD20': p['kon_CD3'] * cd3 * cd20_ab,
            'trimer_dissociation_CD20': p['koff_CD20'] * tri,
            'trimer_dissociation_CD3': p['koff_CD3'] * tri,
            'CD20_tumor_binding': p['kon_CD20'] * ab * t20,
            'CD20_tumor_unbinding': p['koff_CD20'] * t20_ab,
            'trimer_tumor_formation': p['kon_CD20'] * t20 * cd3_ab,
            'trimer_tumor_dissociation': p['koff_CD20'] * t_tri,
            'trimer_tumor_formation_CD3': p['kon_CD3'] * cd3 * t20_ab,
            'trimer_tumor_dissociation_CD3': p['koff_CD3'] * t_tri,
        }

    def rhs(self, y, ab, tc, bc, tumor, kout_tc, kout_bc, kill_bc, kill_tumor,
            boost, kin_tc, kin_bc, k_traffic):
        """Returns (dy, uptake). uptake is nmol/d of free drug bound in each compartment.

        tc/bc/tumor are the *current* cell counts (they change with trafficking and killing).
        kill_bc is a per-compartment rate (1/d), kill_tumor a scalar (1/d). kin_* in cells/d.
        """
        p = self.parameters
        vols = self.volumes()
        y2 = np.asarray(y).reshape(4, N_STATES)
        dy = np.zeros((4, N_STATES))
        uptake = np.zeros(4)

        for c in range(4):
            x = y2[c]
            cd3, cd3_ab, cd20, cd20_ab, tri, t20, t20_ab, t_tri = x
            b = self.calculate_binding_terms(x, ab[c])
            kill = kill_bc[c]

            synth3 = p['kdeg_CD3'] * self._conc(tc[c], p['RCD3'], vols[c])
            synth20 = p['kdeg_CD20'] * self._conc(bc[c], p['RCD20'], vols[c])
            prod3 = prod20 = 0.0
            if c == 0:
                prod3 = self._conc(kin_tc, p['RCD3'], vols[0])
                prod20 = self._conc(kin_bc, p['RCD20'], vols[0])

            d_cd3 = (-b['CD3_binding'] + b['CD3_unbinding']
                     - b['trimer_formation_CD20'] + b['trimer_dissociation_CD3']
                     + synth3 + prod3 - (p['kdeg_CD3'] + kout_tc) * cd3)
            d_cd3_ab = (b['CD3_binding'] - b['CD3_unbinding']
                        - b['trimer_formation_CD3'] + b['trimer_dissociation_CD20']
                        - (p['kint_CD3'] + kout_tc) * cd3_ab
                        + (kout_bc + kill) * tri)
            d_cd20 = (-b['CD20_binding'] + b['CD20_unbinding']
                      - b['trimer_formation_CD3'] + b['trimer_dissociation_CD20']
                      + synth20 + prod20 - (p['kdeg_CD20'] + kout_bc) * cd20
                      - kill * tri)
            d_cd20_ab = (b['CD20_binding'] - b['CD20_unbinding']
                         - b['trimer_formation_CD20'] + b['trimer_dissociation_CD3']
                         - (p['kint_CD20'] + kout_bc) * cd20_ab
                         + kout_tc * tri - kill * tri)
            d_tri = (b['trimer_formation_CD3'] + b['trimer_formation_CD20']
                     - b['trimer_dissociation_CD20'] - b['trimer_dissociation_CD3']
                     - (kout_tc + kout_bc + kill) * tri)
            d_t20 = d_t20_ab = d_t_tri = 0.0
            bound = (b['CD3_binding'] - b['CD3_unbinding']
                     + b['CD20_binding'] - b['CD20_unbinding'])

            if c == NODE:
                # paper only says "a similar set of equations" for tumor; the tumor-cell death
                # terms (kill_tumor) are my assumption
                t20_0 = self._conc(tumor, p['RCD20_tumor'], vols[c])
                d_t20 = (-b['CD20_tumor_binding'] + b['CD20_tumor_unbinding']
                         - b['trimer_tumor_formation'] + b['trimer_tumor_dissociation']
                         + p['kdeg_CD20'] * (t20_0 - t20) - kill_tumor * t20)
                d_t20_ab = (b['CD20_tumor_binding'] - b['CD20_tumor_unbinding']
                            - b['trimer_tumor_formation_CD3'] + b['trimer_tumor_dissociation_CD3']
                            - (p['kint_CD20'] + kill_tumor) * t20_ab + kout_tc * t_tri)
                d_t_tri = (b['trimer_tumor_formation'] + b['trimer_tumor_formation_CD3']
                           - b['trimer_tumor_dissociation'] - b['trimer_tumor_dissociation_CD3']
                           - (kout_tc + kill_tumor) * t_tri)
                d_cd3 += -b['trimer_tumor_formation_CD3'] + b['trimer_tumor_dissociation_CD3']
                d_cd3_ab += (-b['trimer_tumor_formation'] + b['trimer_tumor_dissociation']
                             + kill_tumor * t_tri)
                bound += b['CD20_tumor_binding'] - b['CD20_tumor_unbinding']

            dy[c] = [d_cd3, d_cd3_ab, d_cd20, d_cd20_ab, d_tri, d_t20, d_t20_ab, d_t_tri]
            uptake[c] = bound * vols[c]

        # everything on a cell moves with it, except tumor-bound species
        for s in range(5):
            dy[:, s] += loop_flux(y2[:, s], k_traffic, boost, vols)

        return dy.ravel(), uptake

    def derivatives(self, y, t, ab_conc, tc_counts, bc_counts, tumor_cells,
                    kout_tc, kout_bc, kill_bc=None, kill_tumor=0.0, boost=1.0,
                    kin_tc=0.0, kin_bc=0.0, k_traffic=(50.0, 1.67, 1672.0, 2.63)):
        kill_bc = np.zeros(4) if kill_bc is None else kill_bc
        dy, _ = self.rhs(y, ab_conc, tc_counts, bc_counts, tumor_cells, kout_tc, kout_bc,
                         kill_bc, kill_tumor, boost, kin_tc, kin_bc, k_traffic)
        return dy.tolist()

    def simulate(self, y0, t_span, ab_conc, tc_counts, bc_counts, tumor_cells,
                 kout_tc, kout_bc, rtol=1e-6, atol=1e-10, **kw):
        """Standalone run with fixed drug levels and cell counts"""
        def f(y, t):
            return self.derivatives(y, t, ab_conc, tc_counts, bc_counts, tumor_cells,
                                    kout_tc, kout_bc, **kw)

        return odeint(f, y0, t_span, rtol=rtol, atol=atol)

    def trimers_per_cell(self, y, bc_counts, tumor_cells):
        """Trimer molecules per B cell (4 compartments) and per tumor cell"""
        y2 = np.asarray(y).reshape(4, N_STATES)
        vols = self.volumes()
        nm = self.parameters['nM_to_molecules']
        per_bc = y2[:, 4] * nm * vols / np.maximum(bc_counts, 1.0)
        per_tumor = y2[NODE, 7] * nm * vols[NODE] / max(tumor_cells, 1.0)
        return per_bc, per_tumor

    def calculate_trimer_counts(self, solution, volumes):
        """Trimer molecules per compartment over time"""
        cols = [4 + N_STATES * c for c in range(4)]
        return solution[:, cols] * self.parameters['nM_to_molecules'] * np.asarray(volumes)
