"""Coupled PK / binding / lymphocyte / activation / killing model"""

import time

import numpy as np
from scipy.integrate import odeint

from src.models.binding_submodel import BindingSubmodel, N_STATES, NODE
from src.models.lymphocyte_trafficking import LymphocyteTrafficking, INJ as LYM_INJ
from src.models.pk_submodel import PKSubmodel
from src.models.tcell_activation import TCellActivation
from src.models.tumor_model import TumorKilling

# state layout
PK = slice(0, 7)
LYM = slice(7, 18)
BIND = slice(18, 18 + 4 * N_STATES)
ACT = slice(50, 59)
TUM = 59
INJ = 7 + LYM_INJ
DEPOT = 6

MW_KDA = 149  # epcoritamab, my number, not from the paper


def mg_to_nmol(mg):
    return mg * 1e3 / MW_KDA


def dose_schedule(full_mg=48.0, priming_mg=0.16, inter_mg=0.8, days=168):
    """Step-up regimen from the paper: prime d0, intermediate d7, full d14/d21,
    then weekly (cycle 2), q2w (cycles 3-6), q4w after. Returns [(day, mg)]."""
    doses = [(0, priming_mg), (7, inter_mg), (14, full_mg), (21, full_mg)]
    doses += [(28 + 7 * i, full_mg) for i in range(4)]
    doses += [(56 + 14 * i, full_mg) for i in range(8)]
    d = 168
    while d < days:
        doses.append((d, full_mg))
        d += 28
    return [(d, mg) for d, mg in doses if d < days]


class EpcoritamabModel:

    def __init__(self, kind='DLBCL', pk=None, lymph=None, binding=None,
                 activation=None, tumor=None):
        self.pk = PKSubmodel(pk)
        self.lymph = LymphocyteTrafficking(lymph)
        self.binding = BindingSubmodel(binding, lymphocyte_params=self.lymph.parameters)
        self.act = TCellActivation(activation)
        self.tumor = TumorKilling(kind, tumor)
        self.vols = self.binding.volumes()
        self.k_traffic = self.lymph.rates()
        self.bc_base = np.array(self.lymph.get_initial_conditions()[4:8])
        self.gate = None
        self.deadline = None

    def initial_state(self):
        lym = self.lymph.get_initial_conditions()
        n0 = self.tumor.initial_cells()
        bnd = self.binding.get_initial_conditions(lym[0:4], lym[4:8], self.tumor.reachable(n0))
        return np.concatenate([self.pk.get_initial_conditions(), lym, bnd,
                               self.act.get_initial_conditions(), [n0]])

    def derivatives(self, y, t):
        if self.deadline and time.time() > self.deadline:
            raise TimeoutError('solver too slow, giving up')
        lp = self.lymph.parameters
        lym, act, n_tum = y[LYM], y[ACT], y[TUM]
        tc, bc = lym[0:4], lym[4:8]
        boost = 1 + lp['INJ_Scaler'] * lym[10]
        reach = self.tumor.reachable(n_tum)

        v_bc, p_atc, v_tum = act[0:4], act[4:8], act[8]
        kill_bc = self.tumor.bc_kill_rate(v_bc, self.vols)
        kill_tum = self.tumor.kill_rate(v_tum, self.vols[NODE])

        # normalize to baseline B cells; dividing by the live count blows up once B cells are gone
        per_bc, per_tum = self.binding.trimers_per_cell(y[BIND], self.bc_base, reach)

        pk = y[PK]
        ab = [pk[0], pk[3], pk[4], pk[5]]
        kin_tc = lp['kin_TC_cells'] * lym[8] ** lp['r']
        kin_bc = lp['kin_BC_cells'] * lym[9] ** lp['r']

        d_bind, uptake = self.binding.rhs(
            y[BIND], ab, tc, bc, reach, lp['kout_TC'], lp['kout_BC'], kill_bc, kill_tum,
            boost, kin_tc, kin_bc, self.k_traffic)

        d_pk = self.pk.derivatives(pk, t, bound=uptake)

        d_lym = self.lymph.derivatives(lym, t, kill=kill_bc * bc, patc_blood=p_atc[0])
        d_act = self.act.derivatives(act, t, per_bc, per_tum, bc, reach, self.k_traffic, boost,
                                     t_gate=self.gate)
        d_tum = self.tumor.derivative(n_tum, v_tum, self.vols[NODE])

        return np.concatenate([d_pk, d_lym, d_bind, d_act, [d_tum]])

    def simulate(self, t_eval, doses, rtol=1e-6, atol=1e-9, max_seconds=None):
        """Integrate through the dosing schedule. doses: [(day, mg)]. Returns (len(t_eval), 60)"""
        t_eval = np.asarray(t_eval, dtype=float)
        t_end = t_eval[-1]
        self.deadline = time.time() + max_seconds if max_seconds else None
        by_day = {}
        for day, mg in doses:
            by_day[day] = by_day.get(day, 0.0) + mg_to_nmol(mg)

        # restart the solver at every dose and at TAD / Tp, those are hard switches
        p = self.act.parameters
        cuts = sorted(c for c in {0.0, t_end, p['TAD'], p['Tp'], *by_day} if 0 <= c <= t_end)

        y = self.initial_state()
        out = np.empty((len(t_eval), len(y)))
        for a, b in zip(cuts[:-1], cuts[1:]):
            if a in by_day:
                y[DEPOT] += by_day[a]
                y[INJ] = 1.0
            self.gate = (a + b) / 2
            inside = t_eval[(t_eval > a) & (t_eval < b)]
            times = np.concatenate([[a], inside, [b]])
            sol = odeint(self.derivatives, y, times, rtol=rtol, atol=atol, mxstep=20000)
            keep = (t_eval >= a) & (t_eval < b)
            if b == t_end:
                keep = (t_eval >= a) & (t_eval <= b)
            idx = np.searchsorted(times, t_eval[keep])
            out[keep] = sol[idx]
            y = sol[-1].copy()
        return out

    def summarize(self, t, sol):
        """Pull the things the paper plots out of the state matrix"""
        lp = self.lymph.parameters
        reach = np.array([self.tumor.reachable(n) for n in sol[:, TUM]])
        nm = self.binding.parameters['nM_to_molecules']
        tri_tumor = sol[:, BIND][:, NODE * N_STATES + 7] * nm * self.vols[NODE] / np.maximum(reach, 1.0)

        tc_total = sol[:, 7] + sol[:, ACT][:, 4]
        return {
            't': t,
            'plasma_nM': sol[:, 0],
            'b_cell_pct': 100 * sol[:, 11] / lp['BC_blood_base'],
            't_cell_fold': tc_total / lp['TC_blood_base'],
            'tumor_pct': np.array([self.tumor.diameter_change(n) for n in sol[:, TUM]]),
            'trimers_per_tumor_cell': tri_tumor,
            'v_atc_tumor': sol[:, ACT][:, 8],
        }
