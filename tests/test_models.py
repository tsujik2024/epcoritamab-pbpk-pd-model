import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.binding_submodel import BindingSubmodel
from src.models.full_model import EpcoritamabModel, dose_schedule, mg_to_nmol
from src.models.lymphocyte_trafficking import LymphocyteTrafficking
from src.models.pk_submodel import PKSubmodel
from src.models.tumor_model import TumorKilling


class TestLymphocytes(unittest.TestCase):

    def test_baseline_split_matches_paper(self):
        ic = LymphocyteTrafficking().get_initial_conditions()
        tc = np.array(ic[:4])
        frac = tc / tc.sum()
        self.assertAlmostEqual(frac[0], 0.02, delta=0.003)  # 2% blood
        self.assertAlmostEqual(frac[1], 0.60, delta=0.01)  # 60% spleen

    def test_steady_state_holds(self):
        m = LymphocyteTrafficking()
        ic = m.get_initial_conditions()
        sol = m.simulate(ic, np.linspace(0, 28, 100), [])
        self.assertLess(abs(sol[-1, 0] / sol[0, 0] - 1), 0.01)
        self.assertLess(abs(sol[-1, 4] / sol[0, 4] - 1), 0.02)

    def test_injection_resets_not_stacks(self):
        m = LymphocyteTrafficking()
        sol = m.simulate(m.get_initial_conditions(), np.linspace(0, 2, 41), [0, 0.5, 1.0])
        self.assertLessEqual(sol[:, 10].max(), 1.0 + 1e-9)


class TestPK(unittest.TestCase):

    def test_dose_conversion(self):
        self.assertAlmostEqual(mg_to_nmol(48), 322, delta=5)

    def test_single_dose_is_sane(self):
        m = PKSubmodel()
        t = np.linspace(0, 28, 200)
        sol = m.simulate(m.get_initial_conditions(mg_to_nmol(48)), t)
        self.assertTrue(np.all(sol >= -1e-9))
        self.assertTrue(np.all(np.diff(sol[:, 6]) <= 1e-9))
        peak_day = t[sol[:, 0].argmax()]
        self.assertTrue(3 < peak_day < 8)


class TestBinding(unittest.TestCase):

    def test_no_drug_stays_at_baseline(self):
        lym = LymphocyteTrafficking()
        ic = lym.get_initial_conditions()
        b = BindingSubmodel(lymphocyte_params=lym.parameters)
        lp = lym.parameters
        y0 = b.get_initial_conditions(ic[:4], ic[4:8], 0.0)
        # production has to be on or blood CD3 slowly empties
        sol = b.simulate(y0, np.linspace(0, 5, 20), [0, 0, 0, 0], ic[:4], ic[4:8], 0.0,
                         lp['kout_TC'], lp['kout_BC'],
                         kin_tc=lp['kin_TC_cells'], kin_bc=lp['kin_BC_cells'],
                         k_traffic=lym.rates())
        self.assertLess(abs(sol[-1, 0] / sol[0, 0] - 1), 0.02)
        self.assertLess(abs(sol[-1, 2] / sol[0, 2] - 1), 0.02)


class TestTumor(unittest.TestCase):

    def test_geometry(self):
        t = TumorKilling('DLBCL')
        n0 = t.initial_cells()
        self.assertAlmostEqual(n0 / 1e10, 1.15, delta=0.02)
        self.assertLess(t.reachable(n0), 0.05 * n0)
        tiny = 1e3  # radius < depth (0.01 cm), everything is reachable
        self.assertAlmostEqual(t.reachable(tiny), tiny, delta=1)

    def test_grows_without_killing(self):
        t = TumorKilling('DLBCL')
        self.assertGreater(t.derivative(t.initial_cells(), 0.0, 0.0082), 0)


class TestFullModel(unittest.TestCase):

    def test_untreated_tumor_grows_lymphocytes_hold(self):
        m = EpcoritamabModel('DLBCL')
        t = np.linspace(0, 28, 29)
        sol = m.simulate(t, [])
        r = m.summarize(t, sol)
        self.assertGreater(r['tumor_pct'][-1], 30)
        self.assertLess(abs(r['t_cell_fold'][-1] - 1), 0.02)

    def test_step_up_dosing(self):
        m = EpcoritamabModel('DLBCL')
        t = np.linspace(0, 42, 169)
        sol = m.simulate(t, dose_schedule(48, days=42))
        r = m.summarize(t, sol)
        self.assertLess(r['t_cell_fold'][:8].min(), 0.5)  # injection dip on day 0-2
        self.assertLess(r['b_cell_pct'][t >= 10].max(), 10)  # B cells wiped out
        self.assertLess(r['tumor_pct'][-1], -30)
        self.assertTrue(np.all(np.isfinite(sol)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
