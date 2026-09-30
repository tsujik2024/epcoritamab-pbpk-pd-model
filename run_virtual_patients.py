import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib

if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.models.full_model import EpcoritamabModel, dose_schedule

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')

# Table S2, CV as a fraction
CV = {
    'k_a': 0.95, 'CL': 0.771, 'V_max': 1.31, 'K_m': 1.40,
    'kin_TC': 0.447, 'kin_BC': 0.464,
    'kout_ATC': 0.224, 'sim_slope': 0.579, 'expand_factor': 0.486,
    'k_kill_bc': 0.5, 'k_kill_tumor': 0.5, 'k_decay': 0.5, 'INJ_Scaler': 0.5,
    'Tp': 0.20, 'k_growth': 0.459, 'r0': 0.587,
    'RCD3': 0.447, 'RCD20': 0.447, 'RCD20_tumor': 1.0,
}
DAYS = 168


def lognormal(rng, cv):
    # median = typical value, so this returns a multiplier around 1
    s = np.sqrt(np.log(1 + cv ** 2))
    return float(np.exp(rng.normal(0, s)))


def draw_patient(seed):
    rng = np.random.default_rng(seed)
    f = {k: lognormal(rng, cv) for k, cv in CV.items()}
    base = EpcoritamabModel('DLBCL')  # just for the nominal numbers
    lp = base.lymph.parameters

    # baseline counts and production are tied (kin comes from the baseline count),
    # so scale them together or the patient drifts away from steady state
    lymph = {
        'TC_plasma_base': lp['TC_plasma_base'] * f['kin_TC'],
        'kin_TC': lp['kin_TC'] * f['kin_TC'],
        'BC_plasma_base': lp['BC_plasma_base'] * f['kin_BC'],
        'kin_BC': lp['kin_BC'] * f['kin_BC'],
        'k_decay': lp['k_decay'] * f['k_decay'],
        'INJ_Scaler': lp['INJ_Scaler'] * f['INJ_Scaler'],
    }
    pk = {k: base.pk.parameters[k] * f[k] for k in ('k_a', 'CL', 'V_max', 'K_m')}
    bp = base.binding.parameters
    binding = {k: bp[k] * f[k] for k in ('RCD3', 'RCD20', 'RCD20_tumor')}
    ap = base.act.parameters
    activation = {
        'kout_ATC': ap['kout_ATC'] * f['kout_ATC'],
        'expand_factor': ap['expand_factor'] * f['expand_factor'],
        'Tp': ap['Tp'] * f['Tp'],
        'sim_slope': ap['sim_slope'] * f['sim_slope'],
        'sim_slope_tumor': ap['sim_slope_tumor'] * f['sim_slope'],  # same draw, Table S2 says associated
    }
    tp = base.tumor.parameters
    tumor = {
        'k_growth': tp['k_growth'] * f['k_growth'],
        'r0': tp['r0'] * f['r0'],
        'k_kill_bc': tp['k_kill_bc'] * f['k_kill_bc'],
        'k_kill_tumor': tp['k_kill_tumor'] * f['k_kill_tumor'],
    }
    return dict(pk=pk, lymph=lymph, binding=binding, activation=activation, tumor=tumor)


def run_patient(args):
    seed, kind, full_mg = args
    try:
        model = EpcoritamabModel(kind, **draw_patient(seed))
        t = np.linspace(0, DAYS, DAYS + 1)
        sol = model.simulate(t, dose_schedule(full_mg, days=DAYS), max_seconds=90)
        return model.summarize(t, sol)
    except (TimeoutError, RuntimeError) as e:  # a few extreme draws wedge the solver, skip them
        print(f'patient {seed} failed: {e}')
        return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    kind = sys.argv[2] if len(sys.argv) > 2 else 'DLBCL'
    full_mg = float(sys.argv[3]) if len(sys.argv) > 3 else 48.0
    os.makedirs(OUT, exist_ok=True)

    with Pool(2) as pool:
        runs = [r for r in pool.map(run_patient, [(s, kind, full_mg) for s in range(n)]) if r]
    if not runs:
        sys.exit('every patient failed')

    t = runs[0]['t']
    tumor = np.array([r['tumor_pct'] for r in runs])
    responders = (tumor[:, 84] <= -30).mean() * 100
    print(f"{len(runs)}/{n} patients ran, {kind}, {full_mg:g} mg")
    print(f"tumor change at day 84: median {np.median(tumor[:, 84]):.0f}%, "
          f"5-95%: {np.percentile(tumor[:, 84], 5):.0f}% to {np.percentile(tumor[:, 84], 95):.0f}%")
    print(f"patients with >=30% shrinkage at day 84: {responders:.0f}%  (paper's simulated ORR is ~85% for DLBCL at 48 mg)")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for a, key, label in [(ax[0, 0], 'b_cell_pct', 'Blood B cells (% of baseline)'),
                          (ax[0, 1], 't_cell_fold', 'Blood T cells (fold of baseline)'),
                          (ax[1, 0], 'tumor_pct', 'Tumor diameter (% change)'),
                          (ax[1, 1], 'plasma_nM', 'Plasma epcoritamab (nM)')]:
        y = np.array([r[key] for r in runs])
        lo, mid, hi = np.percentile(y, [5, 50, 95], axis=0)
        a.fill_between(t, lo, hi, alpha=0.25)
        a.plot(t, mid, linewidth=2)
        a.set_xlabel('Day')
        a.set_ylabel(label)
        a.grid(alpha=0.3)
    ax[1, 1].set_yscale('log')
    fig.suptitle(f'{len(runs)} virtual {kind} patients, {full_mg:g} mg step-up (median and 5-95%)')
    fig.tight_layout()
    path = os.path.join(OUT, f'virtual_patients_{kind}.png')
    fig.savefig(path, dpi=150)
    print('saved', path)


if __name__ == '__main__':
    main()
