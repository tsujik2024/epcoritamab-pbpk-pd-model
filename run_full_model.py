import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib

if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.models.full_model import EpcoritamabModel, dose_schedule

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')


def run(kind, full_mg, days):
    model = EpcoritamabModel(kind)
    t = np.linspace(0, days, days * 4 + 1)
    sol = model.simulate(t, dose_schedule(full_mg, days=days))
    return model.summarize(t, sol)


def main():
    full_mg = float(sys.argv[1]) if len(sys.argv) > 1 else 48.0
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 168
    os.makedirs(OUT, exist_ok=True)

    res = {kind: run(kind, full_mg, days) for kind in ('DLBCL', 'FL')}

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        ('plasma_nM', 'Plasma epcoritamab (nM)', True),
        ('b_cell_pct', 'Blood B cells (% of baseline)', False),
        ('t_cell_fold', 'Blood T cells (fold of baseline)', False),
        ('tumor_pct', 'Tumor diameter (% change)', False),
        ('trimers_per_tumor_cell', 'Trimers per tumor cell', False),
        ('v_atc_tumor', 'Tumor-directed ATCs (cells)', True),
    ]
    for a, (key, label, log) in zip(ax.ravel(), panels):
        for kind, r in res.items():
            a.plot(r['t'], r[key], label=kind, linewidth=1.8)
        a.set_xlabel('Day')
        a.set_ylabel(label)
        a.grid(alpha=0.3)
        if log:
            a.set_yscale('log')
    ax[0, 0].legend()
    ax[1, 0].axhline(-30, color='gray', linestyle='--', linewidth=1)
    fig.suptitle(f'Epcoritamab step-up dosing to {full_mg:g} mg, one typical patient')
    fig.tight_layout()
    path = os.path.join(OUT, f'full_model_{full_mg:g}mg.png')
    fig.savefig(path, dpi=150)
    print('saved', path)

    for kind, r in res.items():
        t = r['t']
        d84 = np.argmin(abs(t - 84))
        print(f"\n{kind}")
        print(f"  peak plasma:           {r['plasma_nM'].max():.1f} nM (day {t[r['plasma_nM'].argmax()]:.0f})")
        print(f"  B cells, day 14:       {r['b_cell_pct'][np.argmin(abs(t - 14))]:.1f}% of baseline")
        print(f"  lowest T cells:        {r['t_cell_fold'].min():.2f}x baseline")
        print(f"  tumor at day 28/84:    {r['tumor_pct'][np.argmin(abs(t - 28))]:.0f}% / {r['tumor_pct'][d84]:.0f}%")
        hit = np.nonzero(r['tumor_pct'] <= -30)[0]
        print(f"  first day <= -30%:     {t[hit[0]]:.0f}" if len(hit) else "  never reaches -30%")


if __name__ == '__main__':
    main()
