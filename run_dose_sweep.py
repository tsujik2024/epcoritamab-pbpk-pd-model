import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib

if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.models.full_model import EpcoritamabModel

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
DOSES = [0.0128, 0.04, 0.12, 0.38, 0.76, 1.5, 3, 6, 12, 24, 48, 96, 192, 384]


def run_one(args):
    kind, mg, kill_tumor = args
    tumor = None if kill_tumor else {'k_kill_tumor': 0.0}
    model = EpcoritamabModel(kind, tumor=tumor)
    t = np.linspace(0, 84, 337)
    sol = model.simulate(t, [(d, mg) for d in range(0, 84, 7)])
    r = model.summarize(t, sol)
    late = t >= 14
    return {
        'mg': mg,
        'cavg': r['plasma_nM'].mean(),
        'trimers': r['trimers_per_tumor_cell'][late].mean(),
        'tumor84': r['tumor_pct'][-1],
    }


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else 'DLBCL'
    os.makedirs(OUT, exist_ok=True)

    # weekly flat dosing for 12 weeks, like the paper's trimer simulations.
    # Second pass turns tumor killing off so you can see the binding curve on its own (hook effect)
    with Pool(2) as pool:
        with_kill = pool.map(run_one, [(kind, d, True) for d in DOSES])
        no_kill = pool.map(run_one, [(kind, d, False) for d in DOSES])

    print(f"{'dose mg':>8} {'Cavg nM':>9} {'trimers/tumor (no kill)':>24} {'tumor @84d %':>13}")
    for a, b in zip(with_kill, no_kill):
        print(f"{a['mg']:>8g} {a['cavg']:>9.3g} {b['trimers']:>24.1f} {a['tumor84']:>13.0f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    mg = [r['mg'] for r in with_kill]
    ax[0].semilogx(mg, [r['trimers'] for r in no_kill], 'o-')
    ax[0].set_xlabel('Weekly dose (mg)')
    ax[0].set_ylabel('Trimers per tumor cell (day 14-84 mean)')
    ax[0].set_title('Binding only (tumor killing off)')
    ax[1].semilogx(mg, [r['tumor84'] for r in with_kill], 'o-')
    ax[1].axhline(-30, color='gray', linestyle='--', linewidth=1)
    ax[1].set_xlabel('Weekly dose (mg)')
    ax[1].set_ylabel('Tumor diameter change at day 84 (%)')
    ax[1].set_title(f'{kind}, one typical patient')
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, f'dose_sweep_{kind}.png')
    fig.savefig(path, dpi=150)
    print('saved', path)


if __name__ == '__main__':
    main()
