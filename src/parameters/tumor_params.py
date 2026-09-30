"""Tumor growth / killing and B-cell killing parameters (Table S1)"""

import numpy as np


def get_tumor_default_parameters(kind='DLBCL'):
    growth = {'DLBCL': 0.0301, 'FL': 0.0038}
    if kind not in growth:
        raise ValueError(f"tumor kind must be DLBCL or FL, got {kind}")

    return {
        'k_growth': growth[kind],  # 1/d
        'r0': 1.40,  # cm, baseline radius
        'capacity': 1e-6,  # 1/(1e6 cells)
        'depth': 0.01,  # cm, how far in T cells can reach
        'cells_per_ml': 1e9,
        'k_kill_tumor': 0.165,  # L^2/(1e6 cells)/d
        'k_kill_bc': 0.544,  # same units
    }


def initial_tumor_cells(p):
    return 4 / 3 * np.pi * p['r0'] ** 3 * p['cells_per_ml']
