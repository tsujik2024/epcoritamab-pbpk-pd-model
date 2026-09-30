import numpy as np


def loop_flux(x, k, boost=1.0, vols=None):
    """Net change per compartment on the blood > spleen > node > lymph > blood loop.

    k = (k_pt, k_tn, k_nl, k_lp). If x is a concentration pass vols so the
    moles move, not the concentration.
    """
    a = np.asarray(x, dtype=float)
    if vols is not None:
        a = a * vols
    f0 = k[0] * boost * a[0]
    f1 = k[1] * a[1]
    f2 = k[2] * a[2]
    f3 = k[3] * a[3]
    d = np.array([f3 - f0, f0 - f1, f1 - f2, f2 - f3])
    return d if vols is None else d / vols
