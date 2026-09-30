"""Lymphocyte trafficking parameters (Table S1)."""


def get_lymphocyte_default_parameters():
    return {
        'V_blood': 4.73,  # L
        'V_spleen_tissue': 0.221,  # L

        'TC_plasma_base': 1469,  # cells/mm^3
        'BC_plasma_base': 236,

        # 1e6 cells/day/L blood. kin * V_blood == kout * total cells at baseline,
        # so it has to be scaled by V_blood or the whole pool decays.
        'kin_TC': 310,
        'kin_BC': 223,

        'kout_TC': 0.00422,  # 1/day
        'kout_BC': 0.0189,

        'k_pt': 50.0,  # blood -> spleen
        'k_tn': 1.67,  # spleen -> node
        'k_nl': 1672.0,  # node -> lymph
        'k_lp': 2.63,  # lymph -> blood

        'k_decay': 0.681,
        'INJ_Scaler': 9.08,

        'k_t': 0.00027,
        'r': 2.24,

        'mm3_to_L': 1e6,  # mm^3 per L
        'million_cells': 1e6,
    }


def calculate_derived_parameters(base):
    p = base.copy()
    scale = p['mm3_to_L'] * p['V_blood']

    for cell, kin in (('TC', 'kin_TC'), ('BC', 'kin_BC')):
        blood = p[f'{cell}_plasma_base'] * scale
        kout = p[f'kout_{cell}']

        # Steady state going down the chain, so the trafficking rates and the
        # starting distribution agree (a fixed % split doesn't).
        spleen = p['k_pt'] * blood / (p['k_tn'] + kout)
        node = p['k_tn'] * spleen / (p['k_nl'] + kout)
        lymph = p['k_nl'] * node / (p['k_lp'] + kout)

        p[f'{cell}_blood_base'] = blood
        p[f'{cell}_spleen_base'] = spleen
        p[f'{cell}_node_base'] = node
        p[f'{cell}_lymph_base'] = lymph
        p[f'{kin}_cells'] = p[kin] * p['million_cells'] * p['V_blood']

    return p
