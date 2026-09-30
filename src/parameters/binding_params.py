"""Epcoritamab binding parameters (Table S1)"""


def get_binding_default_parameters():
    return {
        'kon_CD3': 18.1,  # L/nmol/d
        'koff_CD3': 285.0,  # 1/d
        'kon_CD20': 4.15,
        'koff_CD20': 22.5,

        'kdeg_CD3': 1.584,  # 1/d
        'kdeg_CD20': 1.584,
        'kint_CD3': 1.584,
        'kint_CD20': 1.584,

        'RCD3': 30000,  # receptors/cell
        'RCD20': 100000,
        'RCD20_tumor': 30000,

        'nM_to_molecules': 6.022e14,  # molecules per (nM * L)

        'V_blood': 4.73,  # L
        'V_spleen_tissue': 0.221,
        'V_node': 0.0082,  # from PK model
        'V_lymph': 5.2,
        'mm3_to_L': 1e6,
    }


def calculate_binding_derived_parameters(base, lymphocyte_params):
    p = base.copy()
    for k in ('V_blood', 'V_spleen_tissue', 'mm3_to_L'):
        if k in lymphocyte_params:
            p[k] = lymphocyte_params[k]
    return p
