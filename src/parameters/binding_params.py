"""Parameters for epcoritamab-binding submodel"""

from typing import Dict


def get_binding_default_parameters() -> Dict:
    """Get default parameters for binding submodel from Table S1"""
    return {
        # Binding kinetics
        'kon_CD3': 18.1,  # L/nmol/d (association rate to CD3)
        'koff_CD3': 285.0,  # day^-1 (dissociation rate from CD3)
        'kon_CD20': 4.15,  # L/nmol/d (association rate to CD20)
        'koff_CD20': 22.5,  # day^-1 (dissociation rate from CD20)

        # Degradation and internalization
        'kdeg_CD3': 1.584,  # day^-1 (CD3 degradation rate)
        'kdeg_CD20': 1.584,  # day^-1 (CD20 degradation rate)
        'kint_CD3': 1.584,  # day^-1 (CD3-Ab internalization rate)
        'kint_CD20': 1.584,  # day^-1 (CD20-Ab internalization rate)

        # Receptor expression levels
        'RCD3': 30000,  # receptors/cell (CD3 on T-cells)
        'RCD20': 100000,  # molecules/cell (CD20 on B-cells)
        'RCD20_tumor': 30000,  # molecules/cell (CD20 on tumor cells)

        # Conversion factors
        'nM_to_molecules': 6.022e14,  # Convert nM to molecules (Avogadro * 1e-9 L)
        'L_to_nL': 1e9,  # Convert L to nL
    }


def calculate_binding_derived_parameters(base_params: Dict, lymphocyte_params: Dict) -> Dict:
    """Calculate derived parameters for binding submodel"""
    params = base_params.copy()

    # Add lymphocyte parameters needed for calculations
    params.update({
        'V_blood': lymphocyte_params.get('V_blood', 4.73),
        'V_spleen_tissue': lymphocyte_params.get('V_spleen_tissue', 0.221),
        'V_node': 0.0082,  # From PK model
        'V_lymph': 5.2,  # From PK model
        'mm3_to_L': lymphocyte_params.get('mm3_to_L', 1e6),
    })

    return params