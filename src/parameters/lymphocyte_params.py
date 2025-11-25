"""Parameters for lymphocyte trafficking and turnover submodel"""

from typing import Dict


def get_lymphocyte_default_parameters() -> Dict:
    """Get default parameters for lymphocyte trafficking from Table S1"""
    return {
        # Volumes
        'V_blood': 4.73,  # L (calculated from V_plasma, assuming plasma is 55% of blood)
        'V_spleen_tissue': 0.221,  # L

        # Baseline cell counts (cells/mm^3)
        'TC_plasma_base': 1469,  # Baseline T-cell count in blood (cells/mm^3)
        'BC_plasma_base': 236,  # Baseline B-cell count in blood (cells/mm^3)

        # Production rates (10^6 cells/d)
        'kin_TC': 310,  # Production rate of T cells (10^6 cells/day)
        'kin_BC': 223,  # Production rate of B cells (10^6 cells/day)

        # Natural death rates (day^-1)
        'kout_TC': 0.00422,  # Natural death rate of T cells
        'kout_BC': 0.0189,  # Natural death rate of B cells

        # Trafficking rates (day^-1)
        'k_pt': 50.0,  # Blood/plasma to spleen tissue
        'k_tn': 1.67,  # Spleen tissue to lymph node
        'k_nl': 1672.0,  # Lymph node to lymph
        'k_lp': 2.63,  # Lymph to blood/plasma

        # Injection effect parameters
        'k_decay': 0.681,  # Duration of injection effect (day^-1)
        'INJ_Scaler': 9.08,  # Magnitude of injection effect

        # Homeostasis parameters
        'k_t': 0.00027,  # Lymphocyte homeostasis control rate
        'r': 2.24,  # Strength of lymphocyte homeostasis control

        # Conversion factors - FIXED THESE!
        'mm3_to_L': 1e6,  # 1e6 mm^3 in 1 L (NOT 1e-6!)
        'million_cells': 1e6,  # Conversion for kin parameters
    }


def calculate_derived_parameters(base_params: Dict) -> Dict:
    """Calculate derived parameters based on base parameters"""
    params = base_params.copy()

    # Convert baseline counts from cells/mm^3 to total cells in blood compartment
    # cells/mm^3 * (mm^3/L) * L = total cells
    params['TC_blood_base'] = params['TC_plasma_base'] * params['mm3_to_L'] * params['V_blood']
    params['BC_blood_base'] = params['BC_plasma_base'] * params['mm3_to_L'] * params['V_blood']

    # Convert kin from 10^6 cells/day to cells/day
    params['kin_TC_cells'] = params['kin_TC'] * params['million_cells']
    params['kin_BC_cells'] = params['kin_BC'] * params['million_cells']

    # FIXED: Calculate total body lymphocytes correctly
    # Blood contains 2% of total lymphocytes, so total = blood_count / 0.02
    # But we need to be careful about units and scaling

    # Let's use more realistic distributions based on literature:
    # Blood: 2%, Spleen: 25-30%, Lymph nodes: 15-20%, Lymph: rest

    # For T-cells (more in blood and lymphoid organs)
    total_TC_body = params['TC_blood_base'] / 0.02  # 2% in blood

    # Distribute T-cells
    params['TC_spleen_base'] = total_TC_body * 0.25  # 25% in spleen (more realistic)
    params['TC_node_base'] = total_TC_body * 0.15  # 15% in lymph nodes
    params['TC_lymph_base'] = total_TC_body * 0.58  # 58% in lymph (remaining)

    # For B-cells (more in spleen and lymph nodes)
    total_BC_body = params['BC_blood_base'] / 0.02  # 2% in blood

    # Distribute B-cells
    params['BC_spleen_base'] = total_BC_body * 0.30  # 30% in spleen
    params['BC_node_base'] = total_BC_body * 0.20  # 20% in lymph nodes
    params['BC_lymph_base'] = total_BC_body * 0.48  # 48% in lymph (remaining)

    # Print debug info
    print(f"\n=== DEBUG: Cell Counts (in millions) ===")
    print(f"TC blood: {params['TC_blood_base'] / 1e6:.1f} million")
    print(f"TC spleen: {params['TC_spleen_base'] / 1e6:.1f} million")
    print(f"TC node: {params['TC_node_base'] / 1e6:.1f} million")
    print(f"TC lymph: {params['TC_lymph_base'] / 1e6:.1f} million")
    print(f"BC blood: {params['BC_blood_base'] / 1e6:.1f} million")
    print(f"BC spleen: {params['BC_spleen_base'] / 1e6:.1f} million")
    print(f"BC node: {params['BC_node_base'] / 1e6:.1f} million")
    print(f"BC lymph: {params['BC_lymph_base'] / 1e6:.1f} million")

    # Verify distributions sum to approximately 100%
    TC_total_calc = (params['TC_blood_base'] + params['TC_spleen_base'] +
                     params['TC_node_base'] + params['TC_lymph_base'])
    TC_blood_percent = params['TC_blood_base'] / TC_total_calc * 100

    print(f"\nVerification: TC blood % = {TC_blood_percent:.1f}% (should be ~2%)")

    return params