from typing import Dict


def get_tcell_activation_default_parameters() -> Dict:
    """Get default parameters for T-cell activation from Table S1"""
    return {
        # Activation rates
        'sim_slope': 0.007,  # (10^6 cells/day) × (cell/molecule) - activation vs B-cells
        'sim_slope_tumor': 7e-6,  # (10^6 cells/day) × (cell/molecule) - activation vs tumor cells

        # Activation threshold
        'Trimer_Threshold': 24.0,  # molecules/cell - threshold for full activation

        # Clonal expansion
        'expand_factor': 9.27,  # day^-1 - clonal expansion rate

        # Activated T-cell turnover
        'kout_ATC': 0.05,  # day^-1 - activated T-cell death rate

        # Lag times
        'TAD': 3.0,  # days - T-cell activation delay
        'Tp': 10.8,  # days - proliferation duration

        # Conversion factors
        'million_cells': 1e6,  # Conversion for sim_slope parameters
    }