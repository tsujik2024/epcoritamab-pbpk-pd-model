def get_tcell_activation_default_parameters():
    """Table S1"""
    return {
        'sim_slope': 0.007,  # 1e6 cells/day per trimer/cell, vs B-cells
        'sim_slope_tumor': 7e-6,  # same, vs tumor cells
        'Trimer_Threshold': 24.0,  # molecules/cell
        'expand_factor': 9.27,  # 1/day
        'kout_ATC': 0.05,  # 1/day
        'TAD': 3.0,  # days, activation delay
        'Tp': 10.8,  # days, proliferation duration
        'million_cells': 1e6,
    }
