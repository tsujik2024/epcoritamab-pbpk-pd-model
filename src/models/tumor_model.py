"""Tumor growth and T-cell killing (spherical lesion, only the outer shell is reachable)"""

import numpy as np

from src.parameters.tumor_params import get_tumor_default_parameters, initial_tumor_cells


class TumorKilling:

    def __init__(self, kind='DLBCL', parameters=None):
        self.parameters = {**get_tumor_default_parameters(kind), **(parameters or {})}

    def initial_cells(self):
        return initial_tumor_cells(self.parameters)

    def radius(self, cells):
        p = self.parameters
        vol = max(cells, 0.0) / p['cells_per_ml']
        return (3 / (4 * np.pi) * vol) ** (1 / 3)

    def reachable(self, cells):
        p = self.parameters
        vol = max(cells, 0.0) / p['cells_per_ml']
        r = self.radius(cells)
        core = 4 / 3 * np.pi * max(r - p['depth'], 0.0) ** 3
        return (vol - core) * p['cells_per_ml']

    def growth(self, cells):
        p = self.parameters
        # capacity is per 1e6 cells
        return p['k_growth'] * cells * (1 - p['capacity'] * cells / 1e6)

    def kill_rate(self, v_atc_tumor, v_node):
        """Per reachable cell, 1/d. Concentrations are 1e6 cells/L."""
        return self.parameters['k_kill_tumor'] * (v_atc_tumor / 1e6) / v_node ** 2

    def bc_kill_rate(self, v_atc_bc, vols):
        """Per B cell, 1/d, in each compartment"""
        return self.parameters['k_kill_bc'] * (np.asarray(v_atc_bc) / 1e6) / np.asarray(vols) ** 2

    def derivative(self, cells, v_atc_tumor, v_node):
        kill = self.kill_rate(v_atc_tumor, v_node) * self.reachable(cells)
        return self.growth(cells) - kill

    def diameter_change(self, cells):
        """% change in longest diameter vs baseline"""
        r0 = self.parameters['r0']
        return (self.radius(cells) / r0 - 1) * 100
