import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from src.models.lymphocyte_trafficking import LymphocyteTrafficking


def run_lymphocyte_demo():
    """Run a demonstration of the lymphocyte trafficking model"""

    # Initialize model
    model = LymphocyteTrafficking()

    # Set up simulation with weekly injections
    initial_conditions = model.get_initial_conditions()
    t_span = np.linspace(0, 28, 1000)  # 4 weeks
    injection_times = [0, 7, 14, 21]  # Weekly injections

    print("Running lymphocyte trafficking simulation...")
    solution = model.simulate(initial_conditions, t_span, injection_times)

    # Calculate concentrations
    TC_blood_conc, BC_blood_conc = model.calculate_concentrations(solution)

    # Extract key variables for plotting
    TC_blood = solution[:, 0]
    TC_spleen = solution[:, 1]
    BC_blood = solution[:, 4]
    BC_spleen = solution[:, 5]
    INJ = solution[:, 10]

    # Plot results
    plt.figure(figsize=(14, 10))

    # T-cell dynamics
    plt.subplot(2, 2, 1)
    plt.plot(t_span, TC_blood, label='Blood', linewidth=2)
    plt.plot(t_span, TC_spleen, label='Spleen', linewidth=2)
    plt.ylabel('T-cell Count')
    plt.title('T-cell Trafficking Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # B-cell dynamics
    plt.subplot(2, 2, 2)
    plt.plot(t_span, BC_blood, label='Blood', linewidth=2)
    plt.plot(t_span, BC_spleen, label='Spleen', linewidth=2)
    plt.ylabel('B-cell Count')
    plt.title('B-cell Trafficking Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Blood concentrations
    plt.subplot(2, 2, 3)
    plt.plot(t_span, TC_blood_conc, label='T-cells', linewidth=2)
    plt.plot(t_span, BC_blood_conc, label='B-cells', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration (cells/mm³)')
    plt.title('Blood Lymphocyte Concentrations')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Injection effect
    plt.subplot(2, 2, 4)
    plt.plot(t_span, INJ, label='INJ compartment', linewidth=2, color='red')
    for inj_time in injection_times:
        plt.axvline(x=inj_time, color='red', linestyle='--', alpha=0.5,
                    label='Injection' if inj_time == injection_times[0] else "")
    plt.xlabel('Time (days)')
    plt.ylabel('INJ Value')
    plt.title('Injection Effect Compartment')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/lymphocyte_trafficking.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print key metrics
    print("\n=== Lymphocyte Trafficking Results ===")
    print(f"Initial T-cell blood count: {initial_conditions[0]:.0f} cells")
    print(f"Final T-cell blood count: {TC_blood[-1]:.0f} cells")
    print(f"Initial B-cell blood count: {initial_conditions[4]:.0f} cells")
    print(f"Final B-cell blood count: {BC_blood[-1]:.0f} cells")
    print(f"Maximum injection effect: {np.max(INJ):.2f}")

    return solution, t_span


if __name__ == "__main__":
    run_lymphocyte_demo()