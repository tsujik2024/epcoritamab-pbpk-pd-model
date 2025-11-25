import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from src.models.pk_submodel import PKSubmodel


def test_pk_model():
    """Test the PK submodel with a single SC dose"""

    # Initialize model
    pk_model = PKSubmodel()

    # Set up simulation - single 48 mg dose (approximately 320 nM)
    dose_nM = 320.0
    initial_conditions = pk_model.get_initial_conditions(dose=dose_nM)

    # Time points (28 days)
    t_span = np.linspace(0, 28, 1000)

    # Run simulation
    print("Running PK simulation...")
    solution = pk_model.simulate(initial_conditions, t_span)

    # Extract results
    C_plasma = solution[:, 0]
    C_leaky = solution[:, 1]
    C_tight = solution[:, 2]
    C_spleen = solution[:, 3]
    C_node = solution[:, 4]
    C_lymph = solution[:, 5]

    # Plot results
    plt.figure(figsize=(12, 10))

    # Main tissue concentrations
    plt.subplot(2, 2, 1)
    plt.plot(t_span, C_plasma, label='Plasma', linewidth=2)
    plt.plot(t_span, C_leaky, label='Leaky Tissue', linewidth=2)
    plt.plot(t_span, C_tight, label='Tight Tissue', linewidth=2)
    plt.ylabel('Concentration (nM)')
    plt.title('PK Submodel - Major Tissue Compartments')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Lymphoid tissue concentrations
    plt.subplot(2, 2, 2)
    plt.plot(t_span, C_spleen, label='Spleen', linewidth=2)
    plt.plot(t_span, C_node, label='Lymph Node', linewidth=2)
    plt.plot(t_span, C_lymph, label='Lymph', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration (nM)')
    plt.title('PK Submodel - Lymphoid Tissues')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plasma concentration semi-log
    plt.subplot(2, 2, 3)
    plt.semilogy(t_span, C_plasma, label='Plasma', linewidth=2, color='blue')
    plt.xlabel('Time (days)')
    plt.ylabel('Log Concentration (nM)')
    plt.title('Plasma Concentration (Semi-log)')
    plt.grid(True, alpha=0.3)

    # Total drug in system
    plt.subplot(2, 2, 4)
    total_drug = (C_plasma * pk_model.parameters['V_plasma'] +
                  C_leaky * pk_model.parameters['V_leaky'] +
                  C_tight * pk_model.parameters['V_tight'] +
                  C_spleen * pk_model.parameters['V_spleen'] +
                  C_node * pk_model.parameters['V_node'] +
                  C_lymph * pk_model.parameters['V_lymph'])
    plt.plot(t_span, total_drug, label='Total Drug', linewidth=2, color='red')
    plt.xlabel('Time (days)')
    plt.ylabel('Total Drug (nmol)')
    plt.title('Total Drug in System')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/pk_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print key metrics
    print("\n=== PK Simulation Results ===")
    print(f"Maximum plasma concentration: {np.max(C_plasma):.2f} nM")
    print(f"Time to max concentration: {t_span[np.argmax(C_plasma)]:.2f} days")
    print(f"Plasma concentration at day 7: {C_plasma[t_span >= 7][0]:.2f} nM")
    print(f"Plasma concentration at day 14: {C_plasma[t_span >= 14][0]:.2f} nM")
    print(f"Plasma concentration at day 21: {C_plasma[t_span >= 21][0]:.2f} nM")

    return solution, t_span


if __name__ == "__main__":
    test_pk_model()