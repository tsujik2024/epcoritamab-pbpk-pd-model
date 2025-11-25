import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from src.models.binding_submodel import BindingSubmodel


def run_binding_demo():
    """Run a demonstration of the binding submodel"""

    # Initialize model with lymphocyte parameters
    lymphocyte_params = {
        'V_blood': 4.73,
        'V_spleen_tissue': 0.221,
        'mm3_to_L': 1e6
    }

    binding_model = BindingSubmodel(lymphocyte_params=lymphocyte_params)

    # Set up cell counts (using realistic numbers from lymphocyte model)
    print("Setting up cell counts...")
    TC_counts = [6.95e9, 8.69e10, 5.21e9, 2.02e11]  # From lymphocyte model debug output
    BC_counts = [1.12e9, 1.67e10, 1.12e9, 2.68e10]  # From lymphocyte model debug output
    tumor_cell_count = 1e9  # 1 billion tumor cells

    # Get initial conditions
    initial_conditions = binding_model.get_initial_conditions(
        TC_counts, BC_counts, tumor_cell_count
    )

    # Set up antibody concentrations (simulate SC administration with distribution)
    print("Setting up antibody concentrations...")
    t_span = np.linspace(0, 14, 1000)  # 2 weeks

    # Simulate antibody PK (simplified - in real use, this would come from PK model)
    # Peak around day 2-3, then slow decline
    Ab_blood = 50.0 * np.exp(-0.2 * t_span) * (1 - np.exp(-1.5 * t_span))  # nM
    Ab_spleen = 30.0 * np.exp(-0.15 * t_span) * (1 - np.exp(-1.2 * t_span))  # nM
    Ab_node = 20.0 * np.exp(-0.1 * t_span) * (1 - np.exp(-1.0 * t_span))  # nM
    Ab_lymph = 10.0 * np.exp(-0.05 * t_span) * (1 - np.exp(-0.8 * t_span))  # nM

    Ab_concentrations = np.column_stack([Ab_blood, Ab_spleen, Ab_node, Ab_lymph])

    # Cell death rates from lymphocyte model
    kout_TC = 0.00422
    kout_BC = 0.0189

    print("Running binding simulation...")
    # Run simulation - we need to simulate each time point separately due to changing Ab concentrations
    solution = np.zeros((len(t_span), len(initial_conditions)))
    solution[0] = initial_conditions

    for i in range(1, len(t_span)):
        # Use the current antibody concentrations
        current_Ab = Ab_concentrations[i]

        # Simulate one time step
        t_step = [t_span[i - 1], t_span[i]]
        step_solution = binding_model.simulate(
            solution[i - 1], t_step, current_Ab,
            TC_counts, BC_counts, tumor_cell_count, kout_TC, kout_BC
        )
        solution[i] = step_solution[1]  # Take the end point

    print("Simulation completed!")

    # Analyze results
    print("\n=== Binding Simulation Results ===")

    # Extract key results from blood compartment
    CD3_free_blood = solution[:, 0]
    CD3_Ab_blood = solution[:, 1]  # CD3-epcoritamab dimers
    CD20_free_blood = solution[:, 2]
    CD20_Ab_blood = solution[:, 3]  # CD20-epcoritamab dimers
    trimer_blood = solution[:, 4]  # CD3-epcoritamab-CD20 trimers

    # Extract from lymph node compartment (where tumor binding occurs)
    lymph_node_start_idx = 2 * 8  # 3rd compartment (index 2) × 8 states
    CD20_free_tumor_node = solution[:, lymph_node_start_idx + 5]
    CD20_Ab_tumor_node = solution[:, lymph_node_start_idx + 6]
    trimer_tumor_node = solution[:, lymph_node_start_idx + 7]

    # Calculate trimer counts
    compartment_volumes = [4.73, 0.221, 0.0082, 5.2]  # V_blood, V_spleen, V_node, V_lymph
    trimer_counts = binding_model.calculate_trimer_counts(solution, compartment_volumes)

    # Plot results
    plt.figure(figsize=(16, 12))

    # Plot 1: Antibody concentrations
    plt.subplot(3, 3, 1)
    plt.plot(t_span, Ab_blood, label='Blood', linewidth=2)
    plt.plot(t_span, Ab_spleen, label='Spleen', linewidth=2)
    plt.plot(t_span, Ab_node, label='Lymph Node', linewidth=2)
    plt.plot(t_span, Ab_lymph, label='Lymph', linewidth=2)
    plt.ylabel('Antibody Concentration (nM)')
    plt.title('Epcoritamab Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: CD3 binding in blood
    plt.subplot(3, 3, 2)
    plt.plot(t_span, CD3_free_blood, label='Free CD3', linewidth=2)
    plt.plot(t_span, CD3_Ab_blood, label='CD3-Ab Dimer', linewidth=2)
    plt.ylabel('Concentration (nM)')
    plt.title('CD3 Binding in Blood')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: CD20 binding in blood
    plt.subplot(3, 3, 3)
    plt.plot(t_span, CD20_free_blood, label='Free CD20', linewidth=2)
    plt.plot(t_span, CD20_Ab_blood, label='CD20-Ab Dimer', linewidth=2)
    plt.ylabel('Concentration (nM)')
    plt.title('CD20 Binding in Blood')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Trimer formation in blood
    plt.subplot(3, 3, 4)
    plt.plot(t_span, trimer_blood, label='Blood Trimers', linewidth=2, color='red')
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration (nM)')
    plt.title('Trimer Formation in Blood\n(T-cell - B-cell crosslinking)')
    plt.grid(True, alpha=0.3)

    # Plot 5: Tumor binding in lymph node
    plt.subplot(3, 3, 5)
    plt.plot(t_span, CD20_free_tumor_node, label='Free CD20 (Tumor)', linewidth=2)
    plt.plot(t_span, CD20_Ab_tumor_node, label='CD20-Ab (Tumor)', linewidth=2)
    plt.plot(t_span, trimer_tumor_node, label='Tumor Trimers', linewidth=2, color='purple')
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration (nM)')
    plt.title('Tumor Cell Binding in Lymph Node')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Total trimer counts by compartment
    plt.subplot(3, 3, 6)
    compartments = ['Blood', 'Spleen', 'Lymph Node', 'Lymph']
    colors = ['blue', 'green', 'red', 'orange']

    for i, (comp, color) in enumerate(zip(compartments, colors)):
        plt.plot(t_span, trimer_counts[:, i], label=comp, linewidth=2, color=color)

    plt.xlabel('Time (days)')
    plt.ylabel('Trimer Molecules')
    plt.title('Total Trimer Counts by Compartment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 7: Binding kinetics overview
    plt.subplot(3, 3, 7)
    peak_time_idx = np.argmax(Ab_blood)
    peak_time = t_span[peak_time_idx]

    # Show concentrations at peak antibody time
    states_at_peak = {
        'Free CD3': CD3_free_blood[peak_time_idx],
        'CD3-Ab Dimer': CD3_Ab_blood[peak_time_idx],
        'Free CD20': CD20_free_blood[peak_time_idx],
        'CD20-Ab Dimer': CD20_Ab_blood[peak_time_idx],
        'Trimers': trimer_blood[peak_time_idx]
    }

    bars = plt.bar(states_at_peak.keys(), states_at_peak.values(),
                   color=['lightblue', 'blue', 'lightgreen', 'green', 'red'])
    plt.ylabel('Concentration (nM)')
    plt.title(f'Binding States at Peak Antibody (Day {peak_time:.1f})')
    plt.xticks(rotation=45)

    # Add values on bars
    for bar, value in zip(bars, states_at_peak.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 8: Receptor occupancy
    plt.subplot(3, 3, 8)
    CD3_occupancy = (CD3_Ab_blood / (CD3_free_blood + CD3_Ab_blood + 1e-10)) * 100
    CD20_occupancy = (CD20_Ab_blood / (CD20_free_blood + CD20_Ab_blood + 1e-10)) * 100

    plt.plot(t_span, CD3_occupancy, label='CD3 Occupancy', linewidth=2, color='blue')
    plt.plot(t_span, CD20_occupancy, label='CD20 Occupancy', linewidth=2, color='green')
    plt.xlabel('Time (days)')
    plt.ylabel('Receptor Occupancy (%)')
    plt.title('Receptor Occupancy in Blood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    # Plot 9: Trimer-to-cell ratios (approximate)
    plt.subplot(3, 3, 9)
    # Convert trimer concentrations to approximate molecules per cell
    trimer_molecules_blood = trimer_blood * binding_model.parameters['nM_to_molecules'] * compartment_volumes[0] * 1e-9
    TC_count_blood = TC_counts[0]
    BC_count_blood = BC_counts[0]

    trimers_per_TC = trimer_molecules_blood / TC_count_blood
    trimers_per_BC = trimer_molecules_blood / BC_count_blood

    plt.plot(t_span, trimers_per_TC, label='Trimers per T-cell', linewidth=2, color='blue')
    plt.plot(t_span, trimers_per_BC, label='Trimers per B-cell', linewidth=2, color='green')
    plt.xlabel('Time (days)')
    plt.ylabel('Trimers per Cell')
    plt.title('Trimer-to-Cell Ratios in Blood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('../results/binding_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print key metrics
    print(f"\n=== Key Binding Metrics ===")
    print(f"Maximum antibody concentration: {np.max(Ab_blood):.2f} nM")
    print(f"Time to peak antibody: {t_span[np.argmax(Ab_blood)]:.2f} days")
    print(f"Maximum CD3-Ab dimers: {np.max(CD3_Ab_blood):.3f} nM")
    print(f"Maximum CD20-Ab dimers: {np.max(CD20_Ab_blood):.3f} nM")
    print(f"Maximum blood trimers: {np.max(trimer_blood):.4f} nM")
    print(f"Maximum tumor trimers: {np.max(trimer_tumor_node):.4f} nM")

    print(f"\n=== At Peak Antibody (Day {peak_time:.1f}) ===")
    print(f"CD3 occupancy: {CD3_occupancy[peak_time_idx]:.1f}%")
    print(f"CD20 occupancy: {CD20_occupancy[peak_time_idx]:.1f}%")
    print(f"Trimers per T-cell: {trimers_per_TC[peak_time_idx]:.1f}")
    print(f"Trimers per B-cell: {trimers_per_BC[peak_time_idx]:.1f}")

    print(f"\n=== Total Trimer Formation ===")
    total_trimers_end = np.sum(trimer_counts[-1, :])
    print(f"Total trimers at day 14: {total_trimers_end:.2e} molecules")

    return solution, t_span, trimer_counts


def run_simple_binding_test():
    """Run a simpler test case for quick verification"""
    print("\n" + "=" * 50)
    print("RUNNING SIMPLE BINDING TEST")
    print("=" * 50)

    lymphocyte_params = {
        'V_blood': 4.73,
        'V_spleen_tissue': 0.221,
        'mm3_to_L': 1e6
    }

    binding_model = BindingSubmodel(lymphocyte_params=lymphocyte_params)

    # Simple case: only blood compartment with cells
    TC_counts = [1e9, 0, 0, 0]  # 1 billion T-cells in blood only
    BC_counts = [1e8, 0, 0, 0]  # 100 million B-cells in blood only
    tumor_cell_count = 0

    initial_conditions = binding_model.get_initial_conditions(
        TC_counts, BC_counts, tumor_cell_count
    )

    # Short simulation with constant antibody
    t_span = np.linspace(0, 3, 100)
    Ab_concentrations = [1.0, 0, 0, 0]  # 1 nM in blood only

    solution = binding_model.simulate(
        initial_conditions, t_span, Ab_concentrations,
        TC_counts, BC_counts, tumor_cell_count, 0.00422, 0.0189
    )

    # Quick plot of results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t_span, solution[:, 0], label='Free CD3', linewidth=2)
    plt.plot(t_span, solution[:, 1], label='CD3-Ab Dimer', linewidth=2)
    plt.ylabel('Concentration (nM)')
    plt.title('CD3 Binding Kinetics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(t_span, solution[:, 2], label='Free CD20', linewidth=2)
    plt.plot(t_span, solution[:, 3], label='CD20-Ab Dimer', linewidth=2)
    plt.ylabel('Concentration (nM)')
    plt.title('CD20 Binding Kinetics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(t_span, solution[:, 4], label='Trimers', linewidth=2, color='red')
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration (nM)')
    plt.title('Trimer Formation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    CD3_bound = solution[:, 1] / (solution[:, 0] + solution[:, 1] + 1e-10) * 100
    CD20_bound = solution[:, 3] / (solution[:, 2] + solution[:, 3] + 1e-10) * 100
    plt.plot(t_span, CD3_bound, label='CD3 Bound', linewidth=2)
    plt.plot(t_span, CD20_bound, label='CD20 Bound', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Receptor Bound (%)')
    plt.title('Receptor Occupancy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/simple_binding_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Simple test completed!")
    print(f"Final CD3 occupancy: {CD3_bound[-1]:.1f}%")
    print(f"Final CD20 occupancy: {CD20_bound[-1]:.1f}%")
    print(f"Final trimer concentration: {solution[-1, 4]:.6f} nM")


if __name__ == "__main__":
    # Run the full demo
    solution, t_span, trimer_counts = run_binding_demo()

    # Also run the simple test
    run_simple_binding_test()