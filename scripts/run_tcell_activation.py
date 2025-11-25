import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from src.models.tcell_activation import TCellActivation


def run_activation_demo():
    """Run a demonstration of the T-cell activation submodel"""

    # Initialize model
    activation_model = TCellActivation()

    # Set up cell counts (using realistic numbers)
    print("Setting up cell counts...")
    TC_counts = [6.95e9, 8.69e10, 5.21e9, 2.02e11]  # From lymphocyte model
    BC_counts = [1.12e9, 1.67e10, 1.12e9, 2.68e10]  # From lymphocyte model
    tumor_cell_count = 1e9  # 1 billion tumor cells

    # Get initial conditions
    initial_conditions = activation_model.get_initial_conditions(
        TC_counts, BC_counts, tumor_cell_count
    )

    # Set up simulation
    print("Setting up activation simulation...")
    t_span = np.linspace(0, 28, 1000)  # 4 weeks

    # Simulate trimer dynamics (in real use, this would come from binding model)
    # Peak around day 7-14 after TAD
    p = activation_model.parameters
    trimers_blood = 5.0 * np.exp(-0.1 * (t_span - p['TAD'])) * (t_span >= p['TAD'])
    trimers_spleen = 10.0 * np.exp(-0.08 * (t_span - p['TAD'])) * (t_span >= p['TAD'])
    trimers_node = 20.0 * np.exp(-0.06 * (t_span - p['TAD'])) * (t_span >= p['TAD'])
    trimers_lymph = 2.0 * np.exp(-0.12 * (t_span - p['TAD'])) * (t_span >= p['TAD'])

    trimers_per_BC = np.column_stack([trimers_blood, trimers_spleen, trimers_node, trimers_lymph])

    # Tumor trimers (in lymph node) - simulate reaching threshold around day 10
    trimers_per_tumor = 50.0 * (1 - np.exp(-0.2 * (t_span - p['TAD']))) * (t_span >= p['TAD'])

    # Trafficking rates (from lymphocyte model)
    trafficking_rates = {
        'blood_to_spleen': 50.0,
        'spleen_to_node': 1.67,
        'node_to_lymph': 1672.0,
        'lymph_to_blood': 2.63
    }

    print("Running T-cell activation simulation...")
    solution = activation_model.simulate(
        initial_conditions, t_span, trimers_per_BC, trimers_per_tumor,
        TC_counts, BC_counts, tumor_cell_count, trafficking_rates
    )

    print("Simulation completed!")

    # Calculate metrics
    metrics = activation_model.calculate_activation_metrics(solution, TC_counts)

    # Plot results
    plt.figure(figsize=(16, 12))

    # Plot 1: Trimer levels
    plt.subplot(3, 3, 1)
    plt.plot(t_span, trimers_blood, label='Blood', linewidth=2)
    plt.plot(t_span, trimers_spleen, label='Spleen', linewidth=2)
    plt.plot(t_span, trimers_node, label='Lymph Node', linewidth=2)
    plt.plot(t_span, trimers_lymph, label='Lymph', linewidth=2)
    plt.axhline(y=p['Trimer_Threshold'], color='red', linestyle='--',
                label=f'Threshold ({p["Trimer_Threshold"]})', alpha=0.7)
    plt.ylabel('Trimers per B-cell')
    plt.title('Trimer Levels for T-cell Activation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Tumor trimer levels and RELU function
    plt.subplot(3, 3, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Tumor trimers
    ax1.plot(t_span, trimers_per_tumor, label='Tumor Trimers', linewidth=2, color='blue')
    ax1.axhline(y=p['Trimer_Threshold'], color='red', linestyle='--',
                label=f'Threshold ({p["Trimer_Threshold"]})', alpha=0.7)
    ax1.set_ylabel('Trimers per Tumor Cell', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Calculate RELU values
    RELU_values = np.where(trimers_per_tumor > p['Trimer_Threshold'], 1.0, 0.01)
    ax2.plot(t_span, RELU_values, label='RELU', linewidth=2, color='green', linestyle=':')
    ax2.set_ylabel('RELU Value', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1.1)

    plt.title('Tumor Trimer Levels and RELU Function')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 3: Virtual ATCs against B-cells
    plt.subplot(3, 3, 3)
    vATC_BC_blood = solution[:, 0]
    vATC_BC_spleen = solution[:, 1]
    vATC_BC_node = solution[:, 2]
    vATC_BC_lymph = solution[:, 3]

    plt.plot(t_span, vATC_BC_blood, label='Blood', linewidth=2)
    plt.plot(t_span, vATC_BC_spleen, label='Spleen', linewidth=2)
    plt.plot(t_span, vATC_BC_node, label='Lymph Node', linewidth=2)
    plt.plot(t_span, vATC_BC_lymph, label='Lymph', linewidth=2)
    plt.ylabel('Virtual ATCs (vs B-cells)')
    plt.title('Activated T-cells Against B-cells')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Proliferating ATCs
    plt.subplot(3, 3, 4)
    pATC_blood = solution[:, 4]
    pATC_spleen = solution[:, 5]
    pATC_node = solution[:, 6]
    pATC_lymph = solution[:, 7]

    plt.plot(t_span, pATC_blood, label='Blood', linewidth=2)
    plt.plot(t_span, pATC_spleen, label='Spleen', linewidth=2)
    plt.plot(t_span, pATC_node, label='Lymph Node', linewidth=2)
    plt.plot(t_span, pATC_lymph, label='Lymph', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Proliferating ATCs')
    plt.title('Clonally Expanded T-cells')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: Tumor-specific ATCs
    plt.subplot(3, 3, 5)
    vATC_tumor = solution[:, 8]

    plt.plot(t_span, vATC_tumor, label='Tumor ATCs', linewidth=2, color='purple')
    plt.axvline(x=p['TAD'], color='red', linestyle='--', alpha=0.7, label='TAD')
    plt.axvline(x=p['Tp'], color='orange', linestyle='--', alpha=0.7, label='Tp')
    plt.xlabel('Time (days)')
    plt.ylabel('Virtual ATCs (vs Tumor)')
    plt.title('Activated T-cells Against Tumor Cells')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Total activated T-cells
    plt.subplot(3, 3, 6)
    total_ATC = metrics['total_ATC']

    plt.plot(t_span, total_ATC[:, 0], label='Blood', linewidth=2)
    plt.plot(t_span, total_ATC[:, 1], label='Spleen', linewidth=2)
    plt.plot(t_span, total_ATC[:, 2], label='Lymph Node', linewidth=2)
    plt.plot(t_span, total_ATC[:, 3], label='Lymph', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Total Activated T-cells')
    plt.title('Total Activated T-cell Population')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 7: Activation fraction
    plt.subplot(3, 3, 7)
    activation_fraction = metrics['activation_fraction']

    plt.plot(t_span, activation_fraction[:, 0] * 100, label='Blood', linewidth=2)
    plt.plot(t_span, activation_fraction[:, 1] * 100, label='Spleen', linewidth=2)
    plt.plot(t_span, activation_fraction[:, 2] * 100, label='Lymph Node', linewidth=2)
    plt.plot(t_span, activation_fraction[:, 3] * 100, label='Lymph', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Activation Fraction (%)')
    plt.title('Percentage of T-cells Activated')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 8: Key time points
    plt.subplot(3, 3, 8)
    time_points = {
        'TAD': p['TAD'],
        'Tp': p['Tp'],
        'Peak Trimers': t_span[np.argmax(trimers_node)],
        'Peak Activation': t_span[np.argmax(total_ATC[:, 2])]
    }

    bars = plt.bar(time_points.keys(), time_points.values(),
                   color=['red', 'orange', 'blue', 'green'])
    plt.ylabel('Time (days)')
    plt.title('Key Time Points')
    plt.xticks(rotation=45)

    # Add values on bars
    for bar, value in zip(bars, time_points.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{value:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 9: Activation kinetics overview
    plt.subplot(3, 3, 9)
    # Show the progression of activation
    plt.axvspan(0, p['TAD'], alpha=0.2, color='red', label='Pre-activation')
    plt.axvspan(p['TAD'], p['Tp'], alpha=0.2, color='orange', label='Proliferation')
    plt.axvspan(p['Tp'], t_span[-1], alpha=0.2, color='green', label='Turnover')

    # Normalize for comparison
    norm_trimers = trimers_node / np.max(trimers_node)
    norm_ATC = total_ATC[:, 2] / np.max(total_ATC[:, 2])

    plt.plot(t_span, norm_trimers, label='Normalized Trimers', linewidth=2)
    plt.plot(t_span, norm_ATC, label='Normalized ATCs', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized Value')
    plt.title('Activation Kinetics Overview')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/tcell_activation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print key metrics
    print(f"\n=== T-cell Activation Results ===")
    print(f"Maximum trimers per B-cell: {np.max(trimers_node):.1f}")
    print(f"Maximum trimers per tumor cell: {np.max(trimers_per_tumor):.1f}")
    print(f"Time above trimer threshold: {np.sum(trimers_per_tumor > p['Trimer_Threshold'])} days")

    print(f"\n=== Activated T-cell Populations ===")
    max_blood_ATC = np.max(total_ATC[:, 0])
    max_spleen_ATC = np.max(total_ATC[:, 1])
    max_node_ATC = np.max(total_ATC[:, 2])
    max_lymph_ATC = np.max(total_ATC[:, 3])

    print(f"Maximum blood ATCs: {max_blood_ATC:.2e} cells")
    print(f"Maximum spleen ATCs: {max_spleen_ATC:.2e} cells")
    print(f"Maximum lymph node ATCs: {max_node_ATC:.2e} cells")
    print(f"Maximum lymph ATCs: {max_lymph_ATC:.2e} cells")

    print(f"\n=== Activation Fractions ===")
    max_activation_blood = np.max(activation_fraction[:, 0]) * 100
    max_activation_spleen = np.max(activation_fraction[:, 1]) * 100
    max_activation_node = np.max(activation_fraction[:, 2]) * 100
    max_activation_lymph = np.max(activation_fraction[:, 3]) * 100

    print(f"Maximum blood activation: {max_activation_blood:.2f}%")
    print(f"Maximum spleen activation: {max_activation_spleen:.2f}%")
    print(f"Maximum lymph node activation: {max_activation_node:.2f}%")
    print(f"Maximum lymph activation: {max_activation_lymph:.2f}%")

    return solution, t_span, metrics


if __name__ == "__main__":
    solution, t_span, metrics = run_activation_demo()