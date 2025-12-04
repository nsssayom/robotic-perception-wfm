import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Publication-quality plot settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

# Color scheme - professional and colorblind-friendly
COLORS = {
    'error': '#E74C3C',      # Red
    'left': '#3498DB',        # Blue
    'right': '#E67E22',       # Orange
    'grid': '#BDC3C7'         # Light gray
}

def create_scenario_plots(csv_file, scenario_name, output_prefix):
    """Create perception and actuation plots for a scenario"""

    # Load data
    df = pd.read_csv(csv_file)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{scenario_name} Scenario: Perception-Control Validation',
                 fontsize=22, fontweight='bold', y=0.995)

    # ===== PLOT 1: Perception Error =====
    ax1.plot(df['Time'], df['Error'], color=COLORS['error'], linewidth=2.5, label='Steering Error')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Target (Centered)')
    ax1.fill_between(df['Time'], df['Error'], 0, alpha=0.15, color=COLORS['error'])

    ax1.set_xlabel('Simulation Time (s)', fontweight='bold')
    ax1.set_ylabel('Normalized Lateral Error\n(+Right / -Left)', fontweight='bold')
    ax1.set_title('Vision Perception Output', fontsize=18, pad=15)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    # Smart legend placement to avoid overlap
    ax1.legend(loc='best', framealpha=0.95, edgecolor='black', fancybox=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Set reasonable y-limits with extra margin for legend
    error_range = df['Error'].max() - df['Error'].min()
    y_margin = error_range * 0.2
    ax1.set_ylim(df['Error'].min() - y_margin, df['Error'].max() + y_margin)

    # ===== PLOT 2: Motor Commands =====
    ax2.plot(df['Time'], df['Motor_L'], color=COLORS['left'], linewidth=2.5,
             label='Left Motor', marker='o', markevery=max(1, len(df)//20), markersize=5)
    ax2.plot(df['Time'], df['Motor_R'], color=COLORS['right'], linewidth=2.5,
             label='Right Motor', marker='s', markevery=max(1, len(df)//20), markersize=5)

    # Add base speed reference line
    base_speed = 3.0
    ax2.axhline(y=base_speed, color='black', linestyle=':', linewidth=1.5,
                alpha=0.4, label='Base Speed')

    ax2.set_xlabel('Simulation Time (s)', fontweight='bold')
    ax2.set_ylabel('Motor Velocity (rad/s)', fontweight='bold')
    ax2.set_title('Actuation Response (Control Output)', fontsize=18, pad=15)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    # Smart legend placement - try upper center to avoid curve overlap
    ax2.legend(loc='upper center', framealpha=0.95, edgecolor='black',
               fancybox=False, ncol=3, bbox_to_anchor=(0.5, -0.15))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Set motor velocity limits with margin
    all_velocities = pd.concat([df['Motor_L'], df['Motor_R']])
    vel_margin = (all_velocities.max() - all_velocities.min()) * 0.15
    ax2.set_ylim(all_velocities.min() - vel_margin, all_velocities.max() + vel_margin)

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save high-resolution figure
    output_file = f'{output_prefix}_perception_control.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'✓ Saved: {output_file}')
    plt.close()


def create_combined_comparison(csv_files, scenario_names, colors_list):
    """Create a comparison plot showing all three scenarios"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('GenAI World Model Validation: Multi-Scenario Testing',
                 fontsize=22, fontweight='bold', y=0.995)

    for csv_file, scenario_name, color in zip(csv_files, scenario_names, colors_list):
        df = pd.read_csv(csv_file)

        # Plot perception errors
        ax1.plot(df['Time'], df['Error'], color=color, linewidth=2.5,
                 label=scenario_name, alpha=0.85)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('Simulation Time (s)', fontweight='bold')
    ax1.set_ylabel('Normalized Lateral Error', fontweight='bold')
    ax1.set_title('Vision Perception Across Test Scenarios', fontsize=18, pad=15)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax1.legend(loc='best', framealpha=0.95, edgecolor='black', fancybox=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot motor commands (left motor only for clarity)
    for csv_file, scenario_name, color in zip(csv_files, scenario_names, colors_list):
        df = pd.read_csv(csv_file)
        ax2.plot(df['Time'], df['Motor_L'], color=color, linewidth=2.5,
                 label=f'{scenario_name} (Left)', alpha=0.85, linestyle='-')
        ax2.plot(df['Time'], df['Motor_R'], color=color, linewidth=2.0,
                 alpha=0.5, linestyle='--')

    ax2.axhline(y=3.0, color='black', linestyle=':', linewidth=1.5, alpha=0.4)
    ax2.set_xlabel('Simulation Time (s)', fontweight='bold')
    ax2.set_ylabel('Motor Velocity (rad/s)', fontweight='bold')
    ax2.set_title('Control Response Comparison (Solid: Left, Dashed: Right)', fontsize=18, pad=15)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax2.legend(loc='upper center', framealpha=0.95, edgecolor='black',
               fancybox=False, ncol=3, bbox_to_anchor=(0.5, -0.15))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = 'figures/combined_scenario_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'✓ Saved: {output_file}')
    plt.close()


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Publication-Quality Validation Plots")
    print("=" * 60)

    scenarios = [
        ('data/data_baseline.csv', 'Baseline (Asphalt)', 'figures/baseline'),
        ('data/data_glare.csv', 'Glare Condition', 'figures/glare'),
        ('data/data_mars.csv', 'Mars Terrain', 'figures/mars')
    ]

    # Create individual plots for each scenario
    print("\n[1/2] Creating individual scenario plots...")
    for csv_file, scenario_name, prefix in scenarios:
        try:
            create_scenario_plots(csv_file, scenario_name, prefix)
        except Exception as e:
            print(f'✗ Error processing {csv_file}: {e}')

    # Create comparison plot
    print("\n[2/2] Creating combined comparison plot...")
    try:
        csv_files = [s[0] for s in scenarios]
        names = [s[1] for s in scenarios]
        colors = ['#27AE60', '#F39C12', '#8E44AD']  # Green, Orange, Purple
        create_combined_comparison(csv_files, names, colors)
    except Exception as e:
        print(f'✗ Error creating comparison: {e}')

    print("\n" + "=" * 60)
    print("Plot generation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • figures/baseline_perception_control.png")
    print("  • figures/glare_perception_control.png")
    print("  • figures/mars_perception_control.png")
    print("  • figures/combined_scenario_comparison.png")
    print("\nAll plots are 300 DPI and optimized for presentation slides.")
    print("Legend placement optimized to avoid curve overlap.")
