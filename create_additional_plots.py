import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

# Color scheme
SCENARIO_COLORS = {
    'baseline': '#27AE60',    # Green
    'glare': '#F39C12',       # Orange
    'mars': '#8E44AD'         # Purple
}

# Perception constants (from vision_controller.py)
YELLOW_LOWER = np.array([15, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])


def reconstruct_path_from_odometry(csv_file, wheel_base=0.052):
    """
    Reconstruct 2D path from differential drive odometry.

    Args:
        csv_file: Path to CSV with odometry data
        wheel_base: Distance between wheels in meters (e-puck = 0.052m)

    Returns:
        x, y: Arrays of position coordinates
    """
    df = pd.read_csv(csv_file)

    # Initialize pose
    x, y, theta = [0], [0], [0]

    # Differential drive kinematics
    for i in range(1, len(df)):
        # Get odometry deltas (already in meters from wheel encoder * radius)
        dl = df['Odom_L'].iloc[i] - df['Odom_L'].iloc[i-1]
        dr = df['Odom_R'].iloc[i] - df['Odom_R'].iloc[i-1]

        # Compute forward and angular displacement
        ds = (dl + dr) / 2.0  # Forward distance
        dtheta = (dr - dl) / wheel_base  # Angular change

        # Update pose (using midpoint integration)
        theta_new = theta[-1] + dtheta
        x_new = x[-1] + ds * np.cos(theta[-1] + dtheta/2)
        y_new = y[-1] + ds * np.sin(theta[-1] + dtheta/2)

        x.append(x_new)
        y.append(y_new)
        theta.append(theta_new)

    return np.array(x), np.array(y)


def create_trajectory_comparison():
    """Create path trajectory comparison plot"""

    scenarios = [
        ('data/data_baseline.csv', 'Baseline (Asphalt)', 'baseline'),
        ('data/data_glare.csv', 'Glare Condition', 'glare'),
        ('data/data_mars.csv', 'Mars Terrain', 'mars')
    ]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('Robot Path Reconstruction from Odometry:\nGenAI Video Testing Across Scenarios',
                 fontsize=22, fontweight='bold')

    for csv_file, name, key in scenarios:
        x, y = reconstruct_path_from_odometry(csv_file)

        # Plot trajectory
        ax.plot(x, y, color=SCENARIO_COLORS[key], linewidth=3.0,
                label=name, alpha=0.85)

        # Mark start and end points
        ax.scatter(x[0], y[0], color=SCENARIO_COLORS[key], s=150,
                  marker='o', edgecolor='black', linewidth=2, zorder=10)
        ax.scatter(x[-1], y[-1], color=SCENARIO_COLORS[key], s=150,
                  marker='s', edgecolor='black', linewidth=2, zorder=10)

    # Add start/end indicators to legend
    ax.scatter([], [], color='gray', s=150, marker='o',
              edgecolor='black', linewidth=2, label='Start Position')
    ax.scatter([], [], color='gray', s=150, marker='s',
              edgecolor='black', linewidth=2, label='End Position')

    ax.set_xlabel('X Position (m)', fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontweight='bold')
    ax.set_title('Reconstructed 2D Trajectories from Wheel Odometry',
                fontsize=18, pad=15)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.legend(loc='best', framealpha=0.95, edgecolor='black', fancybox=False)
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output_file = 'figures/trajectory_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'✓ Saved: {output_file}')
    plt.close()


def create_perception_visualization():
    """Create video frame + detection overlay visualization"""

    scenarios = [
        ('videos/path_asphalt.mp4', 'Baseline (Asphalt)', 'baseline', 30),
        ('videos/path_glare.mp4', 'Glare Condition', 'glare', 30),
        ('videos/path_mars.mp4', 'Mars Terrain', 'mars', 30)
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Vision Perception Pipeline: GenAI Video Processing',
                 fontsize=24, fontweight='bold', y=0.98)

    column_titles = ['Original Video Frame', 'HSV Yellow Detection Mask', 'Detected Line Centroid']

    for row_idx, (video_path, scenario_name, key, frame_num) in enumerate(scenarios):
        # Read video
        cap = cv2.VideoCapture(video_path)

        # Skip to desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f'✗ Could not read frame from {video_path}')
            continue

        # COLUMN 1: Original frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[row_idx, 0].imshow(frame_rgb)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(column_titles[0], fontsize=16,
                                       fontweight='bold', pad=10)
        axes[row_idx, 0].text(0.02, 0.98, scenario_name,
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=16, fontweight='bold', color='white',
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor=SCENARIO_COLORS[key],
                                      alpha=0.8, edgecolor='black', linewidth=2))

        # COLUMN 2: HSV mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        axes[row_idx, 1].imshow(mask, cmap='gray')
        axes[row_idx, 1].axis('off')
        if row_idx == 0:
            axes[row_idx, 1].set_title(column_titles[1], fontsize=16,
                                       fontweight='bold', pad=10)

        # COLUMN 3: Detection overlay
        moments = cv2.moments(mask)
        height, width = mask.shape

        overlay = frame_rgb.copy()
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            # Draw vertical line at centroid
            cv2.line(overlay, (cx, 0), (cx, height), (0, 255, 0), 4)

            # Draw center reference line
            center_x = width // 2
            cv2.line(overlay, (center_x, 0), (center_x, height), (255, 0, 0), 3)

            # Draw circle at centroid
            cv2.circle(overlay, (cx, cy), 15, (0, 255, 0), -1)
            cv2.circle(overlay, (cx, cy), 15, (0, 0, 0), 3)

            # Calculate and display error
            error = (cx - width/2) / (width/2)
            error_text = f'Error: {error:+.4f}'
            cv2.putText(overlay, error_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(overlay, error_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].axis('off')
        if row_idx == 0:
            axes[row_idx, 2].set_title(column_titles[2], fontsize=16,
                                       fontweight='bold', pad=10)

        # Add legend to first row
        if row_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lime', edgecolor='black', label='Detected Line'),
                Patch(facecolor='red', edgecolor='black', label='Image Center')
            ]
            axes[row_idx, 2].legend(handles=legend_elements, loc='upper right',
                                   framealpha=0.9, edgecolor='black')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = 'figures/perception_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'✓ Saved: {output_file}')
    plt.close()


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Additional Validation Visualizations")
    print("=" * 60)

    print("\n[1/2] Creating trajectory comparison plot...")
    try:
        create_trajectory_comparison()
    except Exception as e:
        print(f'✗ Error creating trajectory plot: {e}')
        import traceback
        traceback.print_exc()

    print("\n[2/2] Creating perception visualization...")
    try:
        create_perception_visualization()
    except Exception as e:
        print(f'✗ Error creating perception visualization: {e}')
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Additional visualizations complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • figures/trajectory_comparison.png")
    print("  • figures/perception_visualization.png")
