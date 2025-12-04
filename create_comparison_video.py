import cv2
import numpy as np
import pandas as pd

# Perception constants
YELLOW_LOWER = np.array([15, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

# Clean professional color scheme with WHITE background
COLORS = {
    'baseline': (76, 175, 80),      # Material Green (BGR)
    'glare': (255, 152, 0),         # Material Orange (BGR)
    'mars': (156, 39, 176),         # Material Purple (BGR)
    'accent': (33, 150, 243),       # Material Blue (BGR)
    'text_dark': (33, 33, 33),      # Dark gray for text
    'text_light': (255, 255, 255),  # White for text on dark
    'bg_white': (250, 250, 250),    # Off-white background
    'card_bg': (255, 255, 255),     # White cards
    'border': (224, 224, 224),      # Light gray borders
    'detected': (0, 200, 0),        # Green
    'center': (244, 67, 54)         # Red
}


def draw_metric_row(canvas, x, y, width, label, value, color):
    """Draw a single metric row with modern design and proper spacing"""
    row_height = 65  # Generous height for breathing room
    padding_left = 16
    padding_top = 20
    
    # Background with subtle shadow effect
    cv2.rectangle(canvas, (x, y), (x + width, y + row_height),
                 COLORS['card_bg'], -1)

    # Border with rounded corners effect (simulated with thicker border)
    cv2.rectangle(canvas, (x, y), (x + width, y + row_height),
                 COLORS['border'], 2)

    # Left accent bar (thicker and more prominent)
    cv2.rectangle(canvas, (x, y), (x + 6, y + row_height), color, -1)

    # Label - modern font, medium size, proper weight
    label_y = y + padding_top
    cv2.putText(canvas, label, (x + padding_left, label_y),
               cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    # Value - larger, bold, clear separation from label
    value_y = y + padding_top + 30  # 30px below label for clear separation
    cv2.putText(canvas, value, (x + padding_left, value_y),
               cv2.FONT_HERSHEY_DUPLEX, 0.75, COLORS['text_dark'], 2, cv2.LINE_AA)

    return row_height


def process_frame_with_detection(frame, scenario_data, frame_idx):
    """Process frame with yellow line detection"""

    if frame_idx < len(scenario_data):
        row = scenario_data.iloc[frame_idx]
        error = row['Error']
        motor_l = row['Motor_L']
        motor_r = row['Motor_R']
        sim_time = row['Time']
    else:
        error = 0.0
        motor_l = 3.0
        motor_r = 3.0
        sim_time = 0.0

    # Convert to HSV and detect yellow
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    # Calculate centroid
    moments = cv2.moments(mask)
    h, w = frame.shape[:2]

    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])

        # Draw detection line (green)
        cv2.line(frame, (cx, 0), (cx, h), COLORS['detected'], 3)

        # Draw center reference (red, dashed)
        center_x = w // 2
        dash_length = 15
        gap_length = 8
        for y in range(0, h, dash_length + gap_length):
            cv2.line(frame, (center_x, y), (center_x, min(y + dash_length, h)),
                    COLORS['center'], 2)

    return frame, error, motor_l, motor_r, sim_time


def draw_live_error_plot(canvas, x, y, width, height, error_history, colors, labels, frame_idx):
    """Draw a live error plot showing recent history"""

    # Plot background
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLORS['card_bg'], -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLORS['border'], 2)

    # Title
    cv2.putText(canvas, "Real-Time Steering Error", (x + 15, y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text_dark'], 2, cv2.LINE_AA)

    # Plot area
    plot_x = x + 40
    plot_y = y + 50
    plot_w = width - 60
    plot_h = height - 80

    # Draw axes
    cv2.line(canvas, (plot_x, plot_y + plot_h), (plot_x + plot_w, plot_y + plot_h),
            COLORS['text_dark'], 2)  # X-axis
    cv2.line(canvas, (plot_x, plot_y), (plot_x, plot_y + plot_h),
            COLORS['text_dark'], 2)  # Y-axis

    # Draw zero line
    zero_y = plot_y + plot_h // 2
    cv2.line(canvas, (plot_x, zero_y), (plot_x + plot_w, zero_y),
            (150, 150, 150), 1, cv2.LINE_AA)

    # Y-axis labels
    cv2.putText(canvas, "+", (plot_x - 25, plot_y + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_dark'], 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (plot_x - 25, zero_y + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_dark'], 1, cv2.LINE_AA)
    cv2.putText(canvas, "-", (plot_x - 25, plot_y + plot_h - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_dark'], 1, cv2.LINE_AA)

    # Plot error history for each scenario
    window_size = 50  # Show last 50 frames

    for idx, (history, color, label) in enumerate(zip(error_history, colors, labels)):
        if len(history) < 2:
            continue

        # Get recent data
        recent_data = history[-window_size:] if len(history) > window_size else history

        # Normalize to plot area
        max_error = 0.15  # Scale for error values

        points = []
        for i, err in enumerate(recent_data):
            px = plot_x + int((i / window_size) * plot_w)
            # Flip Y axis (positive errors go up)
            py = zero_y - int((err / max_error) * (plot_h // 2))
            py = max(plot_y, min(plot_y + plot_h, py))  # Clamp
            points.append((px, py))

        # Draw line
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], color, 2, cv2.LINE_AA)

    # Legend
    legend_x = x + width - 180
    legend_y = y + 50
    for idx, (color, label) in enumerate(zip(colors, labels)):
        ly = legend_y + idx * 25
        cv2.rectangle(canvas, (legend_x, ly), (legend_x + 20, ly + 12), color, -1)
        cv2.putText(canvas, label, (legend_x + 28, ly + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_dark'], 1, cv2.LINE_AA)


def create_side_by_side_comparison_video():
    """Create 3-column video with metrics below"""

    scenarios = [
        {
            'video': 'videos/path_asphalt.mp4',
            'data': 'data/data_baseline.csv',
            'name': 'Baseline',
            'subtitle': 'Asphalt Road',
            'key': 'baseline'
        },
        {
            'video': 'videos/path_glare.mp4',
            'data': 'data/data_glare.csv',
            'name': 'Glare',
            'subtitle': 'High Reflection',
            'key': 'glare'
        },
        {
            'video': 'videos/path_mars.mp4',
            'data': 'data/data_mars.csv',
            'name': 'Mars',
            'subtitle': 'Extreme Domain',
            'key': 'mars'
        }
    ]

    print("Loading video captures and data...")

    # Initialize error history tracking
    error_history = [[], [], []]  # One for each scenario

    # Open all videos
    caps = []
    dfs = []
    total_frames = float('inf')

    for scenario in scenarios:
        cap = cv2.VideoCapture(scenario['video'])
        if not cap.isOpened():
            print(f"✗ Error opening {scenario['video']}")
            return

        caps.append(cap)
        df = pd.read_csv(scenario['data'])
        dfs.append(df)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = min(total_frames, frame_count, len(df))
        print(f"  {scenario['name']}: {frame_count} frames")

    # Get video properties
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    orig_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 16:9 Full HD
    output_width = 1920
    output_height = 1080

    # Layout calculation
    header_height = 80
    footer_height = 60
    hud_height = 310  # Increased to accommodate taller metric rows with proper spacing
    margin = 15
    gap = 12  # Slightly wider gap between video columns

    # Calculate video dimensions
    available_height = output_height - header_height - footer_height - hud_height - 2 * margin
    available_width = output_width - 2 * margin - 2 * gap

    # 3 columns
    video_width = available_width // 3
    video_aspect = orig_width / orig_height
    video_height = int(video_width / video_aspect)

    # Adjust if too tall
    if video_height > available_height:
        video_height = available_height
        video_width = int(video_height * video_aspect)

    print(f"\nOutput: {output_width}x{output_height} (16:9) @ {fps} fps")
    print(f"Video panels: {video_width}x{video_height}")
    print(f"Total frames: {total_frames}")

    # Create video writer
    output_path = 'figures/comparison_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    if not out.isOpened():
        print("✗ Error creating output video")
        return

    print("\nProcessing frames...")
    frame_idx = 0

    while frame_idx < total_frames:
        # Create white canvas
        canvas = np.full((output_height, output_width, 3),
                        COLORS['bg_white'], dtype=np.uint8)

        # === HEADER ===
        cv2.rectangle(canvas, (0, 0), (output_width, header_height),
                     (255, 255, 255), -1)

        cv2.putText(canvas, "GenAI World Model Testing: Vision Perception Validation",
                   (margin, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                   COLORS['text_dark'], 2, cv2.LINE_AA)

        cv2.line(canvas, (0, header_height), (output_width, header_height),
                COLORS['border'], 2)

        # === VIDEOS (3 COLUMNS) ===
        video_y = header_height + margin

        for idx, (scenario, cap, df) in enumerate(zip(scenarios, caps, dfs)):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize video
            frame = cv2.resize(frame, (video_width, video_height))

            # Process detection
            frame, error, motor_l, motor_r, sim_time = process_frame_with_detection(
                frame, df, frame_idx)

            # Track error history for plotting
            error_history[idx].append(error)

            # Calculate X position (3 columns)
            video_x = margin + idx * (video_width + gap)

            # Place video
            canvas[video_y:video_y+video_height, video_x:video_x+video_width] = frame

            # Video border with scenario color
            cv2.rectangle(canvas, (video_x, video_y),
                         (video_x + video_width, video_y + video_height),
                         COLORS[scenario['key']], 4)

            # Scenario label banner on video
            label_height = 40
            cv2.rectangle(canvas, (video_x, video_y),
                         (video_x + 160, video_y + label_height),
                         COLORS[scenario['key']], -1)
            cv2.putText(canvas, scenario['name'],
                       (video_x + 15, video_y + 27), cv2.FONT_HERSHEY_DUPLEX,
                       0.75, COLORS['text_light'], 2, cv2.LINE_AA)

            # === HUD METRICS BELOW VIDEO ===
            hud_y = video_y + video_height + 15
            row_gap = 8  # Comfortable gap between rows

            # Metric 1: Steering Error
            metric_height = draw_metric_row(canvas, video_x, hud_y,
                          video_width, "Steering Error",
                          f"{error:+.4f}", COLORS[scenario['key']])

            # Metric 2: Left Motor
            hud_y += metric_height + row_gap
            draw_metric_row(canvas, video_x, hud_y,
                          video_width, "Left Motor",
                          f"{motor_l:.2f} rad/s", COLORS['baseline'])

            # Metric 3: Right Motor
            hud_y += metric_height + row_gap
            draw_metric_row(canvas, video_x, hud_y,
                          video_width, "Right Motor",
                          f"{motor_r:.2f} rad/s", COLORS['glare'])

            # Metric 4: Time and Frame
            hud_y += metric_height + row_gap
            draw_metric_row(canvas, video_x, hud_y,
                          video_width, "Time | Frame",
                          f"{sim_time:.2f}s  |  Frame {frame_idx + 1}/{total_frames}",
                          COLORS['accent'])

        # === LIVE ERROR PLOT (in the empty space) ===
        plot_x = margin
        # Calculate proper position: video_y + video_height + initial_gap + (metric_height + row_gap) * 4 + extra_spacing
        plot_y = video_y + video_height + 15 + (65 + 8) * 4 + 35  # Below metrics with generous spacing
        plot_width = output_width - 2 * margin
        plot_height = 180

        scenario_colors = [COLORS['baseline'], COLORS['glare'], COLORS['mars']]
        scenario_labels = ['Baseline', 'Glare', 'Mars']

        draw_live_error_plot(canvas, plot_x, plot_y, plot_width, plot_height,
                           error_history, scenario_colors, scenario_labels, frame_idx)

        # === FOOTER WITH PROGRESS ===
        footer_y = output_height - footer_height
        cv2.line(canvas, (0, footer_y), (output_width, footer_y),
                COLORS['border'], 2)

        # Progress bar
        progress = (frame_idx + 1) / total_frames
        bar_height = 12
        bar_y = footer_y + 20
        bar_x = margin
        bar_width = output_width - 2 * margin

        # Background
        cv2.rectangle(canvas, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     COLORS['border'], -1)

        # Progress fill
        progress_width = int(bar_width * progress)
        cv2.rectangle(canvas, (bar_x, bar_y),
                     (bar_x + progress_width, bar_y + bar_height),
                     COLORS['accent'], -1)

        # Progress text
        progress_text = f"Progress: {progress*100:.1f}%"
        cv2.putText(canvas, progress_text, (bar_x, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_dark'], 1, cv2.LINE_AA)

        # Write frame
        out.write(canvas)

        if (frame_idx + 1) % 10 == 0:
            print(f"  Progress: {frame_idx + 1}/{total_frames} ({progress*100:.1f}%)", end='\r')

        frame_idx += 1

    print(f"\n  Progress: {frame_idx}/{total_frames} (100.0%)")

    # Cleanup
    for cap in caps:
        cap.release()
    out.release()

    print(f"\n✓ Saved: {output_path}")
    print(f"  Duration: {frame_idx / fps:.2f} seconds")
    print(f"  Resolution: {output_width}x{output_height} (16:9)")


if __name__ == "__main__":
    print("=" * 70)
    print("Creating Side-by-Side Comparison Video (3 Columns)")
    print("=" * 70)
    print()

    try:
        create_side_by_side_comparison_video()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Video generation complete!")
    print("=" * 70)
