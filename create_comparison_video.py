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
    """Draw a single metric row with proper text spacing"""
    row_height = 50  # Increased from 40 to prevent overlap
    padding = 12

    # Background
    cv2.rectangle(canvas, (x, y), (x + width, y + row_height),
                 COLORS['card_bg'], -1)

    # Border
    cv2.rectangle(canvas, (x, y), (x + width, y + row_height),
                 COLORS['border'], 1)

    # Left accent bar
    cv2.rectangle(canvas, (x, y), (x + 4, y + row_height), color, -1)

    # Label - small, gray, at top with more spacing
    label_y = y + 18  # Increased from 16 for more top spacing
    cv2.putText(canvas, label, (x + padding, label_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1, cv2.LINE_AA)

    # Value - larger, bold, at bottom with more spacing from label
    value_y = y + row_height - 12  # Increased spacing from bottom
    cv2.putText(canvas, value, (x + padding, value_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_dark'], 2, cv2.LINE_AA)

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
    hud_height = 240  # Increased from 200 to accommodate taller metric rows
    margin = 15
    gap = 10

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
            hud_y = video_y + video_height + 12
            row_gap = 6  # Increased gap between rows from 4 to 6

            # Metric 1: Steering Error
            metric_height = draw_metric_row(canvas, video_x, hud_y,
                          video_width, "STEERING ERROR",
                          f"{error:+.4f}", COLORS[scenario['key']])

            # Metric 2: Left Motor
            hud_y += metric_height + row_gap
            draw_metric_row(canvas, video_x, hud_y,
                          video_width, "LEFT MOTOR",
                          f"{motor_l:.2f} rad/s", COLORS['baseline'])

            # Metric 3: Right Motor
            hud_y += metric_height + row_gap
            draw_metric_row(canvas, video_x, hud_y,
                          video_width, "RIGHT MOTOR",
                          f"{motor_r:.2f} rad/s", COLORS['glare'])

            # Metric 4: Time and Frame
            hud_y += metric_height + row_gap
            draw_metric_row(canvas, video_x, hud_y,
                          video_width, "TIME / FRAME",
                          f"{sim_time:.2f}s | {frame_idx + 1}/{total_frames}",
                          COLORS['accent'])

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
