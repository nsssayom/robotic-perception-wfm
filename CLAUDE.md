# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Webots robotics controller implementing a vision-based line-following system for testing GenAI-generated video from Nvidia Cosmos Predict 2.5 Model. The project demonstrates that GenAI World Foundation Models can be used for robotics vehicle perception node testing by feeding synthetic video into a robot controller as if it were real sensor data.

## Architecture

### Core Components

**ResearchRobot Class** (extends Webots `Robot`)
- Integrates Webots hardware control with OpenCV vision processing
- Main control loop runs at simulation timestep (typically 32ms)
- Flow: Video Frame → HSV Conversion → Yellow Line Detection → Steering Command → Motor Control → Data Logging

**Perception Pipeline**
- Reads pre-recorded MP4 videos instead of live camera feed
- Uses HSV color space for yellow line detection (range: [15,100,100] to [35,255,255])
- Calculates centroid using image moments
- Computes steering error as normalized horizontal offset from center

**Control System**
- Differential drive controller with proportional steering (gain: 2.5)
- Base speed: 3.0 rad/s
- Motor commands: `v_left = base_speed + turn_cmd`, `v_right = base_speed - turn_cmd`

**Data Logging**
- CSV format: Time, Frame, Error, Motor_L, Motor_R, Odom_L, Odom_R
- Odometry values calculated from wheel encoders with 0.02m wheel radius

## Development Commands

### Running the Controller

This controller is designed to run within Webots simulation environment:
```bash
# From Webots: Robot → Edit Controller → Select vision_controller.py
# Then click Run or use Ctrl+5
```

### Python Environment

```bash
# Activate virtual environment
source .venv/bin/activate  # or .venv/Scripts/activate on Windows

# Install dependencies (if needed)
pip install opencv-python numpy
```

Note: The `controller` module is provided by Webots at runtime and is not in the venv.

### Testing Different Scenarios

The controller uses configuration constants at the top of the file:
- `VIDEO_PATH`: Select input video (path_asphalt.mp4, path_glare.mp4, or path_mars.mp4)
- `LOG_FILE`: Output CSV filename
- `DISPLAY_SCALE`: OpenCV window size multiplier

Change these values to test different Cosmos-generated scenarios.

### Data Analysis

```bash
# View logged data
head data_baseline.csv
cat data_glare.csv
```

Each test scenario generates a separate CSV file for comparison between real-world, baseline, and GenAI-generated conditions.

## Key Implementation Details

### Video Integration Pattern

Unlike typical Webots controllers that use camera devices, this implementation uses `cv2.VideoCapture()` to read pre-recorded video files. This allows testing with GenAI-generated videos without modifying the simulation world.

### Error Handling

- When video ends: Calls `close_and_exit()` to cleanly shutdown
- When no line detected: Maintains last steering command (error = 0.0)
- Press 'q' key in OpenCV window to manually exit

### Critical Constraint

Standard Webots `Robot` controllers cannot pause/control the simulator (only `Supervisor` nodes can). The controller avoids `simulationSetMode()` calls that would cause crashes.

## Testing Methodology

The project compares robot behavior across three video types:
1. **path_asphalt.mp4**: Baseline real-world or high-fidelity scenario
2. **path_glare.mp4**: Challenging lighting conditions (glare)
3. **path_mars.mp4**: Extreme domain shift (Mars-like terrain)

Compare corresponding CSV files to evaluate how perception degrades under GenAI video artifacts.
