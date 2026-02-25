# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ball Shot Tracker** — An FRC Team 41 camera-side Python application that:
- Tracks yellow ball trajectories during shooting tests
- Calibrates a planar field using 4 AprilTag markers
- Coordinates shots with the roboRIO via NetworkTables 4 (NT4)
- Records trajectory data (world coordinates + time) to JSON
- Supports both networked (NT4) and standalone modes

The system uses a strict state machine to synchronize shooting: the camera waits for the robot to set `shoot_ready`, detects the ball launch, tracks the trajectory, and notifies the robot when complete.

## Architecture

### Main Components

**ballshots6.py** (primary entry point)
- Camera capture + OpenCV processing pipeline
- State machine controller (CALIBRATING → WAITING → ARMED → TRACKING → COOLDOWN → DONE)
- AprilTag detection (4 tags define world coordinate system via perspective-n-point)
- HSV-based ball detection and contour filtering
- Trajectory recording with real-time world coordinate transformation
- NetworkTables client (connects to roboRIO at 10.0.41.2)
- Real-time visualization (matplotlib graph of tracking history)

**process_shots.py**
- Post-processes `shots.json` trajectory data
- Extracts computed fields: distance, peak rise, time-of-flight
- Fits projectile motion model across all shots to find launch speed coefficient (k) and hood-angle offset (δ)
- Outputs CSV with per-shot summaries and predictions

### Data Flow

```
Camera Frame
    ↓
[Undistort via K_CALIB, D_CALIB]
    ↓
[AprilTag Detection + PnP] → World calibration (X/Y/Z axes)
    ↓
[HSV Mask + Contour] → Ball detection (pixel space)
    ↓
[3D Ray Projection] → Ball position in world space (–1.0 m plane)
    ↓
[State Machine Logic + NT Coordination]
    ↓
shots.json (persisted, rewritten atomically per shot)
```

### State Machine

The camera publishes its state via `camera/state` and responds to `robot/shoot_ready`:

| State | Camera Action | Robot Action | Exit |
|-------|---------------|--------------|------|
| CALIBRATING | Detect AprilTags, accumulate PnP poses | Idle | 10 s elapsed + ≥1 pose |
| WAITING | Set ok_to_shoot=true | Spin flywheel, set angle, then set shoot_ready | shoot_ready → true |
| ARMED | Watch for ball (HSV contour) | Slowly feed ball toward shooter | Ball detected in frame |
| TRACKING | Record trajectory + world coords | Stop all motors (ball in flight) | Ball pixel-Y ≥ start_py OR lost >0.5s |
| COOLDOWN | Wait for ball to leave frame | Stay stopped | Ball absent 2 s continuously |
| DONE | Save JSON, publish landing coords, increment shot_count | Idle | Immediate → WAITING |

### Coordinate Systems

- **Pixel space**: (0,0) at top-left; px_y increases downward
- **World space**: Origin at Tag 0 center (bottom-left); X right, Y depth (toward shooter), Z up
  - Tags: 0 (top-left), 1 (top-right), 2 (bot-left), 3 (bot-right)
  - Spacing: 1.0 m horizontal, 1.0 m vertical
  - Bottom tags at Z = 0.1397 m (US letter height / 2)
- **Trajectory plane**: Y = –1.0 m (1 meter in front of tag plane)

## Configuration & Setup

### Python Environment
```bash
python -m venv .venv
source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

**Key dependencies:**
- `opencv-contrib-python` — Computer vision (AprilTag via cv2.aruco)
- `pyntcore>=2026.0.0` — NetworkTables 4 client
- `numpy`, `scipy` — Numerical computing
- `matplotlib` — Real-time graph visualization

### Camera Calibration

Hard-coded in ballshots6.py (lines 38–48):
- **K_CALIB**: Camera intrinsic matrix (1280×720 resolution, PhotonVision-calibrated)
- **D_CALIB**: Radial & tangential distortion coefficients (8 parameters)

Camera matrix from PhotonVision JSON if recalibration needed.

### Key Constants (ballshots6.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| TAG_SIZE | 0.2 m | AprilTag marker size |
| TAG_SPACING_H | 1.0 m | Horizontal tag spacing |
| TAG_SPACING_V | 1.0 m | Vertical tag spacing |
| TAG_Z_BOTTOM | 0.1397 m | Height of bottom tags |
| LO_YELLOW / HI_YELLOW | [25,100,100] / [35,255,255] | HSV ball color range |
| MIN_RISE_PX | 20 px | Ball must rise ≥20px before descent accepted |
| BALL_LOST_TIMEOUT | 0.5 s | Trajectory recording stops if ball undetected |
| BALL_GONE_TIMEOUT | 2.0 s | Shot finalized when ball absent for 2s |
| CALIB_DURATION | 10.0 s | AprilTag calibration time |
| PLANE_Y | –1.0 m | Plane for ball-to-world projection |

## Running the Application

### Basic Modes

```bash
# Connect to Team 41 roboRIO (default)
python ballshots6.py

# Connect to specific server (bench testing)
python ballshots6.py --nt-server 192.168.1.100

# Standalone mode (no NetworkTables)
python ballshots6.py --no-nt

# Show HSV debug window for ball color tuning
python ballshots6.py --show-mask
```

### Output Files

**shots.json** — Atomic JSON array of shot objects (example structure):
```json
[
  {
    "shot_id": 1,
    "timestamp": 1708200000.123,
    "speed_rpm": 3500.0,
    "angle_deg": 45.0,
    "n_frames": 42,
    "duration_s": 1.234,
    "land_x_cm": 108.5,
    "land_y_cm": -32.6,
    "trajectory": [
      { "frame": 0, "time_s": 0.0000, "px_x": 640.0, "px_y": 400.0,
        "world_x_cm": 50.0, "world_y_cm": 10.0 },
      { "frame": 1, "time_s": 0.0333, "px_x": 638.2, "px_y": 385.1,
        "world_x_cm": 52.1, "world_y_cm": 11.3 }
    ]
  }
]
```

**ball_positions.csv** — Optional per-frame export (format TBD)

## NetworkTables Protocol

See [protocol.md](protocol.md) for full NT4 handshake. Key points:

- **Server**: roboRIO at 10.0.41.2, team 41
- **Client identity**: `"balltracker"`
- **Table root**: `/BallTracker/`

### Robot → Camera (inputs)
- `robot/shoot_ready` (bool) — Rising edge triggers ARMED state
- `robot/shot_id` (int) — Monotonic shot counter (1, 2, 3…)
- `robot/shot_speed_rpm` (double, optional) — Logged per shot
- `robot/shot_angle_deg` (double, optional) — Logged per shot

### Camera → Robot (outputs)
- `camera/state` (string) — Current state name
- `camera/ok_to_shoot` (bool) — Safe to start next shot cycle
- `camera/ball_detected` (bool) — Ball in flight (robot can stop shooter)
- `camera/shot_count` (int) — Total shots completed this session
- `camera/last_land_x_cm`, `camera/last_land_y_cm` (double) — Landing position
- `camera/fps` (double) — Processing frame rate

## Key Implementation Details

### Ball Detection

1. **HSV mask**: Isolate yellow using `cv2.inRange(hsv, LO_YELLOW, HI_YELLOW)`
2. **Morphology**: Apply opening/closing with 5×5 kernel to denoise
3. **Contour filtering**: Find contours, filter by area (200–60000 px²) and circularity (≥0.4)
4. **Centroid**: Compute mass-center of largest contour as ball position

### Trajectory End Condition

1. Record `start_py` when ball first detected
2. Ball travels upward (px_y decreases)
3. Require upward travel ≥20 px (debounce)
4. When px_y exceeds `start_py` → ball below start height → **stop recording**
5. Safety: If ball lost >0.5 s → also stop
6. Finalization: Ball absent for 2 s → save JSON, publish landing, reset to WAITING

### Coordinate Transformation

For each tracked frame:
1. Detect ball in pixel space (px_x, px_y)
2. Create ray from camera through image plane (intrinsic K_CALIB)
3. Project ray onto world plane (Y = PLANE_Y = –1.0 m) using PnP-derived extrinsics
4. Clip trajectory at landing point (world_y = 0, the 0-1 tag line)

### GPU Acceleration

- Auto-detects CUDA; falls back to OpenCL (cv2.UMat) if unavailable
- Printed to console on startup

## History & Iteration

- **ballshots5.py** → **ballshots6.py**: Current stable version
- Previous versions in `_old/` (ballshots, ballshots2–5, analyze_calibration, recalibrate)
- `photonvision_config/` contains calibration history and PhotonVision logs

## Common Tasks

### Tune HSV Ball Color
```bash
python ballshots6.py --show-mask
```
Adjust `LO_YELLOW` and `HI_YELLOW` in ballshots6.py (lines 77–78), then re-run.

### Re-calibrate Camera
1. Place 4 AprilTags in test geometry
2. Run ballshots6.py with `--no-nt` or standalone
3. Wait for CALIBRATING state to complete (≥1 valid pose)
4. Monitor console for calibration success
5. If PnP fails, check tag layout against TAG_CENTRES dictionary

### Analyze Shots Post-Session
```bash
python process_shots.py
# Reads shots.json, outputs CSV summary + fitted model parameters
```

### Bench Testing (Without Robot)
```bash
python ballshots6.py --no-nt
# Runs full pipeline without NetworkTables
# Manually set WAITING state or modify code to auto-transition
```

## Notes for Future Contributors

- **Atomic JSON writes**: shots.json is rewritten in full after each shot to survive restarts
- **Frame sync**: All trajectory times are relative to first detection frame (0.000, 0.033, 0.066…)
- **Bounce handling**: After trajectory recording stops (COOLDOWN), any bounces are ignored; only frame absence (2 s) finalizes the shot
- **NT fallback**: If roboRIO unreachable, camera still runs; states are published but robot cannot respond
- **GPU optional**: CUDA not required; code gracefully degrades to OpenCL/CPU
- **Real-time graph**: Matplotlib shows 10 s rolling window of trajectory; updates every 0.15 s
