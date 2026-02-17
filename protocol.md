# Ball Shot Tracking — NetworkTables Protocol

## Overview

A camera-side Python program tracks yellow ball trajectories on a planar field
defined by 4 AprilTags.  The robot (roboRIO, Team 41) coordinates shots via
**NetworkTables 4** so the camera knows *when* to expect a ball and can signal
back *when* tracking is complete.

```
┌─────────┐       NT4 (10.0.41.2)        ┌────────────┐
│  roboRIO│◄────────────────────────────►│  Camera PC │
│  (robot)│   /BallTracker/...           │ ballshots6 │
└─────────┘                              └────────────┘
```

---

## NT Table

All keys live under **`/BallTracker/`**.

### Robot → Camera  (written by robot code)

| Key | Type | Description |
|-----|------|-------------|
| `robot/shoot_ready` | `boolean` | Robot sets **true** when it is about to shoot. Camera watches for this rising edge. |
| `robot/shot_id` | `int` | Monotonically increasing shot counter (1, 2, 3 …). Set before `shoot_ready`. |
| `robot/shot_speed_rpm` | `double` | *(optional)* Flywheel speed for this shot. Logged to CSV. |
| `robot/shot_angle_deg` | `double` | *(optional)* Hood/pivot angle for this shot. Logged to CSV. |

### Camera → Robot  (written by camera code)

| Key | Type | Description |
|-----|------|-------------|
| `camera/state` | `string` | Current state: `CALIBRATING`, `WAITING`, `ARMED`, `TRACKING`, `COOLDOWN`, `DONE` |
| `camera/ok_to_shoot` | `boolean` | **true** = camera is ready for the next shot. Robot must wait for this. |
| `camera/ball_detected` | `boolean` | **true** = ball has been detected in flight (robot can stop shooter). |
| `camera/shot_count` | `int` | Total shots tracked so far this session. |
| `camera/last_land_x_cm` | `double` | X landing position of the most recent shot (cm). |
| `camera/last_land_y_cm` | `double` | Y landing position of the most recent shot (cm). |
| `camera/fps` | `double` | Current processing FPS. |

---

## State Machine

```
                  calibration
    ┌──────────┐  complete    ┌─────────┐
    │CALIBRATING├────────────►│ WAITING  │◄─────────────┐
    └──────────┘              └────┬─────┘              │
                                   │                    │
                        robot sets │                    │
                       shoot_ready │                    │
                                   ▼                    │
                              ┌────────┐                │
                              │ ARMED  │                │
                              └───┬────┘                │
                                  │                     │
                       ball first │                     │
                        detected  │                     │
                                  ▼                     │
                             ┌─────────┐                │
                             │TRACKING │                │
                             └───┬─────┘                │
                                 │                      │
                      ball below │                      │
                    start height │                      │
                      or lost    │                      │
                                 ▼                      │
                            ┌──────────┐                │
                            │ COOLDOWN │                │
                            └───┬──────┘                │
                                │                       │
                     ball gone  │                       │
                      for 2 s   │                       │
                                ▼                       │
                             ┌───────┐   save JSON,     │
                             │ DONE  ├──────────────────┘
                             └───────┘  set ok_to_shoot
```

### State details

| State | Camera action | Robot action | Exit condition |
|-------|--------------|--------------|----------------|
| **CALIBRATING** | Detect AprilTags, collect PnP poses | Idle — do not shoot | `calib_time` seconds elapsed and ≥1 pose collected |
| **WAITING** | Set `ok_to_shoot = true`, `ball_detected = false`, `shoot_ready = false` | Prepare for next shot (spin up flywheel, set angle, etc.). Set `shoot_ready = true` when ready. | `robot/shoot_ready` becomes **true** |
| **ARMED** | Set `ok_to_shoot = false` | **Slowly spin indexer** to feed ball into shooter. Do **not** fire full-speed. | Ball detected in frame (HSV + contour) |
| **TRACKING** | Set `ball_detected = true`; record trajectory | **Stop all shooting** — stop indexer, stop flywheel. Ball is in flight; do not interfere. | Ball pixel-Y exceeds start pixel-Y (fell below launch height) **OR** ball lost for >0.5 s |
| **COOLDOWN** | Stop recording trajectory; wait for ball to leave frame | **Stay stopped** — ball may be bouncing, do not shoot. | Ball absent from frame for **2 s** continuously |
| **DONE** | Save trajectory to JSON; publish landing position; increment `shot_count` | Idle — wait for next cycle | Immediate transition → **WAITING** |

### Robot behaviour summary

The robot reads `camera/state` to decide motor actions:

| `camera/state` | Flywheel | Indexer | Notes |
|----------------|----------|---------|-------|
| `CALIBRATING`  | Off      | Off     | Camera is still calibrating |
| `WAITING`      | Spin up  | Off     | Prepare — set speed/angle, then set `shoot_ready` |
| `ARMED`        | Hold RPM | **Slow** | Slowly feed ball toward shooter; camera is watching for it |
| `TRACKING`     | **Off**  | **Off** | Ball in flight — do NOT shoot another ball |
| `COOLDOWN`     | **Off**  | **Off** | Ball may be bouncing — stay stopped until it's gone |
| `DONE`         | Off      | Off     | Momentary — resets to WAITING immediately |

---

## Trajectory end condition

The ball is tracked in **pixel space** (pixel-Y):

1. When the ball is first detected, record `start_py` (its pixel-Y).
2. The ball travels upward → pixel-Y *decreases*.
3. The ball must first travel upward by at least **20 px** (debounce).
4. Once the ball's pixel-Y exceeds `start_py` again, it has fallen below
   its original height → **stop recording** trajectory (→ COOLDOWN).
5. Safety timeout: if ball is lost for **0.5 s** during flight, also stop
   recording (→ COOLDOWN).
6. **Bounce handling (COOLDOWN):** after recording stops, the ball may
   bounce and remain visible.  The camera ignores it — no more trajectory
   points are recorded.  Only once the ball has been **completely absent
   from the frame for 2 seconds** does the shot finalize (→ DONE → WAITING).

---

## JSON Output

All shots are stored in a single file: **`shots.json`** — an array of shot
objects.  The file is rewritten atomically after each shot so it always
contains the full session history (and survives restarts — existing shots
are loaded on startup).

```jsonc
[
  {
    "shot_id":    1,
    "timestamp":  1708200000.123,
    "speed_rpm":  3500.0,
    "angle_deg":  45.0,
    "n_frames":   42,
    "duration_s": 1.234,
    "land_x_cm":  108.5,
    "land_y_cm":  -32.6,
    "trajectory": [
      { "frame": 0, "time_s": 0.0000, "px_x": 640.0, "px_y": 400.0,
        "world_x_cm": 50.0, "world_y_cm": 10.0 },
      { "frame": 1, "time_s": 0.0333, "px_x": 638.2, "px_y": 385.1,
        "world_x_cm": 52.1, "world_y_cm": 11.3 }
      // … one entry per tracked frame
    ]
  }
  // … one object per shot
]
```

`time_s` is relative to the first detection frame (0.000, 0.033, 0.066 …).

---

## Connection

| Parameter | Value |
|-----------|-------|
| Protocol | NT4 (WebSocket) |
| Team | 41 |
| Server | roboRIO (10.0.41.2) |
| Client identity | `"balltracker"` |
| Fallback | `--nt-server IP` flag for bench testing |

If the roboRIO is unreachable the camera still runs — it just won't
coordinate shots.  A `--no-nt` flag disables NT entirely for standalone
testing.

---