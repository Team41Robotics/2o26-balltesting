"""Visualize shots.json trajectory data with matplotlib."""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

SHOTS_FILE = "shots.json"

with open(SHOTS_FILE) as f:
    shots = json.load(f)

if not shots:
    print("No shots in shots.json")
    sys.exit(1)

colors = cm.tab10(np.linspace(0, 1, len(shots)))

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(f"Ball Trajectory Analysis — {len(shots)} shot(s)", fontsize=14, fontweight="bold")

# ── Plot 1: Top-down field view (X vs Y) ──────────────────────────────────────
ax1 = axes[0]
ax1.set_title("Top-Down Field View")
ax1.set_xlabel("World X (cm)  →  right")
ax1.set_ylabel("World Y (cm)  →  depth")

for i, shot in enumerate(shots):
    traj = shot["trajectory"]
    xs = [p["world_x_cm"] for p in traj]
    ys = [p["world_y_cm"] for p in traj]
    label = f"Shot {shot['shot_id']} (t={shot['timestamp']:.0f})"
    ax1.plot(xs, ys, "o-", color=colors[i], label=label, markersize=3, linewidth=1.5)
    # Mark start and landing
    ax1.plot(xs[0], ys[0], "^", color=colors[i], markersize=8)
    ax1.plot(shot["land_x_cm"], shot["land_y_cm"], "X", color=colors[i], markersize=10)

ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Tag line (Y=0)")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=7, loc="best")

# ── Plot 2: X position vs time ────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_title("X Position vs Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("World X (cm)")

for i, shot in enumerate(shots):
    traj = shot["trajectory"]
    ts = [p["time_s"] for p in traj]
    xs = [p["world_x_cm"] for p in traj]
    ax2.plot(ts, xs, "o-", color=colors[i], markersize=3, linewidth=1.5,
             label=f"Shot {shot['shot_id']}")

ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7)

# ── Plot 3: Y position vs time ────────────────────────────────────────────────
ax3 = axes[2]
ax3.set_title("Y (Depth) Position vs Time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("World Y (cm)")

for i, shot in enumerate(shots):
    traj = shot["trajectory"]
    ts = [p["time_s"] for p in traj]
    ys = [p["world_y_cm"] for p in traj]
    ax3.plot(ts, ys, "o-", color=colors[i], markersize=3, linewidth=1.5,
             label=f"Shot {shot['shot_id']}")

ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=7)

# ── Summary text ──────────────────────────────────────────────────────────────
summary_lines = []
for shot in shots:
    summary_lines.append(
        f"Shot {shot['shot_id']}: {shot['n_frames']} frames, "
        f"{shot['duration_s']:.3f}s, "
        f"land=({shot['land_x_cm']:.1f}, {shot['land_y_cm']:.1f}) cm, "
        f"{shot['speed_rpm']:.0f} RPM"
    )
print("\n".join(summary_lines))

plt.tight_layout()
plt.savefig("shots_plot.png", dpi=150, bbox_inches="tight")
print(f"\nSaved shots_plot.png")
plt.show()
