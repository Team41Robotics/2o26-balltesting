"""
process_shots.py — Post-process shots.json into (angle, speed, distance, height, tof) tuples.

All spatial units are METRES.  Time is in seconds.

Computed fields
───────────────
distance_m
    Straight-line XY distance (m) between the ball's starting world position
    and the last trajectory point still "above" the AprilTag 0-1 line.

    Tag layout:
        Tag 0: (0.0 m, 0.0 m) ──── Tag 1: (1.0 m, 0.0 m)  ← the 0-1 line (Y = 0)
        Tag 3: (0.0 m, 1.0 m)      Tag 2: (1.0 m, 1.0 m)

    "Above the 0-1 line" means world_y > 0.

peak_rise_px
    Maximum upward pixel displacement during the trajectory.
    Computed as:  start_px_y − min(px_y across all frames).
    Lower px_y = higher in the image = higher physical position.
    Reported in pixels because converting to metres requires per-shot
    camera depth, which is not stored in shots.json.

time_of_flight_s
    Duration from first detection to the last tracked frame that is still
    above the 0-1 line (seconds).

Projectile motion model
───────────────────────
Assumes constant-gravity ideal projectile motion and a linear
flywheel-speed → launch-speed relationship:

    v0 = k · rpm                          (k in m s⁻¹ / RPM)

The fit uses every individual trajectory point, not just per-shot
summaries.  For each frame j in shot i the horizontal distance from
the launch position evolves as:

    d(t_j) = k · rpm_i · cos(θ_i + δ) · t_j          (per-point model)

Two free parameters are fitted globally across all frames of all shots:

    k     — launch speed per RPM  (m s⁻¹ / RPM)
    δ     — systematic hood-angle → launch-angle offset  (degrees)

Derived quantities reported per shot:

    distance_pred  = k² · rpm² · sin(2·(θ+δ)) / g
    height_pred    = k² · rpm² · sin²(θ+δ)   / (2g)
    tof_pred       = 2 · k · rpm · sin(θ+δ)  / g

Usage:
    python process_shots.py                  # reads shots.json in same dir
    python process_shots.py path/to/shots.json
"""

import json
import math
import os
import sys
from typing import NamedTuple

import numpy as np

G = 9.81  # m/s²


# ── Data model ─────────────────────────────────────────────────────────────

class ShotResult(NamedTuple):
    shot_id:          int
    angle_deg:        float
    speed_rpm:        float
    distance_m:       float   # straight-line XY distance in metres
    peak_rise_px:     float   # pixels; start_py − min(px_y) — height proxy
    time_of_flight_s: float   # seconds, first frame → last frame above 0-1 line
    start_x_m:        float
    start_y_m:        float
    end_x_m:          float
    end_y_m:          float
    n_traj_pts:       int


# ── Core processing ────────────────────────────────────────────────────────

def process_shot(shot: dict) -> ShotResult | None:
    """
    Extract one ShotResult from a shot dict.
    Returns None if the shot has no usable trajectory.
    """
    traj = shot.get("trajectory", [])
    if not traj:
        return None

    # Starting position (first frame)
    start = traj[0]
    sx = start["world_x_cm"] / 100.0
    sy = start["world_y_cm"] / 100.0
    start_py = start["px_y"]

    # ── Distance: last point still above the tag 0-1 line (Y = 0) ──
    last_above = None
    for pt in traj:
        if pt["world_y_cm"] > 0.0:
            last_above = pt

    if last_above is None:
        return None  # ball never crossed above Y = 0; skip

    ex = last_above["world_x_cm"] / 100.0
    ey = last_above["world_y_cm"] / 100.0
    distance_m = math.hypot(ex - sx, ey - sy)

    # ── Peak height proxy: max pixel rise ──
    min_py = min(pt["px_y"] for pt in traj)
    peak_rise_px = start_py - min_py  # positive = ball rose

    # ── Time of flight to the last above-line frame ──
    time_of_flight_s = last_above["time_s"] - start["time_s"]

    return ShotResult(
        shot_id=shot.get("shot_id", -1),
        angle_deg=shot.get("angle_deg", 0.0),
        speed_rpm=shot.get("speed_rpm", 0.0),
        distance_m=round(distance_m, 4),
        peak_rise_px=round(peak_rise_px, 1),
        time_of_flight_s=round(time_of_flight_s, 4),
        start_x_m=round(sx, 4),
        start_y_m=round(sy, 4),
        end_x_m=round(ex, 4),
        end_y_m=round(ey, 4),
        n_traj_pts=len(traj),
    )


def process_shots_file(path: str) -> tuple[list[ShotResult], list[dict]]:
    """Load shots.json and return (results, raw_shots).

    raw_shots is the full list of shot dicts, needed for the all-points fit.
    Only shots that produce a valid ShotResult are included in both lists
    (they are kept in sync by index).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"shots.json not found at: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("shots.json must contain a JSON array at the top level")

    results, raw_shots, skipped = [], [], 0
    for shot in data:
        r = process_shot(shot)
        if r is None:
            skipped += 1
        else:
            results.append(r)
            raw_shots.append(shot)

    print(f"Processed {len(results)} shot(s), skipped {skipped} "
          "(no trajectory points above the 0-1 line).")
    return results, raw_shots


# ── Projectile motion model ────────────────────────────────────────────────

def _build_allpoints_arrays(
    results: list[ShotResult], raw_shots: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays from every trajectory point in every shot.

    X columns: [angle_deg, speed_rpm, time_s]
    y        : horizontal distance from shot start (m)
    """
    rows_X, rows_y = [], []
    for result, shot in zip(results, raw_shots):
        traj = shot["trajectory"]
        t0   = traj[0]["time_s"]
        sx   = traj[0]["world_x_cm"] / 100.0
        sy   = traj[0]["world_y_cm"] / 100.0
        for pt in traj:
            t = pt["time_s"] - t0
            x = pt["world_x_cm"] / 100.0
            y = pt["world_y_cm"] / 100.0
            d = math.hypot(x - sx, y - sy)
            rows_X.append([result.angle_deg, result.speed_rpm, t])
            rows_y.append(d)
    return np.array(rows_X, dtype=np.float64), np.array(rows_y, dtype=np.float64)


def _model_allpoints(X: np.ndarray, k: float, delta_deg: float) -> np.ndarray:
    """
    Per-trajectory-point projectile model.

    d(t) = k · rpm · cos(θ + δ) · t

    X columns: [angle_deg, speed_rpm, time_s]
    """
    theta = np.deg2rad(X[:, 0] + delta_deg)
    rpm   = X[:, 1]
    t     = X[:, 2]
    return k * rpm * np.cos(theta) * t


def fit_projectile_model(
    results: list[ShotResult], raw_shots: list[dict]
) -> dict | None:
    """
    Fit k (launch speed per RPM) and δ (angle offset) using every
    trajectory point from every shot.

    Model per frame:  d(t) = k · rpm · cos(θ + δ) · t

    Returns a results dict, or None on failure.
    """
    if len(results) < 2:
        print("[FIT] Need ≥ 2 shots to fit the model.")
        return None

    try:
        from scipy.optimize import curve_fit
    except ImportError:
        print("[FIT] scipy not available — install it with: pip install scipy")
        return None

    X, y = _build_allpoints_arrays(results, raw_shots)
    n_pts = len(y)
    print(f"[FIT] Fitting over {n_pts} trajectory points "
          f"from {len(results)} shot(s).")

    try:
        (k_fit, delta_fit), pcov = curve_fit(
            _model_allpoints, X, y,
            p0=[1e-4, 0.0],
            bounds=([0.0, -45.0], [np.inf, 45.0]),
            maxfev=20_000,
        )
    except RuntimeError as exc:
        print(f"[FIT] curve_fit failed: {exc}")
        return None

    perr = np.sqrt(np.diag(pcov))
    k_err, delta_err = perr[0], perr[1]

    # ── R² on all trajectory points ──
    y_pred_all = _model_allpoints(X, k_fit, delta_fit)
    ss_res = float(np.sum((y - y_pred_all) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_allpts = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # ── Derived per-shot predictions ──
    C_fit = k_fit ** 2 / G
    angles = np.array([r.angle_deg        for r in results])
    rpms   = np.array([r.speed_rpm        for r in results])
    tofs   = np.array([r.time_of_flight_s for r in results])
    dists  = np.array([r.distance_m       for r in results])

    theta = np.deg2rad(angles + delta_fit)
    dist_pred   = C_fit * rpms**2 * np.sin(2.0 * theta)
    height_pred = (C_fit / 2.0) * rpms**2 * np.sin(theta)**2
    tof_pred    = 2.0 * k_fit * rpms * np.sin(theta) / G

    dist_residuals = dists - dist_pred

    # R² on per-shot endpoint distances (informational)
    ss_res_d = float(np.sum(dist_residuals**2))
    ss_tot_d = float(np.sum((dists - dists.mean())**2))
    r2_dist  = 1.0 - ss_res_d / ss_tot_d if ss_tot_d > 0 else float("nan")

    # R² on per-shot TOF
    tof_resid  = tofs - tof_pred
    ss_res_tof = float(np.sum(tof_resid**2))
    ss_tot_tof = float(np.sum((tofs - tofs.mean())**2))
    r2_tof     = 1.0 - ss_res_tof / ss_tot_tof if ss_tot_tof > 0 else float("nan")

    return {
        "k":             k_fit,
        "k_err":         k_err,
        "delta_deg":     delta_fit,
        "delta_deg_err": delta_err,
        "C":             C_fit,          # = k²/g, for reference
        "n_points":      n_pts,
        "r2_allpoints":  r2_allpts,
        "r2_distance":   r2_dist,
        "r2_tof":        r2_tof,
        "dist_pred":     dist_pred.tolist(),
        "dist_residuals":dist_residuals.tolist(),
        "height_pred_m": height_pred.tolist(),
        "tof_pred_s":    tof_pred.tolist(),
    }


def print_fit_report(results: list[ShotResult], fit: dict) -> None:
    """Print the projectile model fit summary."""
    print("\n" + "=" * 72)
    print("PROJECTILE MOTION MODEL FIT  (all trajectory points)")
    print("=" * 72)
    print(f"  Per-point model:  d(t) = k · rpm · cos(θ + δ) · t")
    print(f"  Derived range:    R    = k² · rpm² · sin(2·(θ+δ)) / g")
    print()
    print(f"  k         = {fit['k']:.6e} ± {fit['k_err']:.2e}  m·s⁻¹/RPM")
    print(f"  δ (offset)= {fit['delta_deg']:+.3f} ± {fit['delta_deg_err']:.3f}  °")
    print(f"  C (=k²/g) = {fit['C']:.4e}  m/RPM²")
    print()
    print(f"  R² (all {fit['n_points']} traj points) = {fit['r2_allpoints']:.4f}")
    print(f"  R² (per-shot endpoint distances)      = {fit['r2_distance']:.4f}")
    print(f"  R² (per-shot time-of-flight)          = {fit['r2_tof']:.4f}")
    print()

    # Per-shot breakdown
    col_w = 70
    hdr = (f"{'ID':>4}  {'θ(°)':>7}  {'RPM':>7}  "
           f"{'dist_meas':>10}  {'dist_pred':>10}  {'resid':>8}  "
           f"{'h_pred(m)':>10}  {'tof_meas':>9}  {'tof_pred':>9}")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r, dp, dr, hp, tp in zip(
        results,
        fit["dist_pred"], fit["dist_residuals"],
        fit["height_pred_m"], fit["tof_pred_s"],
    ):
        print(
            f"{r.shot_id:>4}  {r.angle_deg:>7.2f}  {r.speed_rpm:>7.0f}  "
            f"{r.distance_m:>10.4f}  {dp:>10.4f}  {dr:>+8.4f}  "
            f"{hp:>10.4f}  {r.time_of_flight_s:>9.3f}  {tp:>9.3f}"
        )
    print("-" * len(hdr))


# ── Output helpers ─────────────────────────────────────────────────────────

def print_table(results: list[ShotResult]) -> None:
    """Print a human-readable table of measured results."""
    if not results:
        print("No valid shots to display.")
        return

    header = (
        f"{'ID':>4}  {'Angle (°)':>10}  {'Speed (rpm)':>12}  "
        f"{'Distance (m)':>13}  {'Rise (px)':>10}  {'ToF (s)':>8}  "
        f"{'Start (x,y) m':>18}  {'End (x,y) m':>18}  {'Pts':>4}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.shot_id:>4}  {r.angle_deg:>10.2f}  {r.speed_rpm:>12.1f}  "
            f"{r.distance_m:>13.4f}  {r.peak_rise_px:>10.1f}  "
            f"{r.time_of_flight_s:>8.3f}  "
            f"{f'({r.start_x_m:.3f}, {r.start_y_m:.3f})':>18}  "
            f"{f'({r.end_x_m:.3f}, {r.end_y_m:.3f})':>18}  "
            f"{r.n_traj_pts:>4}"
        )
    print(sep)


def as_tuples(
    results: list[ShotResult],
) -> list[tuple[float, float, float, float, float]]:
    """Return (angle_deg, speed_rpm, distance_m, peak_rise_px, time_of_flight_s)."""
    return [
        (r.angle_deg, r.speed_rpm, r.distance_m, r.peak_rise_px, r.time_of_flight_s)
        for r in results
    ]


def save_csv(results: list[ShotResult], fit: dict | None, out_path: str) -> None:
    """Write results (and optional model predictions) to a CSV file."""
    import csv

    has_fit = fit is not None
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        base_cols = [
            "shot_id", "angle_deg", "speed_rpm",
            "distance_m", "peak_rise_px", "time_of_flight_s",
            "start_x_m", "start_y_m", "end_x_m", "end_y_m",
            "n_traj_pts",
        ]
        pred_cols = ["dist_pred_m", "dist_resid_m",
                     "height_pred_m", "tof_pred_s"] if has_fit else []
        writer.writerow(base_cols + pred_cols)

        for i, r in enumerate(results):
            row = [
                r.shot_id, r.angle_deg, r.speed_rpm,
                r.distance_m, r.peak_rise_px, r.time_of_flight_s,
                r.start_x_m, r.start_y_m, r.end_x_m, r.end_y_m,
                r.n_traj_pts,
            ]
            if has_fit:
                row += [
                    round(fit["dist_pred"][i], 4),
                    round(fit["dist_residuals"][i], 4),
                    round(fit["height_pred_m"][i], 4),
                    round(fit["tof_pred_s"][i], 4),
                ]
            writer.writerow(row)

    print(f"CSV saved to: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    shots_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "shots.json")
    )

    results, raw_shots = process_shots_file(shots_path)
    print_table(results)

    tuples = as_tuples(results)
    print("\n(angle_deg, speed_rpm, distance_m, peak_rise_px, time_of_flight_s) tuples:")
    for t in tuples:
        print(f"  {t}")

    fit = fit_projectile_model(results, raw_shots)
    if fit:
        print_fit_report(results, fit)

    csv_path = os.path.splitext(shots_path)[0] + "_processed.csv"
    save_csv(results, fit, csv_path)


if __name__ == "__main__":
    main()
