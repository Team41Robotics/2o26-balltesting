"""Analyze PhotonVision calibration JSON for quality assessment."""
import json, math, sys, os
import numpy as np

JSON_PATH = r"C:\Users\Robotics41\Downloads\photon_calibration_95d7cf32-22f6-4d6a-86c6-818f62277d71_1280x720.json"

print(f"Loading: {JSON_PATH}")
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# ── Discover top-level keys ────────────────────────────────────────
print(f"\nTop-level type: {type(data).__name__}")
if isinstance(data, dict):
    print(f"Top-level keys: {list(data.keys())}")
    observations = data.get("observations", data.get("images", []))
    calib_size = data.get("calobjectSize")
    spacing = data.get("calobjectSpacing")
    lens_model = data.get("lensmodel")
    intrinsics = data.get("cameraIntrinsics") or data.get("intrinsics") or data.get("camera_matrix")
    distortion = data.get("distCoeffs") or data.get("distortion") or data.get("dist_coeffs")
    print(f"\ncalobjectSize: {calib_size}")
    print(f"calobjectSpacing: {spacing}")
    print(f"lensmodel: {lens_model}")
    print(f"intrinsics key found: {intrinsics is not None}")
    print(f"distortion key found: {distortion is not None}")
    if intrinsics is not None:
        print(f"  intrinsics: {intrinsics}")
    if distortion is not None:
        print(f"  distortion: {distortion}")
elif isinstance(data, list):
    print(f"Top-level is a list with {len(data)} elements")
    # Check if last few elements are metadata
    if len(data) > 0 and isinstance(data[-1], dict):
        last = data[-1]
        if "snapshotName" in last:
            print("All elements appear to be per-image observations")
        else:
            print(f"Last element keys: {list(last.keys())}")
    observations = data
    calib_size = None
    spacing = None
    lens_model = None
else:
    print("Unknown structure")
    sys.exit(1)

# ── Per-image analysis ─────────────────────────────────────────────
print("\n" + "="*70)
print("PER-IMAGE REPROJECTION ERROR ANALYSIS")
print("="*70)

all_errors = []
per_image_stats = []
images_with_no_errors = []
images_with_few_corners = []
total_corners_detected = 0
total_corners_inlier = 0
total_corners_outlier = 0

for i, obs in enumerate(observations):
    if not isinstance(obs, dict):
        continue
    if "snapshotName" not in obs:
        continue
    
    name = obs["snapshotName"]
    reproj = obs.get("reprojectionErrors", [])
    corners_used = obs.get("cornersUsed", [])
    n_corners = sum(corners_used) if corners_used else 0
    n_total = len(corners_used) if corners_used else 0
    n_inlier = len(reproj)
    n_outlier = n_corners - n_inlier
    
    total_corners_detected += n_corners
    total_corners_inlier += n_inlier
    total_corners_outlier += max(0, n_outlier)
    
    if not reproj:
        images_with_no_errors.append((name, n_corners, n_total))
        continue
    
    # Compute per-image errors
    errs = []
    for e in reproj:
        ex, ey = e["x"], e["y"]
        errs.append(math.sqrt(ex**2 + ey**2))
        all_errors.append((ex, ey, math.sqrt(ex**2 + ey**2), name))
    
    rms = math.sqrt(sum(e**2 for e in errs) / len(errs))
    mean_err = sum(errs) / len(errs)
    max_err = max(errs)
    
    per_image_stats.append({
        "name": name,
        "n_corners_used": n_corners,
        "n_total_corners": n_total,
        "n_errors": len(errs),
        "n_outlier": max(0, n_outlier),
        "outlier_pct": 100 * max(0, n_outlier) / n_corners if n_corners > 0 else 0,
        "rms": rms,
        "mean": mean_err,
        "max": max_err,
    })
    
    if n_corners < 6:
        images_with_few_corners.append((name, n_corners, n_total))

# Sort by RMS descending
per_image_stats.sort(key=lambda x: x["rms"], reverse=True)

print(f"\nTotal images: {len(observations)}")
print(f"Images with reprojection errors: {len(per_image_stats)}")
print(f"Images with NO errors (all outliers): {len(images_with_no_errors)}")

print(f"\n--- MRCAL OUTLIER REJECTION ---")
print(f"Total corners detected across all images: {total_corners_detected}")
print(f"Corners kept as INLIERS:   {total_corners_inlier} ({100*total_corners_inlier/total_corners_detected:.1f}%)")
print(f"Corners REJECTED as outliers: {total_corners_outlier} ({100*total_corners_outlier/total_corners_detected:.1f}%)")
print(f"Entire images rejected (0 inliers): {len(images_with_no_errors)}/{len(observations)} ({100*len(images_with_no_errors)/len(observations):.1f}%)")

# Show outlier rate per image
per_image_stats.sort(key=lambda x: x["outlier_pct"], reverse=True)
print(f"\n--- TOP 20 IMAGES BY OUTLIER REJECTION RATE ---")
print(f"{'Image':<15} {'Detected':>8} {'Inlier':>6} {'Outlier':>7} {'Out%':>6} {'RMS':>8}")
for s in per_image_stats[:20]:
    print(f"{s['name']:<15} {s['n_corners_used']:>8} {s['n_errors']:>6} {s['n_outlier']:>7}"
          f" {s['outlier_pct']:>5.1f}% {s['rms']:>8.2f}")

# Re-sort by RMS for remaining output
per_image_stats.sort(key=lambda x: x["rms"], reverse=True)

print(f"\n--- TOP 15 WORST IMAGES (by RMS) ---")
print(f"{'Image':<15} {'Corners':>7} {'Errors':>6} {'RMS':>8} {'Mean':>8} {'Max':>8}")
for s in per_image_stats[:15]:
    print(f"{s['name']:<15} {s['n_corners_used']:>4}/{s['n_total_corners']:<3}"
          f" {s['n_errors']:>5} {s['rms']:>8.2f} {s['mean']:>8.2f} {s['max']:>8.2f}")

print(f"\n--- TOP 15 BEST IMAGES (by RMS) ---")
for s in per_image_stats[-15:]:
    print(f"{s['name']:<15} {s['n_corners_used']:>4}/{s['n_total_corners']:<3}"
          f" {s['n_errors']:>5} {s['rms']:>8.2f} {s['mean']:>8.2f} {s['max']:>8.2f}")

# ── Overall statistics ─────────────────────────────────────────────
print("\n" + "="*70)
print("OVERALL REPROJECTION ERROR STATISTICS")
print("="*70)

if all_errors:
    all_dists = [e[2] for e in all_errors]
    all_x = [e[0] for e in all_errors]
    all_y = [e[1] for e in all_errors]
    
    overall_rms = math.sqrt(sum(d**2 for d in all_dists) / len(all_dists))
    mean_dist = sum(all_dists) / len(all_dists)
    median_dist = sorted(all_dists)[len(all_dists)//2]
    max_dist = max(all_dists)
    max_err_entry = max(all_errors, key=lambda e: e[2])
    
    print(f"Total corner measurements: {len(all_dists)}")
    print(f"Overall RMS error:  {overall_rms:.4f} px")
    print(f"Mean error:         {mean_dist:.4f} px")
    print(f"Median error:       {median_dist:.4f} px")
    print(f"Max error:          {max_dist:.4f} px (in {max_err_entry[3]})")
    print(f"Mean X error:       {sum(all_x)/len(all_x):.4f} px")
    print(f"Mean Y error:       {sum(all_y)/len(all_y):.4f} px")
    print(f"Std X error:        {np.std(all_x):.4f} px")
    print(f"Std Y error:        {np.std(all_y):.4f} px")
    
    # Error distribution
    print(f"\n--- ERROR DISTRIBUTION ---")
    thresholds = [0.5, 1, 2, 5, 10, 15, 20, 30, 50]
    for t in thresholds:
        count = sum(1 for d in all_dists if d <= t)
        pct = 100 * count / len(all_dists)
        print(f"  ≤ {t:>5.1f} px: {count:>5} / {len(all_dists)} ({pct:.1f}%)")

# ── Images with no errors ─────────────────────────────────────────
if images_with_no_errors:
    print(f"\n--- IMAGES WITH EMPTY REPROJECTION ERRORS ({len(images_with_no_errors)}) ---")
    for name, nc, nt in images_with_no_errors[:20]:
        print(f"  {name}: {nc}/{nt} corners used")

# ── Images with few corners ───────────────────────────────────────
if images_with_few_corners:
    print(f"\n--- IMAGES WITH < 6 CORNERS ({len(images_with_few_corners)}) ---")
    for name, nc, nt in images_with_few_corners[:20]:
        print(f"  {name}: {nc}/{nt} corners used")

# ── Corner coverage analysis ──────────────────────────────────────
print(f"\n--- CORNERS USED DISTRIBUTION ---")
corner_counts = [s["n_corners_used"] for s in per_image_stats]
if corner_counts:
    print(f"  Min corners used: {min(corner_counts)}")
    print(f"  Max corners used: {max(corner_counts)}")
    print(f"  Mean corners used: {sum(corner_counts)/len(corner_counts):.1f}")
    print(f"  Images with <10 corners: {sum(1 for c in corner_counts if c < 10)}")
    print(f"  Images with <20 corners: {sum(1 for c in corner_counts if c < 20)}")
    print(f"  Images with ≥30 corners: {sum(1 for c in corner_counts if c >= 30)}")

# ── Quality Assessment ────────────────────────────────────────────
print("\n" + "="*70)
print("CALIBRATION QUALITY ASSESSMENT")
print("="*70)

if all_errors:
    if overall_rms < 0.5:
        quality = "EXCELLENT"
    elif overall_rms < 1.0:
        quality = "GOOD"
    elif overall_rms < 2.0:
        quality = "ACCEPTABLE"
    elif overall_rms < 5.0:
        quality = "POOR"
    else:
        quality = "VERY POOR / UNUSABLE"
    
    print(f"\n  Overall RMS: {overall_rms:.2f} px → Quality: {quality}")
    print(f"\n  NOTE: These are the reprojection errors from the MRCAL/PhotonVision")
    print(f"  optimization, NOT OpenCV. Good calibration should have RMS < 1.0 px.")
    print(f"  Values > 5 px indicate the model is not fitting the data well.")
    
    if overall_rms > 5:
        print(f"\n  ⚠️  CRITICAL: RMS = {overall_rms:.2f} px is extremely high!")
        print(f"  This means the LENSMODEL_OPENCV ({lens_model}) distortion model")
        print(f"  cannot adequately represent this camera's actual distortion.")
        print(f"  The calibration is essentially UNUSABLE for precision work.")
