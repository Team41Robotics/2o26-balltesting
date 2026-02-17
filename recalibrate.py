"""
recalibrate.py — Re-run camera calibration from PhotonVision ChArUco images
and compare 5-param vs 8-param distortion models.

Board config (from PhotonVision log):
  ChArUcoBoard  8×8 squares, Dict_4X4_1000
  squareSize = 1.0 in  = 0.0254 m
  markerSize = 0.75 in = 0.01905 m

PhotonVision used MRCAL (splined stereographic model) then converted to
OpenCV 8-coeff rational format — that conversion is likely why it's off.
mrcal is Linux-only so we compare OpenCV 5-param vs 8-param here.
"""

import cv2
import numpy as np
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Board parameters ────────────────────────────────────────────
SQUARES_X       = 8
SQUARES_Y       = 8
SQUARE_LENGTH   = 0.0254      # 1.0 inch in metres
MARKER_LENGTH   = 0.01905     # 0.75 inch in metres
ARUCO_DICT      = cv2.aruco.DICT_4X4_1000

MAX_IMAGES      = 30          # subsample to this many for speed

IMG_DIR = r"photonvision_config\calibration\95d7cf32-22f6-4d6a-86c6-818f62277d71\imgs\1280x720"

# PhotonVision / MRCAL result for reference
PV_K = np.array([[545.1385498353699, 0.0,               619.6410585498498],
                 [0.0,               545.0370023702498,  384.2873791498498],
                 [0.0,               0.0,               1.0]])
PV_D = np.array([0.20868612511672218, -0.03661075688672098,
                 0.00023831497063498988, -0.00012144558658498987,
                 -0.0008276839786269898, 0.5646002553667212,
                 -0.03300946316362098,  -0.007130591896524989])

# ── Helpers ─────────────────────────────────────────────────────
def make_board():
    """Create the ChArUco board object."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                   SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    return dictionary, board


# Global detector — created once, reused across threads
_dictionary, _board = make_board()
_detector = cv2.aruco.ArucoDetector(
    _dictionary, cv2.aruco.DetectorParameters()
)


def _detect_one(path):
    """Detect ChArUco corners in a single image. Thread-safe."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    marker_corners, marker_ids, _ = _detector.detectMarkers(img)
    if marker_ids is None or len(marker_ids) < 4:
        return None
    ret, corners, ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, img, _board
    )
    if ret < 6:
        return None
    return (corners, ids, (w, h), path)


def detect_charuco_parallel(img_paths, max_workers=8):
    """Detect ChArUco corners in parallel using threads."""
    all_corners, all_ids, good_paths = [], [], []
    img_size = None
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_detect_one, p): p for p in img_paths}
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result is not None:
                corners, ids, sz, path = result
                all_corners.append(corners)
                all_ids.append(ids)
                good_paths.append(path)
                if img_size is None:
                    img_size = sz
            if done % 20 == 0:
                print(f"  {done}/{len(img_paths)} done  "
                      f"({len(all_corners)} usable)", flush=True)

    print(f"  Detection complete: {len(all_corners)}/{len(img_paths)} usable")
    return all_corners, all_ids, img_size, good_paths


def calibrate_charuco(charuco_corners, charuco_ids, board, img_size, flags):
    """Run OpenCV ChArUco calibration with given flags."""
    ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charuco_corners, charuco_ids, board, img_size,
        None, None, flags=flags
    )
    return ret, K, D.ravel(), rvecs, tvecs


def per_image_errors(charuco_corners, charuco_ids, board, K, D, rvecs, tvecs):
    """Compute per-image reprojection error."""
    errors = []
    obj_points_board = board.getChessboardCorners()

    for i in range(len(charuco_corners)):
        ids = charuco_ids[i].ravel()
        obj_pts = obj_points_board[ids].reshape(-1, 1, 3)
        img_pts = charuco_corners[i].reshape(-1, 1, 2)

        projected, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], K, D)
        err = np.sqrt(np.mean((projected.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2))
        errors.append(err)
    return np.array(errors)


def print_results(label, rms, K, D, errors, good_paths):
    """Pretty-print calibration results."""
    n_coeffs = len(D)
    print(f"\n{'='*60}")
    print(f"  {label}  ({n_coeffs} distortion coefficients)")
    print(f"{'='*60}")
    print(f"  RMS reprojection error: {rms:.4f} px")
    print(f"  fx = {K[0,0]:.2f}   fy = {K[1,1]:.2f}   fx/fy = {K[0,0]/K[1,1]:.4f}")
    print(f"  cx = {K[0,2]:.2f}   cy = {K[1,2]:.2f}")
    print(f"  Distortion: {np.array2string(D, precision=6, separator=', ')}")

    # getOptimalNewCameraMatrix test
    img_size = (1280, 720)
    K0, _ = cv2.getOptimalNewCameraMatrix(K, D, img_size, alpha=0, newImgSize=img_size)
    print(f"\n  getOptimalNewCameraMatrix(alpha=0):")
    print(f"    fx = {K0[0,0]:.2f}   fy = {K0[1,1]:.2f}   ratio = {K0[0,0]/K0[1,1]:.4f}")

    # Per-image error stats
    print(f"\n  Per-image error stats (px):")
    print(f"    mean={errors.mean():.3f}  median={np.median(errors):.3f}  "
          f"std={errors.std():.3f}  min={errors.min():.3f}  max={errors.max():.3f}")

    # Worst 10 images
    worst_idx = np.argsort(errors)[-10:][::-1]
    print(f"\n  Top 10 worst images:")
    for idx in worst_idx:
        print(f"    {os.path.basename(good_paths[idx]):>12s}  error={errors[idx]:.3f} px")

    # Outlier count (>2× median)
    threshold = 2.0 * np.median(errors)
    outliers = np.sum(errors > threshold)
    print(f"\n  Outliers (>{threshold:.3f} px): {outliers}/{len(errors)}")

    return K0


def compare_with_photonvision(K5, D5, K8, D8):
    """Compare our results with PhotonVision's MRCAL result."""
    print(f"\n{'='*60}")
    print(f"  Comparison with PhotonVision (MRCAL)")
    print(f"{'='*60}")
    print(f"  PhotonVision:")
    print(f"    fx = {PV_K[0,0]:.2f}   fy = {PV_K[1,1]:.2f}   fx/fy = {PV_K[0,0]/PV_K[1,1]:.4f}")
    print(f"    D  = {np.array2string(PV_D, precision=6, separator=', ')}")

    img_size = (1280, 720)
    PV_K0, _ = cv2.getOptimalNewCameraMatrix(PV_K, PV_D, img_size, alpha=0, newImgSize=img_size)
    print(f"    getOptimalNewCameraMatrix(alpha=0):")
    print(f"      fx = {PV_K0[0,0]:.2f}   fy = {PV_K0[1,1]:.2f}   ratio = {PV_K0[0,0]/PV_K0[1,1]:.4f}")

    print(f"\n  OpenCV 5-param:")
    print(f"    fx = {K5[0,0]:.2f}   fy = {K5[1,1]:.2f}   fx/fy = {K5[0,0]/K5[1,1]:.4f}")
    print(f"    D  = {np.array2string(D5, precision=6, separator=', ')}")

    print(f"\n  OpenCV 8-param:")
    print(f"    fx = {K8[0,0]:.2f}   fy = {K8[1,1]:.2f}   fx/fy = {K8[0,0]/K8[1,1]:.4f}")
    print(f"    D  = {np.array2string(D8, precision=6, separator=', ')}")


def analyse_distortion_profile(K, D, label):
    """Show how the distortion model warps radii from center."""
    print(f"\n  Radial distortion profile ({label}):")
    cx, cy = K[0, 2], K[1, 2]
    max_r = np.sqrt(cx**2 + cy**2)
    fx, fy = K[0, 0], K[1, 1]

    steps = 10
    for i in range(steps + 1):
        r_px = max_r * i / steps
        r_norm = r_px / ((fx + fy) / 2)
        r2 = r_norm ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3

        if len(D) >= 8:
            k1, k2, p1, p2, k3, k4, k5, k6 = D[:8]
            num = 1 + k1*r2 + k2*r4 + k3*r6
            den = 1 + k4*r2 + k5*r4 + k6*r6
            factor = num / den
        else:
            k1, k2, p1, p2, k3 = D[:5]
            factor = 1 + k1*r2 + k2*r4 + k3*r6

        print(f"    r={r_px:6.1f}px  r_norm={r_norm:.4f}  factor={factor:.6f}  "
              f"({'barrel' if factor > 1 else 'pincushion'})")


# ── Main ────────────────────────────────────────────────────────
def main():
    print("ChArUco Camera Calibration Analysis")
    print("=" * 60)

    # Gather images
    pattern = os.path.join(IMG_DIR, "img*.png")
    all_paths = sorted(glob.glob(pattern),
                       key=lambda p: int(os.path.basename(p)[3:-4]))
    print(f"Found {len(all_paths)} calibration images")

    # Subsample evenly for speed
    if len(all_paths) > MAX_IMAGES:
        step = len(all_paths) / MAX_IMAGES
        img_paths = [all_paths[int(i * step)] for i in range(MAX_IMAGES)]
        print(f"Subsampled to {len(img_paths)} images (every ~{step:.1f}th)")
    else:
        img_paths = all_paths

    # Detect corners (parallel)
    print("\nDetecting ChArUco corners (threaded)...")
    t0 = time.time()
    corners, ids, img_size, good_paths = detect_charuco_parallel(img_paths)
    print(f"  Detection took {time.time()-t0:.1f}s")

    if len(corners) < 10:
        print("ERROR: Not enough usable images for calibration!")
        return

    # ── Calibrate: 5-parameter model ────────────────────────────
    print("\n\nCalibrating with 5-parameter model...")
    t0 = time.time()
    rms5, K5, D5, rvecs5, tvecs5 = calibrate_charuco(
        corners, ids, _board, img_size, flags=0
    )
    print(f"  Calibration took {time.time()-t0:.1f}s")

    errors5 = per_image_errors(corners, ids, _board, K5, D5, rvecs5, tvecs5)
    K0_5 = print_results("5-Parameter Model", rms5, K5, D5, errors5, good_paths)
    analyse_distortion_profile(K5, D5, "5-param")

    # ── Calibrate: 8-parameter rational model ───────────────────
    print("\n\nCalibrating with 8-parameter rational model...")
    t0 = time.time()
    rms8, K8, D8, rvecs8, tvecs8 = calibrate_charuco(
        corners, ids, _board, img_size,
        flags=cv2.CALIB_RATIONAL_MODEL
    )
    print(f"  Calibration took {time.time()-t0:.1f}s")

    errors8 = per_image_errors(corners, ids, _board, K8, D8, rvecs8, tvecs8)
    K0_8 = print_results("8-Parameter Rational Model", rms8, K8, D8, errors8, good_paths)
    analyse_distortion_profile(K8, D8, "8-param")

    # ── Comparison ──────────────────────────────────────────────
    compare_with_photonvision(K5, D5, K8, D8)
    analyse_distortion_profile(PV_K, PV_D, "PhotonVision/MRCAL")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Images used: {len(corners)}/{len(img_paths)}")
    print(f"")
    print(f"  {'Model':<25s} {'RMS(px)':<10s} {'fx/fy':<10s} {'alpha=0 fx/fy':<15s}")
    print(f"  {'-'*60}")

    K0_pv, _ = cv2.getOptimalNewCameraMatrix(PV_K, PV_D, img_size, alpha=0, newImgSize=img_size)
    print(f"  {'PhotonVision (MRCAL)':<25s} {'N/A':<10s} "
          f"{PV_K[0,0]/PV_K[1,1]:<10.4f} {K0_pv[0,0]/K0_pv[1,1]:<15.4f}")
    print(f"  {'OpenCV 5-param':<25s} {rms5:<10.4f} "
          f"{K5[0,0]/K5[1,1]:<10.4f} {K0_5[0,0]/K0_5[1,1]:<15.4f}")
    print(f"  {'OpenCV 8-param':<25s} {rms8:<10.4f} "
          f"{K8[0,0]/K8[1,1]:<10.4f} {K0_8[0,0]/K0_8[1,1]:<15.4f}")

    improvement = (rms5 - rms8) / rms5 * 100
    print(f"\n  8-param RMS improvement over 5-param: {improvement:.1f}%")
    if improvement < 5:
        print(f"  → Marginal improvement — 8-param model is OVERFITTING")
    else:
        print(f"  → Significant improvement — 8-param may be justified")

    # ── Recommendation ──────────────────────────────────────────
    ratio_5 = K0_5[0, 0] / K0_5[1, 1]
    ratio_8 = K0_8[0, 0] / K0_8[1, 1]
    print(f"\n  RECOMMENDATION:")
    if abs(ratio_5 - 1.0) < 0.05:
        print(f"  ✓ Use the 5-parameter model (alpha=0 ratio {ratio_5:.4f} ≈ 1.0)")
        print(f"    Update ballshots6.py K_CALIB and D_CALIB with:")
        k5_str = np.array2string(K5, separator=', ', prefix='    ')
        d5_str = np.array2string(D5, separator=', ', prefix='    ')
        print(f"    K = np.{repr(K5)}")
        print(f"    D = np.{repr(D5)}")
    else:
        print(f"  ⚠ Both models produce skewed alpha=0 results.")
        print(f"    5-param ratio: {ratio_5:.4f}")
        print(f"    8-param ratio: {ratio_8:.4f}")
        print(f"    Consider recalibrating with fewer, higher-quality images.")

    # ── Why MRCAL is different ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WHY MRCAL RESULT IS DIFFERENT")
    print(f"{'='*60}")
    print(f"  PhotonVision uses mrcal's splined-stereographic model")
    print(f"  internally, then converts to OpenCV 8-coeff format.")
    print(f"  The conversion is an APPROXIMATION — mrcal's splined")
    print(f"  model has ~600+ parameters vs OpenCV's 8. The fit to")
    print(f"  a rational polynomial can introduce artefacts,")
    print(f"  especially in the denominator coefficients (k4-k6).")
    print(f"  This is why PV's k4={PV_D[5]:.4f} >> k1={PV_D[0]:.4f}.")


if __name__ == "__main__":
    main()
