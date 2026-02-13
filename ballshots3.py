"""
ballshots3.py — AprilTag detection only (no ball tracking).
Purpose: verify tags are detected reliably with the new fisheye camera.
Press 'q' to quit.
"""
import argparse
import cv2
import cv2.aruco as aruco
import numpy as np

# ─── Camera calibration (Global Shutter Camera, 1920x1080) ───
K = np.array([[1634.435694951629,    0.0,              977.0095538990062],
              [0.0,                  1617.0040060442561, 593.2899951200168],
              [0.0,                  0.0,                1.0]])
D = np.array([0.13479540133744689,
              -0.566065131074218,
              0.00018389150325768655,
              0.003664585948224397,
              0.8656340100414646])

IMG_SIZE = (1920, 1080)

# Undistortion maps (alpha=1 keeps all pixels)
NEW_K, ROI = cv2.getOptimalNewCameraMatrix(K, D, IMG_SIZE, alpha=1, newImgSize=IMG_SIZE)
MAP1, MAP2 = cv2.initUndistortRectifyMap(K, D, None, NEW_K, IMG_SIZE, cv2.CV_16SC2)
D_ZERO = np.zeros(5)

# ─── AprilTag 16h5 detector with tuned parameters ───
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
params = aruco.DetectorParameters()

# --- Adaptive thresholding: try more window sizes for varying lighting ---
params.adaptiveThreshWinSizeMin  = 3
params.adaptiveThreshWinSizeMax  = 53
params.adaptiveThreshWinSizeStep = 4

# --- Relax filtering so small / angled tags aren't rejected ---
params.minMarkerPerimeterRate    = 0.01   # detect smaller tags (default 0.03)
params.maxMarkerPerimeterRate    = 4.0    # allow large tags too
params.polygonalApproxAccuracyRate = 0.05 # default 0.03; more forgiving quad fit
params.minCornerDistanceRate     = 0.01   # allow corners closer together
params.minDistanceToBorder       = 1      # detect tags near frame edge

# --- Corner refinement for sub-pixel accuracy ---
params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize   = 5
params.cornerRefinementMaxIterations = 50
params.cornerRefinementMinAccuracy   = 0.01

# --- Reduce the error threshold for bit extraction ---
params.maxErroneousBitsInBorderRate = 0.5   # default 0.35; be more tolerant
params.errorCorrectionRate          = 0.6   # default 0.6

detector = aruco.ArucoDetector(aruco_dict, params)

TAG_SIZE = 4.0  # inches


def open_camera(preferred=0, probe_range=5):
    cap = cv2.VideoCapture(preferred)
    if cap.isOpened():
        return cap, preferred
    cap.release()
    for i in range(probe_range):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap, i
        cap.release()
    return None, None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='AprilTag detection test')
    ap.add_argument('--cam', type=int, default=0, help='camera index')
    ap.add_argument('--no-undistort', action='store_true', help='skip undistortion (debug)')
    args = ap.parse_args()

    cap, idx = open_camera(args.cam, probe_range=6)
    if cap is None:
        raise RuntimeError('No camera found')
    print(f'Camera index: {idx}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── Undistort ──
        if not args.no_undistort:
            frame = cv2.remap(frame, MAP1, MAP2, cv2.INTER_LINEAR)

        # ── Detect on grayscale (NO blur — blur kills tag edges) ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        # ── Draw results ──
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose and draw axes for each tag
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, TAG_SIZE, NEW_K, D_ZERO)
            for rvec, tvec, tag_id in zip(rvecs, tvecs, ids.flatten()):
                cv2.drawFrameAxes(frame, NEW_K, D_ZERO,
                                  rvec.reshape(3, 1), tvec.reshape(3, 1),
                                  TAG_SIZE)
                # Print distance to tag
                dist = float(np.linalg.norm(tvec))
                # Label on frame
                c = corners[list(ids.flatten()).index(tag_id)][0]
                cx_tag, cy_tag = int(c[:, 0].mean()), int(c[:, 1].mean())
                cv2.putText(frame, f"id={tag_id} d={dist:.1f}in",
                            (cx_tag - 40, cy_tag - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            tag_list = ids.flatten().tolist()
            print(f"[DETECTED] {len(tag_list)} tag(s): {tag_list}")
        else:
            print("[DETECTED] 0 tags")

        # Show rejected candidates count (useful for debugging)
        n_rejected = len(rejected) if rejected else 0
        cv2.putText(frame, f"Detected: {len(ids) if ids is not None else 0}  "
                           f"Rejected: {n_rejected}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw rejected candidates in red for debugging
        if rejected:
            for rej in rejected:
                pts = rej.reshape(-1, 2).astype(int)
                for j in range(4):
                    cv2.line(frame, tuple(pts[j]), tuple(pts[(j+1) % 4]),
                             (0, 0, 255), 1)

        cv2.imshow("Tag Detection", frame)
        cv2.imshow("Grayscale", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
