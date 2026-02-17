"""
ballshots3.py — AprilTag + yellow-ball tracker.
Detects AprilTag 16h5 tags, fits a plane through them, finds a yellow ball,
projects it onto the tag plane, and reports its position in the reference-tag
coordinate frame.  Press 'q' to quit; CSV is saved on exit.

Key design: ALL detection happens on the RAW (distorted) image.  Pixel
coordinates are analytically undistorted via cv2.undistortPoints which
iteratively inverts the full Brown-Conrady model to machine precision —
no lookup-table interpolation error at the edges.  The image is remapped
ONLY for the display window.
"""
import argparse, time, csv, os
import cv2
import cv2.aruco as aruco
import numpy as np

# ─── Sanity limits ───
MAX_RANGE    = 120.0   # in — max ball-to-tag distance kept
MAX_TAG_DIST = 200.0   # in — max camera-to-tag distance kept

# ─── Camera calibration (Global Shutter Camera, calibrated at 1920×1080) ───
#     These are the ORIGINAL intrinsics and distortion from calibration.
#     They describe the RAW image coming off the sensor at calibration res.
K_CALIB = np.array([[1634.435694951629,    0.0,               977.0095538990062],
                    [0.0,                  1617.0040060442561, 593.2899951200168],
                    [0.0,                  0.0,                1.0]])
D = np.array([0.13479540133744689,
              -0.566065131074218,
              0.00018389150325768655,
              0.003664585948224397,
              0.8656340100414646])

assert K_CALIB.shape == (3, 3) and K_CALIB[2, 2] == 1.0
assert K_CALIB[0, 0] > 0 and K_CALIB[1, 1] > 0
assert D.shape[0] == 5

CALIB_SIZE = (1920, 1080)   # resolution calibration was performed at


def scale_K(K_orig, orig_size, new_size):
    """Scale a camera matrix from one resolution to another.

    Intrinsics scale linearly:  fx' = fx * (w'/w),  cx' = cx * (w'/w)
    Distortion coefficients are resolution-independent (normalised coords).
    """
    sx = new_size[0] / orig_size[0]
    sy = new_size[1] / orig_size[1]
    K_new = K_orig.copy()
    K_new[0, 0] *= sx   # fx
    K_new[0, 2] *= sx   # cx
    K_new[1, 1] *= sy   # fy
    K_new[1, 2] *= sy   # cy
    return K_new

# K, IMG_SIZE, DISP_K, MAP1, MAP2, DISP_ROI are set at runtime after
# we know the actual camera resolution.  See __main__ below.
K = None
IMG_SIZE = None
DISP_K = None
DISP_ROI = None
MAP1 = None
MAP2 = None


def undistort_pts(pts_px):
    """Undistort pixel coordinates → normalised camera rays.

    Args:
        pts_px: (N,2) array of pixel coords in the RAW distorted image.
    Returns:
        (N,2) array of (x/z, y/z) normalised coordinates — i.e. the true
        ray direction with z=1.  Uses cv2.undistortPoints which iteratively
        solves the inverse Brown-Conrady model (k1..k3, p1,p2) to full
        float64 precision.  No lookup table, no interpolation artefacts.
    """
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    # P=identity → output is normalised (x/z, y/z)
    out = cv2.undistortPoints(pts, K, D, P=np.eye(3))
    return out.reshape(-1, 2)


def undistort_pts_to_pixel(pts_px):
    """Undistort raw pixels → undistorted pixels in DISP_K frame.

    Useful for drawing on the display image at the correct location.
    """
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, D, P=DISP_K)
    return out.reshape(-1, 2)


# ─── AprilTag 16h5 detector with tuned parameters ───
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
params = aruco.DetectorParameters()

params.adaptiveThreshWinSizeMin  = 3
params.adaptiveThreshWinSizeMax  = 53
params.adaptiveThreshWinSizeStep = 4

params.minMarkerPerimeterRate    = 0.01
params.maxMarkerPerimeterRate    = 4.0
params.polygonalApproxAccuracyRate = 0.05
params.minCornerDistanceRate     = 0.01
params.minDistanceToBorder       = 1

params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize   = 5
params.cornerRefinementMaxIterations = 50
params.cornerRefinementMinAccuracy   = 0.01

params.maxErroneousBitsInBorderRate = 0.5
params.errorCorrectionRate          = 0.6

detector = aruco.ArucoDetector(aruco_dict, params)

TAG_SIZE = 4.0  # inches

# ─── Yellow-ball HSV thresholds ───
LO_YELLOW = np.array([14,  70, 180])
HI_YELLOW = np.array([35, 255, 255])
MORPH_K   = np.ones((5, 5), np.uint8)
BALL_RADIUS = 1.3  # inches — tennis ball radius for contact-point correction

# ─── State ───
positions = []   # list of (timestamp, x_in, y_in, z_in)

# Output smoothing — EMA on final projected XY to reduce jitter
OUT_ALPHA = 0.3           # 0 = full smooth, 1 = no smooth (raw)
_smooth = {'xy': None}    # smoothed (x, y) output in mutable container


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
    ap = argparse.ArgumentParser(description='Ball + AprilTag tracker')
    ap.add_argument('--cam', type=int, default=0, help='camera index')
    args = ap.parse_args()

    cap, cam_idx = open_camera(args.cam, probe_range=6)
    if cap is None:
        raise RuntimeError('No camera found (probed 0..5)')
    print(f'Camera index: {cam_idx}')

    # ── Try to get 1920×1080; accept whatever the camera gives ──
    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.read()  # dummy read — some drivers need this before settings stick
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    IMG_SIZE = (actual_w, actual_h)
    print(f"Resolution: {actual_w}×{actual_h}")

    # ── Scale calibration matrix to actual resolution ──
    K = scale_K(K_CALIB, CALIB_SIZE, IMG_SIZE)
    print(f"Scaled K from {CALIB_SIZE[0]}×{CALIB_SIZE[1]} → "
          f"{IMG_SIZE[0]}×{IMG_SIZE[1]}:")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # ── Build display undistortion maps for the actual resolution ──
    DISP_K, DISP_ROI = cv2.getOptimalNewCameraMatrix(
        K, D, IMG_SIZE, alpha=1, newImgSize=IMG_SIZE)
    MAP1, MAP2 = cv2.initUndistortRectifyMap(
        K, D, None, DISP_K, IMG_SIZE, cv2.CV_16SC2)

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break

            h, w = raw.shape[:2]
            assert (w, h) == IMG_SIZE, f"Frame {w}×{h} != {IMG_SIZE}"

            # ── 1. Undistort for DISPLAY only ──
            disp_frame = cv2.remap(raw, MAP1, MAP2, cv2.INTER_LINEAR)

            # ── 2. Yellow-ball detection on RAW image ──
            ball_blur = cv2.GaussianBlur(raw, (11, 11), 0)
            hsv  = cv2.cvtColor(ball_blur, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LO_YELLOW, HI_YELLOW)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_K)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_K)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

            ball_px_raw = None          # pixel in the RAW distorted image
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M['m00'] > 200:
                    bx = M['m10'] / M['m00']
                    by = M['m01'] / M['m00']
                    assert 0 <= bx < w and 0 <= by < h, \
                        f"Ball centroid ({bx:.0f},{by:.0f}) outside frame"
                    ball_px_raw = (bx, by)
                    r_px = int(np.sqrt(M['m00'] / np.pi))
                    # draw on display frame at undistorted location
                    disp_pt = undistort_pts_to_pixel(
                        np.array([[bx, by]]))[0]
                    cv2.circle(disp_frame,
                               (int(disp_pt[0]), int(disp_pt[1])),
                               r_px, (0, 255, 0), 2)
                    cv2.circle(disp_frame,
                               (int(disp_pt[0]), int(disp_pt[1])),
                               3, (0, 0, 255), -1)

            # ── 3. AprilTag detection on RAW grayscale ──
            gray_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray_raw)

            if ids is not None:
                # draw detected markers on display frame
                # undistort each corner set for drawing
                disp_corners = []
                for c in corners:
                    pts_raw = c.reshape(-1, 2)
                    pts_und = undistort_pts_to_pixel(pts_raw)
                    disp_corners.append(pts_und.reshape(1, -1, 2).astype(np.float32))
                aruco.drawDetectedMarkers(disp_frame, disp_corners, ids)

            # ── 4. Pose → plane → ray → projection ──
            if ids is not None and ball_px_raw is not None:
                # estimatePoseSingleMarkers with ORIGINAL K,D
                # This tells OpenCV the corners are in the distorted image
                # and it internally undistorts them for the PnP solve.
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, TAG_SIZE, K, D)
                assert rvecs.shape[0] == tvecs.shape[0] == len(ids)

                # validate tag distances
                tag_ok = True
                for tv in tvecs:
                    d = float(np.linalg.norm(tv))
                    assert d > 0
                    if d > MAX_TAG_DIST:
                        tag_ok = False
                if not tag_ok:
                    # show display + skip
                    rx, ry, rw, rh = DISP_ROI
                    if rw > 0 and rh > 0:
                        cv2.imshow("Ball Tracking",
                                   disp_frame[ry:ry+rh, rx:rx+rw])
                    else:
                        cv2.imshow("Ball Tracking", disp_frame)
                    cv2.imshow("Mask", mask)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # draw axes for every tag on display frame
                for rv, tv in zip(rvecs, tvecs):
                    cv2.drawFrameAxes(disp_frame, DISP_K, np.zeros(5),
                                      rv.reshape(3, 1), tv.reshape(3, 1),
                                      TAG_SIZE)

                # ── build plane from tag centres + normals ──
                tag_centers = []
                tag_normals = []
                for rv, tv in zip(rvecs, tvecs):
                    R, _ = cv2.Rodrigues(rv)
                    tag_centers.append(tv.ravel())
                    tag_normals.append(R[:, 2])
                tag_centers = np.array(tag_centers)
                tag_normals = np.array(tag_normals)

                n_tags = len(tag_centers)
                if n_tags == 1:
                    centroid = tag_centers[0]
                    n = tag_normals[0].copy()
                else:
                    centroid = tag_centers.mean(axis=0)
                    _, _, vt = np.linalg.svd(tag_centers - centroid)
                    n = vt[-1].copy()
                    avg_z = tag_normals.mean(axis=0)
                    if n @ avg_z < 0:
                        n = -n

                n /= np.linalg.norm(n)
                assert abs(np.linalg.norm(n) - 1.0) < 1e-6
                if n @ centroid > 0:
                    n = -n

                # ── ray from ball pixel (analytically undistorted) ──
                # undistortPoints gives normalised coords (x/z, y/z)
                # which IS the ray direction with z=1.
                norm_pt = undistort_pts(
                    np.array([[ball_px_raw[0], ball_px_raw[1]]]))[0]
                ray = np.array([norm_pt[0], norm_pt[1], 1.0])

                denom = ray @ n

                if abs(denom) > 0.1:
                    t_ray = (centroid @ n + BALL_RADIUS) / denom

                    if t_ray > 0:
                        ball_centre_3d = t_ray * ray
                        P = ball_centre_3d - BALL_RADIUS * n
                        if not (np.isfinite(P).all() and
                                np.linalg.norm(P) < MAX_TAG_DIST * 2):
                            rx, ry, rw, rh = DISP_ROI
                            if rw > 0 and rh > 0:
                                cv2.imshow("Ball Tracking",
                                           disp_frame[ry:ry+rh, rx:rx+rw])
                            else:
                                cv2.imshow("Ball Tracking", disp_frame)
                            cv2.imshow("Mask", mask)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            continue

                        # ── transform into reference-tag frame ──
                        ids_flat = ids.flatten().tolist()
                        ref = ids_flat.index(0) if 0 in ids_flat else 0
                        R_ref, _ = cv2.Rodrigues(rvecs[ref])
                        t_ref    = tvecs[ref].reshape(3, 1)

                        p_tag = (R_ref.T @ (P.reshape(3, 1) - t_ref)).ravel()
                        assert np.isfinite(p_tag).all()
                        p_tag[2] = 0.0

                        # ── sanity gate ──
                        if (abs(p_tag[0]) < MAX_RANGE and
                            abs(p_tag[1]) < MAX_RANGE):

                            raw_xy = np.array([p_tag[0], p_tag[1]])
                            if _smooth['xy'] is None:
                                _smooth['xy'] = raw_xy.copy()
                            else:
                                _smooth['xy'] = (OUT_ALPHA * raw_xy +
                                                 (1 - OUT_ALPHA) * _smooth['xy'])
                            sx, sy = float(_smooth['xy'][0]), float(_smooth['xy'][1])

                            row = (time.time(), sx, sy, 0.0)
                            assert all(np.isfinite(v) for v in row)
                            positions.append(row)
                            print(f"[POS] X:{sx:7.2f}  "
                                  f"Y:{sy:7.2f}  "
                                  f"Z:   0.00 in")

                            cv2.drawFrameAxes(disp_frame, DISP_K,
                                              np.zeros(5),
                                              rvecs[ref], tvecs[ref], 2)
                            cv2.putText(
                                disp_frame,
                                f"X:{sx:.1f}  Y:{sy:.1f} in",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 0, 0), 2)

            # ── 5. HUD ──
            n_det = len(ids) if ids is not None else 0
            n_rej = len(rejected) if rejected else 0
            cv2.putText(disp_frame,
                        f"Tags:{n_det}  Rej:{n_rej}  Pts:{len(positions)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

            # ── 6. Display: crop to valid ROI ──
            # Remap mask to undistorted space so it aligns with disp_frame
            disp_mask = cv2.remap(mask, MAP1, MAP2, cv2.INTER_NEAREST)
            rx, ry, rw, rh = DISP_ROI
            if rw > 0 and rh > 0:
                cv2.imshow("Ball Tracking",
                           disp_frame[ry:ry+rh, rx:rx+rw])
                cv2.imshow("Mask", disp_mask[ry:ry+rh, rx:rx+rw])
            else:
                cv2.imshow("Ball Tracking", disp_frame)
                cv2.imshow("Mask", disp_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ── Save CSV ──
        csv_path = os.path.join(os.path.dirname(__file__),
                                "ball_positions.csv")
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["time", "x_in", "y_in", "z_in"])
            if positions:
                wr.writerows(positions)
                print(f"Saved {len(positions)} rows → {csv_path}")
            else:
                print("No positions recorded.")
