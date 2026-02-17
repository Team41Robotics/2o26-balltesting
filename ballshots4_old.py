import argparse, time, csv, os
import cv2
import cv2.aruco as aruco
import numpy as np

# ───────────────────────────────────────────────────────────────────────────
#  Camera calibration (PhotonVision, 1280×720)
# ───────────────────────────────────────────────────────────────────────────
K_CALIB = np.array([
    [545.1385372907386, 0.0,             619.6410820347659],
    [0.0,              545.037239791495, 384.28731732363195],
    [0.0,              0.0,              1.0               ]])
D_CALIB = np.array([
    0.20856149225304135,   # k1
    -0.03663050900029196,  # k2
    0.00023804707578467353,# p1
    -0.0001210337939646964,# p2
    -0.0008277767967346465,# k3
    0.5645927654152898,    # k4
    -0.03301555724510292,  # k5
    -0.007126953259440723])# k6

CALIB_SIZE = (1280, 720)


def scale_K(K_orig, orig_size, new_size):
    """Scale intrinsics linearly when resolution changes."""
    sx = new_size[0] / orig_size[0]
    sy = new_size[1] / orig_size[1]
    K = K_orig.copy()
    K[0, 0] *= sx;  K[0, 2] *= sx
    K[1, 1] *= sy;  K[1, 2] *= sy
    return K


# ───────────────────────────────────────────────────────────────────────────
#  Tag layout — world coordinates (metres)
# ───────────────────────────────────────────────────────────────────────────
TAG_SIZE  = 0.1016 # 4 in
TAG_SPACING = 1.0 # centre‑to‑centre distance between adjacent tags

# World positions of each tag's CENTRE  (Z = 0, all on the floor)
#
#   Tag 0 = origin        Tag 1 = +X
#   Tag 3 = +Y            Tag 2 = +X +Y
TAG_CENTRES = {
    0: np.array([0.0,         0.0,         0.0]),
    1: np.array([TAG_SPACING, 0.0,         0.0]),
    2: np.array([TAG_SPACING, TAG_SPACING, 0.0]),
    3: np.array([0.0,         TAG_SPACING, 0.0]),
}

def tag_corner_world(tag_id):
    """Return the 4 corner positions (in world frame) for a given tag.

    OpenCV ArUco corner order:
        0 = top‑left, 1 = top‑right, 2 = bottom‑right, 3 = bottom‑left
    when looking at the tag from the front (camera side).

    We define the tag face as lying in the Z = 0 plane with its local
    X along world‑X and local Y along world‑Y.
    """
    c = TAG_CENTRES[tag_id]
    s = TAG_SIZE / 2.0
    return np.array([
        c + np.array([-s, -s, 0]),   # corner 0  (top-left)
        c + np.array([ s, -s, 0]),   # corner 1  (top-right)
        c + np.array([ s,  s, 0]),   # corner 2  (bottom-right)
        c + np.array([-s,  s, 0]),   # corner 3  (bottom-left)
    ], dtype=np.float64)


# ───────────────────────────────────────────────────────────────────────────
#  Reference grid (16.5 cm spacing, centred at tag 0)
# ───────────────────────────────────────────────────────────────────────────
GRID_STEP  = 0.1085         # 10.85 cm
GRID_HALF  = 12             # number of steps in each direction from origin
                            # → covers ±1.98 m  (plenty for 2 m square)

def build_grid_points():
    """Build a 3D point array for the reference grid on the Z = 0 plane."""
    pts = []
    for i in range(-GRID_HALF, GRID_HALF + 1):
        for j in range(-GRID_HALF, GRID_HALF + 1):
            pts.append([i * GRID_STEP, j * GRID_STEP, 0.0])
    return np.array(pts, dtype=np.float64)

def build_grid_lines():
    """Return lists of 3D line segments (start, end) for horizontal and
    vertical grid lines."""
    lo = -GRID_HALF * GRID_STEP
    hi =  GRID_HALF * GRID_STEP
    lines = []
    for i in range(-GRID_HALF, GRID_HALF + 1):
        coord = i * GRID_STEP
        # horizontal line  (constant Y, sweep X)
        lines.append((np.array([lo, coord, 0.0]),
                       np.array([hi, coord, 0.0])))
        # vertical line  (constant X, sweep Y)
        lines.append((np.array([coord, lo, 0.0]),
                       np.array([coord, hi, 0.0])))
    return lines

GRID_LINES = build_grid_lines()


# ───────────────────────────────────────────────────────────────────────────
#  Yellow‑ball HSV thresholds
# ───────────────────────────────────────────────────────────────────────────
LO_YELLOW = np.array([ 18, 120, 120])
HI_YELLOW = np.array([ 35, 255, 255])
MORPH_K   = np.ones((7, 7), np.uint8)
MIN_BALL_AREA = 300
MAX_BALL_AREA = 50000


# ───────────────────────────────────────────────────────────────────────────
#  AprilTag detector
# ───────────────────────────────────────────────────────────────────────────
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
det_params = aruco.DetectorParameters()

det_params.adaptiveThreshWinSizeMin  = 3
det_params.adaptiveThreshWinSizeMax  = 53
det_params.adaptiveThreshWinSizeStep = 10
det_params.adaptiveThreshConstant    = 7
det_params.minMarkerPerimeterRate    = 0.005
det_params.maxMarkerPerimeterRate    = 4.0
det_params.polygonalApproxAccuracyRate = 0.08
det_params.minCornerDistanceRate     = 0.005
det_params.minDistanceToBorder       = 0
det_params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX
det_params.cornerRefinementWinSize   = 5
det_params.cornerRefinementMaxIterations = 50
det_params.cornerRefinementMinAccuracy   = 0.01
det_params.maxErroneousBitsInBorderRate  = 0.65
det_params.errorCorrectionRate           = 0.8

detector = aruco.ArucoDetector(aruco_dict, det_params)


# ───────────────────────────────────────────────────────────────────────────
#  State
# ───────────────────────────────────────────────────────────────────────────
positions = []              # (timestamp, x_m, y_m)
OUT_ALPHA = 0.3             # EMA smoothing factor (1 = raw, 0 = full smooth)
_smooth_xy = None           # EMA accumulator
_frame_n   = 0              # frame counter for throttled prints


# ───────────────────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────────────────
def open_camera(preferred=0, probe_range=6):
    for idx in ([preferred] + [i for i in range(probe_range) if i != preferred]):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap, idx
        cap.release()
    return None, None


def draw_grid(frame, rvec, tvec, K, D_zero):
    """Project the world‑frame grid lines onto *frame* and draw them.
    Batches all points into a single projectPoints call for speed."""
    h, w = frame.shape[:2]

    # Collect all line-segment endpoints into one array
    all_pts = np.empty((len(GRID_LINES) * 2, 3), dtype=np.float64)
    for i, (p0, p1) in enumerate(GRID_LINES):
        all_pts[i * 2]     = p0
        all_pts[i * 2 + 1] = p1

    # Add axis endpoints: X, Y, Z  (3 axes × 2 pts = 6 pts)
    axis_len = 1.0
    axis_pts = np.array([
        [0,0,0], [axis_len,0,0],       # X
        [0,0,0], [0,axis_len,0],       # Y
        [0,0,0], [0,0,axis_len/2],     # Z
    ], dtype=np.float64)

    batch = np.vstack([all_pts, axis_pts])
    proj, _ = cv2.projectPoints(batch, rvec, tvec, K, D_zero)
    px = proj.reshape(-1, 2).astype(int)

    margin = 2000
    def _ok(pt):
        return -margin <= pt[0] <= w + margin and -margin <= pt[1] <= h + margin

    # Draw grid lines
    n_lines = len(GRID_LINES)
    for i in range(n_lines):
        pt1 = tuple(px[i * 2])
        pt2 = tuple(px[i * 2 + 1])
        if _ok(pt1) or _ok(pt2):
            cv2.line(frame, pt1, pt2, (0, 0, 180), 1, cv2.LINE_AA)

    # Draw axes (after grid lines in the array)
    base = n_lines * 2
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue
    for a in range(3):
        pt1 = tuple(px[base + a * 2])
        pt2 = tuple(px[base + a * 2 + 1])
        if _ok(pt1) or _ok(pt2):
            cv2.line(frame, pt1, pt2, axis_colors[a], 1, cv2.LINE_AA)


def solve_plane_pose(corners_list, ids, K, D_zero):
    """Given detected ArUco corners + IDs, build 2D–3D correspondences
    for tags 0–3 and solve a single solvePnP.

    Returns (success, rvec, tvec).
    """
    obj_pts = []   # 3D world
    img_pts = []   # 2D image (undistorted)

    ids_flat = ids.flatten().tolist()
    for idx, tag_id in enumerate(ids_flat):
        if tag_id not in TAG_CENTRES:
            continue
        world_corners = tag_corner_world(tag_id)     # (4, 3)
        image_corners = corners_list[idx].reshape(4, 2)  # (4, 2)
        for wc, ic in zip(world_corners, image_corners):
            obj_pts.append(wc)
            img_pts.append(ic)

    if len(obj_pts) < 4:
        return False, None, None

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    # solvePnP with IPPE for coplanar points (more stable than iterative)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, D_zero, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        # fallback to iterative
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, D_zero, flags=cv2.SOLVEPNP_ITERATIVE)
    return ok, rvec, tvec


def ray_plane_intersect(px, py, rvec, tvec, K_new):
    """Intersect the camera ray through pixel (px, py) with the Z = 0 plane.

    Returns the 3D world point (x, y, 0) or None.
    """
    R, _ = cv2.Rodrigues(rvec)
    # camera ray in camera frame (undistorted, so D = 0)
    fx, fy = K_new[0, 0], K_new[1, 1]
    cx, cy = K_new[0, 2], K_new[1, 2]
    ray_cam = np.array([(px - cx) / fx,
                        (py - cy) / fy,
                        1.0])
    # transform ray into world frame
    ray_world = R.T @ ray_cam
    cam_pos   = (-R.T @ tvec).ravel()  # camera position in world

    # intersect with Z = 0:  cam_pos.z + t * ray_world.z = 0
    if abs(ray_world[2]) < 1e-8:
        return None
    t = -cam_pos[2] / ray_world[2]
    if t < 0:
        return None
    hit = cam_pos + t * ray_world
    return hit   # (x, y, ~0)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='ballshots4 — grid + ball tracker')
    ap.add_argument('--cam', type=int, default=0, help='camera index')
    args = ap.parse_args()

    # ── Open camera ──
    cap, cam_idx = open_camera(args.cam)
    if cap is None:
        raise RuntimeError('No camera found')
    print(f'Camera index: {cam_idx}')

    # Set MJPG and request 1280×720 (native for this camera)
    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    IMG_SIZE = (actual_w, actual_h)
    print(f'Resolution: {actual_w}×{actual_h}')

    # ── Scale calibration to actual resolution ──
    K = scale_K(K_CALIB, CALIB_SIZE, IMG_SIZE)
    print(f'Scaled K  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  '
          f'cx={K[0,2]:.1f}  cy={K[1,2]:.1f}')

    # ── Build FULL undistortion remap (alpha=0 → crop to valid pixels) ──
    NEW_K, ROI = cv2.getOptimalNewCameraMatrix(
        K, D_CALIB, IMG_SIZE, alpha=0, newImgSize=IMG_SIZE)
    MAP1, MAP2 = cv2.initUndistortRectifyMap(
        K, D_CALIB, None, NEW_K, IMG_SIZE, cv2.CV_16SC2)
    D_ZERO = np.zeros(5)

    print(f'New K     fx={NEW_K[0,0]:.1f}  fy={NEW_K[1,1]:.1f}  '
          f'cx={NEW_K[0,2]:.1f}  cy={NEW_K[1,2]:.1f}')
    print(f'ROI: {ROI}')

    _fps_time = time.perf_counter()
    _fps_val  = 0.0

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            _frame_n += 1

            # ── 1. UNDISTORT everything ──
            frame = cv2.remap(raw, MAP1, MAP2, cv2.INTER_LINEAR)

            # ── 2. Detect AprilTags on undistorted frame ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)

            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)

            # ── 3. Solve PnP for the tag plane ──
            rvec, tvec = None, None
            if ids is not None:
                ok_pnp, rvec, tvec = solve_plane_pose(
                    corners, ids, NEW_K, D_ZERO)
                if ok_pnp:
                    # ── 4. Draw reference grid ──
                    draw_grid(frame, rvec, tvec, NEW_K, D_ZERO)

                    # Debug: camera height + PnP info  (every 15 frames)
                    if _frame_n % 15 == 0:
                        R, _ = cv2.Rodrigues(rvec)
                        cam_pos = (-R.T @ tvec).ravel()
                        ids_flat = ids.flatten().tolist()
                        obj_dbg, img_dbg = [], []
                        for ii, tid in enumerate(ids_flat):
                            if tid in TAG_CENTRES:
                                wc = tag_corner_world(tid)
                                ic = corners[ii].reshape(4, 2)
                                for w, p in zip(wc, ic):
                                    obj_dbg.append(w)
                                    img_dbg.append(p)
                        reproj, _ = cv2.projectPoints(
                            np.array(obj_dbg), rvec, tvec, NEW_K, D_ZERO)
                        err = np.sqrt(np.mean(
                            (reproj.reshape(-1, 2) - np.array(img_dbg))**2))
                        print(f'[DBG] cam=({cam_pos[0]:.3f},{cam_pos[1]:.3f},'
                              f'{cam_pos[2]:.3f})  reproj={err:.2f}px  '
                              f'tags={[t for t in ids_flat if t in TAG_CENTRES]}')

            # ── 5. Yellow-ball detection ──
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LO_YELLOW, HI_YELLOW)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_K)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_K)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

            ball_px = None
            if cnts:
                # Filter by area and circularity, pick best
                best_cnt = None
                best_area = 0
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
                        continue
                    perim = cv2.arcLength(cnt, True)
                    if perim < 1:
                        continue
                    circ = 4 * np.pi * area / (perim * perim)
                    if circ < 0.4:          # reject non-round blobs
                        continue
                    if area > best_area:
                        best_area = area
                        best_cnt = cnt

                if best_cnt is not None:
                    (bx, by), r_px = cv2.minEnclosingCircle(best_cnt)
                    ball_px = (bx, by)
                    cv2.circle(frame, (int(bx), int(by)),
                               int(r_px), (0, 255, 0), 2)
                    cv2.circle(frame, (int(bx), int(by)),
                               3, (0, 0, 255), -1)

            # ── 6. Ray-plane intersection → world position ──
            if rvec is not None and ball_px is not None:
                hit = ray_plane_intersect(
                    ball_px[0], ball_px[1], rvec, tvec, NEW_K)
                if hit is not None:
                    wx, wy = hit[0], hit[1]

                    # EMA smoothing
                    raw_xy = np.array([wx, wy])
                    if _smooth_xy is None:
                        _smooth_xy = raw_xy.copy()
                    else:
                        _smooth_xy = (OUT_ALPHA * raw_xy +
                                      (1 - OUT_ALPHA) * _smooth_xy)
                    sx, sy = float(_smooth_xy[0]), float(_smooth_xy[1])

                    # convert to cm for display
                    sx_cm = sx * 100.0
                    sy_cm = sy * 100.0

                    positions.append((time.time(), sx, sy))
                    if _frame_n % 15 == 0:
                        print(f'[POS] X:{sx_cm:7.1f} cm   Y:{sy_cm:7.1f} cm')

                    cv2.putText(frame,
                                f'X:{sx_cm:.1f}cm  Y:{sy_cm:.1f}cm',
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 0, 0), 2)

            # ── 7. HUD ──
            now = time.perf_counter()
            dt = now - _fps_time
            _fps_time = now
            if dt > 0:
                _fps_val = 0.9 * _fps_val + 0.1 * (1.0 / dt)
            n_det = len(ids) if ids is not None else 0
            n_rej = len(rejected) if rejected else 0
            cv2.putText(frame,
                        f'FPS:{_fps_val:.0f}  Tags:{n_det}  '
                        f'Pts:{len(positions)}',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            if rvec is not None:
                cv2.putText(frame, 'PnP OK', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 0), 2)

            cv2.imshow('ballshots4', frame)
            cv2.imshow('Mask', mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        cap.release()
        cv2.destroyAllWindows()

        csv_path = os.path.join(os.path.dirname(__file__),
                                'ball_positions.csv')
        with open(csv_path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['time', 'x_m', 'y_m'])
            if positions:
                wr.writerows(positions)
                print(f'Saved {len(positions)} rows → {csv_path}')
            else:
                print('No positions recorded.')
