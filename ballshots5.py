"""ballshots5 — max-FPS ball tracker with CUDA + threaded capture."""
import argparse, time, csv, os, threading
import cv2
import cv2.aruco as aruco
import numpy as np

# ── GPU back-end selection ──
USE_CUDA = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        USE_CUDA = True
        cv2.cuda.printShortCudaDeviceInfo(0)
        print('[GPU] CUDA back-end ENABLED')
except Exception:
    pass
if not USE_CUDA:
    cv2.ocl.setUseOpenCL(True)
    print('[GPU] CUDA not available — using OpenCL/UMat fallback')

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
    sx = new_size[0] / orig_size[0]
    sy = new_size[1] / orig_size[1]
    K = K_orig.copy()
    K[0, 0] *= sx;  K[0, 2] *= sx
    K[1, 1] *= sy;  K[1, 2] *= sy
    return K


# ───────────────────────────────────────────────────────────────────────────
#  Tag layout — world coordinates (metres)
# ───────────────────────────────────────────────────────────────────────────
TAG_SIZE    = 0.1016    # 4 in
TAG_SPACING = 1.0       # centre-to-centre distance between adjacent tags

TAG_CENTRES = {
    0: np.array([0.0,         0.0,         0.0]),
    1: np.array([TAG_SPACING, 0.0,         0.0]),
    2: np.array([TAG_SPACING, TAG_SPACING, 0.0]),
    3: np.array([0.0,         TAG_SPACING, 0.0]),
}

def tag_corner_world(tag_id):
    c = TAG_CENTRES[tag_id]
    s = TAG_SIZE / 2.0
    return np.array([
        c + np.array([-s, -s, 0]),
        c + np.array([ s, -s, 0]),
        c + np.array([ s,  s, 0]),
        c + np.array([-s,  s, 0]),
    ], dtype=np.float64)


# ───────────────────────────────────────────────────────────────────────────
#  Reference grid (smaller for speed — 6 lines each way instead of 12)
# ───────────────────────────────────────────────────────────────────────────
GRID_STEP = 0.1085
GRID_HALF = 12

def build_grid_lines():
    lo = -GRID_HALF * GRID_STEP
    hi =  GRID_HALF * GRID_STEP
    lines = []
    for i in range(-GRID_HALF, GRID_HALF + 1):
        coord = i * GRID_STEP
        lines.append((np.array([lo, coord, 0.0]),
                       np.array([hi, coord, 0.0])))
        lines.append((np.array([coord, lo, 0.0]),
                       np.array([coord, hi, 0.0])))
    return lines

GRID_LINES = build_grid_lines()

_N_GRID_PTS = len(GRID_LINES) * 2
_grid_all_pts = np.empty((_N_GRID_PTS + 6, 3), dtype=np.float64)
for _i, (_p0, _p1) in enumerate(GRID_LINES):
    _grid_all_pts[_i * 2]     = _p0
    _grid_all_pts[_i * 2 + 1] = _p1
_grid_all_pts[_N_GRID_PTS + 0] = [0, 0, 0]
_grid_all_pts[_N_GRID_PTS + 1] = [1, 0, 0]
_grid_all_pts[_N_GRID_PTS + 2] = [0, 0, 0]
_grid_all_pts[_N_GRID_PTS + 3] = [0, 1, 0]
_grid_all_pts[_N_GRID_PTS + 4] = [0, 0, 0]
_grid_all_pts[_N_GRID_PTS + 5] = [0, 0, 0.5]

# ───────────────────────────────────────────────────────────────────────────
#  Yellow-ball HSV thresholds
# ───────────────────────────────────────────────────────────────────────────
LO_YELLOW = np.array([ 25, 120, 120], dtype=np.uint8)
HI_YELLOW = np.array([ 35, 255, 255], dtype=np.uint8)
MORPH_K   = np.ones((5, 5), np.uint8)          # 5×5 instead of 7×7

# ───────────────────────────────────────────────────────────────────────────
#  AprilTag detector  (only used during calibration)
# ───────────────────────────────────────────────────────────────────────────
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
det_params = aruco.DetectorParameters()
det_params.adaptiveThreshWinSizeMin    = 3
det_params.adaptiveThreshWinSizeMax    = 53
det_params.adaptiveThreshWinSizeStep   = 10
det_params.adaptiveThreshConstant      = 7
det_params.minMarkerPerimeterRate      = 0.005
det_params.maxMarkerPerimeterRate      = 4.0
det_params.polygonalApproxAccuracyRate = 0.08
det_params.minCornerDistanceRate       = 0.005
det_params.minDistanceToBorder         = 0
det_params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
det_params.cornerRefinementWinSize     = 5
det_params.cornerRefinementMaxIterations = 50
det_params.cornerRefinementMinAccuracy   = 0.01
det_params.maxErroneousBitsInBorderRate  = 0.65
det_params.errorCorrectionRate           = 0.8
detector = aruco.ArucoDetector(aruco_dict, det_params)

# ───────────────────────────────────────────────────────────────────────────
#  Threaded camera reader  (decouples USB I/O from processing)
# ───────────────────────────────────────────────────────────────────────────
class CamReader:
    """Reads frames in a background thread so cap.read() never blocks."""
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            with self._lock:
                self.ok, self.frame = ok, f

    def read(self):
        with self._lock:
            return self.ok, self.frame

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)


# ───────────────────────────────────────────────────────────────────────────
#  State
# ───────────────────────────────────────────────────────────────────────────
positions  = []
OUT_ALPHA  = 0.3
_smooth_xy = None
_frame_n   = 0

CALIB_DURATION = 10.0
_calib_rvecs   = []
_calib_tvecs   = []
_calib_done    = False
_frozen_rvec   = None
_frozen_tvec   = None
_frozen_grid_px = None
R_frozen       = None
R_T_frozen     = None
cam_pos_frozen = None

# Pre-rendered grid overlay (created once after calibration freeze)
_grid_overlay  = None     # BGR image with grid lines drawn
_grid_mask     = None     # single-channel mask of where grid pixels are


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


def build_grid_overlay(h, w, grid_px):
    """Pre-render the grid into an overlay image (done ONCE)."""
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    n = len(GRID_LINES)
    M = 2000
    for i in range(n):
        x1, y1 = int(grid_px[i * 2][0]), int(grid_px[i * 2][1])
        x2, y2 = int(grid_px[i * 2 + 1][0]), int(grid_px[i * 2 + 1][1])
        if (-M <= x1 <= w+M and -M <= y1 <= h+M) or \
           (-M <= x2 <= w+M and -M <= y2 <= h+M):
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 180), 1, cv2.LINE_4)
    base = n * 2
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for a in range(3):
        x1, y1 = int(grid_px[base + a*2][0]), int(grid_px[base + a*2][1])
        x2, y2 = int(grid_px[base + a*2 + 1][0]), int(grid_px[base + a*2 + 1][1])
        if (-M <= x1 <= w+M and -M <= y1 <= h+M) or \
           (-M <= x2 <= w+M and -M <= y2 <= h+M):
            cv2.line(overlay, (x1, y1), (x2, y2), colors[a], 2, cv2.LINE_4)
    mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return overlay, mask


def solve_plane_pose(corners_list, ids, K, D_zero):
    obj_pts = []
    img_pts = []
    ids_flat = ids.flatten().tolist()
    for idx, tag_id in enumerate(ids_flat):
        if tag_id not in TAG_CENTRES:
            continue
        wc = tag_corner_world(tag_id)
        ic = corners_list[idx].reshape(4, 2)
        for w, p in zip(wc, ic):
            obj_pts.append(w)
            img_pts.append(p)
    if len(obj_pts) < 4:
        return False, None, None
    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, D_zero, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, D_zero, flags=cv2.SOLVEPNP_ITERATIVE)
    return ok, rvec, tvec


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--calib-time', type=float, default=CALIB_DURATION,
                    help='Seconds to collect tag poses before freezing')
    ap.add_argument('--show-mask', action='store_true',
                    help='Show mask debug window (costs ~2-3 FPS)')
    args = ap.parse_args()
    CALIB_DURATION = args.calib_time

    # ── Open camera ──
    cap, cam_idx = open_camera(args.cam)
    if cap is None:
        raise RuntimeError('No camera found')
    print(f'Camera index: {cam_idx}')

    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 120)             # request max FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # minimize capture latency

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    IMG_SIZE = (actual_w, actual_h)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Resolution: {actual_w}x{actual_h}  cam-FPS: {actual_fps}')

    K = scale_K(K_CALIB, CALIB_SIZE, IMG_SIZE)
    NEW_K, ROI = cv2.getOptimalNewCameraMatrix(
        K, D_CALIB, IMG_SIZE, alpha=0, newImgSize=IMG_SIZE)

    if USE_CUDA:
        MAP1, MAP2 = cv2.initUndistortRectifyMap(
            K, D_CALIB, None, NEW_K, IMG_SIZE, cv2.CV_32FC1)
    else:
        MAP1, MAP2 = cv2.initUndistortRectifyMap(
            K, D_CALIB, None, NEW_K, IMG_SIZE, cv2.CV_16SC2)
    D_ZERO = np.zeros(5)

    # Pre-extract ray constants (avoids per-frame dict/array lookups)
    _ray_fx = NEW_K[0, 0]
    _ray_fy = NEW_K[1, 1]
    _ray_cx = NEW_K[0, 2]
    _ray_cy = NEW_K[1, 2]

    # Upload remap LUTs to GPU (done once)
    if USE_CUDA:
        MAP1_G = cv2.cuda.GpuMat(MAP1)
        MAP2_G = cv2.cuda.GpuMat(MAP2)
        _morph_open  = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_OPEN,  cv2.CV_8UC1, MORPH_K)
        _morph_close = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_CLOSE, cv2.CV_8UC1, MORPH_K)
        _gpu_raw   = cv2.cuda.GpuMat()
        _gpu_frame = cv2.cuda.GpuMat()
    else:
        MAP1_U = cv2.UMat(MAP1)
        MAP2_U = cv2.UMat(MAP2)

    print(f'K  fx={NEW_K[0,0]:.1f} fy={NEW_K[1,1]:.1f} '
          f'cx={NEW_K[0,2]:.1f} cy={NEW_K[1,2]:.1f}')

    # Start threaded capture
    reader = CamReader(cap)
    # Let the reader fill at least one frame
    time.sleep(0.1)

    _fps_time    = time.perf_counter()
    _fps_val     = 0.0
    _calib_start = None
    _4pi = 4.0 * np.pi                         # constant

    try:
        while True:
            ok, raw = reader.read()
            if not ok or raw is None:
                continue                        # reader hasn't delivered yet

            _frame_n += 1

            # ── 1. UNDISTORT ──
            if USE_CUDA:
                _gpu_raw.upload(raw)
                _gpu_frame = cv2.cuda.remap(
                    _gpu_raw, MAP1_G, MAP2_G,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
            else:
                raw_u   = cv2.UMat(raw)
                frame_u = cv2.remap(raw_u, MAP1_U, MAP2_U,
                                    cv2.INTER_LINEAR)

            # ==============================================================
            #  CALIBRATION PHASE
            # ==============================================================
            if not _calib_done:
                frame = _gpu_frame.download() if USE_CUDA else frame_u.get()
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)

                if _calib_start is None:
                    _calib_start = time.perf_counter()

                elapsed   = time.perf_counter() - _calib_start
                remaining = max(0, CALIB_DURATION - elapsed)

                if ids is not None:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    ok_pnp, rv, tv = solve_plane_pose(
                        corners, ids, NEW_K, D_ZERO)
                    if ok_pnp:
                        _calib_rvecs.append(rv.ravel().copy())
                        _calib_tvecs.append(tv.ravel().copy())

                n_poses = len(_calib_rvecs)
                cv2.putText(frame,
                    f'CALIBRATING  {remaining:.1f}s  poses:{n_poses}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 180, 255), 2)

                if elapsed >= CALIB_DURATION:
                    if n_poses > 0:
                        all_rv = np.array(_calib_rvecs)
                        all_tv = np.array(_calib_tvecs)
                        _frozen_rvec = np.median(all_rv, axis=0).reshape(3, 1)
                        _frozen_tvec = np.median(all_tv, axis=0).reshape(3, 1)

                        proj, _ = cv2.projectPoints(
                            _grid_all_pts, _frozen_rvec, _frozen_tvec,
                            NEW_K, D_ZERO)
                        _frozen_grid_px = proj.reshape(-1, 2).astype(np.int32)

                        R_frozen, _  = cv2.Rodrigues(_frozen_rvec)
                        R_T_frozen   = R_frozen.T.astype(np.float64,
                                                          copy=True)
                        cam_pos_frozen = (
                            -R_T_frozen @ _frozen_tvec).ravel().copy()

                        # Pre-render grid overlay (ONE TIME)
                        _grid_overlay, _grid_mask = build_grid_overlay(
                            actual_h, actual_w, _frozen_grid_px)

                        _calib_done = True
                        print(f'[CALIB] LOCKED from {n_poses} poses')
                        print(f'[CALIB] cam=({cam_pos_frozen[0]:.3f},'
                              f'{cam_pos_frozen[1]:.3f},'
                              f'{cam_pos_frozen[2]:.3f})')
                    else:
                        print('[CALIB] No tags seen — retrying...')
                        _calib_start = time.perf_counter()

                cv2.imshow('ballshots5', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ==============================================================
            #  TRACKING PHASE
            # ==============================================================

            # ── 2. HSV mask ──
            if USE_CUDA:
                _gpu_hsv = cv2.cuda.cvtColor(_gpu_frame, cv2.COLOR_BGR2HSV)
                h_g, s_g, v_g = cv2.cuda.split(_gpu_hsv)
                _, h_lo = cv2.cuda.threshold(
                    h_g, int(LO_YELLOW[0]) - 1, 255, cv2.THRESH_BINARY)
                _, h_hi = cv2.cuda.threshold(
                    h_g, int(HI_YELLOW[0]),     255, cv2.THRESH_BINARY_INV)
                _, s_lo = cv2.cuda.threshold(
                    s_g, int(LO_YELLOW[1]) - 1, 255, cv2.THRESH_BINARY)
                _, v_lo = cv2.cuda.threshold(
                    v_g, int(LO_YELLOW[2]) - 1, 255, cv2.THRESH_BINARY)
                mask_g = cv2.cuda.bitwise_and(h_lo, h_hi)
                mask_g = cv2.cuda.bitwise_and(mask_g, s_lo)
                mask_g = cv2.cuda.bitwise_and(mask_g, v_lo)
                mask_g = _morph_open.apply(mask_g)
                mask_g = _morph_close.apply(mask_g)
                mask  = mask_g.download()
                frame = _gpu_frame.download()
            else:
                hsv_u  = cv2.cvtColor(frame_u, cv2.COLOR_BGR2HSV)
                mask_u = cv2.inRange(hsv_u, LO_YELLOW, HI_YELLOW)
                mask_u = cv2.morphologyEx(mask_u, cv2.MORPH_OPEN,  MORPH_K)
                mask_u = cv2.morphologyEx(mask_u, cv2.MORPH_CLOSE, MORPH_K)
                mask  = mask_u.get()
                frame = frame_u.get()

            # ── 3. Contours + ball detection ──
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

            ball_px = None
            if cnts:
                best_cnt  = None
                best_area = 0
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area < 200 or area > 60000:  # quick area gate
                        continue
                    perim = cv2.arcLength(cnt, True)
                    if perim < 1:
                        continue
                    circ = _4pi * area / (perim * perim)
                    if circ < 0.4:
                        continue
                    if area > best_area:
                        best_area = area
                        best_cnt  = cnt

                if best_cnt is not None:
                    (bx, by), r_px = cv2.minEnclosingCircle(best_cnt)
                    ball_px = (bx, by)
                    cv2.circle(frame, (int(bx), int(by)),
                               int(r_px), (0, 255, 0), 2)
                    cv2.circle(frame, (int(bx), int(by)),
                               3, (0, 0, 255), -1)

            # ── 4. Ray → world position (inlined, no function call) ──
            if ball_px is not None:
                ray_cam_x = (ball_px[0] - _ray_cx) / _ray_fx
                ray_cam_y = (ball_px[1] - _ray_cy) / _ray_fy
                # R_T @ [rx, ry, 1]  — manual dot products (3 muls + 2 adds × 3)
                rw0 = R_T_frozen[0,0]*ray_cam_x + R_T_frozen[0,1]*ray_cam_y + R_T_frozen[0,2]
                rw1 = R_T_frozen[1,0]*ray_cam_x + R_T_frozen[1,1]*ray_cam_y + R_T_frozen[1,2]
                rw2 = R_T_frozen[2,0]*ray_cam_x + R_T_frozen[2,1]*ray_cam_y + R_T_frozen[2,2]
                if abs(rw2) > 1e-8:
                    t = -cam_pos_frozen[2] / rw2
                    if t > 0:
                        hx = cam_pos_frozen[0] + t * rw0
                        hy = cam_pos_frozen[1] + t * rw1
                        if _smooth_xy is None:
                            _smooth_xy = np.array([hx, hy])
                        else:
                            _smooth_xy[0] += OUT_ALPHA * (hx - _smooth_xy[0])
                            _smooth_xy[1] += OUT_ALPHA * (hy - _smooth_xy[1])
                        sx = float(_smooth_xy[0])
                        sy = float(_smooth_xy[1])
                        sx_cm = sx * 100.0
                        sy_cm = sy * 100.0

                        positions.append((time.time(), sx, sy))
                        if _frame_n % 60 == 0:
                            print(f'[POS] X:{sx_cm:7.1f} cm  '
                                  f'Y:{sy_cm:7.1f} cm')

                        cv2.putText(frame,
                                    f'X:{sx_cm:.1f}cm  Y:{sy_cm:.1f}cm',
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 0, 0), 2)

            # ── 5. Composite pre-rendered grid (fast copyTo) ──
            np.copyto(frame, _grid_overlay, where=_grid_mask[:,:,None] > 0)

            # ── 6. HUD ──
            now = time.perf_counter()
            dt  = now - _fps_time
            _fps_time = now
            if dt > 0:
                _fps_val = 0.9 * _fps_val + 0.1 / dt
            cv2.putText(frame,
                        f'FPS:{_fps_val:.0f}  LOCKED  Pts:{len(positions)}',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 0), 2)

            cv2.imshow('ballshots5', frame)
            if args.show_mask:
                cv2.imshow('Mask', mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        reader.stop()
        cap.release()
        cv2.destroyAllWindows()

        csv_path = os.path.join(os.path.dirname(__file__),
                                'ball_positions.csv')
        with open(csv_path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['time', 'x_m', 'y_m'])
            if positions:
                wr.writerows(positions)
                print(f'Saved {len(positions)} rows -> {csv_path}')
            else:
                print('No positions recorded.')
