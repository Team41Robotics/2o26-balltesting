"""
ballshots6 — NT-coordinated ball-shot tracker.

See protocol.md for the full NetworkTables handshake.

Usage:
    python ballshots6.py                        # connect to team 41 roboRIO
    python ballshots6.py --nt-server 10.0.41.2  # explicit server IP
    python ballshots6.py --no-nt                 # standalone (no NT)
    python ballshots6.py --show-mask             # show HSV debug window
"""
import argparse, time, json, os, threading, enum, collections, sys
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# ── Camera calibration (PhotonVision/MRCAL, 1280×720) ──
K_CALIB = np.array([
    [545.1385372907386, 0.0,             619.6410820347659],
    [0.0,              545.037239791495, 384.28731732363195],
    [0.0,              0.0,              1.0               ]])
D_CALIB = np.array([
    0.20856149225304135, -0.03663050900029196,
    0.00023804707578467353, -0.0001210337939646964,
    -0.0008277767967346465, 0.5645927654152898,
    -0.03301555724510292, -0.007126953259440723])
CALIB_SIZE = (1280, 720)

# ── Tag layout (metres) ──
# World frame: X right, Y depth (+Y toward the shooter structure), Z up.
# Tags are on a VERTICAL face (XZ plane, Y ≈ 0).
# Layout:
#   Tag 0 (top-left)  ──────  Tag 1 (top-right)
#   Tag 2 (bot-left)  ──────  Tag 3 (bot-right)
#
# Horizontal separation (tag-centre to tag-centre): TAG_SPACING_H = 1.0 m
# Vertical   separation (tag-centre to tag-centre): TAG_SPACING_V = 1.0 m
# Bottom-tag height above floor: half of US letter paper (11 in / 2 = 5.5 in = 0.1397 m)
TAG_SIZE      = 0.1016              # marker side length (metres)
TAG_SPACING_H = 1.0                 # horizontal centre-to-centre (metres)
TAG_SPACING_V = 1.0                 # vertical   centre-to-centre (metres)
TAG_Z_BOTTOM  = (11 * 0.0254) / 2  # = 0.1397 m — bottom-tag Z height
TAG_CENTRES = {
    0: np.array([0.0,          0.0, TAG_Z_BOTTOM + TAG_SPACING_V]),  # top-left
    1: np.array([TAG_SPACING_H, 0.0, TAG_Z_BOTTOM + TAG_SPACING_V]),  # top-right
    2: np.array([0.0,          0.0, TAG_Z_BOTTOM]),                   # bot-left
    3: np.array([TAG_SPACING_H, 0.0, TAG_Z_BOTTOM]),                  # bot-right
}

# ── Grid (metres) ──
GRID_MINOR = 0.25          # quarter-metre lines (dotted)
GRID_MAJOR = 1.0           # full-metre lines (solid)
GRID_EXTENT = 6.0          # ±3 m from origin

# ── Ball detection ──
LO_YELLOW = np.array([25, 100, 100], dtype=np.uint8)
HI_YELLOW = np.array([35, 255, 255], dtype=np.uint8)
MORPH_K   = np.ones((5, 5), np.uint8)

# ── Tracking parameters ──
MIN_RISE_PX       = 20      # ball must rise ≥20 px before we accept descent
BALL_LOST_TIMEOUT = 0.5     # seconds without detection → end trajectory recording
BALL_GONE_TIMEOUT = 2.0     # seconds ball must be absent after trajectory → finalize
MIN_AREA          = 200
MAX_AREA          = 60000
MIN_CIRCULARITY   = 0.4
FOUR_PI           = 4.0 * np.pi

# ── Graph ──
GRAPH_WINDOW      = 10.0    # seconds of history shown
GRAPH_UPDATE_INTERVAL = 0.15  # seconds between plot refreshes

# ── Calibration ──
CALIB_DURATION = 10.0
# PLANE_Y        = -1.0  # world Y (metres from tags) for ball-tracking plane intersection
PLANE_Y        = -1.0  # world Y (metres from tags) for ball-tracking plane intersection


# ═══════════════════════════════════════════════════════════════════════════
#  STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════
class State(enum.Enum):
    CALIBRATING = 'CALIBRATING'
    WAITING     = 'WAITING'
    ARMED       = 'ARMED'
    TRACKING    = 'TRACKING'
    COOLDOWN    = 'COOLDOWN'
    DONE        = 'DONE'


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def scale_K(K_orig, orig_size, new_size):
    sx = new_size[0] / orig_size[0]
    sy = new_size[1] / orig_size[1]
    K = K_orig.copy()
    K[0, 0] *= sx;  K[0, 2] *= sx
    K[1, 1] *= sy;  K[1, 2] *= sy
    return K


def tag_corner_world(tag_id):
    """World coords of the 4 ArUco corners for tag_id.

    Tags are on a VERTICAL face (world XZ plane, Y ≈ 0).
    World: X right, Z up.

    The markers are placed rotated 180° on the structure, so ArUco's corner 0
    (which is image-TL for an upright marker) appears at image-BR here.
    The world assignment therefore starts at the bottom-right physical corner:
      0: bottom-right → (+s, 0, -s)   ← image BR
      1: bottom-left  → (-s, 0, -s)   ← image BL
      2: top-left     → (-s, 0, +s)   ← image TL
      3: top-right    → (+s, 0, +s)   ← image TR
    """
    c = TAG_CENTRES[tag_id]
    s = TAG_SIZE / 2.0
    return np.array([
        c + [+s, 0, -s],   # corner 0: image bottom-right
        c + [-s, 0, -s],   # corner 1: image bottom-left
        c + [-s, 0, +s],   # corner 2: image top-left
        c + [+s, 0, +s],   # corner 3: image top-right
    ], dtype=np.float64)


def build_grid_lines():
    """Return list of (p0, p1, is_major) for the vertical tracking plane.
    
    The plane is at Y = PLANE_Y, spanning X from -EXTENT to +EXTENT 
    and Z from 0 (floor) to +EXTENT.
    """
    lines = []
    n_steps = int(GRID_EXTENT / GRID_MINOR)
    for i in range(-n_steps, n_steps + 1):
        coord = i * GRID_MINOR
        major = (abs(coord % GRID_MAJOR) < 1e-6 or
                 abs(coord % GRID_MAJOR - GRID_MAJOR) < 1e-6)
        
        # Horizontal lines (parallel to X axis, at constant Z)
        # Only draw if Z >= 0
        if coord >= 0:
            lines.append((np.array([-GRID_EXTENT, PLANE_Y, coord]),
                           np.array([ GRID_EXTENT, PLANE_Y, coord]), major))
        
        # Vertical lines (parallel to Z axis, at constant X)
        lines.append((np.array([coord, PLANE_Y, 0.0]),
                       np.array([coord, PLANE_Y, GRID_EXTENT]), major))
    return lines


def _draw_dashed_line(img, pt1, pt2, color, thickness=1,
                      dash=8, gap=6):
    """Draw a dashed line on img between pt1 and pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = (dx*dx + dy*dy) ** 0.5
    if length < 1:
        return
    ux = dx / length
    uy = dy / length
    d = 0.0
    while d < length:
        sx = int(x1 + ux * d)
        sy = int(y1 + uy * d)
        d2 = min(d + dash, length)
        ex = int(x1 + ux * d2)
        ey = int(y1 + uy * d2)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_4)
        d += dash + gap


def build_grid_overlay(h, w, grid_lines, R, tvec, K, D):
    """Build the world-space grid overlay in distorted pixel space.

    Projects world-space grid lines (subdivided) using K and D so that
    they follow the lens distortion of the raw frame.
    XYZ axis arrows are rendered in blue/green/red.
    """
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    rvec, _ = cv2.Rodrigues(R)
    M = 2000
    subdiv = 200  # points per line segment

    # ── Grid lines ──
    for p0, p1, is_major in grid_lines:
        # Subdivide line for smooth distortion curves
        pts_w = np.array([p0 + (p1 - p0) * t for t in np.linspace(0, 1, subdiv)], dtype=np.float64)
        proj, _ = cv2.projectPoints(pts_w, rvec, tvec, K, D)
        px = proj.reshape(-1, 2).astype(np.int32)

        # Check if line is at least partially in view
        in_view = np.any((px[:, 0] >= -M) & (px[:, 0] <= w+M) & 
                         (px[:, 1] >= -M) & (px[:, 1] <= h+M))
        if not in_view:
            continue

        color = (0, 0, 200) if is_major else (0, 0, 120)
        if is_major:
            cv2.polylines(overlay, [px], False, color, 1, cv2.LINE_4)
        else:
            # For dashed lines, we just draw the segments
            for j in range(0, subdiv - 1, 2):
                cv2.line(overlay, tuple(px[j]), tuple(px[j+1]), color, 1, cv2.LINE_4)

    # ── Axis arrows (straight segments are fine for small axis markers) ──
    axis_pts = np.array([
        [0, 0, 0], [1, 0, 0],   # X
        [0, 0, 0], [0, 1, 0],   # Y
        [0, 0, 0], [0, 0, 0.5]  # Z
    ], dtype=np.float64)
    proj_axis, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, D)
    px_axis = proj_axis.reshape(-1, 2).astype(np.int32)
    
    for a, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(overlay, tuple(px_axis[a*2]), tuple(px_axis[a*2+1]), col, 2, cv2.LINE_4)

    mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return overlay, mask


def build_calib_grid_overlay(h, w, K, D, spacing=80, n_samples=200):
    """Grid regular in *undistorted* pixel space, projected through distortion.

    Shows how the calibration model warps a perfectly uniform grid —
    lines are straight at the centre and curve toward the edges.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    rvec0 = tvec0 = np.zeros((3, 1), dtype=np.float64)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    margin = int(max(w, h) * 2)
    sweep = np.linspace(-margin, max(w, h) + margin, n_samples, dtype=np.float64)

    # horiz=False → vertical lines (u fixed, v sweeps)
    # horiz=True  → horizontal lines (v fixed, u sweeps)
    for horiz in (False, True):
        size   = h if horiz else w
        coords = np.arange(-margin, size + margin + 1, spacing, dtype=np.float64)
        for coord in coords:
            pts = np.empty((n_samples, 3), dtype=np.float64)
            pts[:, 0] = (sweep - cx) / fx if horiz else (coord - cx) / fx
            pts[:, 1] = (coord - cy) / fy if horiz else (sweep - cy) / fy
            pts[:, 2] = 1.0
            proj, _ = cv2.projectPoints(pts, rvec0, tvec0, K, D)
            cv2.polylines(overlay, [proj.reshape(-1, 2).astype(np.int32)],
                          False, (255, 220, 60), 1, cv2.LINE_AA)

    # Cross-hair at principal point.
    cp_px, _ = cv2.projectPoints(
        np.array([[[0.0, 0.0, 1.0]]]), rvec0, tvec0, K, D)
    cpx, cpy = int(cp_px[0, 0, 0]), int(cp_px[0, 0, 1])
    cv2.drawMarker(overlay, (cpx, cpy), (255, 255, 100),
                   cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

    mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return overlay, mask


def solve_plane_pose(corners_list, ids, K, D):
    """Return (ok, R 3×3, tvec 3×1) for the tag plane, or (False, None, None).

    Uses SQPNP (robust, no planar ambiguity, any N≥3 points) with an
    ITERATIVE fallback.  Rejects solutions where the camera appears below
    the floor (cam_pos Z ≤ 0) or the reprojection error is too large.
    """
    obj_pts, img_pts = [], []
    for idx, tag_id in enumerate(ids.flatten().tolist()):
        if tag_id not in TAG_CENTRES:
            continue
        wc = tag_corner_world(tag_id)
        ic = corners_list[idx].reshape(4, 2)
        for w, p in zip(wc, ic):
            obj_pts.append(w); img_pts.append(p)
    if len(obj_pts) < 4:
        return False, None, None
    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    # SQPNP: modern, unambiguous solver — works well for planar + non-planar
    ok, rv, tv = cv2.solvePnP(obj_pts, img_pts, K, D,
                               flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        ok, rv, tv = cv2.solvePnP(obj_pts, img_pts, K, D,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return False, None, None

    R, _ = cv2.Rodrigues(rv)

    # Sanity: camera must be in front of the tags (Y < 0 in world frame)
    # Using [1, 0] or .item() to ensure we get a scalar from the tvec array
    cam_y = float((-R.T @ tv).ravel()[1])
    if cam_y >= 0:
        return False, None, None

    # Sanity: reprojection error must be reasonable (< 5 px mean)
    proj, _ = cv2.projectPoints(obj_pts, rv, tv, K, D)
    err = float(np.mean(np.linalg.norm(
        proj.reshape(-1, 2) - img_pts, axis=1)))
    if err > 5.0:
        return False, None, None

    return True, R, tv


def detect_ball(mask):
    """Find the best circular yellow blob in a binary mask.

    Returns ((cx, cy), r_px) or (None, 0).

    Centre is found via distance-transform peak: the pixel furthest
    from any blob edge.  This lands on the thickest (roundest) part of
    the blob, which is the actual ball body even when motion blur
    smears the mask into a long streak.  The peak value is the radius
    of the largest inscribed circle — the ball's true radius, not the
    half-length of the streak.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < 1 or FOUR_PI * area / (perim * perim) < MIN_CIRCULARITY:
            continue
        if area > best_area:
            best_area, best = area, cnt
    if best is None:
        return None, 0

    # Distance-transform peak within the contour's bounding-box ROI.
    bx, by, bw, bh = cv2.boundingRect(best)
    roi  = mask[by:by+bh, bx:bx+bw]
    dist = cv2.distanceTransform(roi, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, r, _, loc = cv2.minMaxLoc(dist)
    return (float(bx + loc[0]), float(by + loc[1])), r


def ray_to_world(ball_px, K, D, R_T, cam_pos, plane_y=PLANE_Y):
    """Unproject a distorted pixel to world XZ at y=plane_y.

    Returns (x_cm, z_cm) or None if the ray misses the plane.
    """
    pts = np.array([[[ball_px[0], ball_px[1]]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, D)[0, 0]
    rw  = R_T @ np.array([und[0], und[1], 1.0])
    if abs(rw[1]) < 1e-8:
        return None
    t = (plane_y - cam_pos[1]) / rw[1]
    if t <= 0:
        return None
    pt = cam_pos + t * rw
    return (pt[0] * 100.0, pt[2] * 100.0)


class CamReader:
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


def default_pose():
    """Synthetic pose: camera ~1.5 m in front of the tag wall, at floor level, looking horizontally.

    World: X right, Y depth (+Y toward the structure), Z up.
    Rotation Rx(+90°): camera-Y = world-(-Z), camera-Z = world(+Y)
      → camera is level, looking along world +Y.
    tvec corresponds to cam_pos ≈ (0.5, -1.5, 0.15) m in world.

    Returns (R 3×3, tvec 3×1).
    """
    R = np.array([[1.,  0.,  0.],
                  [0.,  0., -1.],
                  [0.,  1.,  0.]], dtype=np.float64)   # Rx(+90°)
    tvec = np.array([[-0.5], [0.15], [1.5]], dtype=np.float64)
    return R, tvec


# ═══════════════════════════════════════════════════════════════════════════
#  NETWORKTABLES
# ═══════════════════════════════════════════════════════════════════════════
class NTInterface:
    """Wraps all NT4 reads/writes.  If --no-nt, all methods are no-ops."""

    def __init__(self, team=41, server=None, enabled=True):
        self.enabled = enabled
        if not enabled:
            print('[NT] Disabled (--no-nt)')
            return
        import ntcore                       # type: ignore
        self.inst = ntcore.NetworkTableInstance.getDefault()
        if server:
            self.inst.setServer(server)
        else:
            self.inst.setServerTeam(team)
        self.inst.startClient4('balltracker')

        tbl = self.inst.getTable('BallTracker')
        # ── camera → robot ──
        self.pub_state       = tbl.getStringTopic('camera/state').publish()
        self.pub_ok_shoot    = tbl.getBooleanTopic('camera/ok_to_shoot').publish()
        self.pub_ball_det    = tbl.getBooleanTopic('camera/ball_detected').publish()
        self.pub_shot_count  = tbl.getIntegerTopic('camera/shot_count').publish()
        self.pub_land_x      = tbl.getDoubleTopic('camera/last_land_x_cm').publish()
        self.pub_land_y      = tbl.getDoubleTopic('camera/last_land_y_cm').publish()
        self.pub_fps         = tbl.getDoubleTopic('camera/fps').publish()
        # ── robot → camera ──
        self.sub_shoot_ready = tbl.getBooleanTopic('robot/shoot_ready').subscribe(False)
        self.sub_shot_id     = tbl.getIntegerTopic('robot/shot_id').subscribe(0)
        self.sub_speed       = tbl.getDoubleTopic('robot/shot_speed_rpm').subscribe(0.0)
        self.sub_angle       = tbl.getDoubleTopic('robot/shot_angle_deg').subscribe(0.0)

        # initialise
        self.pub_state.set(State.CALIBRATING.value)
        self.pub_ok_shoot.set(False)
        self.pub_ball_det.set(False)
        self.pub_shot_count.set(0)
        self.pub_land_x.set(0.0)
        self.pub_land_y.set(0.0)
        self.pub_fps.set(0.0)
        print(f'[NT] Client started → team {team}'
              + (f' ({server})' if server else ''))

    # ── publish helpers ──
    def set_state(self, s: State):
        if self.enabled:
            self.pub_state.set(s.value)

    def set_ok_to_shoot(self, v: bool):
        if self.enabled:
            self.pub_ok_shoot.set(v)

    def set_ball_detected(self, v: bool):
        if self.enabled:
            self.pub_ball_det.set(v)

    def set_shot_count(self, n: int):
        if self.enabled:
            self.pub_shot_count.set(n)

    def set_landing(self, x_cm: float, y_cm: float):
        if self.enabled:
            self.pub_land_x.set(x_cm)
            self.pub_land_y.set(y_cm)

    def set_fps(self, fps: float):
        if self.enabled:
            self.pub_fps.set(fps)

    # ── subscribe helpers ──
    def get_shoot_ready(self) -> bool:
        return self.sub_shoot_ready.get() if self.enabled else False

    def get_shot_id(self) -> int:
        return int(self.sub_shot_id.get()) if self.enabled else 0

    def get_speed(self) -> float:
        return self.sub_speed.get() if self.enabled else 0.0

    def get_angle(self) -> float:
        return self.sub_angle.get() if self.enabled else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  JSON HELPERS
# ═══════════════════════════════════════════════════════════════════════════
SHOTS_FILE = 'shots.json'


def load_shots(base):
    """Load existing shots list from JSON, or return empty list."""
    path = os.path.join(base, SHOTS_FILE)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_shot(base, shot_id, speed, angle, traj, all_shots):
    """
    Append this shot to the in-memory list and write the full JSON file.
    traj: list of (abs_time, px_x, px_y, world_x_cm, world_y_cm)
    Returns (land_x, land_y, duration).
    """
    if not traj:
        return 0.0, 0.0, 0.0

    t0 = traj[0][0]
    duration = traj[-1][0] - t0
    land_x = traj[-1][3]
    land_y = traj[-1][4]

    shot_obj = {
        'shot_id':    shot_id,
        'timestamp':  round(t0, 3),
        'speed_rpm':  round(speed, 1),
        'angle_deg':  round(angle, 1),
        'n_frames':   len(traj),
        'duration_s': round(duration, 3),
        'land_x_cm':  round(land_x, 1),
        'land_y_cm':  round(land_y, 1),
        'trajectory': [
            {
                'frame':      i,
                'time_s':     round(t - t0, 4),
                'px_x':       round(px, 1),
                'px_y':       round(py, 1),
                'world_x_cm': round(wx, 1),
                'world_y_cm': round(wy, 1),
            }
            for i, (t, px, py, wx, wy) in enumerate(traj)
        ],
    }

    all_shots.append(shot_obj)

    path = os.path.join(base, SHOTS_FILE)
    with open(path, 'w') as f:
        json.dump(all_shots, f, indent=2)

    print(f'[SHOT {shot_id}] {len(traj)} frames, '
          f'{duration:.2f}s  land=({land_x:.1f}, {land_y:.1f}) cm')
    return land_x, land_y, duration


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--image', type=str, default=None,
                    help='Load an image for one-shot calibration test')
    ap.add_argument('--calib-time', type=float, default=CALIB_DURATION)
    ap.add_argument('--show-mask', action='store_true')
    ap.add_argument('--no-graph', action='store_true',
                    help='Disable live matplotlib graph')
    ap.add_argument('--no-nt', action='store_true',
                    help='Disable NetworkTables (standalone mode)')
    ap.add_argument('--nt-server', type=str, default=None,
                    help='NT server IP (default: auto team 41)')
    ap.add_argument('--team', type=int, default=41)
    args = ap.parse_args()
    CALIB_DURATION = args.calib_time
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_shots = load_shots(BASE_DIR)

    # ── NetworkTables ──
    nt = NTInterface(team=args.team, server=args.nt_server,
                     enabled=not args.no_nt)

    # ── Input source ──
    if args.image:
        cap = None
        cam_idx = -1
        raw_image = cv2.imread(args.image)
        if raw_image is None:
            raise RuntimeError(f'Could not load {args.image}')
        actual_h, actual_w = raw_image.shape[:2]
        print(f'Image mode: {args.image} ({actual_w}x{actual_h})')
    else:
        # ── Open camera ──
        def open_camera(preferred=0, probe_range=6):
            for idx in ([preferred] +
                        [i for i in range(probe_range) if i != preferred]):
                c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if c.isOpened():
                    return c, idx
                c.release()
            return None, None

        cap, cam_idx = open_camera(args.cam)
        if cap is None:
            raise RuntimeError('No camera found')
        print(f'Camera index: {cam_idx}')

        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter.fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 120)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_GAIN, 0)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Resolution: {actual_w}x{actual_h}  '
              f'cam-FPS: {cap.get(cv2.CAP_PROP_FPS)}')

    K = scale_K(K_CALIB, CALIB_SIZE, (actual_w, actual_h))

    # Point-only undistortion: display the raw (distorted) image and
    # undistort only the individual points we care about.
    print(f'[CAM] {actual_w}x{actual_h}  '
          f'fx={K[0,0]:.1f} fy={K[1,1]:.1f}  '
          f'cx={K[0,2]:.1f} cy={K[1,2]:.1f}')

    if USE_CUDA:
        _morph_open  = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_OPEN,  cv2.CV_8UC1, MORPH_K)
        _morph_close = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_CLOSE, cv2.CV_8UC1, MORPH_K)
        _gpu_raw = cv2.cuda.GpuMat()

    # ── AprilTag detector (calibration only) ──
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
    det_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, det_params)

    if args.image:
        print(f"[IMAGE] Analyzing {args.image}...")
        img_raw = cv2.imread(args.image)
        if img_raw is None:
            print(f"Error: Could not read {args.image}")
            sys.exit(1)
        
        corners, ids, rejected = detector.detectMarkers(img_raw)
        if ids is not None and len(ids) > 0:
            print(f"Found {len(ids)} markers.")
            obj_pts = []
            img_pts = []
            for i in range(len(ids)):
                tid = ids[i][0]
                world_corners = tag_corner_world(tid)
                if world_corners is not None:
                    obj_pts.extend(world_corners)
                    img_pts.extend(corners[i][0])
            
            obj_pts = np.array(obj_pts, dtype=np.float32)
            img_pts = np.array(img_pts, dtype=np.float32)
            
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D_CALIB, flags=cv2.SOLVEPNP_SQPNP)
            if success:
                R_mat, _ = cv2.Rodrigues(rvec)
                cam_pos = -R_mat.T @ tvec
                
                # Check reprojection
                proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D_CALIB)
                err = np.sqrt(np.mean(np.sum((proj_pts.squeeze() - img_pts)**2, axis=1)))
                print(f"PnP Success! Err: {err:.2f} px")
                print(f"Cam Pos: X={cam_pos[0,0]:.3f} Y={cam_pos[1,0]:.3f} Z={cam_pos[2,0]:.3f}")

                # Build overlays
                lines = build_grid_lines()
                overlay, mask = build_grid_overlay(actual_h, actual_w, lines, R_mat, tvec, K, D_CALIB)
                
                calib_overlay, calib_mask = build_calib_grid_overlay(actual_h, actual_w, K, D_CALIB)
                
                # Draw markers
                aruco.drawDetectedMarkers(img_raw, corners, ids)
                
                # Combine
                out = img_raw.copy()
                # Apply blue calib grid first
                c_mask = (calib_mask > 0)
                out[c_mask] = cv2.addWeighted(out[c_mask], 0.5, calib_overlay[c_mask], 0.5, 0)
                # Apply red world grid
                w_mask = (mask > 0)
                out[w_mask] = cv2.addWeighted(out[w_mask], 0.3, overlay[w_mask], 0.7, 0)
                
                cv2.imshow("Calibration Result", out)
                print("Press any key to exit.")
                cv2.waitKey(0)
                sys.exit(0)
            else:
                print("PnP failed.")
        else:
            print("No markers found.")
        sys.exit(0)

    GRID_LINES = build_grid_lines()

    # ── Start threaded capture ──
    reader = CamReader(cap)
    time.sleep(0.1)

    # ── Display window (2× upscaled) ──
    DISPLAY_SCALE = 2
    cv2.namedWindow('ballshots6', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ballshots6', actual_w * DISPLAY_SCALE,
                                   actual_h * DISPLAY_SCALE)

    # ── State ──
    state       = State.CALIBRATING
    nt.set_state(state)

    calib_tag_data = collections.defaultdict(list) # tag_id -> list of (4,2) arrays
    calib_start = None

    frozen_R       = None   # 3×3 rotation matrix (world→camera)
    frozen_tvec    = None
    R_T_frozen     = None
    cam_pos_frozen = None
    grid_overlay   = None
    grid_mask_img  = None
    calib_grid_overlay  = None
    calib_grid_mask_img = None

    shot_count     = 0
    cur_traj       = []          # current shot trajectory
    traj_start_py  = 0.0         # pixel-Y when ball first detected
    traj_peaked    = False       # has ball risen ≥ MIN_RISE_PX?
    ball_last_seen = 0.0         # timestamp of last detection
    ball_gone_since = 0.0        # timestamp when ball disappeared (cooldown)
    cur_shot_id    = 0
    cur_speed      = 0.0
    cur_angle      = 0.0

    fps_time = time.perf_counter()
    fps_val  = 0.0
    frame_n  = 0

    # ── Live graph ──
    use_graph = not args.no_graph
    graph_times = collections.deque()   # perf_counter timestamps
    graph_ys    = collections.deque()   # world Y in metres
    graph_last_update = 0.0
    if use_graph:
        plt.ion()
        graph_fig, graph_ax = plt.subplots(figsize=(7, 3))
        graph_fig.canvas.manager.set_window_title('Ball Y — last 10 s')
        graph_line, = graph_ax.plot([], [], 'b-', linewidth=1.5)
        graph_ax.set_xlabel('Time (s)')
        graph_ax.set_ylabel('Y position (m)')
        graph_ax.set_title('Live Ball Y Position')
        graph_ax.grid(True, alpha=0.3)
        graph_fig.tight_layout()
        graph_fig.show()

    try:
        while True:
            ok, raw = reader.read()
            if not ok or raw is None:
                continue
            frame_n += 1

            # ── Raw frame (no remap — point-only undistortion) ──
            frame = raw.copy()
            if USE_CUDA:
                _gpu_raw.upload(raw)

            # ==========================================================
            #  CALIBRATING
            # ==========================================================
            if state == State.CALIBRATING:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)

                if calib_start is None:
                    calib_start = time.perf_counter()
                elapsed   = time.perf_counter() - calib_start
                remaining = max(0, CALIB_DURATION - elapsed)

                if ids is not None:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    for i, tid_arr in enumerate(ids):
                        tid = int(tid_arr[0])
                        calib_tag_data[tid].append(corners[i][0])

                # Count unique tags seen so far
                n_tags = len(calib_tag_data)
                # Count total detections across all tags
                total_dets = sum(len(v) for v in calib_tag_data.values())

                cv2.putText(frame,
                    f'CALIBRATING  {remaining:.1f}s  tags:{n_tags} dets:{total_dets}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 180, 255), 2)

                if elapsed >= CALIB_DURATION:
                    if n_tags > 0:
                        # Average pixel coordinates for each tag
                        avg_corners = []
                        avg_ids = []
                        for tid, corner_list in calib_tag_data.items():
                            if len(corner_list) > 0:
                                # Mean of (N, 4, 2) array over axis 0
                                mean_corner = np.mean(corner_list, axis=0)
                                # solve_plane_pose expects (1, 4, 2) shaped corners per tag
                                avg_corners.append(mean_corner.reshape(1, 4, 2).astype(np.float32))
                                avg_ids.append([tid])
                        
                        avg_ids = np.array(avg_ids, dtype=np.int32)
                        
                        ok_p, R_pose, tv = solve_plane_pose(
                            avg_corners, avg_ids, K, D_CALIB)
                        
                        if ok_p:
                            frozen_R = R_pose
                            frozen_tvec = tv
                            print(f'[CALIB] LOCKED from average of {total_dets} detections')
                        else:
                            frozen_R, frozen_tvec = default_pose()
                            print('[CALIB] PnP failed on averaged corners — using default pose')
                    else:
                        frozen_R, frozen_tvec = default_pose()
                        print('[CALIB] No tags found — using default '
                              'pose (not-to-scale)')

                    R_T_frozen = frozen_R.T.astype(np.float64, copy=True)
                    cam_pos_frozen = (
                        -R_T_frozen @ frozen_tvec).ravel().copy()
                    grid_overlay, grid_mask_img = \
                        build_grid_overlay(actual_h, actual_w,
                                           GRID_LINES,
                                           frozen_R, frozen_tvec, K, D_CALIB)
                    calib_grid_overlay, calib_grid_mask_img = \
                        build_calib_grid_overlay(actual_h, actual_w,
                                                 K, D_CALIB)
                    print(f'[CALIB] cam=({cam_pos_frozen[0]:.3f},'
                          f' {cam_pos_frozen[1]:.3f},'
                          f' {cam_pos_frozen[2]:.3f}) m'
                          f'  [Z should be > 0 = above floor]')
                    state = State.WAITING
                    nt.set_state(state)
                    nt.set_ok_to_shoot(True)
                    nt.set_ball_detected(False)

                cv2.imshow('ballshots6', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ==========================================================
            #  TRACKING PHASES  (WAITING / ARMED / TRACKING / DONE)
            # ==========================================================

            # ── HSV mask (on raw image) ──
            if USE_CUDA:
                gpu_hsv = cv2.cuda.cvtColor(_gpu_raw, cv2.COLOR_BGR2HSV)
                h_g, s_g, v_g = cv2.cuda.split(gpu_hsv)
                _, h_lo = cv2.cuda.threshold(
                    h_g, int(LO_YELLOW[0])-1, 255, cv2.THRESH_BINARY)
                _, h_hi = cv2.cuda.threshold(
                    h_g, int(HI_YELLOW[0]),   255, cv2.THRESH_BINARY_INV)
                _, s_lo = cv2.cuda.threshold(
                    s_g, int(LO_YELLOW[1])-1, 255, cv2.THRESH_BINARY)
                _, v_lo = cv2.cuda.threshold(
                    v_g, int(LO_YELLOW[2])-1, 255, cv2.THRESH_BINARY)
                mask_g = cv2.cuda.bitwise_and(h_lo, h_hi)
                mask_g = cv2.cuda.bitwise_and(mask_g, s_lo)
                mask_g = cv2.cuda.bitwise_and(mask_g, v_lo)
                mask_g = _morph_open.apply(mask_g)
                mask_g = _morph_close.apply(mask_g)
                mask  = mask_g.download()
            else:
                raw_u  = cv2.UMat(raw)
                hsv_u  = cv2.cvtColor(raw_u, cv2.COLOR_BGR2HSV)
                mask_u = cv2.inRange(hsv_u, LO_YELLOW, HI_YELLOW)
                mask_u = cv2.morphologyEx(mask_u, cv2.MORPH_OPEN,  MORPH_K)
                mask_u = cv2.morphologyEx(mask_u, cv2.MORPH_CLOSE, MORPH_K)
                mask  = mask_u.get()

            # ── Contour detection ──
            ball_px, ball_r = detect_ball(mask)
            if ball_px is not None:
                cv2.circle(frame, (int(ball_px[0]), int(ball_px[1])),
                           int(ball_r), (0, 255, 0), 2)
                cv2.circle(frame, (int(ball_px[0]), int(ball_px[1])),
                           3, (0, 0, 255), -1)

            # ── Ray → world ──
            world_xy = (ray_to_world(ball_px, K, D_CALIB,
                                     R_T_frozen, cam_pos_frozen)
                        if ball_px is not None else None)

            now_t = time.time()

            # ==========================================================
            #  STATE: WAITING  — ok_to_shoot is true, watch for robot
            # ==========================================================
            if state == State.WAITING:
                if nt.get_shoot_ready():
                    cur_shot_id = nt.get_shot_id()
                    cur_speed   = nt.get_speed()
                    cur_angle   = nt.get_angle()
                    cur_traj    = []
                    traj_peaked = False
                    state = State.ARMED
                    nt.set_state(state)
                    nt.set_ok_to_shoot(False)
                    nt.set_ball_detected(False)
                    print(f'[ARMED] shot_id={cur_shot_id}  '
                          f'speed={cur_speed:.0f}  angle={cur_angle:.1f}')

            # ==========================================================
            #  STATE: ARMED  — looking for ball to appear
            # ==========================================================
            elif state == State.ARMED:
                if ball_px is not None:
                    traj_start_py  = ball_px[1]   # pixel-Y at first detect
                    ball_last_seen = now_t
                    traj_peaked    = False
                    state = State.TRACKING
                    nt.set_state(state)
                    nt.set_ball_detected(True)
                    print(f'[TRACKING] ball detected at py={traj_start_py:.0f}')

                    # record first point
                    if world_xy:
                        cur_traj.append((now_t, ball_px[0], ball_px[1],
                                         world_xy[0], world_xy[1]))

            # ==========================================================
            #  STATE: TRACKING  — recording trajectory
            # ==========================================================
            elif state == State.TRACKING:
                end_recording = False

                if ball_px is not None:
                    ball_last_seen = now_t

                    # record point
                    wx = world_xy[0] if world_xy else 0.0
                    wy = world_xy[1] if world_xy else 0.0
                    cur_traj.append((now_t, ball_px[0], ball_px[1],
                                     wx, wy))

                    # check if ball has risen enough
                    rise = traj_start_py - ball_px[1]  # positive = upward
                    if rise >= MIN_RISE_PX:
                        traj_peaked = True

                    # check if ball fell back below start height
                    if traj_peaked and ball_px[1] >= traj_start_py:
                        end_recording = True
                        print('[TRACKING] ball fell below start height '
                              '→ COOLDOWN')

                else:
                    # ball lost — check timeout
                    if now_t - ball_last_seen > BALL_LOST_TIMEOUT:
                        end_recording = True
                        print('[TRACKING] ball lost (timeout) → COOLDOWN')

                if end_recording:
                    # stop recording but wait for ball to leave frame
                    ball_gone_since = now_t
                    state = State.COOLDOWN
                    nt.set_state(state)

            # ==========================================================
            #  STATE: COOLDOWN  — trajectory done, wait for ball to
            #                     leave the frame for 2 s before saving
            # ==========================================================
            elif state == State.COOLDOWN:
                if ball_px is not None:
                    # ball still visible (bouncing) — reset the timer
                    ball_gone_since = now_t
                else:
                    # ball not visible — check if gone long enough
                    if now_t - ball_gone_since >= BALL_GONE_TIMEOUT:
                        state = State.DONE
                        print('[COOLDOWN] ball gone for 2 s → DONE')
                        # fall through immediately

            # ==========================================================
            #  STATE: DONE  — save + reset
            # ==========================================================
            if state == State.DONE:
                shot_count += 1
                if cur_shot_id == 0:
                    cur_shot_id = shot_count   # fallback if no NT id

                land_x, land_y, dur = save_shot(
                    BASE_DIR, cur_shot_id, cur_speed, cur_angle,
                    cur_traj, all_shots)

                nt.set_landing(land_x, land_y)
                nt.set_shot_count(shot_count)

                # reset for next shot
                cur_traj = []
                state = State.WAITING
                nt.set_state(state)
                nt.set_ok_to_shoot(True)
                nt.set_ball_detected(False)
                print(f'[WAITING] ready for next shot  '
                      f'({shot_count} total)')

            # ── Grid overlay ──
            np.copyto(frame, calib_grid_overlay,
                      where=calib_grid_mask_img[:, :, None] > 0)
            np.copyto(frame, grid_overlay,
                      where=grid_mask_img[:, :, None] > 0)

            # ── HUD ──
            now_pc = time.perf_counter()
            dt = now_pc - fps_time
            fps_time = now_pc
            if dt > 0:
                fps_val = 0.9 * fps_val + 0.1 / dt
            nt.set_fps(fps_val)

            # state colour
            state_colors = {
                State.WAITING:  (0, 200, 0),
                State.ARMED:    (0, 180, 255),
                State.TRACKING: (0, 0, 255),
                State.COOLDOWN: (180, 0, 180),
                State.DONE:     (255, 0, 0),
            }
            sc = state_colors.get(state, (200, 200, 200))
            cv2.putText(frame,
                f'FPS:{fps_val:.0f}  {state.value}  '
                f'shots:{shot_count}',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc, 2)

            if state == State.TRACKING and cur_traj:
                cv2.putText(frame,
                    f'traj: {len(cur_traj)} pts',
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

            if state == State.COOLDOWN:
                wait_left = BALL_GONE_TIMEOUT - (now_t - ball_gone_since)
                cv2.putText(frame,
                    f'waiting for ball to leave ({wait_left:.1f}s)',
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (180, 0, 180), 2)

            if world_xy and ball_px:
                cv2.putText(frame,
                    f'X:{world_xy[0]/100:.2f}m  Y:{world_xy[1]/100:.2f}m',
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

            # ── Live graph update ──
            if use_graph:
                if world_xy is not None:
                    # world_xy is in cm → convert to metres
                    graph_times.append(now_pc)
                    graph_ys.append(world_xy[1] / 100.0)

                # prune points older than GRAPH_WINDOW
                cutoff = now_pc - GRAPH_WINDOW
                while graph_times and graph_times[0] < cutoff:
                    graph_times.popleft()
                    graph_ys.popleft()

                # throttle plot refreshes
                if now_pc - graph_last_update >= GRAPH_UPDATE_INTERVAL:
                    graph_last_update = now_pc
                    if graph_times:
                        ts = np.array(graph_times)
                        ts = ts - ts[-1]           # relative, 0 = now
                        ys = np.array(graph_ys)
                        graph_line.set_data(ts, ys)
                        graph_ax.set_xlim(-GRAPH_WINDOW, 0.5)
                        y_lo = float(ys.min()) - 0.3
                        y_hi = float(ys.max()) + 0.3
                        if y_hi - y_lo < 0.6:
                            mid = (y_hi + y_lo) / 2
                            y_lo, y_hi = mid - 0.3, mid + 0.3
                        graph_ax.set_ylim(y_lo, y_hi)
                    try:
                        graph_fig.canvas.draw_idle()
                        graph_fig.canvas.flush_events()
                    except Exception:
                        pass

            cv2.imshow('ballshots6', frame)
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
        if use_graph:
            plt.close('all')
        print(f'Session: {shot_count} shots tracked.')
