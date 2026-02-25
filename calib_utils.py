"""
calib_utils.py — AprilTag layout, PnP solving, and grid overlay helpers.

Imported by ballshots7.py; can also be used standalone for testing.
"""
import cv2
import numpy as np

# ── Tag layout (metres) ──
# World frame: X right, Y depth (+Y toward the shooter), Z up.
# Tags are on a VERTICAL face (XZ plane, Y ≈ 0).
#   Tag 0 (top-left)  ──  Tag 1 (top-right)
#   Tag 2 (bot-left)  ──  Tag 3 (bot-right)
TAG_SIZE      = 0.2               # marker side length (metres) — 4 inches
TAG_SPACING_H = 1.0                  # horizontal centre-to-centre (metres)
TAG_SPACING_V = 1.0                  # vertical centre-to-centre (metres)
TAG_Z_BOTTOM  = (11 * 0.0254) / 2   # = 0.1397 m — bottom-tag centre height
TAG_CENTRES = {
    0: np.array([0.0,          0.0, TAG_Z_BOTTOM + TAG_SPACING_V]),  # top-left
    1: np.array([TAG_SPACING_H, 0.0, TAG_Z_BOTTOM + TAG_SPACING_V]),  # top-right
    2: np.array([0.0,          0.0, TAG_Z_BOTTOM]),                   # bot-left
    3: np.array([TAG_SPACING_H, 0.0, TAG_Z_BOTTOM]),                  # bot-right
}

# ── Tracking plane ──
PLANE_Y = -1.0   # world Y where ball positions are projected (metres from tag plane)

# ── Grid display ──
GRID_MINOR  = 0.25   # minor grid spacing (m)
GRID_MAJOR  = 1.0    # major grid spacing (m)
GRID_EXTENT = 6.0    # half-width/height of grid (m)


def scale_K(K_orig, orig_size, new_size):
    """Scale camera intrinsic matrix for a different resolution."""
    sx = new_size[0] / orig_size[0]
    sy = new_size[1] / orig_size[1]
    K = K_orig.copy()
    K[0, 0] *= sx;  K[0, 2] *= sx
    K[1, 1] *= sy;  K[1, 2] *= sy
    return K


def tag_corner_world(tag_id):
    """Return 4×3 world coords for tag_id corners (TL, TR, BR, BL order)."""
    c = TAG_CENTRES[tag_id]
    s = TAG_SIZE / 2.0
    return np.array([
        c + [-s, 0, +s],   # TL
        c + [+s, 0, +s],   # TR
        c + [+s, 0, -s],   # BR
        c + [-s, 0, -s],   # BL
    ], dtype=np.float64)


def default_pose():
    """Synthetic warm-start pose: camera ~1.5 m in front of tag wall, level.

    Returns (R 3×3, tvec 3×1).
    """
    R = np.array([[1.,  0.,  0.],
                  [0.,  0., -1.],
                  [0.,  1.,  0.]], dtype=np.float64)
    tvec = np.array([[-0.5], [0.15], [1.5]], dtype=np.float64)
    return R, tvec


def solve_plane_pose(corners_list, ids, K, D):
    """Return (ok, R 3×3, tvec 3×1) for the tag plane, or (False, None, None).

    Tries SQPNP first, then ITERATIVE with warm-start, then cold ITERATIVE.
    Rejects solutions where cam_y >= 0 (camera behind tags).
    """
    obj_pts, img_pts = [], []
    for i, tag_id in enumerate(ids.flatten().tolist()):
        if tag_id not in TAG_CENTRES:
            continue
        for wp, ip in zip(tag_corner_world(tag_id), corners_list[i].reshape(4, 2)):
            obj_pts.append(wp)
            img_pts.append(ip)
    if len(obj_pts) < 4:
        return False, None, None

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    R0, tv0 = default_pose()
    rv0, _ = cv2.Rodrigues(R0)

    attempts = [
        (cv2.SOLVEPNP_SQPNP,     None,        None,        False),
        (cv2.SOLVEPNP_ITERATIVE,  rv0.copy(),  tv0.copy(),  True),
        (cv2.SOLVEPNP_ITERATIVE,  None,        None,        False),
    ]
    labels = {cv2.SOLVEPNP_SQPNP: 'SQPNP', cv2.SOLVEPNP_ITERATIVE: 'ITER'}

    for flags, init_rv, init_tv, use_guess in attempts:
        lbl = labels[flags] + ('+guess' if use_guess else '')
        if use_guess:
            ok, rv, tv = cv2.solvePnP(
                obj_pts, img_pts, K, D,
                rvec=init_rv, tvec=init_tv,
                useExtrinsicGuess=True, flags=flags)
        else:
            ok, rv, tv = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flags)
        if not ok:
            print(f'[PnP/{lbl}] solver returned False')
            continue

        R, _ = cv2.Rodrigues(rv)
        cam_pos = (-R.T @ tv).ravel()
        proj, _ = cv2.projectPoints(obj_pts, rv, tv, K, D)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))
        # print(f'[PnP/{lbl}] npts={len(obj_pts)}'
            #   f'  cam=({cam_pos[0]:.3f},{cam_pos[1]:.3f},{cam_pos[2]:.3f})m'
            #   f'  reproj={err:.2f}px', end='')

        if cam_pos[1] >= 0:
            print('  → REJECT (cam behind tags)')
            continue
        print('  → OK')
        return True, R, tv

    return False, None, None


def build_grid_lines():
    """Return list of (p0, p1, is_major) world-space line segments for the tracking plane."""
    lines = []
    n = int(GRID_EXTENT / GRID_MINOR)
    for i in range(-n, n + 1):
        coord = i * GRID_MINOR
        major = abs(coord % GRID_MAJOR) < 1e-6 or abs(coord % GRID_MAJOR - GRID_MAJOR) < 1e-6
        if coord >= 0:  # horizontal lines at Z=coord
            lines.append((np.array([-GRID_EXTENT, PLANE_Y, coord]),
                          np.array([ GRID_EXTENT, PLANE_Y, coord]), major))
        # vertical lines at X=coord
        lines.append((np.array([coord, PLANE_Y, 0.0]),
                      np.array([coord, PLANE_Y, GRID_EXTENT]), major))
    return lines


def build_grid_overlay(h, w, grid_lines, R, tvec, K, D):
    """Project world-space grid onto distorted image space.

    Returns (overlay BGR image, binary mask).
    """
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    rvec, _ = cv2.Rodrigues(R)
    subdiv = 200
    M = 2000

    for p0, p1, is_major in grid_lines:
        pts_w = np.array([p0 + (p1 - p0) * t
                          for t in np.linspace(0, 1, subdiv)], dtype=np.float64)
        proj, _ = cv2.projectPoints(pts_w, rvec, tvec, K, D)
        px = proj.reshape(-1, 2).astype(np.int32)
        if not np.any((px[:, 0] >= -M) & (px[:, 0] <= w + M) &
                      (px[:, 1] >= -M) & (px[:, 1] <= h + M)):
            continue
        color = (0, 0, 200) if is_major else (0, 0, 120)
        if is_major:
            cv2.polylines(overlay, [px], False, color, 1, cv2.LINE_4)
        else:
            for j in range(0, subdiv - 1, 2):
                cv2.line(overlay, tuple(px[j]), tuple(px[j + 1]), color, 1, cv2.LINE_4)

    # XYZ axis arrows (1m each, origin at world [0,0,0])
    axis_pts = np.array([[0,0,0],[1,0,0],      # X axis (red)
                         [0,0,0],[0,1,0],      # Y axis (green)
                         [0,0,0],[0,0,1]], dtype=np.float64)  # Z axis (blue)
    proj_ax, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, D)
    px_ax = proj_ax.reshape(-1, 2).astype(np.int32)

    # Draw axis lines with arrowheads
    axis_colors = [(0,0,255), (0,255,0), (255,0,0)]  # BGR: red, green, blue
    axis_labels = ['X', 'Y', 'Z']
    for a, (col, label) in enumerate(zip(axis_colors, axis_labels)):
        p0, p1 = tuple(px_ax[a*2]), tuple(px_ax[a*2+1])
        # Draw line
        cv2.line(overlay, p0, p1, col, 3, cv2.LINE_AA)
        # Draw arrowhead at endpoint
        cv2.circle(overlay, p1, 4, col, -1, cv2.LINE_AA)
        # Draw label near endpoint (offset for visibility)
        offset = np.array([8, -8])
        label_pos = tuple(p1 + offset)
        cv2.putText(overlay, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)

    _, mask = cv2.threshold(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    return overlay, mask


def build_calib_grid_overlay(h, w, K, D, spacing=80, n_samples=200):
    """Draw a uniform undistorted grid re-projected through the distortion model.

    Visualises how the calibration warps straight lines.
    Returns (overlay BGR image, binary mask).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = np.zeros((3, 1), dtype=np.float64)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    margin = int(max(w, h) * 2)
    sweep = np.linspace(-margin, max(w, h) + margin, n_samples, dtype=np.float64)

    for horiz in (False, True):
        size   = h if horiz else w
        coords = np.arange(-margin, size + margin + 1, spacing, dtype=np.float64)
        for coord in coords:
            pts = np.empty((n_samples, 3), dtype=np.float64)
            pts[:, 0] = (sweep - cx) / fx if horiz else (coord - cx) / fx
            pts[:, 1] = (coord - cy) / fy if horiz else (sweep - cy) / fy
            pts[:, 2] = 1.0
            proj, _ = cv2.projectPoints(pts, z, z, K, D)
            cv2.polylines(overlay, [proj.reshape(-1, 2).astype(np.int32)],
                          False, (255, 220, 60), 1, cv2.LINE_AA)

    cp, _ = cv2.projectPoints(np.array([[[0., 0., 1.]]]), z, z, K, D)
    cv2.drawMarker(overlay, (int(cp[0,0,0]), int(cp[0,0,1])),
                   (255, 255, 100), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

    _, mask = cv2.threshold(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    return overlay, mask
