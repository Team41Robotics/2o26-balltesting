import argparse
import cv2, cv2.aruco as aruco, numpy as np, time, csv, os

# --- Sanity limits ---
MAX_RANGE = 120.0       # inches — max plausible distance from tag to ball
MAX_TAG_DIST = 200.0    # inches — max plausible camera-to-tag distance

# --- Camera (from provided calibration) ---
K = np.array([[1634.435694951629, 0.0, 977.0095538990062],
              [0.0, 1617.0040060442561, 593.2899951200168],
              [0.0, 0.0, 1.0]])
D = np.array([  0.13479540133744689,
                -0.566065131074218,
                0.00018389150325768655,
                0.003664585948224397,
                0.8656340100414646])

# Validate calibration data
assert K.shape == (3, 3), f"K must be 3x3, got {K.shape}"
assert K[2, 2] == 1.0, "K[2,2] must be 1"
assert K[0, 0] > 0 and K[1, 1] > 0, "Focal lengths must be positive"
assert D.shape[0] == 5, f"Expected 5 distortion coefficients, got {D.shape[0]}"
# Compute optimal undistortion map once (alpha=1 keeps all pixels; 0 crops aggressively)
IMG_SIZE = (1920, 1080)
NEW_K, ROI = cv2.getOptimalNewCameraMatrix(K, D, IMG_SIZE, alpha=1, newImgSize=IMG_SIZE)
MAP1, MAP2 = cv2.initUndistortRectifyMap(K, D, None, NEW_K, IMG_SIZE, cv2.CV_16SC2)
# After undistortion the image is distortion-free → use NEW_K and zero distortion downstream
D_ZERO = np.zeros(5)
# derived intrinsics (from the NEW camera matrix used after undistortion)
fx = float(NEW_K[0,0]); fy = float(NEW_K[1,1]); cx = float(NEW_K[0,2]); cy = float(NEW_K[1,2])

# Validate derived intrinsics
assert NEW_K.shape == (3, 3), f"NEW_K must be 3x3, got {NEW_K.shape}"
assert fx > 0 and fy > 0, f"Focal lengths must be positive: fx={fx}, fy={fy}"
assert 0 < cx < IMG_SIZE[0], f"cx out of image bounds: {cx}"
assert 0 < cy < IMG_SIZE[1], f"cy out of image bounds: {cy}"

# --- AprilTag 16h5 ---
detector = aruco.ArucoDetector(
    aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5),
    aruco.DetectorParameters())
TAG_SIZE = 4.0          # inches
HALF = TAG_SIZE / 2.0
# tag-local corner offsets (TL, TR, BR, BL)
TAG_CORNERS_LOCAL = np.float32([[-HALF,HALF,0],[HALF,HALF,0],[HALF,-HALF,0],[-HALF,-HALF,0]])

# --- HSV thresholds for yellow ball ---
LO_YELLOW = np.array([14, 70, 180])
HI_YELLOW = np.array([35, 255, 255])
KERNEL = np.ones((5,5), np.uint8)

# --- State ---
positions = []
def open_camera(preferred=0, probe_range=5):
    """Try to open preferred camera index, else probe 0..probe_range-1 and return (cap, idx)."""
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
    p = argparse.ArgumentParser(description='Ballshots tracker')
    p.add_argument('--cam', type=int, default=0, help='preferred camera index')
    args = p.parse_args()

    cap, cam_idx = open_camera(args.cam, probe_range=6)
    if cap is None:
        raise RuntimeError('No camera available (probed indices 0..5)')
    print(f'Using camera index: {cam_idx}')
    # request the calibrated image size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # undistort entire frame first (fisheye lens correction via precomputed maps)
            frame = cv2.remap(frame, MAP1, MAP2, cv2.INTER_LINEAR)
            assert frame is not None, "Remap produced None frame"
            assert frame.shape[:2] == (IMG_SIZE[1], IMG_SIZE[0]), \
                f"Frame size mismatch after undistort: {frame.shape[:2]} vs expected {(IMG_SIZE[1], IMG_SIZE[0])}"
            blur = cv2.GaussianBlur(frame, (11,11), 0)

            # ---- Ball: find centroid via moments (use heavier blur for color) ----
            ball_blur = cv2.GaussianBlur(frame, (25,25), 0)
            hsv = cv2.cvtColor(ball_blur, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LO_YELLOW, HI_YELLOW)
            mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL), cv2.MORPH_CLOSE, KERNEL)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ball_px = None
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M['m00'] > 200:                       # min-area gate
                    bx = M['m10'] / M['m00']
                    by = M['m01'] / M['m00']
                    ball_px = (bx, by)
                    assert 0 <= bx < IMG_SIZE[0], f"Ball x={bx:.1f} outside image width {IMG_SIZE[0]}"
                    assert 0 <= by < IMG_SIZE[1], f"Ball y={by:.1f} outside image height {IMG_SIZE[1]}"
                    r_px = int(np.sqrt(M['m00'] / np.pi)) # approx radius for drawing
                    cv2.circle(frame, (int(bx),int(by)), r_px, (0,255,0), 2)
                    cv2.circle(frame, (int(bx),int(by)), 3, (0,0,255), -1)

            # ---- AprilTags: detect + fit plane through 3-D corners ----
            # use lighter blur for tag detection so edges stay sharp
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            # debug: print detected tag ids each frame
            if ids is not None:
                try:
                    print(f"[DEBUG] tags={ids.flatten().tolist()}")
                except Exception:
                    print(f"[DEBUG] tags found (ids shape)={ids.shape}")

            if ids is not None and ball_px is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, TAG_SIZE, NEW_K, D_ZERO)
                assert rvecs.shape[0] == tvecs.shape[0] == len(ids), \
                    f"Pose count mismatch: rvecs={rvecs.shape[0]}, tvecs={tvecs.shape[0]}, ids={len(ids)}"

                # Validate each tag pose — distance must be positive and plausible
                tag_dists = []
                for tv in tvecs:
                    d = float(np.linalg.norm(tv))
                    tag_dists.append(d)
                    assert d > 0, f"Tag distance must be positive, got {d}"
                # Skip frame entirely if any tag is implausibly far (bad pose estimate)
                if any(d > MAX_TAG_DIST for d in tag_dists):
                    cv2.imshow("Ball Tracking", frame)
                    cv2.imshow("Mask", mask)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                aruco.drawDetectedMarkers(frame, corners)
                # draw axes for all detected tags (use visible length = TAG_SIZE * 1.5)
                axis_len = float(TAG_SIZE * 1.5)
                for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
                    rvv = rv.reshape((3,1))
                    tvv = tv.reshape((3,1))
                    try:
                        cv2.drawFrameAxes(frame, NEW_K, D_ZERO, rvv, tvv, axis_len)
                    except Exception:
                        # fallback if shapes differ
                        cv2.drawFrameAxes(frame, NEW_K, D_ZERO, rv.reshape(3,), tv.reshape(3,), axis_len)
                    # draw projected corners to help debug pose
                    for corner_tag in TAG_CORNERS_LOCAL:
                        p_img, _ = cv2.projectPoints(corner_tag.reshape((1,3)), rvv, tvv, NEW_K, D_ZERO)
                        px, py = int(p_img[0,0,0]), int(p_img[0,0,1])
                        cv2.circle(frame, (px, py), 3, (255,0,255), -1)

                # --- Build plane from tag centers + tag normals ---
                # Collect each tag's center (tvec) and local Z-axis (normal) in camera frame
                tag_centers = []
                tag_normals = []
                for rv, tv in zip(rvecs, tvecs):
                    R, _ = cv2.Rodrigues(rv)
                    tag_centers.append(tv.ravel())
                    # The tag's local Z-axis in camera coords is the 3rd column of R
                    tag_normals.append(R[:, 2])
                tag_centers = np.array(tag_centers)       # (N_tags) x 3
                tag_normals = np.array(tag_normals)       # (N_tags) x 3
                assert tag_centers.shape[1] == 3, f"Tag centers must be 3D, got {tag_centers.shape}"

                n_tags = len(tag_centers)
                if n_tags == 1:
                    # Single tag: use its local Z-axis directly as plane normal
                    centroid = tag_centers[0]
                    n = tag_normals[0].copy()
                else:
                    # Multiple tags: fit plane through tag centers via SVD
                    centroid = tag_centers.mean(axis=0)
                    _, _, vt = np.linalg.svd(tag_centers - centroid)
                    n = vt[-1].copy()
                    # Refine: orient normal to agree with the average tag Z-axis
                    avg_tag_z = tag_normals.mean(axis=0)
                    if n @ avg_tag_z < 0:
                        n = -n

                n /= np.linalg.norm(n)
                assert abs(np.linalg.norm(n) - 1.0) < 1e-6, f"Plane normal not unit length: {np.linalg.norm(n)}"
                # ensure normal points toward camera (origin) so denom sign is consistent
                if n @ centroid > 0:
                    n = -n

                # ray from camera through ball pixel → intersect plane
                r = np.array([(ball_px[0]-cx)/fx, (ball_px[1]-cy)/fy, 1.0])
                denom = r @ n

                # Guard: reject near-parallel rays (causes huge coordinate blowups)
                if abs(denom) > 0.1:
                    t_ray = (centroid @ n) / denom

                    # Guard: intersection must be in front of camera
                    if t_ray > 0:
                        P = t_ray * r                     # 3-D point on plane (cam frame)
                        assert np.isfinite(P).all(), f"Non-finite intersection point: {P}"
                        assert np.linalg.norm(P) < MAX_TAG_DIST * 2, \
                            f"Intersection too far from camera: {np.linalg.norm(P):.1f} in"

                        # reference tag = id 0 if visible, else first tag
                        ids_flat = ids.flatten().tolist()
                        ref = ids_flat.index(0) if 0 in ids_flat else 0
                        R_ref, _ = cv2.Rodrigues(rvecs[ref])
                        t_ref = tvecs[ref].reshape(3,1)

                        # transform into tag-0 frame
                        p_tag = (R_ref.T @ (P.reshape(3,1) - t_ref)).ravel()
                        assert np.isfinite(p_tag).all(), f"Non-finite tag-frame coords: {p_tag}"

                        # Sanity bounds: reject if any coordinate is unreasonably large
                        # (ball should be within ~MAX_RANGE inches of any tag in practice)
                        if (abs(p_tag[0]) < MAX_RANGE and
                            abs(p_tag[1]) < MAX_RANGE and
                            abs(p_tag[2]) < MAX_RANGE and
                            p_tag[2] > 0.01):             # Z must be meaningfully > 0
                            cv2.drawFrameAxes(frame, NEW_K, D_ZERO, rvecs[ref], tvecs[ref], 2)
                            row = (time.time(), *p_tag)
                            assert all(np.isfinite(v) for v in row), f"Non-finite value in row: {row}"
                            assert all(abs(v) < 1e9 for v in row), f"Absurdly large value in row: {row}"
                            positions.append(row)

                            cv2.putText(frame, f"X:{p_tag[0]:.1f} Y:{p_tag[1]:.1f} Z:{p_tag[2]:.1f} in",
                                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            cv2.imshow("Ball Tracking", frame)
            cv2.imshow("Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # save to CSV (always attempt to save recorded positions)
        path = os.path.join(os.path.dirname(__file__), "ball_positions.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time","x_in","y_in","z_in"])
            if positions:
                w.writerows(positions)
                print(f"Saved {len(positions)} rows → {path}")
            else:
                print("No positions recorded.")
