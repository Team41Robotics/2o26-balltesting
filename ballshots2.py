import cv2, cv2.aruco as aruco, numpy as np, time, csv, os

# --- Camera ---
fx = fy = 853.33
cx, cy = 640.0, 360.0
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
D = np.zeros((5,1))

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
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    blur = cv2.GaussianBlur(frame, (25,25), 0)

    # ---- Ball: find centroid via moments ----
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
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
            r_px = int(np.sqrt(M['m00'] / np.pi)) # approx radius for drawing
            cv2.circle(frame, (int(bx),int(by)), r_px, (0,255,0), 2)
            cv2.circle(frame, (int(bx),int(by)), 3, (0,0,255), -1)

    # ---- AprilTags: detect + fit plane through 3-D corners ----
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and ball_px is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, TAG_SIZE, K, D)
        aruco.drawDetectedMarkers(frame, corners)

        # collect every tag corner in camera coords
        pts3d = []
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            t = tv.reshape(3,1)
            for c in TAG_CORNERS_LOCAL:
                pts3d.append((R @ c.reshape(3,1) + t).ravel())
        pts3d = np.array(pts3d)                   # (N*4) x 3

        # best-fit plane (SVD)
        centroid = pts3d.mean(axis=0)
        _, _, vt = np.linalg.svd(pts3d - centroid)
        n = vt[-1]; n /= np.linalg.norm(n)        # plane normal

        # ray from camera through ball pixel → intersect plane
        r = np.array([(ball_px[0]-cx)/fx, (ball_px[1]-cy)/fy, 1.0])
        denom = r @ n
        if abs(denom) > 1e-6:
            t_ray = (centroid @ n) / denom
            P = t_ray * r                         # 3-D point on plane (cam frame)
            
            # compute ball's actual camera-frame position (assuming it's not on the plane)
            # we need the real 3D point — for now, use a heuristic or just check plane distance
            # Better: we'll back out the ball center from the pixel, assume some Z or use the intersection + height
            
            # For simplicity: compute signed distance from a "guess" ball position to plane
            # Since we don't have real depth, we'll use the projection and add a height check later
            # Actually: let's assume the ball is slightly above the plane in camera Z
            # We'll accept only if the ball pixel projects to a reasonable Z in cam coords
            
            # Alternative: just filter by tag-frame Z > 0 (ball above tag plane in tag coords)
            # reference tag = id 0 if visible, else first tag
            ids_flat = ids.flatten().tolist()
            ref = ids_flat.index(0) if 0 in ids_flat else 0
            R_ref, _ = cv2.Rodrigues(rvecs[ref])
            t_ref = tvecs[ref].reshape(3,1)

            # transform into tag-0 frame
            p_tag = (R_ref.T @ (P.reshape(3,1) - t_ref)).ravel()
            
            # Filter: only record if Z > 0 in tag frame (ball above the tag plane)
            if p_tag[2] > 0:
                cv2.drawFrameAxes(frame, K, D, rvecs[ref], tvecs[ref], 2)
                positions.append((time.time(), *p_tag))

                cv2.putText(frame, f"X:{p_tag[0]:.1f} Y:{p_tag[1]:.1f} Z:{p_tag[2]:.1f} in",
                            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("Ball Tracking", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# save to CSV
if positions:
    path = os.path.join(os.path.dirname(__file__), "ball_positions.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time","x_in","y_in","z_in"])
        w.writerows(positions)
    print(f"Saved {len(positions)} rows → {path}")
else:
    print("No positions recorded.")
