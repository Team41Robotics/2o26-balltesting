import cv2
import cv2.aruco as aruco
import numpy as np
import time
import csv
import os

diameter_avg = []
positions = []  # list to store recorded ball positions (relative to tag 0)
# ----------------------
# Camera parameters (approx. for MSI Sword 15 720p)
# ----------------------
fx = fy = 853.33
cx, cy = 1280 / 2, 720 / 2
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])
dist_coeffs = np.zeros((5, 1))

# ----------------------
# AprilTag 16h5 dictionary
# ----------------------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
TAG_SIZE_INCH = 4.0

# ----------------------
# Ball parameters
# ----------------------
BALL_DIAMETER_IN = 6.0
REF_DISTANCE_IN = 4.0
REF_PIXEL_DIAMETER = 1280.0
FOCAL_LENGTH_PX = REF_PIXEL_DIAMETER * REF_DISTANCE_IN / BALL_DIAMETER_IN

# ----------------------
# Video capture
# ----------------------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# frame counter
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    frame_blur = cv2.GaussianBlur(frame, (25, 25), 0)

    # -------- Yellow ball detection --------
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    # lower_yellow = np.array([17, 140, 150])
    # upper_yellow = np.array([35, 255, 255])
    
    # note: heavy backlighting on blue USB camera; above values 4 latop; below adjusted for camera
    # top half of ball is heavily overexposed
    lower_yellow = np.array([14, 70, 180])
    upper_yellow = np.array([35, 255, 255])


    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_cam = None
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        # average out diameter for less flickering4
        diameter_px = radius * 2
        diameter_avg.append(diameter_px)
        if len(diameter_avg) >= 5:
            diameter_avg.pop(0)
        diameter_px = sum(diameter_avg) / len(diameter_avg)

        if diameter_px > 25:
            Z_cam = FOCAL_LENGTH_PX * BALL_DIAMETER_IN / diameter_px
            X_cam = (x - cx) * Z_cam / fx
            Y_cam = (y - cy) * Z_cam / fy
            ball_cam = np.array([X_cam, Y_cam, Z_cam]).reshape((3,1))
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # -------- AprilTag detection --------
    gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and ball_cam is not None:
        # Estimate pose for all detected tags (tvecs are tag origins in camera coords, in inches)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, TAG_SIZE_INCH, camera_matrix, dist_coeffs)
        aruco.drawDetectedMarkers(frame, corners)

        # Collect tag centers (tvecs) for plane fitting
        tag_centers = [t.reshape((3,)) for t in tvecs]
        pts = np.vstack(tag_centers)  # N x 3

        # Fit best plane through tag centers using SVD
        centroid = pts.mean(axis=0)
        uu, dd, vv = np.linalg.svd(pts - centroid)
        normal = vv[-1, :]
        normal = normal / np.linalg.norm(normal)

        # Project ball center (camera coords) to the plane: P_proj = P - ((P - p0)·n) * n
        P = ball_cam.reshape((3,))
        P_proj = P - ((P - centroid) @ normal) * normal

        # Find index of AprilTag ID 0 to use as reference frame. If not present, fall back to first detected tag
        ids_list = ids.flatten().tolist()
        if 0 in ids_list:
            ref_idx = ids_list.index(0)
            ref_id = 0
        else:
            ref_idx = 0
            ref_id = int(ids_list[0])

        # Pose of reference tag
        rvec_ref = rvecs[ref_idx].reshape((3,1))
        tvec_ref = tvecs[ref_idx].reshape((3,1))
        R_ref, _ = cv2.Rodrigues(rvec_ref)

        # Transform projected point into reference tag frame: p_tag = R_ref^T * (P_proj - tvec_ref)
        p_cam = P_proj.reshape((3,1))
        p_tag = R_ref.T @ (p_cam - tvec_ref)
        X_tag, Y_tag, Z_tag = p_tag.flatten()

        # Draw axis for reference tag
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_ref, tvec_ref, 2)

        # Record position with timestamp and frame number
        positions.append({
            "time": time.time(),
            "frame": frame_no,
            "ref_tag": int(ref_id),
            "x_in": float(X_tag),
            "y_in": float(Y_tag),
            "z_in": float(Z_tag),
        })

        cv2.putText(frame, f"Ball X_tag{ref_id}: {X_tag:.2f} in", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(frame, f"Ball Y_tag{ref_id}: {Y_tag:.2f} in", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(frame, f"Ball Z_tag{ref_id}: {Z_tag:.2f} in", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.imshow("Ball Tracking", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save recorded positions to CSV
if positions:
    out_path = os.path.join(os.path.dirname(__file__), "ball_positions.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "frame", "ref_tag", "x_in", "y_in", "z_in"]) 
        for p in positions:
            writer.writerow([p["time"], p["frame"], p["ref_tag"], p["x_in"], p["y_in"], p["z_in"]])
    print(f"Saved {len(positions)} positions to {out_path}")
else:
    print("No positions recorded.")