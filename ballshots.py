import cv2
import cv2.aruco as aruco
import numpy as np

diameter_avg = []
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, TAG_SIZE_INCH, camera_matrix, dist_coeffs)
        # Use first tag as reference
        R_tag2cam, _ = cv2.Rodrigues(rvecs[0])
        t_tag2cam = tvecs[0].reshape((3,1))
        # Ball position in tag frame
        ball_tag = R_tag2cam.T @ (ball_cam - t_tag2cam)
        X_tag, Y_tag, Z_tag = ball_tag.flatten()
        aruco.drawDetectedMarkers(frame, corners)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 2)
        
        
        cv2.putText(frame, f"Ball X_tag: {X_tag:.2f} in", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        # x axis tested and looks good - measuring from center of april tag to circle
        cv2.putText(frame, f"Ball Y_tag: {Y_tag:.2f} in", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        # y axis tested and looks fine
        cv2.putText(frame, f"Ball Z_tag: {Z_tag:.2f} in", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        # z axis looks to be inaccurate 
    cv2.imshow("Ball Tracking", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()