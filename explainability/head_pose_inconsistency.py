import cv2
import numpy as np
import mediapipe as mp
import math
import sys

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe landmark indices (commonly used stable points)
# nose tip, chin, left eye outer, right eye outer, left mouth corner, right mouth corner
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# 3D model points in an arbitrary model coordinate (units: mm). These are rough average face model points.
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye corner
    (43.3, 32.7, -26.0),    # Right eye corner
    (-28.9, -28.9, -20.0),  # Left mouth corner
    (28.9, -28.9, -20.0)    # Right mouth corner
], dtype=np.float64)

def rotation_vector_to_euler(rv):
    """Convert Rodrigues rotation vector to Euler angles (degrees): yaw, pitch, roll."""
    R, _ = cv2.Rodrigues(rv)
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    # Return in degrees: yaw (y), pitch (x), roll (z)
    return (math.degrees(y), math.degrees(x), math.degrees(z))

def estimate_head_pose(landmarks, img_w, img_h, camera_matrix):
    # Build 2D image points from selected landmarks
    image_points = []
    for idx in LANDMARK_IDS:
        lm = landmarks[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        image_points.append((x, y))
    image_points = np.array(image_points, dtype=np.float64)

    dist_coeffs = np.zeros((4,1))  # assume no lens distortion
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None
    yaw, pitch, roll = rotation_vector_to_euler(rvec)
    return (yaw, pitch, roll, image_points, rvec, tvec)

def analyze_video(video_path,
                  max_jump_threshold_deg=20.0,   # per-frame jump threshold (deg)
                  max_range_threshold_deg=45.0,  # overall allowed head rotation (deg)
                  jump_event_count_threshold=5   # suspicious if > this many jumps
                 ):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: cannot open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    angles_list = []  # list of (yaw,pitch,roll)
    frame_idx = 0
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            angles_list.append(None)
            continue

        # If multiple faces, analyze each — here we take the first face
        landmarks = results.multi_face_landmarks[0].landmark
        if len(landmarks) < max(LANDMARK_IDS)+1:
            angles_list.append(None)
            continue

        pose = estimate_head_pose(landmarks, w, h, camera_matrix)
        if pose is None:
            angles_list.append(None)
            continue

        yaw, pitch, roll, img_pts, rvec, tvec = pose
        angles_list.append((yaw, pitch, roll))
        face_count += 1

    cap.release()
    face_mesh.close()

    # Post-process angles: drop None frames, compute diffs
    valid_angles = [a for a in angles_list if a is not None]
    if len(valid_angles) == 0:
        print("No faces/head poses detected in the video.")
        return

    yaws = np.array([a[0] for a in valid_angles])
    pitchs = np.array([a[1] for a in valid_angles])
    rolls = np.array([a[2] for a in valid_angles])

    # Stats
    yaw_range = np.ptp(yaws)   # peak-to-peak range
    pitch_range = np.ptp(pitchs)
    roll_range = np.ptp(rolls)

    # Frame-to-frame jumps: compute diffs between consecutive valid frames
    diffs = []
    for i in range(1, len(valid_angles)):
        prev = valid_angles[i-1]
        cur = valid_angles[i]
        dy = abs(cur[0] - prev[0])
        dp = abs(cur[1] - prev[1])
        dr = abs(cur[2] - prev[2])
        diffs.append((dy, dp, dr))
    diffs = np.array(diffs) if diffs else np.zeros((0,3))

    jump_events = np.sum((diffs > max_jump_threshold_deg).any(axis=1))

    # Simple heuristic flags
    flags = []
    if yaw_range > max_range_threshold_deg or pitch_range > max_range_threshold_deg or roll_range > max_range_threshold_deg:
        flags.append("Large overall head rotation range")
    if jump_events > jump_event_count_threshold:
        flags.append(f"Many sudden head pose jumps (> {jump_event_count_threshold})")
    if np.mean(np.abs(diffs[:,0])) > (max_jump_threshold_deg/3 if diffs.size else 1e-9):
        flags.append("High average yaw variation")
    # Final decision
    suspicious = len(flags) > 0

    # Print summary
    print("Head pose analysis summary:")
    print(f"  Frames analyzed (with face): {len(valid_angles)} / {len(angles_list)}")
    print(f"  Yaw range (deg):   {yaw_range:.2f}, mean: {np.mean(yaws):.2f}, std: {np.std(yaws):.2f}")
    print(f"  Pitch range (deg): {pitch_range:.2f}, mean: {np.mean(pitchs):.2f}, std: {np.std(pitchs):.2f}")
    print(f"  Roll range (deg):  {roll_range:.2f}, mean: {np.mean(rolls):.2f}, std: {np.std(rolls):.2f}")
    print(f"  Sudden jump events (>{max_jump_threshold_deg}°): {int(jump_events)}")

    if suspicious:
        print("\n Head pose inconsistency detected:")
        for f in flags:
            print("  -", f)
    else:
        print("\nHead pose appears consistent and natural.")

    # Optional: return structured result
    return {
        "frames_with_face": len(valid_angles),
        "total_frames": len(angles_list),
        "yaw_range": float(yaw_range),
        "pitch_range": float(pitch_range),
        "roll_range": float(roll_range),
        "jump_events": int(jump_events),
        "flags": flags
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python head_pose_inconsistency.py <video_path>")
        sys.exit(1)
    analyze_video(sys.argv[1])
