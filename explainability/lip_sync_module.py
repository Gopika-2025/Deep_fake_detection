import cv2
import sys
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def main(video_path):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    lip_movements = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Using landmark indices for upper and lower lip points (MediaPipe face mesh)
            # Upper lip points (example): 13, 14
            # Lower lip points (example): 308, 78
            upper_lip_y = (landmarks[13].y + landmarks[14].y) / 2
            lower_lip_y = (landmarks[308].y + landmarks[78].y) / 2

            lip_distance = abs(upper_lip_y - lower_lip_y)
            lip_movements.append(lip_distance)
        else:
            # No face detected in this frame; skip or append zero
            lip_movements.append(0)

    cap.release()
    face_mesh.close()

    # Filter out zeros (frames with no detection) for averaging
    lip_movements = [m for m in lip_movements if m > 0]

    if len(lip_movements) == 0:
        print("No face detected in any frame.")
        return

    avg_movement = np.mean(lip_movements)
    print(f"Average lip movement: {avg_movement:.4f}")

    # Baseline lip movement threshold (tune this value based on your videos)
    baseline = 0.008

    # Clamp avg_movement to baseline max to avoid negative mismatch
    clamped_movement = min(avg_movement, baseline)

    # Calculate mismatch percentage: less movement = higher mismatch
    mismatch_percent = (1 - (clamped_movement / baseline)) * 100

    print(f"Lip-sync mismatch score: {mismatch_percent:.2f}%")

    if mismatch_percent > 70:
        print("Possible lip-sync mismatch — very little mouth movement detected.")
    else:
        print("Mouth movement detected — likely synced with speech (if any).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lip_sync_mismatch.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    main(video_path)
