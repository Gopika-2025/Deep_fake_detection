import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh

# Indices for left and right eye landmarks (MediaPipe Face Mesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(landmarks, eye_indices):
    coords = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blinks(ear_list, threshold=0.3, consecutive_frames=1):
    blinks = []
    count = 0
    for i, ear in enumerate(ear_list):
        if ear < threshold:
            count += 1
        else:
            if count >= consecutive_frames:
                blinks.append(i - count // 2)
            count = 0
    if count >= consecutive_frames:
        blinks.append(len(ear_list) - count // 2)
    return blinks

def main(video_path, max_faces=5, visualize=False, ear_threshold=0.3, consecutive_frames=1):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=max_faces)

    # Dictionary to store each person's EAR sequences
    person_data = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for person_id, landmarks in enumerate(results.multi_face_landmarks):
                if person_id not in person_data:
                    person_data[person_id] = {"left_seq": [], "right_seq": [], "frames": []}

                left_ear = eye_aspect_ratio(landmarks.landmark, LEFT_EYE_IDX)
                right_ear = eye_aspect_ratio(landmarks.landmark, RIGHT_EYE_IDX)

                person_data[person_id]["left_seq"].append(left_ear)
                person_data[person_id]["right_seq"].append(right_ear)
                person_data[person_id]["frames"].append(frame_idx)

            # Append NaN for faces not detected in this frame
            detected_ids = set(range(len(results.multi_face_landmarks)))
            for pid in person_data.keys():
                if pid not in detected_ids:
                    person_data[pid]["left_seq"].append(np.nan)
                    person_data[pid]["right_seq"].append(np.nan)
                    person_data[pid]["frames"].append(frame_idx)
        else:
            # No faces detected, append NaN for all persons
            for pid in person_data.keys():
                person_data[pid]["left_seq"].append(np.nan)
                person_data[pid]["right_seq"].append(np.nan)
                person_data[pid]["frames"].append(frame_idx)

    cap.release()

    # Build result string
    results_str = ""
    for pid, data in person_data.items():
        left_seq = np.array(data["left_seq"])
        right_seq = np.array(data["right_seq"])

        left_mean = np.nanmean(left_seq)
        right_mean = np.nanmean(right_seq)

        left_seq = np.nan_to_num(left_seq, nan=left_mean)
        right_seq = np.nan_to_num(right_seq, nan=right_mean)

        # Detect blinks
        left_blinks = detect_blinks(left_seq, threshold=ear_threshold, consecutive_frames=consecutive_frames)
        right_blinks = detect_blinks(right_seq, threshold=ear_threshold, consecutive_frames=consecutive_frames)

        blink_count_left = len(left_blinks)
        blink_count_right = len(right_blinks)

        # Blink asymmetry index
        if max(blink_count_left, blink_count_right) > 0:
            asymmetry_index = abs(blink_count_left - blink_count_right) / max(blink_count_left, blink_count_right)
        else:
            asymmetry_index = 0

        results_str += f"Person {pid}:\n"
        results_str += f"  Left eye blinks: {blink_count_left}\n"
        results_str += f"  Right eye blinks: {blink_count_right}\n"
        results_str += f"  Blink asymmetry index: {asymmetry_index:.2f}\n"
        if asymmetry_index > 0.3:
            results_str += "  Possible eye blink mismatch — potential morphing.\n"
        else:
            results_str += "  Blink pattern appears natural.\n"

        # Optional visualization
        if visualize:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.plot(data["frames"], left_seq, label="Left EAR")
            plt.plot(data["frames"], right_seq, label="Right EAR")
            plt.axhline(y=ear_threshold, color='r', linestyle='--', label="EAR Threshold")
            plt.title(f"Person {pid} Eye Aspect Ratio over Time")
            plt.xlabel("Frame")
            plt.ylabel("EAR")
            plt.legend()
            plt.show()

    return results_str

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python eye_blink_mismatch.py <video_path> [max_faces] [visualize]")
        print("Example: python eye_blink_mismatch.py video.mp4 2 True")
    else:
        video = sys.argv[1]
        max_faces = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        visualize = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
        output = main(video, max_faces=max_faces, visualize=visualize)
        print(output)
