import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_IDX = [336, 296, 334, 293, 300]

def eyebrow_vertical_position(landmarks, idx_list, image_height):
    ys = [landmarks[i].y * image_height for i in idx_list]
    return np.mean(ys)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)

    left_positions = []
    right_positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_pos = eyebrow_vertical_position(landmarks, LEFT_EYEBROW_IDX, h)
            right_pos = eyebrow_vertical_position(landmarks, RIGHT_EYEBROW_IDX, h)

            left_positions.append(left_pos)
            right_positions.append(right_pos)
        else:
            left_positions.append(np.nan)
            right_positions.append(np.nan)

    cap.release()

    # Clean NaNs
    left_positions = np.array(left_positions)
    right_positions = np.array(right_positions)
    left_positions = np.nan_to_num(left_positions, nan=np.nanmean(left_positions))
    right_positions = np.nan_to_num(right_positions, nan=np.nanmean(right_positions))

    # Calculate difference between left and right eyebrow vertical positions over time
    diff = np.abs(left_positions - right_positions)
    avg_diff = np.mean(diff)

    # Build return string for Streamlit
    result = f"Average eyebrow vertical position difference: {avg_diff:.2f} pixels.\n"
    if avg_diff > 10:  # example threshold
        result += " Possible eyebrow mismatch detected - potential manipulation."
    else:
        result += " Eyebrow movement appears natural."

    return result  # <-- Return instead of print

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python eyebrow_mismatch.py <video_path>")
    else:
        print(main(sys.argv[1]))  # Use print only when running standalone
