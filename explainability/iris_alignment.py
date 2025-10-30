import cv2
import mediapipe as mp
import numpy as np
from skimage.feature import local_binary_pattern

mp_face_mesh = mp.solutions.face_mesh

# Iris landmark indices from MediaPipe (with refine_landmarks=True)
LEFT_IRIS_IDX = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]

def extract_iris_patch(frame, landmarks, iris_indices, patch_size=30):
    h, w, _ = frame.shape
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_indices]

    # Compute bounding rect around iris landmarks
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]

    x_min, x_max = max(min(x_coords) - 5, 0), min(max(x_coords) + 5, w)
    y_min, y_max = max(min(y_coords) - 5, 0), min(max(y_coords) + 5, h)

    iris_patch = frame[y_min:y_max, x_min:x_max]
    iris_patch = cv2.resize(iris_patch, (patch_size, patch_size))
    return iris_patch

def compute_lbp_histogram(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def compare_histograms(hist1, hist2):
    # Chi-squared distance
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-7))

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # IMPORTANT for iris landmarks!
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    distances = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            if len(landmarks) < 478:
                # Iris landmarks not detected
                continue

            left_iris_patch = extract_iris_patch(frame, landmarks, LEFT_IRIS_IDX)
            right_iris_patch = extract_iris_patch(frame, landmarks, RIGHT_IRIS_IDX)

            left_hist = compute_lbp_histogram(left_iris_patch)
            right_hist = compute_lbp_histogram(right_iris_patch)

            dist = compare_histograms(left_hist, right_hist)
            distances.append(dist)

            # Optional: visualize or print
            cv2.imshow("Left Iris", left_iris_patch)
            cv2.imshow("Right Iris", right_iris_patch)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            # No face detected
            continue

    cap.release()
    cv2.destroyAllWindows()

    if distances:
        avg_dist = np.mean(distances)
        print(f"Average iris histogram distance between left and right eye: {avg_dist:.4f}")
        if avg_dist > 0.25:
            print("Possible iris mismatch detected — potential deepfake.")
        else:
            print("Iris patterns appear consistent.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python iris_alignment.py <video_path>")
    else:
        main(sys.argv[1])
