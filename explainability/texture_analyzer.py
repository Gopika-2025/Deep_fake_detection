import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import mediapipe as mp

# Constants for LBP
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Forehead landmark indices (approximate region above eyebrows)
FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]

def extract_forehead_region(frame, landmarks, indices):
    h, w, _ = frame.shape
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    forehead_region = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w_box, h_box = cv2.boundingRect(pts)
    forehead_cropped = forehead_region[y:y+h_box, x:x+w_box]
    return forehead_cropped

def lbp_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, N_POINTS + 3),
                             range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def chi_square_distance(histA, histB):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + 1e-7))

def detect_scar_mole(img, threshold=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    ratio = np.sum(thresh == 255) / (img.shape[0] * img.shape[1])
    return ratio

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    ref_lbp = None
    ref_hsv = None
    scar_ratios = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            forehead = extract_forehead_region(frame, landmarks, FOREHEAD_IDX)
            
            if forehead.size == 0:
                continue

            lbp_hist = lbp_histogram(forehead)
            hsv_hist = hsv_histogram(forehead)
            scar_ratio = detect_scar_mole(forehead)

            if ref_lbp is None:
                ref_lbp = lbp_hist
                ref_hsv = hsv_hist
                continue

            scar_ratios.append(scar_ratio)

    cap.release()

    if scar_ratios:
        avg_lbp_dist = np.mean([chi_square_distance(ref_lbp, lbp_histogram(forehead)) for _ in scar_ratios])
        avg_hsv_dist = np.mean([chi_square_distance(ref_hsv, hsv_histogram(forehead)) for _ in scar_ratios])
        avg_scar_ratio = np.mean(scar_ratios)

        result_str = f"Avg Skin Texture Mismatch (LBP chi-square): {avg_lbp_dist:.4f}\n"
        result_str += f"Avg Skin Color Mismatch (HSV chi-square): {avg_hsv_dist:.4f}\n"
        result_str += f"Avg Scar/Mole ratio on forehead: {avg_scar_ratio:.4%}\n"

        if avg_lbp_dist > 0.3 or avg_hsv_dist > 0.3 or avg_scar_ratio > 0.02:
            result_str += "Possible forehead skin mismatch or anomaly — potential morphing detected.\n"
        else:
            result_str += "Forehead skin texture and color appear consistent.\n"
    else:
        result_str = "No forehead data to analyze.\n"

    return result_str

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python forehead_texture_check.py <video_path>")
    else:
        output = main(sys.argv[1])
        print(output)
