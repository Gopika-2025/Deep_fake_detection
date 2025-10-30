import cv2
import mediapipe as mp
import numpy as np

# Constants for EAR blink detection
EAR_THRESHOLD = 0.4
CONSEC_FRAMES = 2

# Indices for left and right eye landmarks from MediaPipe Face Mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh

def eye_aspect_ratio(landmarks, eye_indices):
    # Get the 6 landmark points for the eye
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    
    # Compute distances
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    # Vertical distances
    A = dist(points[1], points[5])
    B = dist(points[2], points[4])
    # Horizontal distance
    C = dist(points[0], points[3])
    
    ear = (A + B) / (2.0 * C)
    return ear

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    
    frame_num = 0
    left_ear_list = []
    right_ear_list = []
    
    blink_start = None
    blink_end = None
    blink_durations = []
    blink_intervals = []
    
    prev_blink_end = None
    blinking = False
    consec_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
            left_ear_list.append(left_ear)
            right_ear_list.append(right_ear)
            
            ear = (left_ear + right_ear) / 2.0
            
            # Blink detection logic
            if ear < EAR_THRESHOLD:
                consec_frames += 1
                if not blinking and consec_frames >= CONSEC_FRAMES:
                    blinking = True
                    blink_start = frame_num - CONSEC_FRAMES + 1
            else:
                if blinking:
                    blink_end = frame_num - 1
                    blinking = False
                    duration = blink_end - blink_start + 1
                    blink_durations.append(duration)
                    if prev_blink_end is not None:
                        interval = blink_start - prev_blink_end
                        blink_intervals.append(interval)
                    prev_blink_end = blink_end
                consec_frames = 0
        else:
            # No face detected, reset counters
            consec_frames = 0
            if blinking:
                blink_end = frame_num - 1
                blinking = False
                duration = blink_end - blink_start + 1
                blink_durations.append(duration)
                if prev_blink_end is not None:
                    interval = blink_start - prev_blink_end
                    blink_intervals.append(interval)
                prev_blink_end = blink_end
        
    cap.release()
    face_mesh.close()
    
    # Summary stats
    total_blinks = len(blink_durations)
    avg_blink_duration = np.mean(blink_durations) if blink_durations else 0
    std_blink_duration = np.std(blink_durations) if blink_durations else 0
    avg_blink_interval = np.mean(blink_intervals) if blink_intervals else 0
    std_blink_interval = np.std(blink_intervals) if blink_intervals else 0
    
    print(f"Total blinks detected: {total_blinks}")
    print(f"Average blink duration (frames): {avg_blink_duration:.2f}")
    print(f"Blink duration std dev (frames): {std_blink_duration:.2f}")
    print(f"Average blink interval (frames): {avg_blink_interval:.2f}")
    print(f"Blink interval std dev (frames): {std_blink_interval:.2f}")
    
    # Return blink features for further use
    blink_features = {
        "total_blinks": total_blinks,
        "avg_blink_duration": avg_blink_duration,
        "std_blink_duration": std_blink_duration,
        "avg_blink_interval": avg_blink_interval,
        "std_blink_interval": std_blink_interval,
        "left_ear_list": left_ear_list,
        "right_ear_list": right_ear_list
    }
    return blink_features

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python blink_analysis.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    features = process_video(video_path)
