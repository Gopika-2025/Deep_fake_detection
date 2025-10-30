import cv2
import numpy as np

def detect_flicker(video_path, threshold=15):
    """
    Detect flicker events in a video based on frame brightness differences.

    Parameters:
        video_path (str): Path to input video
        threshold (float): Difference threshold to consider as flicker

    Returns:
        dict: flicker analysis results
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    prev_gray = None
    brightness_diffs = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
            brightness_diffs.append(diff)
        prev_gray = gray

    cap.release()

    brightness_diffs = np.array(brightness_diffs)
    avg_diff = np.mean(brightness_diffs)
    max_diff = np.max(brightness_diffs)
    flicker_events = np.sum(brightness_diffs > threshold)

    results = {
        "average_diff": avg_diff,
        "max_diff": max_diff,
        "flicker_events": int(flicker_events)
    }

    return results

def main(video_path):
    results = detect_flicker(video_path)
    results_str = f"Flicker Detection Results for {video_path}:\n"
    results_str += f"Average brightness difference between frames: {results['average_diff']:.2f}\n"
    results_str += f"Maximum brightness difference between frames: {results['max_diff']:.2f}\n"
    results_str += f"Number of flicker events (diff > 15): {results['flicker_events']}\n"

    if results['flicker_events'] > 0:
        results_str += "Significant flicker detected — possible manipulation!\n"
    else:
        results_str += "No significant flicker detected.\n"

    return results_str

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python flicker_detection.py <video_path>")
    else:
        output = main(sys.argv[1])
        print(output)
