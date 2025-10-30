import cv2
import os

def extract_limited_frames(video_path, output_dir, max_frames=10, frame_gap=30):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_dir, video_name)
    os.makedirs(save_path, exist_ok=True)

    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or saved >= max_frames:
            break

        if count % frame_gap == 0:
            filename = os.path.join(save_path, f"frame_{saved:03d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"[DONE] {saved} frames saved from {video_name} to {save_path}")

def process_video_folder(source_folder, output_folder, max_frames=10, frame_gap=30):
    if not os.path.exists(source_folder):
        print(f"[WARNING] Skipping folder – not found: {source_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(source_folder):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(source_folder, video_file)
            print(f"[INFO] Extracting from: {video_file}")
            extract_limited_frames(video_path, output_folder, max_frames, frame_gap)

if __name__ == "__main__":
    # ✅ Corrected folder name with space
    real_video_dir = r"D:\Deep_fake_morphing\data\videos\DFD_original sequences"
    fake_video_dir = r"D:\Deep_fake_morphing\data\videos\DFD_manipulated_sequences"

    real_frame_output = r"D:\Deep_fake_morphing\data\frames\real"
    fake_frame_output = r"D:\Deep_fake_morphing\data\frames\fake"

    if os.path.exists(real_video_dir):
        process_video_folder(real_video_dir, real_frame_output, max_frames=10, frame_gap=30)
    else:
        print("[INFO] Real video folder not found – skipping real videos.")

    process_video_folder(fake_video_dir, fake_frame_output, max_frames=10, frame_gap=30)
