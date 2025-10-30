import os
import shutil
import cv2

# ==== Configurable paths ====
DATA_DIR = r"D:\Deep_fake_morphing\data"
REAL_DIR = os.path.join(DATA_DIR, "videos", "DFD_original_sequences")
FAKE_DIR = os.path.join(DATA_DIR, "videos", "DFD_manipulated_sequences")

METADATA_DIR = os.path.join(DATA_DIR, "metadata")
CORRUPT_DIR = os.path.join(DATA_DIR, "corrupt")

LOG_FILE = os.path.join(METADATA_DIR, "corrupt_files.txt")

# Supported video/image extensions to check
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def is_video_file(filename):
    return os.path.splitext(filename.lower())[1] in VIDEO_EXTS

def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTS

def check_video_corrupt(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return True  # Cannot open file

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return True  # No readable frames
    return False

def check_image_corrupt(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return True
    return False

def log_and_move_corrupt(src_path, base_dir, log_f):
    # Relative path to base_dir
    rel_path = os.path.relpath(src_path, base_dir)
    dest_path = os.path.join(CORRUPT_DIR, rel_path)

    # Make sure destination dir exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Move file
    shutil.move(src_path, dest_path)

    # Log
    log_f.write(f"{rel_path}\n")
    print(f"[CORRUPT] Moved: {rel_path}")

def scan_and_check(base_dir, log_f):
    for root, _, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)

            # Skip zero byte files outright
            if os.path.getsize(full_path) == 0:
                log_and_move_corrupt(full_path, base_dir, log_f)
                continue

            if is_video_file(file):
                if check_video_corrupt(full_path):
                    log_and_move_corrupt(full_path, base_dir, log_f)
            elif is_image_file(file):
                if check_image_corrupt(full_path):
                    log_and_move_corrupt(full_path, base_dir, log_f)
            else:
                # Unknown file type - optionally skip or check
                pass

def main():
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(CORRUPT_DIR, exist_ok=True)

    with open(LOG_FILE, "w") as log_f:
        print(f"Checking real videos/images in {REAL_DIR} ...")
        scan_and_check(REAL_DIR, log_f)

        print(f"Checking fake videos/images in {FAKE_DIR} ...")
        scan_and_check(FAKE_DIR, log_f)

    print("Corrupt file check complete.")
    print(f"See log: {LOG_FILE}")

if __name__ == "__main__":
    main()
