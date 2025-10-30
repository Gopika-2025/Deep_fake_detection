import os
import cv2
from mtcnn import MTCNN

input_real_dir = "data/frames/real"
input_fake_dir = "data/frames/fake"

output_real_dir = "data/cropped_faces/real"
output_fake_dir = "data/cropped_faces/fake"

os.makedirs(output_real_dir, exist_ok=True)
os.makedirs(output_fake_dir, exist_ok=True)

detector = MTCNN()

def crop_and_save_faces(input_dir, output_dir):
    count_total = 0
    count_saved = 0

    # Walk through each subfolder (video folders) inside input_dir
    for root, _, files in os.walk(input_dir):
        # Get video folder name (last part of root path)
        video_folder = os.path.basename(root)

        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                count_total += 1
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue

                results = detector.detect_faces(img)
                if len(results) == 0:
                    print(f"No face detected in {img_path}")
                    continue

                for i, face in enumerate(results):
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    cropped_face = img[y:y+h, x:x+w]

                    # Create subfolder inside output_dir for this video
                    save_subfolder = os.path.join(output_dir, video_folder)
                    os.makedirs(save_subfolder, exist_ok=True)

                    # Unique filename: videoFolder_frameFile_faceIndex.jpg
                    base_name = os.path.splitext(filename)[0]
                    save_name = f"{video_folder}_{base_name}_face{i}.jpg"
                    save_path = os.path.join(save_subfolder, save_name)

                    cv2.imwrite(save_path, cropped_face)
                    count_saved += 1

    print(f"Processed {count_total} images from {input_dir}")
    print(f"Saved {count_saved} cropped faces to {output_dir}")

print("Cropping faces from real frames...")
crop_and_save_faces(input_real_dir, output_real_dir)

print("Cropping faces from fake frames...")
crop_and_save_faces(input_fake_dir, output_fake_dir)
