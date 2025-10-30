import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = r"D:\Deep_fake_morphing\models\mobilenetv2_finetuned.h5"
VIDEO_PATH = r"D:\Deep_fake_morphing\data\test_videos\sample_video.mp4"  # Update to your video

IMG_SIZE = (224, 224)
FRAME_SKIP = 5  # Use every 5th frame to speed up

def load_model(path):
    print("Loading model...")
    model = tf.keras.models.load_model(path)
    print("Model loaded.")
    return model

def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    preds = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            img = preprocess_frame(frame)
            pred = model.predict(img, verbose=0)[0]  # [prob_real, prob_fake]
            preds.append(pred)

        frame_idx += 1

    cap.release()
    return np.array(preds)

def aggregate_predictions(predictions):
    avg_pred = np.mean(predictions, axis=0)
    real_prob, fake_prob = avg_pred[0], avg_pred[1]
    final_label = "FAKE" if fake_prob > real_prob else "REAL"
    return final_label, real_prob, fake_prob

def main():
    model = load_model(MODEL_PATH)
    predictions = predict_video(model, VIDEO_PATH)
    final_label, real_prob, fake_prob = aggregate_predictions(predictions)

    print("\n=== Video Level Prediction ===")
    print(f"Final Label: {final_label}")
    print(f"Average Real Probability: {real_prob:.4f}")
    print(f"Average Fake Probability: {fake_prob:.4f}")

if __name__ == "__main__":
    main()
