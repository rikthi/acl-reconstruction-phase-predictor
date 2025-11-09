import cv2
import numpy as np
import tensorflow as tf
import os
import json
import glob
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = 'surgical_phase_model.keras'
LABEL_MAP_PATH = 'label_map.json'
PREDICTIONS_DIR = 'predictions'
TARGET_FPS = 30  # We will process at the video's full FPS
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # Must match the model's training size


def temporal_smooth(probabilities, window=5):
    """
    Applies a simple moving average to the prediction probabilities.
    :param probabilities: (N_frames, N_classes) array of probabilities.
    :param window: The number of frames on *each side* to average.
    :return: (N_frames, N_classes) smoothed probabilities.
    """
    smoothed = np.zeros_like(probabilities)
    for i in range(len(probabilities)):
        start = max(0, i - window)
        end = min(len(probabilities), i + window + 1)  # +1 to be inclusive
        # Average the probabilities in the window
        smoothed[i] = np.mean(probabilities[start:end], axis=0)
    return smoothed


def predict_video(model, video_path, label_map, output_path):
    """
    Runs prediction on a single video and saves the -phase.txt file.
    """
    print(f"--- Processing: {os.path.basename(video_path)} ---")

    # 1. Load video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video Info: {total_frames} frames @ {fps:.2f} FPS")

    frames = []
    all_predictions = []

    pbar = tqdm(total=total_frames, desc="  Reading & Predicting")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Read frame and add to batch
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

        # Preprocess input for the model
        preprocessed_frame = tf.keras.applications.efficientnet.preprocess_input(resized_frame)
        frames.append(preprocessed_frame)

        # 2. Predict in batches
        if len(frames) == BATCH_SIZE:
            batch_array = np.array(frames)
            preds = model.predict(batch_array, verbose=0)
            all_predictions.extend(preds)
            frames = []  # Clear the batch
            pbar.update(BATCH_SIZE)

    # 3. Predict any remaining frames
    if frames:
        batch_array = np.array(frames)
        preds = model.predict(batch_array, verbose=0)
        all_predictions.extend(preds)
        pbar.update(len(frames))

    cap.release()
    pbar.close()

    if not all_predictions:
        print("  Error: Could not get any predictions from video.")
        return

    # --- Step 3: Temporal Smoothing ---
    print("  Applying temporal smoothing...")
    # all_predictions is (N_frames, N_classes)
    smoothed_probs = temporal_smooth(np.array(all_predictions), window=5)

    # Get the final class index from the *smoothed* probabilities
    predicted_indices = np.argmax(smoothed_probs, axis=1)
    # ----------------------------------

    # 4. Write to output file
    try:
        with open(output_path, 'w') as f:
            for i, class_index in enumerate(predicted_indices):
                # Map index back to phase name
                phase_name = label_map.get(str(class_index), "Unknown")
                f.write(f"{i}\t{phase_name}\n")

        print(f"  Prediction complete!\n  Output saved to: {output_path}")

    except Exception as e:
        print(f"  Error writing to output file: {e}")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please run 'train.py' first!")
        return

    if not os.path.exists(LABEL_MAP_PATH):
        print(f"Error: Label map not found at '{LABEL_MAP_PATH}'")
        return

    # 1. Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Load label map
    print(f"Loading label map from {LABEL_MAP_PATH}...")
    with open(LABEL_MAP_PATH, 'r') as f:
        label_data = json.load(f)
    # Use the int_to_label map for predictions
    int_to_label = label_data['int_to_label']

    # 3. Find videos to process
    video_paths = glob.glob(os.path.join(PREDICTIONS_DIR, "*.mp4")) + \
                  glob.glob(os.path.join(PREDICTIONS_DIR, "*.avi")) + \
                  glob.glob(os.path.join(PREDICTIONS_DIR, "*.mov"))

    if not video_paths:
        print(f"No videos (.mp4, .avi, .mov) found in '{PREDICTIONS_DIR}'.")
        return

    print(f"Found {len(video_paths)} videos to process...")

    # 4. Process each video
    for video_path in video_paths:
        base_name = os.path.basename(video_path)
        file_name, file_ext = os.path.splitext(base_name)
        output_path = os.path.join(PREDICTIONS_DIR, f"{file_name}-PREDICTED-phases.txt")

        predict_video(model, video_path, int_to_label, output_path)


if __name__ == "__main__":
    main()