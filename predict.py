import tensorflow as tf
import cv2
import numpy as np
import json
import sys
import os
import glob

# --- Configuration ---
IMG_SIZE = (224, 224)
MODEL_PATH = 'surgical_phase_model.keras'
LABEL_MAP_PATH = 'label_map.json'
PREDICTION_DIR = 'predictions'  # Folder to scan for new videos

# Supported video file extensions
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv']


def predict_video(model, int_to_label, video_path, output_path):
    """
    Runs prediction on a single video file and saves the output.
    """

    # 1. Open video and output file
    print(f"--- Processing: {os.path.basename(video_path)} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video Info: {total_frames} frames @ {fps:.2f} FPS")

    frame_count = 0

    with open(output_path, 'w') as out_f:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                # End of video
                break

            # 2. Pre-process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
            frame_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(frame_resized)

            # Add batch dimension
            frame_batch = np.expand_dims(frame_preprocessed, axis=0)

            # 3. Predict
            predictions = model.predict(frame_batch, verbose=0)  # verbose=0 for cleaner output

            # Get the winning class index
            predicted_class_int = np.argmax(predictions[0])

            # Convert index to phase name
            predicted_phase_name = int_to_label.get(predicted_class_int, "Unknown")

            # 4. Write to output file
            # This matches the "frame_index<TAB>phase_name" format
            out_f.write(f"{frame_count}\t{predicted_phase_name}\n")

            frame_count += 1

            # Print progress every 300 frames (e.g., every 10 seconds at 30fps)
            if frame_count % 300 == 0:
                print(f"  Processed {frame_count} / {total_frames} frames...")

    # 5. Clean up
    cap.release()
    print(f"  Prediction complete!")
    print(f"  Output saved to: {output_path}")


def main():
    # 1. Check for model and label map
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print(f"Error: Model ('{MODEL_PATH}') or Label Map ('{LABEL_MAP_PATH}') not found.")
        print("Please run 'train.py' first to train the model.")
        return

    # 2. Check for predictions directory
    if not os.path.exists(PREDICTION_DIR):
        print(f"Error: Directory not found: '{PREDICTION_DIR}'")
        print(f"Please create it and add your videos for prediction.")
        return

    # 3. Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 4. Load the label map
    print(f"Loading label map from {LABEL_MAP_PATH}...")
    with open(LABEL_MAP_PATH, 'r') as f:
        label_data = json.load(f)
    int_to_label = label_data['int_to_label']
    # Convert string keys from JSON back to integers
    int_to_label = {int(k): v for k, v in int_to_label.items()}

    # 5. Find all videos in the prediction directory
    video_files_to_process = []
    for ext in VIDEO_EXTENSIONS:
        video_files_to_process.extend(
            glob.glob(os.path.join(PREDICTION_DIR, ext))
        )

    if not video_files_to_process:
        print(f"No videos found in '{PREDICTION_DIR}'.")
        return

    print(f"Found {len(video_files_to_process)} videos to process...")

    # 6. Loop and process each video
    for video_path in video_files_to_process:
        # Create an output file name
        base_name = os.path.basename(video_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_file_name = f"{file_name_no_ext}-PREDICTED-phases.txt"

        # Save the output file in the SAME directory
        output_file_path = os.path.join(PREDICTION_DIR, output_file_name)

        predict_video(model, int_to_label, video_path, output_file_path)


if __name__ == "__main__":
    main()