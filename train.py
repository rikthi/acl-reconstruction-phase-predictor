import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from DataGenerator import SurgicalDataGenerator  # Import our NEW generator

# --- Configuration ---
# Point to the pre-processed image folder
PREPROCESSED_DIR = 'data_preprocessed'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2  # 20% of videos for validation
RANDOM_SEED = 42
# WORKERS = 8 # We will remove this from model.fit()

# Your 5 specific phases.
KNOWN_PHASES = sorted([
    'Preparation',
    'Diagnosis',
    'FemoralTunnelCreation',
    'TibialTunnelCreation',
    'ACLReconstruction'
])


def parse_preprocessed_data(data_dir):
    """
    Parses all pre-processed image files.
    """
    # Create label mappings
    label_to_int = {name: i for i, name in enumerate(KNOWN_PHASES)}
    int_to_label = {i: name for i, name in enumerate(KNOWN_PHASES)}

    master_frame_list = []

    # Find all 'video*' subfolders
    video_folders = glob.glob(os.path.join(data_dir, "video*"))

    print(f"Found {len(video_folders)} pre-processed video folders.")

    for video_folder in video_folders:
        # Get all image paths in this folder
        image_paths = glob.glob(os.path.join(video_folder, "*.jpg"))

        for img_path in image_paths:
            # Filename is e.g., '000120_Preparation.jpg'
            filename = os.path.basename(img_path)
            try:
                # Get the phase name from the filename
                phase_name = filename.split('_')[1].split('.jpg')[0]

                if phase_name in label_to_int:
                    label_int = label_to_int[phase_name]
                    # Store the video folder (for splitting) and the image path
                    master_frame_list.append((video_folder, img_path, label_int))
                else:
                    print(f"Warning: Unknown phase '{phase_name}' in filename {img_path}")
            except IndexError:
                print(f"Warning: Malformed filename {img_path}")

    print(f"Total valid frames found: {len(master_frame_list)}")
    print(f"Classes (in order): {KNOWN_PHASES}")

    return master_frame_list, label_to_int, int_to_label


def build_model(input_shape, n_classes):
    """
    Builds a MobileNetV2-based classification model.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def main():
    if not os.path.exists(PREPROCESSED_DIR):
        print(f"Error: Pre-processed data directory not found at '{PREPROCESSED_DIR}'")
        print("Please run 'preprocess.py' first!")
        return

    # 1. Parse all pre-processed data
    master_frame_list, label_to_int, int_to_label = parse_preprocessed_data(PREPROCESSED_DIR)

    if not master_frame_list:
        print("No data found. Exiting.")
        return

    with open('label_map.json', 'w') as f:
        json.dump({'label_to_int': label_to_int, 'int_to_label': int_to_label}, f)

    n_classes = len(label_to_int)

    # 2. Split data by VIDEO
    all_video_folders = sorted(list(set(f[0] for f in master_frame_list)))

    train_paths, val_paths = train_test_split(
        all_video_folders,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED
    )

    print(f"Total videos: {len(all_video_folders)}")
    print(f"Training on {len(train_paths)} videos.")
    print(f"Validating on {len(val_paths)} videos.")

    # 3. Create frame lists for each split
    # We only need the (img_path, label_int) tuples now
    train_frame_list = [(f[1], f[2]) for f in master_frame_list if f[0] in train_paths]
    val_frame_list = [(f[1], f[2]) for f in master_frame_list if f[0] in val_paths]

    print(f"Training with {len(train_frame_list)} frames.")
    print(f"Validating with {len(val_frame_list)} frames.")

    # 4. Create Data Generators
    train_gen = SurgicalDataGenerator(
        frame_list=train_frame_list,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        n_classes=n_classes,
        shuffle=True
    )

    val_gen = SurgicalDataGenerator(
        frame_list=val_frame_list,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        n_classes=n_classes,
        shuffle=False
    )

    # 5. Build and compile the model
    model = build_model(input_shape=(*IMG_SIZE, 3), n_classes=n_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 6. Set up callbacks
    callbacks = [
        ModelCheckpoint('surgical_phase_model.keras', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]

    # 7. Train the model
    print("\n--- Starting Training ---")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
        # --- FIX ---
        # Removed 'workers' and 'use_multiprocessing'
        # as they are not supported in this Keras version.
        # The data generator is already very fast without them.
        # -----------
    )
    print("--- Training Complete ---")
    print("Best model saved as 'surgical_phase_model.keras'")
    print("Label map saved as 'label_map.json'")


if __name__ == "__main__":
    main()