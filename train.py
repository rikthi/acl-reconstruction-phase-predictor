import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0  # Step 6: Switched model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GroupShuffleSplit  # Step 5: Stratified split
from sklearn.utils.class_weight import compute_class_weight  # Step 1: Class weights
from DataGenerator import SurgicalDataGenerator

# --- Configuration ---
PREPROCESSED_DIR = 'data_preprocessed'
IMG_SIZE = (224, 224)  # EfficientNetB0 default size
BATCH_SIZE = 32
RANDOM_SEED = 42

# Step 2: Epochs for two-phase training
PHASE_1_EPOCHS = 5  # Train the top head
PHASE_2_EPOCHS = 15  # Fine-tune the full model (Total 20)

VALIDATION_SPLIT = 0.2

KNOWN_PHASES = sorted([
    'Preparation',
    'Diagnosis',
    'FemoralTunnelCreation',
    'TibialTunnelCreation',
    'ACLReconstruction'
])


def parse_preprocessed_data(data_dir):
    """Parses all pre-processed image files."""
    label_to_int = {name: i for i, name in enumerate(KNOWN_PHASES)}
    int_to_label = {i: name for i, name in enumerate(KNOWN_PHASES)}

    master_frame_list = []
    video_folders = glob.glob(os.path.join(data_dir, "video*"))

    print(f"Found {len(video_folders)} pre-processed video folders.")

    for video_folder in video_folders:
        image_paths = glob.glob(os.path.join(video_folder, "*.jpg"))

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            try:
                phase_name = filename.split('_')[1].split('.jpg')[0]

                if phase_name in label_to_int:
                    label_int = label_to_int[phase_name]
                    # Store (video_folder_path, image_path, label_as_int)
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
    Builds an EfficientNetB0-based classification model.
    """
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # We will control trainability in main()
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  # Your suggested size
    x = Dropout(0.4)(x)  # Your suggested dropout
    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Return both model and base_model to control them
    return model, base_model


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

    # --- Step 1 & 5: Class Weights and Stratified Split ---

    # Create lists for the splitter
    all_labels = [f[2] for f in master_frame_list]
    all_groups = [f[0] for f in master_frame_list]  # Video folder paths as groups

    print("\n--- Checking Class Distribution ---")
    class_counts = np.bincount(all_labels, minlength=n_classes)
    for i, count in enumerate(class_counts):
        phase_name = int_to_label[i]
        percentage = 100 * count / len(all_labels)
        print(f"  {phase_name}: {count} frames ({percentage:.2f}%)")

    # Step 1: Calculate Class Weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(n_classes),
        y=np.array(all_labels)
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"\nCalculated Class Weights: {class_weights_dict}")
    print("-----------------------------------\n")

    # Step 5: Create a stratified group split
    # This ensures videos are not split, and it tries to balance classes
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(X=master_frame_list, y=all_labels, groups=all_groups))

    # We only need the (img_path, label_int) tuples from the master list
    train_frame_list = [master_frame_list[i][1:3] for i in train_idx]
    val_frame_list = [master_frame_list[i][1:3] for i in val_idx]

    print(f"Training with {len(train_frame_list)} frames.")
    print(f"Validating with {len(val_frame_list)} frames.")

    # 4. Create Data Generators
    train_gen = SurgicalDataGenerator(
        frame_list=train_frame_list, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
        n_classes=n_classes, shuffle=True, augment=True
    )

    val_gen = SurgicalDataGenerator(
        frame_list=val_frame_list, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
        n_classes=n_classes, shuffle=False, augment=False
    )

    # 5. Build the model
    model, base_model = build_model(input_shape=(*IMG_SIZE, 3), n_classes=n_classes)

    # --- Step 2: Two-Phase Training ---

    # --- PHASE 1: Train the Head ---
    print("\n--- Starting Training - PHASE 1 (Training Head) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # 0.0001
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_phase1 = [
        ModelCheckpoint('surgical_phase_model.keras', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)  # Shorter patience for head
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE_1_EPOCHS,
        class_weight=class_weights_dict,  # Apply class weights
        callbacks=callbacks_phase1
    )

    # --- PHASE 2: Fine-Tuning ---
    print("\n--- Starting Training - PHASE 2 (Fine-Tuning) ---")

    # Unfreeze the last 20 layers of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # We MUST re-compile the model after changing trainability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 0.00001 (very low LR)
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()  # Show the new trainable parameter count

    callbacks_phase2 = [
        ModelCheckpoint('surgical_phase_model.keras', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)  # Full patience
    ]

    # Continue training
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE_2_EPOCHS,
        class_weight=class_weights_dict,  # Apply class weights
        callbacks=callbacks_phase2
    )

    print("--- Training Complete ---")
    print("Best model saved as 'surgical_phase_model.keras'")
    print("Label map saved as 'label_map.json'")


if __name__ == "__main__":
    main()