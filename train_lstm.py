# train_lstm.py
import os
import json
import glob
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sequence_generator import SequenceDataGenerator
from model_lstm import build_temporal_model

# ----------------- CONFIG -----------------
PREPROCESSED_DIR = 'data_preprocessed'
LABEL_MAP_PATH = 'label_map.json'
MODEL_OUT_PHASE1 = 'temporal_model_phase1.keras'
MODEL_OUT_PHASE2 = 'temporal_model_phase2.keras'
IMG_SIZE = (224,224)
SEQ_LEN = 16
BATCH_SIZE = 8      # sequences are heavier; reduce batch size if OOM
RANDOM_SEED = 42
PHASE_1_EPOCHS = 8  # train head
PHASE_2_EPOCHS = 20 # fine-tune
VALIDATION_SPLIT = 0.2
# Known phases (same as your previous)
KNOWN_PHASES = sorted([
    'Preparation',
    'Diagnosis',
    'FemoralTunnelCreation',
    'TibialTunnelCreation',
    'ACLReconstruction'
])
# ------------------------------------------

# reproducibility
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def parse_preprocessed_data(data_dir):
    label_to_int = {name: i for i, name in enumerate(KNOWN_PHASES)}
    int_to_label = {i: name for i, name in enumerate(KNOWN_PHASES)}
    master_frame_list = []
    video_folders = glob.glob(os.path.join(data_dir, "video*"))
    for vf in video_folders:
        imgs = glob.glob(os.path.join(vf, "*.jpg"))
        for p in imgs:
            filename = os.path.basename(p)
            try:
                phase_name = filename.split('_')[1].split('.jpg')[0]
            except Exception:
                continue
            if phase_name in label_to_int:
                master_frame_list.append((vf, p, label_to_int[phase_name]))
    return master_frame_list, label_to_int, int_to_label

def build_preprocess_fn():
    return tf.keras.applications.efficientnet.preprocess_input

def generate_train_val(master_list):
    all_labels = [m[2] for m in master_list]
    all_groups = [m[0] for m in master_list]
    n_classes = len(KNOWN_PHASES)
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(X=master_list, y=all_labels, groups=all_groups))
    train_list = [master_list[i] for i in train_idx]
    val_list = [master_list[i] for i in val_idx]
    # compute class weights on train split
    train_labels = [t[2] for t in train_list]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(n_classes), y=np.array(train_labels))
    class_weights_dict = dict(enumerate(class_weights))
    return train_list, val_list, class_weights_dict

def seq_collate_for_metrics(generator):
    # collect labels from generator validation data for metrics computation (slow but useful)
    ys = []
    for i in range(len(generator)):
        _, y = generator[i]
        ys.append(y)
    return np.concatenate(ys, axis=0)

def main():
    if not os.path.exists(PREPROCESSED_DIR):
        raise SystemExit("Preprocessed data directory not found. Run your preprocess.py first.")

    master_list, l2i, i2l = parse_preprocessed_data(PREPROCESSED_DIR)
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump({'label_to_int': l2i, 'int_to_label': i2l}, f)
    if len(master_list) == 0:
        raise SystemExit("No frames found. Check your preprocessing and filenames.")

    n_classes = len(l2i)
    print("Found frames:", len(master_list))
    print("Classes:", l2i)

    train_list, val_list, class_weights = generate_train_val(master_list)
    print(f"Train frames: {len(train_list)}  Val frames: {len(val_list)}")
    print("Class weights:", class_weights)

    preprocess_fn = build_preprocess_fn()

    train_gen = SequenceDataGenerator(train_list, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
                                      seq_len=SEQ_LEN, shuffle=True, augment=True, preprocess_fn=preprocess_fn)
    val_gen = SequenceDataGenerator(val_list, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
                                    seq_len=SEQ_LEN, shuffle=False, augment=False, preprocess_fn=preprocess_fn)

    # Build model: outputs predictions for every timestep (we will use center timestep for loss)
    model = build_temporal_model(input_shape=(SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3),
                                 n_classes=n_classes, lstm_units=256, dropout=0.4, train_base=False)

    # ------ PHASE 1: Train head (base frozen)
    # Freeze base EfficientNet layers inside TimeDistributed - the outer function set base trainable False earlier
    for layer in model.layers:
        # ensure base model layers remain non-trainable at phase 1
        if 'efficientnet' in layer.name:
            layer.trainable = False

    # We will compile with a custom loss that extracts the center timestep predictions
    def center_loss(y_true, y_pred):
        # y_pred: (batch, seq, n_classes)
        center = tf.shape(y_pred)[1] // 2
        # take center logit:
        center_pred = y_pred[:, center, :]
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, center_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=center_loss,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')])

    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT_PHASE1, save_best_only=True, monitor='val_acc'),
        tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, restore_best_weights=True)
    ]

    print("=== Phase 1: training head ===")
    model.fit(train_gen, validation_data=val_gen, epochs=PHASE_1_EPOCHS,
              class_weight=class_weights, callbacks=callbacks_phase1)

    # ------ PHASE 2: Fine-tune last layers of base
    print("=== Phase 2: fine-tuning ===")
    # Unfreeze base EfficientNet inside model (we need to identify it)
    # Easier: set all layers trainable then freeze earlier layers manually
    for layer in model.layers:
        layer.trainable = True

    # Now freeze all layers up to the last N percent of layers in the base network to avoid catastrophic forgetting
    # Find base model layers by name containing 'efficientnetb0' or 'top' naming conventions
    # We'll freeze the earliest layers by hierarchical order: freeze the first X% of layers
    total_layers = len(model.layers)
    # Freeze first ~70% of layers (tune as needed)
    freeze_up_to = int(total_layers * 0.7)
    for i, layer in enumerate(model.layers):
        if i < freeze_up_to:
            layer.trainable = False
        else:
            layer.trainable = True

    # Recompile with a very low learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=center_loss,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')])

    callbacks_phase2 = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT_PHASE2, save_best_only=True, monitor='val_acc'),
        tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=6, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=PHASE_2_EPOCHS,
              class_weight=class_weights, callbacks=callbacks_phase2)

    print("Training complete. Best model saved to", MODEL_OUT_PHASE2)

if __name__ == "__main__":
    main()
