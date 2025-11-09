# infer_lstm.py
import os
import cv2
import json
import glob
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet

MODEL_PATH = 'temporal_model_phase2.keras'  # final model
LABEL_MAP_PATH = 'label_map.json'
PREDICTIONS_DIR = 'predictions'  # put videos here
IMG_SIZE = (224,224)
SEQ_LEN = 16
BATCH_SIZE = 8

def temporal_smooth(probs, window=5):
    sm = np.zeros_like(probs)
    N = probs.shape[0]
    for i in range(N):
        a = max(0, i-window)
        b = min(N, i+window+1)
        sm[i] = probs[a:b].mean(axis=0)
    return sm

def load_label_map():
    with open(LABEL_MAP_PATH, 'r') as f:
        data = json.load(f)
    # int_to_label stored as dict of string keys
    int_to_label = data['int_to_label']
    # ensure mapping from int index to label
    mapping = {int(k): v for k, v in int_to_label.items()}
    return mapping

def video_to_frames(video_path, out_tmp_dir, fps_sample=None):
    os.makedirs(out_tmp_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # optional: subsample by fps_sample (None means full)
        if fps_sample is None or idx % fps_sample == 0:
            fname = os.path.join(out_tmp_dir, f"{idx:08d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        idx += 1
    cap.release()
    files = sorted(glob.glob(os.path.join(out_tmp_dir, "*.jpg")))
    return files

def predict_on_video(video_path, model, label_map, output_txt):
    tmp_dir = "_tmp_infer_frames"
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    frame_files = video_to_frames(video_path, tmp_dir)
    if not frame_files:
        print("No frames extracted.")
        return

    preprocess = efficientnet.preprocess_input
    # Build overlapping windows
    N = len(frame_files)
    half = SEQ_LEN // 2
    # We will run predictions on windows centered on each frame index (clamp edges)
    probs_all = []
    batch_inputs = []
    batch_centers = []
    for center in tqdm(range(N), desc="Predicting frames"):
        # build window indices
        idxs = [min(max(0, center - half + k), N-1) for k in range(SEQ_LEN)]
        seq_imgs = []
        for fi in idxs:
            img = cv2.imread(frame_files[fi])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
            seq_imgs.append(preprocess(img.astype('float32')))
        batch_inputs.append(np.stack(seq_imgs, axis=0))
        batch_centers.append(center)
        if len(batch_inputs) == BATCH_SIZE:
            arr = np.array(batch_inputs)  # (B, seq, H, W, C)
            preds = model.predict(arr, verbose=0)  # (B, seq, n_classes)
            # take center timestep
            center_idx = preds.shape[1] // 2
            probs_all.extend(preds[:, center_idx, :].tolist())
            batch_inputs = []
            batch_centers = []
    if batch_inputs:
        arr = np.array(batch_inputs)
        preds = model.predict(arr, verbose=0)
        center_idx = preds.shape[1] // 2
        probs_all.extend(preds[:, center_idx, :].tolist())

    probs_all = np.array(probs_all)  # (N, n_classes)
    # Temporal smoothing
    sm = temporal_smooth(probs_all, window=5)
    preds = np.argmax(sm, axis=1)

    # Write per-frame predictions to file
    with open(output_txt, 'w') as f:
        for i, p in enumerate(preds):
            phase = label_map.get(int(p), "Unknown")
            f.write(f"{i}\t{phase}\n")

    # cleanup
    import shutil
    shutil.rmtree(tmp_dir)
    print("Saved predictions to", output_txt)

def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Model not found. Run train_lstm.py and ensure model exists.")
    label_map = load_label_map()
    print("Loaded label map:", label_map)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # we only need predict

    vids = glob.glob(os.path.join(PREDICTIONS_DIR, "*.mp4")) + glob.glob(os.path.join(PREDICTIONS_DIR, "*.avi"))
    if not vids:
        print("No videos found in", PREDICTIONS_DIR)
        return
    for v in vids:
        outtxt = os.path.splitext(v)[0] + "-PREDICTED-LSTM-phases.txt"
        predict_on_video(v, model, label_map, outtxt)

if __name__ == "__main__":
    main()
