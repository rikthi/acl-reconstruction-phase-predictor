import numpy as np
import tensorflow as tf
import cv2
import math
import os


class SurgicalDataGenerator(tf.keras.utils.Sequence):


    def __init__(self, frame_list, batch_size, img_size, n_classes, shuffle=True, augment=False):
        super().__init__()

        self.frame_list = frame_list
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):

        return math.floor(len(self.frame_list) / self.batch_size)

    def __getitem__(self, index):

        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_samples = [self.frame_list[k] for k in batch_indices]

        # Generate data
        X, y = self.__data_generation(batch_samples)

        return X, y

    def on_epoch_end(self):

        self.indices = np.arange(len(self.frame_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment_image(self, image):


        # 1. Random Horizontal Flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)

        # 2. Random Rotation (between -10 and +10 degrees)
        angle = np.random.uniform(-10, 10)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows), borderValue=(0, 0, 0))  # Fill borders with black

        # 3. Random Zoom (between 100% and 120%)
        scale = np.random.uniform(1.0, 1.2)
        h, w = image.shape[:2]
        ch, cw = int(h / scale), int(w / scale)  # Cropped height/width

        h_start = (h - ch) // 2
        w_start = (w - cw) // 2

        cropped = image[h_start:h_start + ch, w_start:w_start + cw]
        image = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- NEW AUGMENTATIONS ---
        # 4. Random Brightness/Contrast (Step 4)
        if np.random.rand() < 0.3:
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-20, 20)  # brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # 5. Random Gaussian Blur (Step 4)
        if np.random.rand() < 0.2:
            ksize = int(np.random.choice([3, 5]))  # Kernel size must be odd
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        # -------------------------

        return image

    def __data_generation(self, batch_samples):

        X = np.empty((self.batch_size, *self.img_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=int)

        for i, (image_path, label) in enumerate(batch_samples):
            # Load image from disk
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"Error reading image {image_path}, using black frame.")
                frame = np.zeros((*self.img_size, 3), dtype=np.uint8)

            # Convert color
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply augmentation if enabled
            if self.augment:
                frame_rgb = self._augment_image(frame_rgb)

            # Preprocess for the model
            # NOTE: We now call a generic 'preprocess_input'
            X[i,] = tf.keras.applications.efficientnet.preprocess_input(frame_rgb)
            y[i] = label

        return X, y