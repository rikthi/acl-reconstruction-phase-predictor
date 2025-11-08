import numpy as np
import tensorflow as tf
import cv2
import math
import os


class SurgicalDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras data generator to load pre-extracted image frames.
    This is MUCH faster than reading from video files.
    """

    def __init__(self, frame_list, batch_size, img_size, n_classes, shuffle=True):
        """
        Initialization
        :param frame_list: List of (image_path, label_index) tuples.
        :param batch_size: Size of each batch.
        :param img_size: Tuple (height, width) to resize frames.
        :param n_classes: Total number of unique phase labels.
        :param shuffle: Whether to shuffle data at the end of each epoch.
        """
        super().__init__()

        self.frame_list = frame_list
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return math.floor(len(self.frame_list) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_samples = [self.frame_list[k] for k in batch_indices]

        # Generate data
        X, y = self.__data_generation(batch_samples)

        return X, y

    def on_epoch_end(self):
        """Updates indices after each epoch."""
        self.indices = np.arange(len(self.frame_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_samples):
        """Generates data containing batch_size samples."""
        X = np.empty((self.batch_size, *self.img_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=int)

        for i, (image_path, label) in enumerate(batch_samples):
            # Load image from disk
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"Error reading image {image_path}, using black frame.")
                frame = np.zeros((*self.img_size, 3), dtype=np.uint8)

            # Convert color and preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # No resize needed, as preprocess.py already did it
            # frame_resized = cv2.resize(frame_rgb, (self.img_size[1], self.img_size[0]))

            X[i,] = tf.keras.applications.mobilenet_v2.preprocess_input(frame_rgb)
            y[i] = label

        return X, y