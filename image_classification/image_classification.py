# Image classification based on
# https://keras.io/examples/vision/image_classification_from_scratch/
# Dataset: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

# JPEG image files on disk, no leveraging pre-trained weights or
# pre-made Keras Application model
# Use of Keras layers for image standarization and image augmentation

import tensorflow as tf
from tensorflow import keras

# Filter corrupted data
import os
num_skipped = 0
for folder_name in ('Cat', 'Dog'):
    folder_path = os.path.join('datasets/PetImages', folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)



