# data_preprocessing.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical

# Define file paths
raw_data_dir = 'data/raw/'
processed_data_dir = 'data/processed/'
train_data_dir = os.path.join(processed_data_dir, 'train/')
val_data_dir = os.path.join(processed_data_dir, 'val/')
test_data_dir = os.path.join(processed_data_dir, 'test/')
labels_csv = os.path.join(raw_data_dir, 'labels.csv')

# Create directories if they don't exist
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(val_data_dir, exist_ok=True)
os.makedirs(test_data_dir, exist_ok=True)

# Load image labels
print("Loading image labels...")
labels_df = pd.read_csv(labels_csv)
print(labels_df.head())

# Define image parameters
img_width, img_height = 224, 224
batch_size = 32

# Image data generator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Function to preprocess and save images
def preprocess_and_save_images(data_dir, subset):
    generator = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=raw_data_dir,
        x_col='filename',
        y_col='label',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset,
        save_to_dir=data_dir,
        save_prefix='img',
        save_format='jpeg'
    )
    for i in range(len(generator)):
        generator.next()

# Preprocess and save training images
print("Preprocessing and saving training images...")
preprocess_and_save_images(train_data_dir, 'training')

# Preprocess and save validation images
print("Preprocessing and saving validation images...")
preprocess_and_save_images(val_data_dir, 'validation')

print("Data preprocessing completed!")
