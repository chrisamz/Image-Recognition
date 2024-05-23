# exploratory_data_analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Define file paths
processed_data_dir = 'data/processed/'
train_data_dir = os.path.join(processed_data_dir, 'train/')
val_data_dir = os.path.join(processed_data_dir, 'val/')
test_data_dir = os.path.join(processed_data_dir, 'test/')
labels_csv = 'data/raw/labels.csv'

# Load image labels
labels_df = pd.read_csv(labels_csv)

# Display the first few rows of the labels dataframe
print("First few rows of the labels dataframe:")
print(labels_df.head())

# Summary statistics of the labels
print("\nSummary statistics of the labels:")
print(labels_df.describe())

# Distribution of labels
plt.figure(figsize=(10, 6))
sns.countplot(data=labels_df, x='label')
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Load a sample of images and display them
def load_and_display_images(data_dir, sample_size=5):
    sample_files = os.listdir(data_dir)[:sample_size]
    plt.figure(figsize=(15, 5))
    for i, file in enumerate(sample_files):
        img_path = os.path.join(data_dir, file)
        img = load_img(img_path)
        plt.subplot(1, sample_size, i+1)
        plt.imshow(img)
        plt.title(file)
        plt.axis('off')
    plt.show()

print("\nDisplaying a sample of training images:")
load_and_display_images(train_data_dir)

# Image dimensions and statistics
def get_image_stats(data_dir):
    img_dims = []
    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        img = load_img(img_path)
        img_array = img_to_array(img)
        img_dims.append(img_array.shape)
    img_dims = np.array(img_dims)
    print("\nImage dimensions statistics:")
    print(f"Mean dimensions: {img_dims.mean(axis=0)}")
    print(f"Std dimensions: {img_dims.std(axis=0)}")
    print(f"Min dimensions: {img_dims.min(axis=0)}")
    print(f"Max dimensions: {img_dims.max(axis=0)}")

get_image_stats(train_data_dir)

print("\nDisplaying a sample of validation images:")
load_and_display_images(val_data_dir)

# Additional visualizations (e.g., mean image, std image)
def display_mean_image(data_dir):
    img_arrays = []
    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_arrays.append(img_array)
    img_arrays = np.array(img_arrays)
    mean_img = np.mean(img_arrays, axis=0)
    plt.imshow(mean_img.astype('uint8'))
    plt.title('Mean Image')
    plt.axis('off')
    plt.show()

print("\nDisplaying the mean image of the training set:")
display_mean_image(train_data_dir)

print("\nDisplaying the mean image of the validation set:")
display_mean_image(val_data_dir)

print("EDA completed!")
