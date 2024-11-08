## UNCOMMENTING THESE TWO LINES WILL FORCE KERAS/TF TO RUN ON CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Conv3D, MaxPooling3D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling3D, Input, AveragePooling3D, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import random


# CONSTANTS
NB_CLASSES = 50  # 75 heart rate classes
LENGTH_VIDEO = 64
IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16
IMAGE_CHANNELS = 1
BATCH_SIZE = 16
EPOCHS = 1000

# Load your dataset here
CSV_PATH = 'BP.csv'
DATASET_PATH = 'dataset/'

# 1. Function to preprocess each video
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < LENGTH_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame / 255.0
        frames.append(frame)
    
    while len(frames) < LENGTH_VIDEO:
        frames.append(frames[-1])  # Pad with the last frame
    
    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=-1)  # Add channel dimension
    return frames

# 2. Load heart rates and video names from CSV
def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    video_names = df['video_name'].values
    heart_rate_values = df['heart_rate'].values
    return video_names, heart_rate_values

# 3. Map heart rate values to the closest heart rate class
def map_heart_rate_to_class(heart_rate):
    heart_rates = np.linspace(55, 240, NB_CLASSES)
    return np.argmin(np.abs(heart_rates - heart_rate))

# 4. Process the dataset
def process_dataset(csv_path, dataset_path):
    video_names, heart_rate_values = load_csv_data(csv_path)
    
    video_data = []
    heart_rate_classes = []
    
    for video_name, heart_rate in zip(video_names, heart_rate_values):
        video_path = f'{dataset_path}/{video_name}'
        processed_video = preprocess_video(video_path)
        video_data.append(processed_video)
        
        heart_rate_class = map_heart_rate_to_class(heart_rate)
        heart_rate_classes.append(heart_rate_class)
    
    video_data = np.array(video_data)
    heart_rate_labels = to_categorical(heart_rate_classes, num_classes=NB_CLASSES + 1)  # +1 for noise class
    
    return video_data, heart_rate_labels

# Load and preprocess the dataset
x_train, y_train = process_dataset(CSV_PATH, DATASET_PATH)

# Split into training and validation sets
x_validation = x_train[-9:]  # Last 9 for validation
y_validation = y_train[-9:]  # Last 9 for validation
x_train = x_train[:-9]  # All but last 9 for training
y_train = y_train[:-9]  # All but last 9 for training

class VideoDataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, augment=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augment  # Enable augmentation flag
    
    def __len__(self):
        return len(self.x) // self.batch_size
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if self.augment:
            batch_x = self.apply_augmentation(batch_x)
        
        return np.array(batch_x), np.array(batch_y)
    
    def apply_augmentation(self, videos):
        """Apply spatial and temporal augmentations to the video data."""
        augmented_videos = []
        
        for video in videos:
            augmented_video = []
            for frame in video:
                augmented_frame = self.spatial_augment(frame)
                augmented_video.append(augmented_frame)
            
            augmented_video = np.array(augmented_video)
            augmented_video = self.temporal_augment(augmented_video)
            augmented_videos.append(augmented_video)
        
        return np.array(augmented_videos)
    
    def spatial_augment(self, frame):
        """Perform random spatial augmentations on a single frame."""
        frame = tf.image.random_flip_left_right(frame)
        frame = tf.image.random_brightness(frame, max_delta=0.2)
        frame = tf.image.rot90(frame, k=np.random.choice([0, 1, 2, 3]))  # Rotate by 90, 180, or 270 degrees
        
        return frame
    
    def temporal_augment(self, video):
        """Perform random temporal augmentations on a video sequence."""
        if np.random.rand() < 0.5:
            # Randomly drop some frames (frame skipping)
            drop_idx = np.random.choice(video.shape[0], size=int(0.1 * video.shape[0]), replace=False)
            video = np.delete(video, drop_idx, axis=0)
        
        if np.random.rand() < 0.5:
            # Randomly duplicate some frames (frame duplication)
            dup_idx = np.random.choice(video.shape[0], size=int(0.1 * video.shape[0]), replace=False)
            dup_frames = video[dup_idx]
            video = np.concatenate([video, dup_frames], axis=0)
        
        # Ensure the sequence length remains constant by trimming or padding
        if len(video) > LENGTH_VIDEO:
            video = video[:LENGTH_VIDEO]  # Trim extra frames
        elif len(video) < LENGTH_VIDEO:
            padding = np.zeros((LENGTH_VIDEO - len(video), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
            video = np.concatenate([video, padding], axis=0)  # Pad with zeros
        
        return video

# MODEL DEFINITION
model = Sequential()

# Input layer
model.add(Input(shape=(LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))

# First convolutional block
model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1)))  # Reduced filters
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Second convolutional block
model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1)))  # Reduced filters
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

# Third convolutional block
model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

# Global average pooling
model.add(GlobalAveragePooling3D())

# Flatten the output
model.add(Flatten())  # Flatten to convert to 2D tensor

# Reshaping the output for LSTM
model.add(Reshape((LENGTH_VIDEO, -1)))  # (timesteps, features)

# LSTM Layer
model.add(LSTM(128, return_sequences=False))  # You can adjust the units as needed
model.add(Dropout(0.5))

# Fully connected layers
model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))

# Output layer (softmax for classification)
model.add(Dense(NB_CLASSES + 1, activation='softmax'))  # +1 for noise class

# Compile model with a slightly increased learning rate
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Model summary
model.summary()

# CALLBACKS
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('model_weights_epoch_{epoch:02d}.keras', save_freq=100, monitor='val_accuracy', save_best_only=True)

# DATA AUGMENTATION: Apply augmentation to the training data (rotation, flipping, etc.)
# Define VideoDataGenerator with augmentation
train_generator = VideoDataGenerator(x_train, y_train, batch_size=BATCH_SIZE)
validation_generator = VideoDataGenerator(x_validation, y_validation, batch_size=BATCH_SIZE)

# TRAINING THE MODEL
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=EPOCHS,
                    verbose=2,
                    callbacks=[early_stopping, reduce_lr, checkpoint])


# history = model.fit(x_train,
#                     y_train,
#                     epochs=EPOCHS,
#                     verbose=2,
#                     callbacks=[early_stopping, reduce_lr, checkpoint])
# PLOTTING RESULTS
plt.figure(figsize=(12, 8))

# Plot history for accuracy
plt.subplot(211)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot history for loss
plt.subplot(212)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()
