## UNCOMMENTING THESE TWO LINES WILL FORCE KERAS/TF TO RUN ON CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Conv3D, MaxPooling3D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling3D, Input, AveragePooling3D
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import random


# CONSTANTS
NB_CLASSES = 75  # 75 heart rate classes
LENGTH_VIDEO = 65
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
IMAGE_CHANNELS = 1

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
    def __init__(self, video_data, labels, batch_size=32, shuffle=True):
        self.video_data = video_data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.video_data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.video_data) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_video_data = self.video_data[indices]
        batch_labels = self.labels[indices]
        return self.__data_augmentation(batch_video_data), batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_augmentation(self, batch_video_data):
        augmented_videos = []
        for video in batch_video_data:
            # Randomly flip the video
            if random.random() > 0.5:
                video = np.flip(video, axis=2)  # Flip along the time axis

            # Randomly rotate each frame in the video
            for i in range(video.shape[0]):
                if random.random() > 0.5:
                    angle = random.randint(0, 360)
                    M = cv2.getRotationMatrix2D((IMAGE_WIDTH/2, IMAGE_HEIGHT/2), angle, 1)
                    video[i] = cv2.warpAffine(video[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), M, (IMAGE_WIDTH, IMAGE_HEIGHT)).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

            augmented_videos.append(video)
        
        return np.array(augmented_videos)


# MODEL DEFINITION
model = Sequential()

# Input layer
model.add(Input(shape=(LENGTH_VIDEO, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))

# First convolutional block
model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1)))  # Reduced filters
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))  # Reduced dropout

# Second convolutional block
model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1)))  # Reduced filters
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))  # Reduced dropout

# Third convolutional block
model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

# Global average pooling
model.add(GlobalAveragePooling3D())

# Fully connected layers
model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))  # Reduced neurons
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES + 1, activation='softmax'))  # +1 for noise class

    

# Compile model with a slightly increased learning rate
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
model.summary()

# CALL BACKS
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('model_weights_epoch_{epoch:02d}.keras', save_freq=100, monitor='val_accuracy', save_best_only=True)



# TRAINING THE MODEL
EPOCHS = 1000  # Start with fewer epochs
# history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=EPOCHS, verbose=2, 
#                     callbacks=[early_stopping, reduce_lr, checkpoint])

# Define batch size
BATCH_SIZE = 8  # Adjust based on your memory capacity

# Create the data generator
train_generator = VideoDataGenerator(x_train, y_train, batch_size=BATCH_SIZE)
validation_generator = VideoDataGenerator(x_validation, y_validation, batch_size=BATCH_SIZE)

# Train the model using the generator
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=EPOCHS,
                    verbose=2,
                    callbacks=[early_stopping, reduce_lr, checkpoint])


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
