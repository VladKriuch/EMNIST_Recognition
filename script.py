import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split

import PIL
import seaborn as sns

train_data_path = 'emnist-balanced-train.csv'
validation_data_path = 'emnist-balanced-test.csv'

train_data = pd.read_csv(train_data_path, header=None)
validation_data = pd.read_csv(validation_data_path, header=None)

# The classes of this balanced dataset are as follows. Index into it based on class label
class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

train_imgs = np.transpose(train_data.values[:,1:].reshape(len(train_data), 28, 28, 1), axes=[0,2,1,3])
train_labels = tf.keras.utils.to_categorical(train_data.values[:,0], num_classes)

validation_imgs = np.transpose(validation_data.values[:,1:].reshape(len(validation_data), 28, 28, 1), axes=[0,2,1,3])
validation_labels = tf.keras.utils.to_categorical(validation_data.values[:,0], num_classes)

train_imgs = (train_imgs.astype(np.float64)) / 255
validation_imgs = (validation_imgs.astype(np.float64)) / 255

train_x,test_x,train_y,test_y = train_test_split(train_imgs,train_labels,test_size=0.3,random_state = 42)
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomTranslation(0.2, 0.2),
  tf.keras.layers.RandomZoom(0.2),
])

augmented_imgs = data_augmentation(train_imgs)
labels = np.copy(train_labels)

augmented_imgs = np.append(augmented_imgs, data_augmentation(train_imgs), axis=0)
labels = np.append(labels, train_labels)

train_imgs = np.append(train_imgs, augmented_imgs, axis=0)
train_labels = np.append(train_labels, labels)

del augmented_imgs
del labels

number_of_classes = 47

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(14,kernel_size=2,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2), # loosing data ?
    tf.keras.layers.Conv2D(28, kernel_size=2, activation='relu'),
    tf.keras.layers.Conv2D(56, kernel_size=3),
    tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Conv2D(28,kernel_size=3,input_shape=(28,28,1), padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(248,activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(number_of_classes,activation='softmax')
])

MCP = ModelCheckpoint('Best_points.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=9,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=9,factor=0.2,min_lr=0.0001)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=35, validation_data=(test_x,test_y),callbacks=[MCP,ES,RLP])

model.evaluate(validation_imgs, validation_labels)