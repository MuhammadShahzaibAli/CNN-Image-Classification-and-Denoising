# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 00:57:49 2023

@author: Shahzaib
"""

# Importing libraries

import numpy as np
from numpy import asarray
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Loading Data

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0

# Adding noise to data

noise_factor = 0.2
train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape) 
test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape) 

# Making sure values still in (0,1)

train_imagse_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)

# One hot encoding

train_labels_onehot=tf.keras.utils.to_categorical(train_labels)
test_labels_onehot=tf.keras.utils.to_categorical(test_labels)

# CNN classifier

# Input

model=tf.keras.Input(shape=(28,28,1))

batchnorm=tf.keras.layers.BatchNormalization()(model)

# 1st convolution layer

conv=tf.keras.layers.Conv2D(64, (5,5), strides=1, activation="relu", padding='same')(batchnorm)
batchnorm1=tf.keras.layers.BatchNormalization()(conv)

# Max pooling 1st

pool=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(batchnorm1)


# Flatten layer

flatten=tf.keras.layers.Flatten()(pool)

# Fully connected layer

output=tf.keras.layers.Dense(10)(flatten)

full_model=tf.keras.Model(model, output)

# Compiling model

full_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True) ,optimizer='adam',metrics=['accuracy'])

# Model summary

full_model.summary()

# Fitting the model

history=full_model.fit(train_images,train_labels_onehot,batch_size=32,epochs=20,callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=6,restore_best_weights=True),validation_data=(test_images,test_labels_onehot))

# Plotting loss

plt.plot(history.history['loss'])

# Model Evaluation

loss,accuracy=full_model.evaluate(test_images,test_labels_onehot,verbose=2)

print(f'Classification accuracy for clean test images is {accuracy*100}%')

loss_noisy,accuracy_noisy=full_model.evaluate(test_images_noisy,test_labels_onehot,verbose=2)

print(f'Classification accuracy for noisy test images is {accuracy_noisy*100}%')


# CNN autoencoder structure for denoising


encoder_input=tf.keras.Input(shape=(28,28,1))
encoder_batchnorm=tf.keras.layers.BatchNormalization()(encoder_input)
encoder_conv=tf.keras.layers.Conv2D(16, 3, strides=1, activation="relu", padding='same')(encoder_batchnorm)
encoder_conv1=tf.keras.layers.Conv2D(8, 3, strides=1, activation="relu", padding='same')(encoder_conv)
encoder_batchnorm=tf.keras.layers.BatchNormalization()(encoder_conv1)

decoder_conv=tf.keras.layers.Conv2DTranspose(8, 3, strides=1, activation="relu", padding='same')(encoder_batchnorm)
decoder_conv1=tf.keras.layers.Conv2DTranspose(16, 3, strides=1, activation="relu", padding='same')(decoder_conv)
decoder_dense=tf.keras.layers.Dense(1)(decoder_conv1)

autoencoder=tf.keras.Model(encoder_input, decoder_dense)

autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

autoencoder.fit(train_images_noisy, train_images,batch_size=10, epochs=5, shuffle=True, validation_data=(test_images_noisy, test_images))

autoencoder.summary()

denoised_images=autoencoder.predict(test_images_noisy)

n=10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(test_images_noisy[i]))
    plt.gray()
    
    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(denoised_images[i]))
    plt.gray()


loss_denoised,accuracy_denoised=full_model.evaluate(denoised_images,test_labels_onehot,verbose=2)

print(f'Classification accuracy for the autoencoder denoised test images is {accuracy_denoised*100}%')

noisy_train=full_model.fit(train_images_noisy,train_labels_onehot,batch_size=32,epochs=5,validation_data=(test_images,test_labels_onehot))

plt.plot(noisy_train.history['loss'])

loss_noisy_test,accuracy_noisy_test=full_model.evaluate(test_images_noisy,test_labels_onehot,verbose=2)

print(f'Classification accuracy for noisy test images with model trained on noisy train images is {accuracy_noisy_test*100}%')
