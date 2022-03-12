## Deep Learning for Cross-Sequence MR Image Synthesis
# @author: Alexander F.I. Osman, April 2021

# This code demonstrates a U-Net model for end-to-end MR image translation using BRATS-2018 imaging dataset.

# Import libraries
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os
import random
import SimpleITK as sitk
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from matplotlib.pyplot import imshow
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import skimage
import skimage.metrics
import sklearn.metrics
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from skimage.util import img_as_ubyte
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.models import load_model
from keras.optimizers import Adam

###############################################################################
# 1. LOAD DATA AND PREFORM PRE-PROCESSING #####################################
###############################################################################

# Expand the dimension of the array (:,240,240) to (:,240,240,1)
trainSrc = np.expand_dims(dataSrc_allTrain, -1)
trainSrc = trainSrc.astype('float32')

trainTrgt = np.expand_dims(dataTrgt_allTrain, -1)
trainTrgt = trainTrgt.astype('float32')

X_train, X_test, Y_train, Y_test = train_test_split(trainSrc, trainTrgt, test_size=0.20, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.20, random_state=1)

###############################################################################
# 2. BUILD THE MODEL ARCHITECTURE #############################################
###############################################################################

# For consistency
seed = 42
np.random.seed = seed

height1 = 224
width1 = 224
channel1 = 1

inputs = Input((height1, width1, channel1))
s = inputs
#Contraction path
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(s)
#c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(p1)
#c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(p2)
#c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(p3)
#c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(p4)
#c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c5)

#Expansive path
#u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = UpSampling2D(size=(2, 2))(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c6)

#u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = UpSampling2D(size=(2, 2))(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c7)

#u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = UpSampling2D(size=(2, 2))(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c8)

#u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = UpSampling2D(size=(2, 2))(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c9)

outputs = Conv2D(1, (1, 1), activation='tanh')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
#Summerize layers
print(model.summary())
#Plot graph
#plot_model(model, to_file='simple_model.png')

###############################################################################
# 3. TRAIN AND VALIDATE THE CNN MODEL #########################################
###############################################################################

# Use early stopping method to solve model over-fitting problem
callbacks = tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss', verbose=1)
# The patience parameter is the amount of epochs to check for improvement

# Train the model
start = time.time()
history = model.fit(X_train, Y_train,
                    batch_size=32,
                    epochs=120,
                    verbose=1,
                    callbacks=[callbacks],
                    validation_data=(X_val, Y_val))

finish = time.time()
print('total_time = ', finish - start)
#print(history.history.keys())
print('Training has been finished successfully')

# loss curve: plots the graph of the training loss vs.
# validation loss over the number of epochs.
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('average training loss and validation loss')
plt.ylabel('mean-squared error')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper right')
plt.show()

# Save Model
model.save('saved_model/MRI_Synthesis_T1_T2')
#model.save('MRI_Synthesis_T1_T2.h5')

###############################################################################
# 4. MAKE PREDICTIONS ON TEST DATASET #########################################
###############################################################################

# Set compile=False as we are not loading it for training, only for prediction.
new_model = load_model('saved_model/MRI_Synthesis_T1_T2', compile=False)
#new_model = load_model('MRI_Synthesis_T1_T2.h5')
# Check its architecture
new_model.summary()

# Predict on the test set: Let us see how the model generalize by using the test set.
predTest = new_model.predict(X_test, verbose=1)
predTest = np.reshape(predTest, (len(predTest), width1, height1))

gtTest = np.reshape(Y_test, (len(Y_test), width1, height1))

# Sanity check, view few images
image_number = random.randint(0, len(Y_test-1))
#print(image_number)
plt.figure()
plt.subplot(241)
plt.imshow(predTest[image_number,:,:], cmap='gray')
plt.title('Predicted Test'), #plt.clim(0, 1)
plt.colorbar()
plt.subplot(245)
plt.imshow(gtTest[image_number,:,:], cmap='gray')
plt.title('GT'), #plt.clim(0, 1)
plt.colorbar()

###############################################################################
# 5. EVALUATE THE PREDICTIONS ON TEST DATASET #########################################
###############################################################################

ref_img = gtTest
eval_img = predTest

_psnr = np.zeros(len(ref_img))
_ssim = np.zeros(len(ref_img))
_mse = np.zeros(len(ref_img))
_mae = np.zeros(len(ref_img))

for j in range (len(ref_img)):
    # PSNR: Compute the peak signal to noise ratio (PSNR) for an image.
    _psnr[j] = skimage.metrics.peak_signal_noise_ratio(ref_img[j], eval_img[j],data_range=None)
    # SSIM: Compute the mean structural similarity index between two images.
    ssim = skimage.metrics.structural_similarity(ref_img[j], eval_img[j], full=True)
    _ssim[j] = ssim[0]
    # MSE: Compute the mean-squared error between two images.
    _mse[j] = skimage.metrics.mean_squared_error(eval_img[j], ref_img[j])
    # AE & MAE: Compute the mean absolute error between two images.
    _mae[j] = sklearn.metrics.mean_absolute_error(ref_img[j], eval_img[j])

average_psnr = np.average(_psnr)
std_psnr = np.std(_psnr)
average_ssim = np.average(_ssim)
std_ssim = np.std(_ssim)
average_mse = np.average(_mse)
std_mse = np.std(_mse)
average_mae = np.average(_mae)
std_mae = np.std(_mae)

print(f"average_PSNR value is {average_psnr} dB")
print(f"std_PSNR value is {std_psnr} dB")
print("average_SSIM of input noisy image = ", average_ssim)
print("std_SSIM of input noisy image = ", std_ssim)
print("average_MSE of input noisy image = ", average_mse)
print("std_MSE of input noisy image = ", std_mse)
print("average_MAE of input noisy image = ", average_mae)
print("std_MAE of input noisy image = ", std_mae)

###############
ref_img = gtTest[125,:,:]
#ref_img = img_as_ubyte(ref_img)
print(ref_img.dtype)

eval_img = predTest[125,:,:]
print(eval_img.dtype)

plt.figure()
plt.subplot(121)
plt.imshow(ref_img, cmap='gray')
plt.title('Predicted Test'), #plt.clim(0, 1)
plt.colorbar()
plt.subplot(122)
plt.imshow(eval_img, cmap='gray')
plt.title('GT'), #plt.clim(0, 1)
plt.colorbar()
plt.show()

# PSNR: Compute the peak signal to noise ratio (PSNR) for an image.
psnr = skimage.metrics.peak_signal_noise_ratio(ref_img, eval_img,data_range=None)
#print("PSNR of input noisy image = ", psnr)
print(f"PSNR value is {psnr} dB")

# SSIM: Compute the mean structural similarity index between two images.
ssim = skimage.metrics.structural_similarity(ref_img, eval_img, full=True)
print("SSIM of input noisy image = ", ssim[0])

plt.figure()
plt.imshow(ssim[1], cmap='gray')
plt.title('ssim'), plt.clim(0, 1)
plt.colorbar()
plt.axis('off')
plt.show()

# MSE: Compute the mean-squared error between two images.
mse = skimage.metrics.mean_squared_error(eval_img, ref_img)
print("MSE of input noisy image = ", mse)

# AE & MAE: Compute the mean absolute error between two images.
mae = sklearn.metrics.mean_absolute_error(ref_img, eval_img)
print("MAE of input noisy image = ", mae)

abs_diff = np.abs(eval_img - ref_img)

plt.figure()
plt.imshow(abs_diff, cmap='gray')
plt.title('absolute difference map (residual)'), plt.clim(0, 1)
plt.colorbar()
plt.axis('off')
plt.show()

###############################################################################
###############################################################################
