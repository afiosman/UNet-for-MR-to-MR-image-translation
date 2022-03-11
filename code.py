## Deep Learning for MR Image Synthesis
# @author: Alexander F.I. Osman, April 2021

"""
This code demonstrates a CNN (Convolutional Encoder-Decoder) using
BRATS-2018 imaging dataset.
Training and testing for end-to-end image translation of MRI contrasts
This code uses 256x256 2D images.

The code goes through the following steps:
1. load data and pre-process the images (clean data, split , normalize, etc.)
2. build model architecture
3. train model and validate its performance
4. make predictions for previously unseen/test data
5. evaluate its predictions for previously unseen data
"""

###############################################################################
# 1. LOAD DATA AND PREFORM PRE-PROCESSING #####################################
###############################################################################

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

# For consistency
seed = 42
np.random.seed = seed

# Reading the *nii MR image data files - source and target images
img_pathS_T1 = 'MR_and_MR_images/train_path/Source_images_T1\Brats18_2013_1_1_t1.nii.gz'
img_pathT_T2 = 'MR_and_MR_images/train_path/Target_images_T2\Brats18_2013_1_1_t2.nii.gz'

dataS_T1 = nib.load(img_pathS_T1)
dataT_T2 = nib.load(img_pathT_T2)

dataS_T1 = dataS_T1.get_fdata()
dataT_T2 = dataT_T2.get_fdata()

# Image preprocessing: (A) Get the brain region (dimensionality reduction)
indice_listS_T1 = np.where(dataS_T1 > 0)
indice_listT_T2 = np.where(dataT_T2 > 0)
# calculate the min and max of the indice,  here volume have 3 channels
# channel_0_min = min(indice_list[0])
# channel_0_max = max(indice_list[0])
# channel_1_min = min(indice_list[1])
# channel_1_max = max(indice_list[1])
channel_2_minS_T1 = min(indice_listS_T1[2])
channel_2_maxS_T1 = max(indice_listS_T1[2])

channel_2_minT_T2 = min(indice_listT_T2[2])
channel_2_maxT_T2 = max(indice_listT_T2[2])

brain_volS_T1 = dataS_T1[:,:,channel_2_minS_T1:channel_2_maxS_T1]
brain_volT_T2 = dataT_T2[:,:,channel_2_minT_T2:channel_2_maxT_T2]

# Sanity check, view few images
last = channel_2_maxS_T1 - channel_2_minS_T1

image_number = random.randint(0, last-3)
print(image_number)
plt.figure()
plt.subplot(121)
plt.imshow(brain_volS_T1[:,:,image_number].T, cmap='gray')
plt.title('S'), plt.colorbar()
plt.subplot(122)
plt.imshow(brain_volT_T2[:,:,image_number].T, cmap='gray')
plt.title('T'), plt.colorbar()
plt.show()

"""
# Image preprocessing: (B) N4 Bias Field Correction by simpleITK
#This function carry out BiasFieldCorrection for the files in a specific directory
#src_path: path of the source file
#dst_path: path of the target file

src_path = 'MR_and_MR_images/train_path/Source_images_T1\Brats18_2013_0_1_t1.nii.gz'
inputImage = sitk.ReadImage(src_path)
maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
corrector = sitk.N4BiasFieldCorrectionImageFilter()
output = corrector.Execute(inputImage, maskImage)

#Sanity check, view few images
plt.figure()
plt.subplot(121)
plt.imshow(output[:,:,image_number].T, cmap='gray')
plt.title('S'), plt.colorbar()
plt.subplot(122)
plt.imshow(output[:,:,image_number].T, cmap='gray')
plt.title('T'), plt.colorbar()
plt.show()
"""

# Image preprocessing: (B) Normalize the data (zero mean and unit variance)
axis = None
vol_meanS_T1 = np.mean(brain_volS_T1, axis=axis)
vol_stdS_T1 = np.std(brain_volS_T1, axis=axis)
vol_normS_T1 = np.abs((brain_volS_T1 - vol_meanS_T1) / vol_stdS_T1)

vol_meanT_T2 = np.mean(brain_volT_T2, axis=axis)
vol_stdT_T2 = np.std(brain_volT_T2, axis=axis)
vol_normT_T2 = np.abs((brain_volT_T2 - vol_meanT_T2) / vol_stdT_T2)

#Sanity check, view few images
plt.figure()
plt.subplot(121)
plt.imshow(vol_normS_T1[:,:,image_number].T, cmap='gray')
plt.title('S'), plt.colorbar()
plt.subplot(122)
plt.imshow(vol_normT_T2[:,:,image_number].T, cmap='gray')
plt.title('T'), plt.colorbar()
plt.show()

# Scaling/Normalization: mapping to 0-1
axis = None
vol_maxS_T1 = np.max(vol_normS_T1, axis=axis)
vol_minS_T1 = np.min(vol_normS_T1, axis=axis)
vol_normS_T1 = (vol_normS_T1 - vol_minS_T1)/(vol_maxS_T1 - vol_minS_T1)

vol_maxT_T2 = np.max(vol_normT_T2, axis=axis)
vol_minT_T2 = np.min(vol_normT_T2, axis=axis)
vol_normT_T2 = (vol_normT_T2 - vol_minT_T2)/(vol_maxT_T2 - vol_minT_T2)

#Sanity check, view few images
plt.figure()
plt.subplot(121)
plt.imshow(vol_normS_T1[:,:,image_number].T, cmap='gray')
plt.title('S'), plt.colorbar()
plt.subplot(122)
plt.imshow(vol_normT_T2[:,:,image_number].T, cmap='gray')
plt.title('T'), plt.colorbar()
plt.show()

# Image preprocessing: (C) Resizing the images
width0, height0, depth0 = vol_normS_T1.shape

width1 = 224
height1 = 224
depth1 = depth0

resiedS_T1 = np.zeros((depth1, width1, height1))
resiedT_T2 = np.zeros((depth1, width1, height1))

for i in range (depth1-1):
    imgS_T1 = vol_normS_T1[:,:,i]
    imgT_T2 = vol_normT_T2[:,:,i]
    img_smS_T1 = cv2.resize(imgS_T1, (width1, height1), interpolation=cv2.INTER_CUBIC)
    img_smT_T2 = cv2.resize(imgT_T2, (width1, height1), interpolation=cv2.INTER_CUBIC)
    resiedS_T1[i,:,:] = np.abs(img_smS_T1)
    resiedT_T2[i,:,:] = np.abs(img_smT_T2)

# Sanity check, view few images
plt.figure()
plt.subplot(121)
plt.imshow(resiedS_T1[image_number,:,:].T, cmap='gray')
plt.title('S'), plt.colorbar()
plt.subplot(122)
plt.imshow(resiedT_T2[image_number,:,:].T, cmap='gray')
plt.title('T'), plt.colorbar()
plt.show()

###############################################################################
# 2. BUILD THE MODEL ARCHITECTURE #############################################
###############################################################################

#vol_maxS_T1 = print(np.min(resiedS_T1))
#vol_maxT_T2 = print(np.min(resiedT_T2))
width1 = 224
height1 = 224
channel1 = 1
depth1 = depth0

trainS_T1 = np.expand_dims(resiedS_T1, -1)
trainS_T1 = trainS_T1.astype('float32')

trainT_T2 = np.expand_dims(resiedT_T2, -1)
trainT_T2 = trainT_T2.astype('float32')

X_train, X_test, Y_train, Y_test = train_test_split(trainS_T1, trainT_T2, test_size=0.20, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.20, random_state=1)

start = time.time()

inputs = Input((height1, width1, channel1))
s = inputs
#LeakyReLU
#Contraction path
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c5)

#Expansive path
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#u6 = UpSampling2D(size=(2, 2))(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#u7 = UpSampling2D(size=(2, 2))(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#u8 = UpSampling2D(size=(2, 2))(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#u9 = UpSampling2D(size=(2, 2))(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
model.summary()

###############################################################################
# 3. TRAIN AND VALIDATE THE CNN MODEL #########################################
###############################################################################

# Use early stopping method to solve model over-fitting problem
callbacks = tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss', verbose=1)
# The patience parameter is the amount of epochs to check for improvement

# Train the model
history = model.fit(X_train, Y_train,
                    batch_size=32,
                    epochs=150,
                    verbose=1,
                    callbacks=[callbacks],
                    validation_data=(X_val, Y_val),
                    shuffle=True)

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

# Predict model on the training/same input array.
predTrain = model.predict(X_train, verbose = 1)
predTrain = np.reshape(predTrain, (len(predTrain), width1, height1))

gtTrain = np.reshape(Y_train, (len(Y_train), width1, height1))

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
image_number = random.randint(0, len(Y_test-3))
#print(image_number)
plt.figure()
plt.subplot(241)
plt.imshow(predTest[image_number,:,:].T, cmap='gray')
plt.title('Predicted Test'), #plt.clim(0, 1), plt.colorbar()
plt.subplot(245)
plt.imshow(gtTest[image_number,:,:].T, cmap='gray')
plt.title('GT'), #plt.clim(0, 1), plt.colorbar()

image_number = random.randint(0, len(Y_test-3))
plt.subplot(242)
plt.imshow(predTest[image_number,:,:].T, cmap='gray')
plt.title('Predicted Test'), #plt.clim(0, 1), plt.colorbar()
plt.subplot(246)
plt.imshow(gtTest[image_number,:,:].T, cmap='gray')
plt.title('GT'), #plt.clim(0, 1), plt.colorbar()

image_number = random.randint(0, len(Y_test-3))
plt.subplot(243)
plt.imshow(predTest[image_number,:,:].T, cmap='gray')
plt.title('Predicted Test'), #plt.clim(0, 1), plt.colorbar()
plt.subplot(247)
plt.imshow(gtTest[image_number,:,:].T, cmap='gray')
plt.title('GT'), #plt.clim(0, 1), plt.colorbar()

image_number = random.randint(0, len(Y_test-3))
plt.subplot(244)
plt.imshow(predTest[image_number,:,:].T, cmap='gray')
plt.title('Predicted Test'), #plt.clim(0, 1), plt.colorbar()
plt.subplot(248)
plt.imshow(gtTest[image_number,:,:].T, cmap='gray')
plt.title('GT'), #plt.clim(0, 1), plt.colorbar()
plt.show()

###############################################################################
# 5. EVALUATE THE PREDICTIONS ON TEST DATASET #########################################
###############################################################################

ref_img = gtTest[5,:,:]
#ref_img = img_as_ubyte(ref_img)
print(ref_img.dtype)

eval_img = predTest[5,:,:]
print(eval_img.dtype)

# PSNR: Compute the peak signal to noise ratio (PSNR) for an image.
psnr = skimage.metrics.peak_signal_noise_ratio(ref_img, eval_img,data_range=None)
print("PSNR of input noisy image = ", psnr)
print(f"PSNR value is {psnr} dB")

# SSIM: Compute the mean structural similarity index between two images.
ssim = skimage.metrics.structural_similarity(ref_img, eval_img, full=True)
print("SSIM of input noisy image = ", ssim[0])

plt.figure()
plt.imshow(ssim[1].T, cmap='gray')
plt.title('ssim'), plt.clim(0, 1)
plt.colorbar()
plt.show()

# MSE: Compute the mean-squared error between two images.
mse = skimage.metrics.mean_squared_error(eval_img, ref_img)
print("MSE of input noisy image = ", mse)

# AE & MAE: Compute the mean absolute error between two images.
mae = sklearn.metrics.mean_absolute_error(ref_img, eval_img)
print("MAE of input noisy image = ", mae)

abs_diff_resid = np.abs(eval_img - ref_img)

plt.figure()
plt.imshow(abs_diff_resid.T, cmap='gray')
plt.title('absolute difference map (residual)'), plt.clim(0, 1)
plt.colorbar()
plt.show()


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


