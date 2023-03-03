# Deep Learning for MR Image Synthesis
# @author: Alexander F.I. Osman, April 2021

"""
This code demonstrates a 2D U-Net architecture for cross-sequence MR image
translations across T1, T1c, T2, & T2-FLAIR contrasts.
It takes a source image and converting it a target image.
Architectures: 2D U-Net

Dataset: BRATS-2018 challenge dataset.

The training process goes through the following steps:
1. Load and preprocess the data (crop, resize , & normalize)
2. Build the model architecture (2D U-Net)
3. Train the model for dose prediction and validate its performance
4. Make predictions on a test dataset
5. Evaluate the model performance (SSIM, MAE, etc.)
"""

###############################################################################
# 1. LOADING AND PREPROCESSING DATA ###########################################
###############################################################################

import numpy as np
import nibabel as nib
import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
functions 

# Read and process the images (crop, resize, & normalize)
# Source images
dataset_path = 'E:/Datasets/'
dir_list_sc = sorted(glob.glob(dataset_path + '/MR_images_T1/train-pats/*nii.gz'))
img_sc = image_proc(dir_list_sc)
img_sc = np.expand_dims(img_sc, -1)

# Target images
dataset_path = 'E:/Datasets/'
dir_list_tg = sorted(glob.glob(dataset_path + '/MR_images_T2/train-pats/*nii.gz'))
img_tg = image_proc(dir_list_tg)
img_tg = np.expand_dims(img_tg, -1)

# Split data to training and validation sets
x_train, x_val, y_train, y_val = train_test_split(img_sc.astype('float32'), img_tg.astype('float32'), test_size=0.20, random_state=1)

# Plot
slice_numb = 70
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(img_sc[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight')
plt.subplot(222)
plt.imshow(img_tg[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('target image'), plt.axis('tight')
slice_numb = 100
plt.subplot(223)
plt.imshow(x_train[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight')
plt.subplot(224)
plt.imshow(y_train[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('target image'), plt.axis('tight')
plt.show()

###############################################################################
# 2. BUILD THE MODEL ARCHITECTURE #############################################
###############################################################################

from keras.layers import Conv2D, BatchNormalization, Activation, \
    UpSampling2D, MaxPooling2D, Conv2DTranspose, Dropout, Concatenate, Input
from keras.models import Model
import random


def conv_block(input, num_filters):
    """ Convolutional Layers """
    x = Conv2D(num_filters, 3, kernel_initializer='he_uniform', padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, kernel_initializer='he_uniform', padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(input, num_filters):
    """ Encoder Block """
    x = conv_block(input, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(input)
    # x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_2DUNet_model_v1(input_shape):
    """ U-NET Architecture """
    inputs = Input(input_shape, dtype='float32')
    ini_numb_of_filters = 16

    """ Eecoder 1, 2, 3, 4 """
    s1, p1 = encoder_block(inputs, ini_numb_of_filters)
    s2, p2 = encoder_block(p1, ini_numb_of_filters * 2)
    s3, p3 = encoder_block(p2, ini_numb_of_filters * 4)
    s4, p4 = encoder_block(p3, ini_numb_of_filters * 8)

    """ Bridge """
    b1 = conv_block(p4, ini_numb_of_filters * 16)

    """ Decoder 1, 2, 3, 4 """
    d1 = decoder_block(b1, s4, ini_numb_of_filters * 8)
    d2 = decoder_block(d1, s3, ini_numb_of_filters * 4)
    d3 = decoder_block(d2, s2, ini_numb_of_filters * 2)
    d4 = decoder_block(d3, s1, ini_numb_of_filters)

    """ Outputs  """
    outputs = Conv2D(1, 1, padding="same", activation="linear")(d4)

    from keras.optimizers import Adam
    learning_rate = 0.001
    optimizer = Adam(learning_rate)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy', 'mae'])
    return model


# Print a summary of the model
np.random.seed(42)  # seeding for consistency
img_height = x_train.shape[1]  # 64
img_width = x_train.shape[2]  # 64
img_channels = x_train.shape[3]  # 1

input_shape = (img_height, img_width, img_channels)

model = build_2DUNet_model_v1(input_shape)
# model = build_2DUNet_model_v2(input_shape)
print(model.summary())
print(model.input_shape)
print(model.output_shape)

##############################################################################
# 3. TRAINING THE MODEL ######################################################
##############################################################################

import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import pandas as pd
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt


# Hyperparameters
epochs = 3
batch_size = 32
steps_per_epoch = len(x_train) // batch_size
val_steps_per_epoch = len(x_val) // batch_size

# Callbacks
checkpoint_filepath = 'saved_model/MR_synth_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
callbacks = [
    EarlyStopping(patience=50, monitor='val_loss', restore_best_weights=False, verbose=1),
    ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True),
    CSVLogger('MR_Synth_2D_logs.csv',  separator=',')]
   # ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_lr=1e-7, verbose=1)

# Train the model
start = time.time()
history = model.fit(x_train, y_train,
                    steps_per_epoch=steps_per_epoch,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[callbacks],
                    validation_data=(x_val, y_val),
                    validation_steps=val_steps_per_epoch,
                    shuffle=False)

finish = time.time()
print('total exec. time (h)): ', (finish - start)/3600.)
print('Training has been finished successfully')

# Evaluate the model on training and validation sets
mse_train, acc_train, mae_train = model.evaluate(x_train, y_train)
print('training_loss (mse):', np.round(mse_train, 5))
print('training_metric1 (accuracy):', np.round(acc_train, 5))
print('training_metric2 (mae):', np.round(mae_train, 5))

mse_val, acc_val, mae_val = model.evaluate(x_val, y_val)
print('val_loss (mse):', np.round(mse_val, 5))
print('val_metric1 (accuracy):', np.round(acc_val, 5))
print('val_metric2 (mae):', np.round(mae_val, 5))

# Save the trained model
model.save('saved_model/MRI_Synth_T1_T2_2D.hdf5')

# Plot the Learning Curve
filepath = 'MR_Synth_2D_logs.csv'
plot_learning_curve(filepath)

###############################################################################
# 4. MAKE PREDICTIONS #########################################################
###############################################################################

from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt


# Load and process the test data set 
# Source images
dataset_path = 'E:/Datasets/'
dir_list_sc = sorted(glob.glob(dataset_path + '/MR_images_T1/test-pats/*nii.gz'))
img_sc = image_proc(dir_list_sc)
img_sc = np.expand_dims(img_sc, -1)

# Target images
dataset_path = 'E:/Datasets/'
dir_list_tg = sorted(glob.glob(dataset_path + '/MR_images_T2/test-pats/*nii.gz'))
img_tg = image_proc(dir_list_tg)
img_tg = np.expand_dims(img_tg, -1)

# Test data set
x_test, y_test = img_sc.astype('float32'), img_tg.astype('float32')

# Load the trained model
new_model = load_model('saved_model/MRI_Synth_T1_T2_2D.h5')

# Check the model's architecture
new_model.summary()

# Make predictions from test set
test_pred = new_model.predict(x_test, verbose=1, batch_size=2)
test_pred = test_pred[:,:,:,0]   # strip" the last dim.
test_real = y_test[:,:,:,0]   # strip" the last dim.

# Plot
slice_numb = 50
plt.figure()
plt.subplot(231)
plt.imshow(x_test[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight'), plt.axis('off')
plt.subplot(232)
plt.imshow(test_real[slice_numb,:,:].T, cmap='gray')
plt.colorbar(), plt.title('target image (real)'), plt.axis('tight'), plt.axis('off')
plt.subplot(233)
plt.imshow(test_pred[slice_numb,:,:].T, cmap='gray')
plt.colorbar(), plt.title('target image (pred)'), plt.axis('tight'), plt.axis('off')
slice_numb = 100
plt.subplot(234)
plt.imshow(x_test[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight'), plt.axis('off')
plt.subplot(235)
plt.imshow(test_real[slice_numb,:,:].T, cmap='gray')
plt.colorbar(), plt.title('target image (real)'), plt.axis('tight'), plt.axis('off')
plt.subplot(236)
plt.imshow(test_pred[slice_numb,:,:].T, cmap='gray')
plt.colorbar(), plt.title('target image (pred)'), plt.axis('tight'), plt.axis('off')
plt.show()

###############################################################################
# 5. EVALUATE THE MODEL PERFORMANCE  ##########################################
###############################################################################

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, hausdorff_distance, \
    mean_squared_error, normalized_mutual_information, normalized_root_mse
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt


# Evaluate the model on test data set
mse_test, acc_test, mae_test  = new_model.evaluate(x_test, y_test, verbose=1)
print('test_loss (mse):', np.round(mse_test, 5))
print('test_metric1 (accuracy):', np.round(acc_test, 5))
print('test_metric2 (mae):', np.round(mae_test, 5))

# Compute the PSNR, SSIM, MAE, MSE, HDD, NMI, & NRMSE between two images.
psnr = np.zeros(len(test_real))
ssim = np.zeros(len(test_real))
mse = np.zeros(len(test_real))
mae = np.zeros(len(test_real))

for item in tqdm(range(len(test_real)), desc='Computing'):
    # PSNR
    psnr[item] = peak_signal_noise_ratio(test_real[item], test_pred[item], data_range=None)
    # SSIM
    ssim0 = structural_similarity(test_real[item], test_pred[item], full=True)
    ssim[item] = ssim0[0]
    # MSE
    mse[item] = mean_squared_error(test_real[item], test_pred[item])
    # MAE
    mae[item] = abs(test_real[item] - test_pred[item]).mean()

mean_psnr, std_psnr = np.mean(psnr), np.std(psnr)
print("PSNR (dB) =", np.round(mean_psnr, 3), "±", np.round(std_psnr, 3))
mean_ssim, std_ssim = np.mean(ssim), np.std(ssim)
print("SSIM =", np.round(mean_ssim, 3), "±", np.round(std_ssim, 3))
mean_mse, std_mse = np.mean(mse), np.std(mse)
print("MSE =", np.round(mean_mse, 3), "±", np.round(std_mse, 3))
mean_mae, std_mae = np.mean(mae), np.std(mae)
print("MAE =", np.round(mean_mae, 3), "±", np.round(std_mae, 3))

# Plot
slice_numb = 50
fig = plt.figure()
grid = plt.GridSpec(2, 3, wspace = .15, hspace = .15)
exec (f"plt.subplot(grid{[0]})")
plt.imshow(x_test[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight'), plt.axis('off')
exec (f"plt.subplot(grid{[1]})")
plt.imshow(test_real[slice_numb,:,:].T, cmap='gray')
plt.colorbar(), plt.title('target image (real)'), plt.axis('tight'), plt.axis('off')
exec (f"plt.subplot(grid{[2]})")
plt.imshow(test_pred[slice_numb,:,:].T, cmap='gray')
plt.colorbar(), plt.title('target image (pred)'), plt.axis('tight'), plt.axis('off')
exec (f"plt.subplot(grid{[3]})")
residual = test_real[slice_numb,:,:] - test_pred[slice_numb,:,:]
plt.imshow(residual.T, cmap='gray')
plt.colorbar(), plt.title('difference map'), plt.axis('tight'), plt.axis('off')
exec (f"plt.subplot(grid{[4]})")
ssim = structural_similarity(test_real[slice_numb,:,:], test_pred[slice_numb,:,:], full=True)
print("SSIM val =", np.round(ssim[0], 3))
plt.imshow(ssim[1].T, cmap='gray')
plt.colorbar(), plt.title('SSIM'), plt.axis('tight'), plt.axis('off'), plt.clim(0, 1)
plt.show()

###############################################################################
############################### THE END #######################################
###############################################################################
