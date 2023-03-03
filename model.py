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
2. Pre-process the data (data resizing, cropping, normalization, etc.)
3. Build the model architecture (2D U-Net)
4. Train the model for dose prediction and validate its performance
5. Make predictions on a test dataset
6. Evaluate the model performance (SSIM, MAE, etc.)
"""

###############################################################################
# 1. LOADING DATA #############################################################
###############################################################################

import numpy as np
import nibabel as nib
import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt


def read_nifti_file(filepath):
    """ Data loader (*.nii)
    :param filepath: file path
    :return: 2D array images
    """
    img_data0 = np.zeros((240, 240, 155), dtype='float32')
    img_data = []
    for item in tqdm(sorted(filepath), desc='Loading'):
        img = nib.load(item).get_fdata()
        img_data0 = np.concatenate((img_data0, img), axis=2)
    return np.array(img_data0[:,:,155::]).astype('float32')


# Read source images
dataset_path = 'E:/Datasets/'
dir_list_sc = sorted(glob.glob(dataset_path + '/MR_images_T1/train-pats/*nii.gz'))
img_sc = read_nifti_file(dir_list_sc)

# Read target images
dataset_path = 'E:/Datasets/'
dir_list_tg = sorted(glob.glob(dataset_path + '/MR_images_T2/train-pats/*nii.gz'))
img_tg = read_nifti_file(dir_list_tg)


print("Used memory to store img_sc: ", img_sc.nbytes/(1024*1024), "MB")
print("Used memory to store img_tg: ", img_tg.nbytes/(1024*1024), "MB")

# Plot
slice_numb = 70
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(img_sc[:,:,slice_numb].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight')
plt.subplot(222)
plt.imshow(img_tg[:,:,slice_numb].T, cmap='gray')
plt.colorbar(), plt.title('target image'), plt.axis('tight')
slice_numb = 100
plt.subplot(223)
plt.imshow(img_sc[:,:,slice_numb].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight')
plt.subplot(224)
plt.imshow(img_tg[:,:,slice_numb].T, aspect=0.5, cmap='gray')
plt.colorbar(), plt.title('target image'), plt.axis('tight')
plt.show()

###############################################################################
# 2. DATA PREPROCESSING #######################################################
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


def image_proc(filepath):
    """ Data loader (*.nii)
    :param filepath: file path
    :return: 2D array images
    """
    img_data0 = np.zeros((96, 96, 1), dtype='float32')
    img_data = []
    for item in tqdm(sorted(filepath), desc='Processing'):
        # loading images
        img = nib.load(item).get_fdata()
        # Crop to get the brain region (along z-axis and x & y axes)
        ind = np.where(img > 0)
        ind_min, ind_max = min(ind[2]), max(ind[2])
        ind_mid = round((ind_min + ind_max) / 2)
        img = img[8:232,8:232,ind_mid-32:ind_mid+32]   # to have 224 x 224 x 64 dim.
        # resize
        img = zoom(img, (0.428, 0.428, 1))   # to have 96 x 96 x 64 dim.
        # Normalize using zero mean and unit variance method & scale to 0-1 range
        img = ((img - img.mean()) / img.std())
        img = ((img - img.min()) / (img.max() - img.min()))  # Scale to 0-1 range
        # Convert 3D images to 2D image slices
        img_data0 = np.concatenate((img_data0, img), axis=2)
    img_data0 = np.moveaxis(img_data0, [2], [0])
    return np.array(img_data0[1::,:,:]).astype('float32')


# Process the images (crop, resize, & normalize)
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


print("Used memory to store img_sc: ", img_sc.nbytes/(1024*1024), "MB")
print("Used memory to store img_tg: ", img_tg.nbytes/(1024*1024), "MB")
print("Used memory to store x_train: ", x_train.nbytes/(1024*1024), "MB")
print("Used memory to store y_train: ", y_train.nbytes/(1024*1024), "MB")

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
# 3. BUILD THE MODEL ARCHITECTURE #############################################
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


def build_2DUNet_model_v2(input_shape):
    """ 3D Standard U-NET Architecture
    :param input_shape: (image height, image width, image depth, image channels)
    :return: model
    """
    inputs = Input(input_shape)
    ini_numb_of_filters = 16

    """ Contraction path """
    c1 = Conv2D(ini_numb_of_filters, 3, kernel_initializer='he_uniform', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation("relu")(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(ini_numb_of_filters, 3, kernel_initializer='he_uniform', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation("relu")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(ini_numb_of_filters * 2, 3, kernel_initializer='he_uniform', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(ini_numb_of_filters * 2, 3, kernel_initializer='he_uniform', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(ini_numb_of_filters * 4, 3, kernel_initializer='he_uniform', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation("relu")(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(ini_numb_of_filters * 4, 3, kernel_initializer='he_uniform', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation("relu")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(ini_numb_of_filters * 8, 3, kernel_initializer='he_uniform', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation("relu")(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(ini_numb_of_filters * 8, 3, kernel_initializer='he_uniform', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation("relu")(c4)
    p4 = MaxPooling2D(pool_size=2)(c4)

    c5 = Conv2D(ini_numb_of_filters * 16, 3, kernel_initializer='he_uniform', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation("relu")(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(ini_numb_of_filters * 16, 3, kernel_initializer='he_uniform', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation("relu")(c5)

    """ Expansive path """
    u6 = Conv2DTranspose(ini_numb_of_filters * 8, 2, strides=2, padding='same')(c5)
    # u6 = UpSampling2D((2, 2), data_format="channels_last")(c5)
    u6 = Concatenate([u6, c4])
    c6 = Conv2D(ini_numb_of_filters * 8, 3, kernel_initializer='he_uniform', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation("relu")(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(ini_numb_of_filters * 8, 3, kernel_initializer='he_uniform', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation("relu")(c6)

    u7 = Conv2DTranspose(ini_numb_of_filters * 4, 2, strides=2, padding='same')(c6)
    # u7 = UpSampling2D((2, 2), data_format="channels_last")(c6)
    u7 = Concatenate([u7, c3])
    c7 = Conv2D(ini_numb_of_filters * 4, 3, kernel_initializer='he_uniform', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation("relu")(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(ini_numb_of_filters * 4, 3, kernel_initializer='he_uniform', padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation("relu")(c7)

    u8 = Conv2DTranspose(ini_numb_of_filters * 2, 2, strides=2, padding='same')(c7)
    # u8 = UpSampling2D((2, 2), data_format="channels_last")(c7)
    u8 = Concatenate([u8, c2])
    c8 = Conv2D(ini_numb_of_filters * 2, 3, kernel_initializer='he_uniform', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation("relu")(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(ini_numb_of_filters * 2, 3, kernel_initializer='he_uniform', padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation("relu")(c8)

    u9 = Conv2DTranspose(ini_numb_of_filters, 2, strides=2, padding='same')(c8)
    # u9 = UpSampling2D((2, 2), data_format="channels_last")(c8)
    u9 = Concatenate([u9, c1])
    c9 = Conv2D(ini_numb_of_filters, 3, kernel_initializer='he_uniform', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation("relu")(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(ini_numb_of_filters, 3, kernel_initializer='he_uniform', padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation("relu")(c9)

    outputs = Conv2D(1, 1, activation='linear')(c9)

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
img_channels = x_train.shape[3]  # 12

input_shape = (img_height, img_width, img_channels)

model = build_2DUNet_model_v1(input_shape)
# model = build_2DUNet_model_v2(input_shape)
print(model.summary())
print(model.input_shape)
print(model.output_shape)

##############################################################################
# 4. TRAINING THE MODEL ######################################################
##############################################################################

import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import pandas as pd
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt


def plot_learning_curve(filepath):
    df = pd.read_csv(filepath)
    df_x, df_yt, df_yv = df.values[:, 0], df.values[:, 2], df.values[:, 5]
    plt.figure(figsize=(5, 4))
    plt.plot(df_x, df_yt)
    plt.plot(df_x, df_yv)
    # plt.title('average training loss and validation loss')
    plt.ylabel('mean-squared error', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['training loss', 'validation loss'], fontsize=14, loc='upper right')
    plt.show()
    return


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
################################# THE END #####################################
###############################################################################

# Deep Learning for MR Image Synthesis
# @author: Alexander F.I. Osman, April 2021

"""
PART II: TESTING THE MODEL PERFORMANCE

This code demonstrates a 2D U-Net architecture for cross-sequence MR image
translations across T1, T1c, T2, & T2-FLAIR contrasts.
It takes a source image and converting it a target image.
Architectures: 2D U-Net

Dataset: BRATS-2018 challenge dataset.

The testing process goes through the following steps:
1. Load and preprocess the data (crop, resize , & normalize)
2. Make predictions on a test dataset
3. Evaluate the model performance (SSIM, MAE, etc.)
"""

##############################################################################
# 1. LOAD AND PROCESS THE DATA  ##############################################
##############################################################################

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


def image_proc(filepath):
    """ Data loader (*.nii)
    :param filepath: file path
    :return: 2D array images
    """
    img_data0 = np.zeros((96, 96, 1), dtype='float32')
    img_data = []
    for item in tqdm(sorted(filepath), desc='Processing'):
        # loading images
        img = nib.load(item).get_fdata()
        # Crop to get the brain region (along z-axis and x & y axes)
        ind = np.where(img > 0)
        ind_min, ind_max = min(ind[2]), max(ind[2])
        ind_mid = round((ind_min + ind_max) / 2)
        img = img[8:232,8:232,ind_mid-32:ind_mid+32]   # to have 224 x 224 x 64 dim.
        # resize
        img = zoom(img, (0.428, 0.428, 1))   # to have 96 x 96 x 64 dim.
        # Normalize using zero mean and unit variance method & scale to 0-1 range
        img = ((img - img.mean()) / img.std())
        img = ((img - img.min()) / (img.max() - img.min()))  # Scale to 0-1 range
        # Convert 3D images to 2D image slices
        img_data0 = np.concatenate((img_data0, img), axis=2)
    img_data0 = np.moveaxis(img_data0, [2], [0])
    return np.array(img_data0[1::,:,:]).astype('float32')


# Process the images (crop, resize, & normalize)
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


print("Used memory to store x_test: ", x_test.nbytes/(1024*1024), "MB")
print("Used memory to store y_test: ", y_test.nbytes/(1024*1024), "MB")

# Plot
slice_numb = 70
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(x_test[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight')
plt.subplot(222)
plt.imshow(y_test[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('target image'), plt.axis('tight')
slice_numb = 100
plt.subplot(223)
plt.imshow(x_test[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight')
plt.subplot(224)
plt.imshow(y_test[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('target image'), plt.axis('tight')
plt.show()

###############################################################################
# 2. MAKE PREDICTIONS #########################################################
###############################################################################

from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt


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
# 3. EVALUATE THE MODEL PERFORMANCE  ##########################################
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
hdd = np.zeros(len(test_real))
nmi = np.zeros(len(test_real))
nrmse = np.zeros(len(test_real))

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
    # HDD
    hdd[item] = hausdorff_distance(test_real[item], test_pred[item])
    # NMI
    nmi[item] = normalized_mutual_information(test_real[item], test_pred[item])
    # NRMSE
    nrmse[item] = normalized_root_mse(test_real[item], test_pred[item])

mean_psnr, std_psnr = np.mean(psnr), np.std(psnr)
print("PSNR (dB) =", np.round(mean_psnr, 3), "±", np.round(std_psnr, 3))
mean_ssim, std_ssim = np.mean(ssim), np.std(ssim)
print("SSIM =", np.round(mean_ssim, 3), "±", np.round(std_ssim, 3))
mean_mse, std_mse = np.mean(mse), np.std(mse)
print("MSE =", np.round(mean_mse, 3), "±", np.round(std_mse, 3))
mean_mae, std_mae = np.mean(mae), np.std(mae)
print("MAE =", np.round(mean_mae, 3), "±", np.round(std_mae, 3))
mean_hdd, std_hdd = np.mean(hdd), np.std(hdd)
print("HDD =", np.round(mean_hdd, 3), "±", np.round(std_hdd, 3))
mean_nmi, std_nmi = np.mean(nmi), np.std(nmi)
print("NMI =", np.round(mean_nmi, 3), "±", np.round(std_nmi, 3))
mean_nrmse, std_nrmse = np.mean(nrmse), np.std(nrmse)
print("NRMSE =", np.round(mean_nrmse, 3), "±", np.round(std_nrmse, 3))

# Plot
slice_numb = 50
fig = plt.figure()
grid = plt.GridSpec(2, 5, wspace = .15, hspace = .15)
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

slice_numb = 100
exec (f"plt.subplot(grid{[5]})")
plt.imshow(x_test[slice_numb,:,:,0].T, cmap='gray')
plt.colorbar(), plt.title('source image'), plt.axis('tight'), plt.axis('off')
exec (f"plt.subplot(grid{[6]})")
plt.imshow(test_real[slice_numb,:,:].T, cmap='gray')
plt.colorbar(), plt.title('target image (real)'), plt.axis('tight'), plt.axis('off')
exec (f"plt.subplot(grid{[7]})")
plt.imshow(test_pred[slice_numb,:,:].T, cmap='gray')
plt.colorbar(), plt.title('target image (pred)'), plt.axis('tight'), plt.axis('off')
exec (f"plt.subplot(grid{[8]})")
residual = test_real[slice_numb,:,:] - test_pred[slice_numb,:,:]
plt.imshow(residual.T, cmap='gray')
plt.colorbar(), plt.title('difference map'), plt.axis('tight'), plt.axis('off')
exec (f"plt.subplot(grid{[9]})")
ssim = structural_similarity(test_real[slice_numb,:,:], test_pred[slice_numb,:,:], full=True)
print("SSIM val =", np.round(ssim[0], 3))
plt.imshow(ssim[1].T, cmap='gray')
plt.colorbar(), plt.title('SSIM'), plt.axis('tight'), plt.axis('off'), plt.clim(0, 1)
plt.show()

###############################################################################
############################### THE END #######################################
###############################################################################
