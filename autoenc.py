import os
import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import nibabel as nib #reading MR images
#from sklearn.cross_validation import train_test_split
import math
from sklearn.model_selection import train_test_split
import glob
import tensorflow as tf
from skimage.transform import resize
#import data as data


#from matplotlib import pyplot as plt

img_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/kits19/data/Image/case_*/imaging.nii.gz")
mask_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/kits19/data/Image/case_*/segmentation.nii.gz")

images=[]
for i in range(1):
	a=[]
	a=nib.load(img_path[i])
	a=a.get_data()
	print("image (%d) loaded"%(i))
	a=resize(a,(a.shape[0],512,512))
	a=a[30:80,:,:]
	for j in range(a.shape[0]):
		images.append((a[j,:,:]))

images=np.asarray(images)

masks=[]
for i in range(1):
	b=[]
	b=nib.load(mask_path[i])
	b=b.get_data()
	print("mask (%d) loaded"%(i))
	b=resize(b,(b.shape[0],512,512))
	b=b[30:80,:,:]
	for j in range(b.shape[0]):
		masks.append((b[j,:,:]))

masks=np.asarray(masks)


images = images.reshape(-1, 512,512,1)
masks=masks.reshape(-1,512,512,1)

train_X,valid_X,train_ground,valid_ground = train_test_split(images,
                                                             masks,
                                                             test_size=0.2,
                                                             random_state=13)


print("Dataset (images) shape: {shape}".format(shape=images.shape))
print("Dataset (masks) shape: {shape}".format(shape=masks.shape))




batch_size = 2
epochs = 2
inChannel = 1
x, y = 512, 512
input_img = Input(shape = (x, y, inChannel))


def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)


    #decoder
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
autoencoder.summary()

autoencoder.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])


autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2)
pred = autoencoder.predict(valid_X)
