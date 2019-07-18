import os
import cv2
from keras.layers import Input,concatenate,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
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
from keras.models import load_model

#import data as data
'''
import numpy as np
images=np.load("images.npy")
masks=np.load("masks.npy");
'''

img_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/kits19/data/Image/case_*/imaging.nii.gz")
mask_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/kits19/data/Image/case_*/segmentation.nii.gz")

images=[]
a=[]
for i in range(1,50):
	a=nib.load(img_path[i])
	a=a.get_data()
	print("image (%d) loaded"%(i))
	a=resize(a,(a.shape[0],512,512))
	a=a[:,:,:]
	for j in range(a.shape[0]):
		images.append((a[j,:,:]))

images=np.asarray(images)

masks=[]
b=[]
for i in range(1,50):
	b=nib.load(mask_path[i])
	b=b.get_data()
	print("mask (%d) loaded"%(i))
	b=resize(b,(b.shape[0],512,512))
	b=b[:,:,:]
	for j in range(b.shape[0]):
		masks.append((b[j,:,:]))

masks=np.asarray(masks)


images = images.reshape(-1, 512,512,1)
masks=masks.reshape(-1,512,512,1)


train_X,valid_X,train_ground,valid_ground = train_test_split(images,
                                                             masks,
                                                             test_size=0.2,
                                                             random_state=13)



batch_size = 2
epochs = 1
inChannel = 1
x, y = 512, 512

def unet(pretrained_weights = None,input_size = (512,512,1)):
	inputs = Input(input_size)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
	merge6 = concatenate([drop4,up6], axis = 3)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = 3)
	conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = 3)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = 3)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

	model = Model(input = inputs, output = conv10)


	

	#if(pretrained_weights):
	#	model.load_weights(pretrained_weights)

	return model


model=unet()
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
filepath="weightsbest1-50.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
train = model.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2,callbacks=callbacks_list)
scores = model.evaluate(valid_X, valid_ground, verbose=1)
print("Evaluation %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

'''
import h5py
f = h5py.File('weightsbest.hdf5', 'r+')
del f['optimizer_weights']
f.close()

model=load_model('weightsbest.hdf5')
scores = model.evaluate(valid_X, valid_ground, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
'''
	


'''
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

pred = autoencoder.predict(valid_X)
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(5):
    p=plt.subplot(1, 5, i+1)
    p=plt.imshow(valid_ground[i, ..., 0], cmap='gray')
p.savefig("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/scratch")
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(5):
    q=plt.subplot(1, 5, i+1)
    q=plt.imshow(pred[i, ..., 0], cmap='gray')  
q.savefig("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/scratch")
'''
