import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from keras.layers import Input,concatenate,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers import Input,concatenate,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import nibabel as nib #reading MR images
#from sklearn.cross_validation import train_test_split
import math
from sklearn.model_selection import train_test_split
import glob
import tensorflow as tf
from skimage.transform import resize
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.utils import Sequence
import pandas as pd
import nibabel as nib


class My_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return  int(np.ceil(len(self.image_filenames) / int(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.asarray([
            resize(np.load(file_name)/255., (512,512,1))
               for file_name in batch_x]), np.asarray([
            resize(np.load(file_name)/255., (512,512,1))
               for file_name in batch_y])


def build_unet(shape):
    input_layer = Input(shape = shape)
    
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(input_layer)
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv4)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same')(conv5)
    
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides = (2, 2), padding = 'same')(conv5), conv4], axis = 3)
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(up6)
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(conv6), conv3], axis = 3)
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(up7)
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(conv7), conv2], axis = 3)
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(up8)
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(conv8), conv1], axis = 3)
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(up9)
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)
    
    return Model(input_layer, conv10)





sz=(512,512,1)


model=build_unet(sz)
model.compile(optimizer = Adam(lr = 1e-4), loss = ['mean_squared_error'], metrics = ['accuracy'])
model.summary()
train = pd.read_csv('trainnpy.csv')
valid = pd.read_csv('testnpy.csv')
GT_training = train.iloc[:, 1].values
training_filenames = train.iloc[:, 0].values

GT_validation = valid.iloc[:, 1].values
validation_filenames = valid.iloc[:, 0].values

my_training_batch_generator = My_Generator(training_filenames, GT_training, 5)
my_validation_batch_generator = My_Generator(validation_filenames, GT_validation, 5)
filepath="weightsbestforall-loss.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit_generator(generator=my_training_batch_generator,
                                          steps_per_epoch=10,
                                          epochs=50,
                                          verbose=1,
                                          validation_data=my_validation_batch_generator,
                                          validation_steps=100,
                                          use_multiprocessing=True,
                                          workers=10,
                                          max_queue_size=10,callbacks=callbacks_list)

