import os
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
    
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape, Conv2DTranspose
from keras.layers import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,1,0,2"
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
from keras.layers.core import Dropout, Activation

from skimage.transform import resize
#import data as data


#from matplotlib import pyplot as plt

img_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/scratch/data/case_ (*)/imaging.nii")
mask_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/scratch/data/case_ (*)/segmentation.nii")
images=[]
for i in range(1,100):
	a=[]
	a=nib.load(img_path[i])
	a=a.get_data()
	print("image (%d) loaded"%(i))
	a=resize(a,(a.shape[0],512,512))
	a=a[:,:,:]
	for j in range(a.shape[0]):
		images.append((a[j,:,:]))

images=np.asarray(images)

masks=[]
for i in range(1,100):
	b=[]
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

i_m = np.max(images)
i_mi = np.min(images)
m_m = np.max(masks)
m_mi = np.min(masks)
images = (images - i_mi) / (i_m - i_mi)
masks= (masks - m_mi) / (m_m - m_mi)

train_X,valid_X,train_ground,valid_ground = train_test_split(images,
                                                             masks,
                                                             test_size=0.2,
                                                             random_state=13)


print("Dataset (images) shape: {shape}".format(shape=images.shape))
print("Dataset (masks) shape: {shape}".format(shape=masks.shape))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def depth_softmax(matrix):
    sigmoid = lambda x: 1 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    return softmax_matrix


# In[ ]:


def build_unet(shape):
	input_layer = Input(shape = shape)

	conv = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(input_layer)
	conv = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(conv)
	pool = MaxPooling2D(pool_size = (2, 2))(conv)


	conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv)
	conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv1)
	pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(pool1)
	conv2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv2)
	pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(pool2)
	conv3 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv3)
	pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(pool3)
	conv4 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv4)
	pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(pool4)
	conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides = (2, 2), padding = 'same')(conv5), conv4], axis = 3)
	conv6 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(up6)
	conv6 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(conv6), conv3], axis = 3)
	conv7 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(up7)
	conv7 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(conv7), conv2], axis = 3)
	conv8 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(up8)
	conv8 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(conv8), conv1], axis = 3)
	conv9 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(up9)
	conv9 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv9)

	conv10 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(conv9)
	conv10 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(conv10)
	pool10 = MaxPooling2D(pool_size = (2, 2))(conv10)

	conv11 = Conv2D(1, (1, 1), activation = depth_softmax)(conv10)

	return Model(input_layer, conv11)


# In[ ]:


model = build_unet((512, 512, 1))
model.summary()
model.compile(optimizer =Adam(lr=1e-5), loss = 'binary_crossentropy', metrics = [dice_coef, 'binary_accuracy'])


# In[ ]:


weight_saver = ModelCheckpoint(
    'modeljl10.h5',
    monitor = 'val_dice_coeff',
    save_best_only = True,
    mode = 'min',
    save_weights_only = True
)

reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor = 'val_loss', factor = 0.5,
    patience = 3, verbose = 1,
    mode = 'min',
    cooldown = 2, min_lr = 1e-6
)

early = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 15
)


# In[ ]:




model.fit(train_X, train_ground, validation_split=.2, batch_size=10, epochs=10, verbose = 1,
    callbacks = [
        weight_saver,
        early,
        reduce_lr_on_plateau
    ])

model.save('unetrahul.h5')



'''
batch_size = 2
epochs = 40
inChannel = 1
x, y = 512, 512
input_img = Input(shape = (x, y, inChannel))

def get_fcn_vgg16_8s(inputs, n_classes):
    
    x = BatchNormalization()(inputs)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    block_3 = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    block_4 = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding="same")(x)

    block_5 = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    
    sum_1 = add([block_4, block_5])
    sum_1 = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(sum_1)
    
    sum_2 = add([block_3, sum_1])
    
    x = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), activation='linear', padding='same')(sum_2)
    
    return x
 

def get_model():
    
    inputs = Input((512, 512, 1))
    
    base = get_fcn_vgg16_8s(inputs, 3)
    #base = models.get_fcn_vgg16_16s(inputs, NUMBER_OF_CLASSES)
    #base = models.get_fcn_vgg16_8s(inputs, NUMBER_OF_CLASSES)
    #base = models.get_unet(inputs, NUMBER_OF_CLASSES)
    #base = models.get_segnet_vgg16(inputs, NUMBER_OF_CLASSES)
    
    # softmax
 
    model = Model(inputs=inputs, outputs=(512,512,1))
    model.compile(optimizer=Adadelta(), loss='categorical_crossentropy')
    
    #print(model.summary())
    #sys.exit()
    
    return model


def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall


autoencoder = get_model()
autoencoder.compile(loss = confusion,
              optimizer='Adam',metrics=['accuracy'])
autoencoder.summary()

filepath="weights_autoenc1withloss.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_iou_loss_core', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2,callbacks=callbacks_list)
scores =autoencoder.evaluate(valid_X, valid_ground, verbose=1)
print("Evaluation %s: %.2f%%" % (autoencoder.metrics_names[1], scores[1]*100))'''
'''
def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
	l1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
	l2 = BatchNormalization()(l1)
	l3 = Conv2D(64, (3, 3), activation='relu', padding='same')(l2) #28 x 28 x 32
	l4 = BatchNormalization()(l3)
	conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(l4) #28 x 28 x 32
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
	conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
	conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)


	#decoder
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
	conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)
	r1= Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
	r2=BatchNormalization()(r1)
	r3=Conv2D(64, (3, 3), activation='relu', padding='same')(r2)
	r4=BatchNormalization()(r3)
	up2 = UpSampling2D((2,2))(r4) # 28 x 28 x 64
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
	return decoded

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  # some implementations don't square y_pred
  denominator = tf.reduce_sum(y_true + tf.square(y_pred))

  return numerator / (denominator + tf.keras.backend.epsilon())


def iou_loss_core(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou



autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss = dice_loss,
              optimizer='Adam',
              metrics=[iou_loss_core, 'acc'])
autoencoder.summary()


filepath="weights_autoenc1withloss.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_iou_loss_core', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2,callbacks=callbacks_list)
scores =autoencoder.evaluate(valid_X, valid_ground, verbose=1)
print("Evaluation %s: %.2f%%" % (autoencoder.metrics_names[1], scores[1]*100))
'''