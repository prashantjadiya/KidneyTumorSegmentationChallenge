import glob
import nibabel as nib
import numpy as np
from skimage.transform import resize
from keras.models import load_model
import tensorflow as tf
from keras import backend as K

img_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/scratch/data/case_ (*)/imaging.nii")
mask_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/scratch/data/case_ (*)/segmentation.nii")

images=[]
for i in range(2,3):
	a=[]
	a=nib.load(img_path[i])
	a=a.get_data()
	print("image (%d) loaded"%(i))
	a=resize(a,(a.shape[0],512,512))
	a=a[40:90,:,:]
	for j in range(a.shape[0]):
		images.append((a[j,:,:]))

valid_X=np.asarray(images)
'''
masks=[]
for i in range(200,210):
	b=[]
	b=nib.load(mask_path[i])
	b=b.get_data()
	print("mask (%d) loaded"%(i))
	b=resize(b,(b.shape[0],512,512))
	b=b[30:80,:,:]
	for j in range(b.shape[0]):
		masks.append((b[j,:,:]))

masks=np.asarray(masks)
'''

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



def tversky(y_true, y_pred,smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)




valid_X = valid_X.reshape(-1,512,512,1)
#valid_ground=masks.reshape(-1,512,512,1)

#1-50

model1=load_model('weightsbestforall-loss.hdf5') 
#scores = model1.evaluate(valid_X, valid_ground, verbose=1)
pred=model1.predict(valid_X)
np.save("pred1.npy",pred)
print(pred)

#50-100
'''

model2=load_model('weightsbest50-100.hdf5') 
scores = model2.evaluate(valid_X, valid_ground, verbose=1)


#100-150


model3=load_model('weightsbest100-150.hdf5') 
scores = model3.evaluate(valid_X, valid_ground, verbose=1)



#150-200


model4=load_model('weightsbest150-200.hdf5') 
scores = model4.evaluate(valid_X, valid_ground, verbose=1)




print("model 1:::::::  %s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))
print("model 2:::::::  %s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
print("model 3:::::::  %s: %.2f%%" % (model3.metrics_names[1], scores[1]*100))
print("model 4:::::::  %s: %.2f%%" % (model4.metrics_names[1], scores[1]*100))

'''
