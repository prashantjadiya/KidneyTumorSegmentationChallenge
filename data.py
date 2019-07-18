import numpy as np
import nibabel as nib #reading MR images
import glob
import math
from skimage.transform import resize
import os
from PIL import Image
img_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/kits19/data/Image/case_*/imaging.nii.gz")
mask_path = glob.glob("/storage/research/Intern19_v2/KidneyTumorSegmentationChallenge/kits19/data/Image/case_*/segmentation.nii.gz")


for i in range(1,100,1):
	a=[]
	a=nib.load(img_path[i])
	a=a.get_data()
	print("image (%d) loaded"%(i))
	a=resize(a,(a.shape[0],600,600))
	a=a[40:90,:,:]
	for j in range(a.shape[0]):
		img = Image.fromarray(a[j,:,:])
		img =img.convert("L")
		img.save("trainPNG/Image/"+str(i)+"_"+str(j)+".png")
		#np.save("train/Image/"+str(i)+"_"+str(j)+".npy",a[j])


for i in range(1,100,1):
	b=[]
	b=nib.load(mask_path[i])
	b=b.get_data()
	print("mask (%d) loaded"%(i))
	b=resize(b,(b.shape[0],600,600))
	b=b[40:90,:,:]
	for j in range(b.shape[0]):
		img = Image.fromarray(a[j,:,:])
		img =img.convert("L")

		img.save("trainPNG/Mask/"+str(i)+"_"+str(j)+".png")

		#np.save("train/Mask/"+str(i)+"_"+str(j)+".npy",b[j])


for i in range(100,200,1):
	a=[]
	a=nib.load(img_path[i])
	a=a.get_data()
	print("image (%d) loaded"%(i))
	a=resize(a,(a.shape[0],600,600))
	a=a[40:90,:,:]
	for j in range(a.shape[0]):
		img = Image.fromarray(a[j,:,:])
		img =img.convert("L")

		img.save("trainPNG/Image/"+str(i)+"_"+str(j)+".png")
		#np.save("train/Image/"+str(i)+"_"+str(j)+".npy",a[j])


for i in range(100,200,1):
	b=[]
	b=nib.load(mask_path[i])
	b=b.get_data()
	print("mask (%d) loaded"%(i))
	b=resize(b,(b.shape[0],600,600))
	b=b[40:90,:,:]
	for j in range(b.shape[0]):
		img = Image.fromarray(a[j,:,:])
		img =img.convert("L")

		img.save("trainPNG/Mask/"+str(i)+"_"+str(j)+".png")
		#np.save("train/Mask/"+str(i)+"_"+str(j)+".npy",b[j])


for i in range(200,210,1):
	a=[]
	a=nib.load(img_path[i])
	a=a.get_data()
	print("image (%d) loaded"%(i))
	a=resize(a,(a.shape[0],600,600))
	a=a[40:90,:,:]
	for j in range(a.shape[0]):
		img = Image.fromarray(a[j,:,:])
		img =img.convert("L")

		img.save("testPNG/Image/"+str(i)+"_"+str(j)+".png")
		#np.save("test/Image/"+str(i)+"_"+str(j)+".npy",a[j])


for i in range(200,210,1):
	b=[]
	b=nib.load(mask_path[i])
	b=b.get_data()
	print("mask (%d) loaded"%(i))
	b=resize(b,(b.shape[0],600,600))
	b=b[40:90,:,:]
	for j in range(b.shape[0]):
		img = Image.fromarray(a[j,:,:])
		img =img.convert("L")

		img.save("testPNG/Mask/"+str(i)+"_"+str(j)+".png")
		#np.save("test/Mask/"+str(i)+"_"+str(j)+".npy",b[j])
