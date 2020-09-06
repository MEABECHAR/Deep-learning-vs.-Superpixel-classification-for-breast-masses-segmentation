
from google.colab import drive
drive.mount('/content/drive')


## Imports
import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import cv2
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

dir_img="Data_Train/Data/"
dir_mask="Data_Train/Gr_Th/"

width=240
height=240
channels=3

tab_img=[]
tab_mask=[]

for n, i in tqdm(enumerate(os.listdir(dir_img)), total=len(os.listdir(dir_img))):
    tab_img.append(cv2.resize(cv2.imread(dir_img+i), (width, height))/255)

    img_mask=cv2.resize(cv2.imread(dir_mask+i), (width, height))[:,:,2]
    img_mask_result=np.zeros(shape=(height, width, 1), dtype=np.float32)
    img_mask_result[:,:,0][img_mask==255]=1.
    tab_mask.append(img_mask_result)

tab_img=np.array(tab_img)
tab_mask=np.array(tab_mask)

from keras.preprocessing import image

# Creating the training Image and Mask generator
image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')

seed = 42
image_datagen.fit(tab_img[:int(tab_img.shape[0]*0.9)], augment=True, seed=seed)
mask_datagen.fit(tab_mask[:int(tab_mask.shape[0]*0.9)], augment=True, seed=seed)

BATCH_SIZE = 32
x=image_datagen.flow(tab_img[:int(tab_img.shape[0]*0.9)],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
y=mask_datagen.flow(tab_mask[:int(tab_mask.shape[0]*0.9)],batch_size=BATCH_SIZE,shuffle=True, seed=seed)

# Creating the validation Image and Mask generator
image_datagen_val = image.ImageDataGenerator()
mask_datagen_val = image.ImageDataGenerator()

image_datagen_val.fit(tab_img[int(tab_img.shape[0]*0.9):], augment=True, seed=seed)
mask_datagen_val.fit(tab_mask[int(tab_mask.shape[0]*0.9):], augment=True, seed=seed)

x_val=image_datagen_val.flow(tab_img[int(tab_img.shape[0]*0.9):],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
y_val=mask_datagen_val.flow(tab_mask[int(tab_mask.shape[0]*0.9):],batch_size=BATCH_SIZE,shuffle=True, seed=seed)

#creating a training and validation generator that generate masks and images
train_generator = zip(x, y)
val_generator = zip(x_val, y_val)

Valid_dir_img="Data_Valid/Data/"
Valid_dir_mask="Data_Valid/Gr_Th/"

width=512
height=512
channels=3

Valid_tab_img=[]
Valid_tab_mask=[]

for n, i in tqdm(enumerate(os.listdir(Valid_dir_img)), total=len(os.listdir(Valid_dir_img))):
    Valid_tab_img.append(cv2.resize(cv2.imread(Valid_dir_img+i), (width, height))/255)

    img_mask=cv2.resize(cv2.imread(Valid_dir_mask+i), (width, height))[:,:,2]
    img_mask_result=np.zeros(shape=(height, width, 1), dtype=np.float32)
    img_mask_result[:,:,0][img_mask==255]=1.
    Valid_tab_mask.append(img_mask_result)

tab_img=np.array(tab_img)
tab_mask=np.array(tab_mask)

Valid_tab_img=np.array(Valid_tab_img)
Valid_tab_mask=np.array(Valid_tab_mask)

from keras.preprocessing import image

# Creating the training Image and Mask generator
image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')

seed = 42
image_datagen.fit(tab_img, augment=True, seed=seed)
mask_datagen.fit(tab_mask, augment=True, seed=seed)

BATCH_SIZE = 32
x=image_datagen.flow(tab_img,batch_size=BATCH_SIZE,shuffle=True, seed=seed)
y=mask_datagen.flow(tab_mask,batch_size=BATCH_SIZE,shuffle=True, seed=seed)

# Creating the validation Image and Mask generator
image_datagen_val = image.ImageDataGenerator()
mask_datagen_val = image.ImageDataGenerator()

image_datagen_val.fit(Valid_tab_img, augment=True, seed=seed)
mask_datagen_val.fit(Valid_tab_mask, augment=True, seed=seed)

x_val=image_datagen_val.flow(Valid_tab_img,batch_size=BATCH_SIZE,shuffle=True, seed=seed)
y_val=mask_datagen_val.flow(Valid_tab_mask,batch_size=BATCH_SIZE,shuffle=True, seed=seed)

#creating a training and validation generator that generate masks and images
train_generator = zip(x, y)
val_generator = zip(x_val, y_val)

pip install segmentation-models

from segmentation_models import Unet

model = Unet('resnet34', encoder_weights='imagenet', classes=1, input_shape=(512,512, 3), activation='sigmoid')

model.compile('Adam', loss="binary_crossentropy", metrics=["acc"])

from keras.models import load_model
model.load_weights('Unet_weights.h5')

results = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=500, steps_per_epoch=1000,epochs=10)
model.save_weights('Unet_weights.h5')



def SEG_EVAL(Seg,GT):
    # Seg : Segmented image, must be binary (1 = regions of interest 0 = background)
    # GT : Ground truth, must be binary (1 = regions of interest 0 = background)
    Seg.astype(np.bool)
    GT.astype(np.bool)
    
    #dice_coefficient
    intersection = np.logical_and(Seg, GT)
    dice_coefficient = 2. * intersection.sum() / (Seg.sum() + GT.sum())
    
    #IoU
    TP = np.logical_and(Seg, GT)
    IoU = TP.sum() / (GT.sum() + Seg.sum() - TP.sum())
    
    #recall
    recall = TP.sum() / GT.sum()
    
    #precision
    precision = TP.sum() / Seg.sum()
    
    EVAL = [dice_coefficient,IoU,recall,precision]
    
    return EVAL
    
def VIS_EVAL(Image, Seg, GT, option = 'contour'):
    # Image : Gray image
    # Seg : Segmented image, must be binary (1 = regions of interest 0 = background)
    # GT : Ground truth, must be binary (1 = regions of interest 0 = background)
    # option = 'contour' For contour plotting
    # option = 'region' For color the region
       
    Image = cv2.cvtColor(np.array(Image, np.uint8), cv2.COLOR_GRAY2RGB)
    GT = np.array(255*GT, dtype=np.uint8)
    Seg = np.array(255*Seg, dtype=np.uint8)
    GT = cv2.cvtColor(GT, cv2.COLOR_GRAY2RGB)
    Seg = cv2.cvtColor(Seg, cv2.COLOR_GRAY2RGB)
        
    if option == 'region':
        GT[:, :, 1:2] = 0*GT[:, :, 1:2]
        Seg[:, :, 0] = 0*Seg[:, :, 0]
        Seg[:, :, 2] = 0*Seg[:, :, 2]
        
        VIS_Seg = cv2.addWeighted(Image, 1, Seg, 0.5, 0)
        VIS_GT = cv2.addWeighted(Image, 1, GT, 0.5, 0)
          
    elif option == 'contour':
       
        contours_GT = cv2.Canny(GT, 250, 260) 
        contours_Seg = cv2.Canny(Seg, 250, 260) 
        
        contours_GT = cv2.cvtColor(contours_GT, cv2.COLOR_GRAY2RGB)
        contours_Seg = cv2.cvtColor(contours_Seg, cv2.COLOR_GRAY2RGB)
        
        contours_GT[:, :, 1:2] = 0*contours_GT[:, :, 1:2]
        contours_Seg[:, :, 0] = 0*contours_Seg[:, :, 0]
        contours_Seg[:, :, 2] = 0*contours_Seg[:, :, 2]
        
        VIS_Seg = cv2.addWeighted(Image, 1, contours_Seg, 0.5, 0)
        VIS_GT = cv2.addWeighted(Image, 1, contours_GT, 0.5, 0)
        
    return VIS_Seg,VIS_GT

dir_img = "Data_Test/Data/"
dir_mask = "Data_Test/Gr_Th/"
Eval = []

Dice = []
TP = []
FP = []

width=240
height=240
channels=3

for i in os.listdir(dir_img):
    Image_Test = cv2.resize(cv2.imread(dir_img+i), (width, height))/255
    Image = cv2.resize(cv2.imread(dir_img+i), (width, height))[:,:,2]
    VT=cv2.resize(cv2.imread(dir_mask+i), (width, height))[:,:,2]
    VT=(VT==255).astype(int)

    Image_Test = np.expand_dims(Image_Test, axis=0)
    Seg = model.predict(Image_Test)
    Seg = ((Seg[0,:,:,0]*255)>5).astype(int)

    Eval.append(SEG_EVAL(Seg,VT))

    VIS_Seg,VIS_GT=VIS_EVAL(Image, Seg, VT, option = 'region')
    cv2.imwrite('CNN_Results/Seg_'+i, VIS_Seg)
    cv2.imwrite('CNN_Results/GT_'+i, VIS_GT)
    #cv2.imwrite('FPN_Results/Mask_Seg_'+i, Seg)
    
np.savetxt("CNN_Results.csv", Eval, delimiter=",")

