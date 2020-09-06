## Import libraries
import os
import pandas as pd
import numpy as np
import cv2
from imblearn.over_sampling import SMOTE
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestRegressor
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

#%% Function to calculate evaluation metics
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
#%% Superpixel-Based Segmentation

Height = 512
Width = 512

Train_Img_Dir='Train_Data/Images/'
Train_VT_Dir='Train_Data/VTs/'
Valid_Img_Dir='Valid_Data/Images/'
Valid_VT_Dir='Valid_Data/VTs/'

Train_mask=[]

Trai_Im=os.listdir(Train_Img_Dir)
Trai_VT=os.listdir(Train_VT_Dir)


Nbr_SLIC=500
Compa=0.05


Data_All=np.zeros((1,7))
Data_Mass=np.zeros((1,7))
Data_NoMass=np.zeros((1,7))

Ind_Img=len(Trai_Im)

for i in range(0,Ind_Img):
    print('Processing of '+Trai_Im[i]+' is done.')
    
    Image_Train = cv2.imread(Train_Img_Dir+Trai_Im[i])[:,:,1]
    Image_Train = cv2.resize(Image_Train, (Height, Width))
    
    # Preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    Image_Train = clahe.apply(Image_Train)
    
    ret,im_th = cv2.threshold(Image_Train,50,255,cv2.THRESH_BINARY)
       
    Image_Tst_SP = (ndimage.binary_fill_holes(im_th.astype(np.uint8)).astype(np.uint8))*255
    
    Image_VT = cv2.imread(Train_VT_Dir+Trai_VT[i])[:,:,1]
    Image_VT = cv2.resize(Image_VT, (Height, Width))
    
    # Superpixel Calculation  
    segments = slic(img_as_float(Image_Train), n_segments=Nbr_SLIC, sigma=1, compactness=Compa,multichannel=False)
      
    Data=np.zeros((np.amax(segments),3),dtype=float)
    
    # Features extraction on each superpixel
    for v in np.unique(segments):
        Image_Train = cv2.imread(Train_Img_Dir+Trai_Im[i])[:,:,1]
        Image_Train = cv2.resize(Image_Train, (Height, Width))
    
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
        Image_Train = clahe.apply(Image_Train)/255
        
        mask = segments == v
        Index=np.where(mask == True)
        Indexx=np.array(Index)
        si=np.shape(Indexx)
        
        Mean=0
        SD=0
        Energy=0
        Entropy=0
        Smoothness=0
        Skewness=0
        Kurtosis=0
        
        Vec_Pixel=[]
        Vec_Tst_SP=[]
        Vec_VT=[]

        for j in range(0,si[1]):
            x=Indexx[0,j]
            y=Indexx[1,j]
            Vec_Pixel.append(Image_Train[x,y])
            Vec_Tst_SP.append(Image_Tst_SP[x,y])
            Vec_VT.append(Image_VT[x,y])
        
        Vec_Pixel=np.array(Vec_Pixel)/np.max(Vec_Pixel)
        Vec_VT=np.array(Vec_VT)
        l=len(Vec_Pixel)
        
        if np.average(Vec_Tst_SP)>200:
            
            Mean=np.sum(Vec_Pixel)/l

            for j in range(0,si[1]):
                SD=SD+(np.sum(np.power((Vec_Pixel[j]-Mean),2)))                
                Energy=Energy+(np.sum(np.power(Vec_Pixel[j],2)))
                Entropy=Entropy+np.sum(Vec_Pixel[j]*np.log2((Vec_Pixel[j]+1)))
            
            SD=np.power(SD/l,1/2)
            Smoothness=1-(1/(1+np.power(SD,2)))
            
            for j in range(0,si[1]):
                Skewness=Skewness+(np.sum(np.power((Vec_Pixel[j]-Mean)/SD,3)))
                Kurtosis=Kurtosis+np.sum(Vec_Pixel[j]-Mean/SD)
                
            Skewness=Skewness/l
            Kurtosis=Kurtosis-3
        
            Data_SP=[Mean,SD,Energy,Entropy,Smoothness,Skewness,Kurtosis]
            Data_SP = np.expand_dims(Data_SP, axis = 0)
            
            Data_All=np.concatenate((Data_All,Data_SP),axis=0)
        
            if np.average(Vec_VT) > 150:
                Data_Mass=np.concatenate((Data_Mass,Data_SP),axis=0)
                Train_mask.append(1)
            else :
                Data_NoMass=np.concatenate((Data_NoMass,Data_SP),axis=0) 
                Train_mask.append(2)

#%% Training step
Data_All=np.delete(Data_All, 0, 0)    
Data_Mass=np.delete(Data_Mass, 0, 0)
Data_NoMass=np.delete(Data_NoMass, 0, 0)

Data_All=np.array(Data_All)
Data_Mass=np.array(Data_Mass)
Data_NoMass=np.array(Data_NoMass)
Train_mask=np.array(Train_mask)

oversample = SMOTE()
X, y = oversample.fit_resample(Data_All, Train_mask)

Train_mask = to_categorical(Train_mask)
y = to_categorical(y)

rf = RandomForestRegressor(n_estimators = 100, random_state = 30)
rf.fit(X, y)

#%% Test and Segmentation step
Test_Img_Dir='Test_Data/Images/'
Test_VT_Dir='Test_Data/VTs/'

Test_Im=os.listdir(Test_Img_Dir)
Ind_Img=len(Test_Im)


Eval=[]

for i in range(0,Ind_Img):
    print(Test_Im[i])
    
    Image_Test = cv2.imread(Test_Img_Dir+Test_Im[i])[:,:,1]
    Image_Test = cv2.resize(Image_Test, (Height, Width))
    
    VT = cv2.imread(Test_VT_Dir+Test_Im[i])[:,:,2]
    VT = cv2.resize(VT, (Height, Width))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    Image_Test = clahe.apply(Image_Test) 
    
    ret,im_th = cv2.threshold(Image_Test,50,255,cv2.THRESH_BINARY)
       
    Image_Tst_SP = (ndimage.binary_fill_holes(im_th.astype(np.uint8)).astype(np.uint8))*255
        
    segmentation=Image_Test
    segments = slic(img_as_float(Image_Test), n_segments=Nbr_SLIC, sigma=1, compactness=Compa,multichannel=False)
    

    for v in np.unique(segments) :
    
        segmentss = segments
        Image_Test = cv2.imread(Test_Img_Dir+Test_Im[i])[:,:,1]
        Image_Test = cv2.resize(Image_Test, (Height, Width))
    
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
        Image_Test = clahe.apply(Image_Test) /255
    
        mask = segments == v
        Index=np.where(mask == True)
        Indexx=np.array(Index)
        si=np.shape(Indexx)
        Mean=0
        SD=0
        Energy=0
        Entropy=0
        Smoothness=0
        Skewness=0
        Kurtosis=0
        
        Vec_Pixel=[]
        Vec_Tst_SP=[]
        Vec_VT=[]

        for j in range(0,si[1]):
            x=Indexx[0,j]
            y=Indexx[1,j]
            Vec_Pixel.append(Image_Test[x,y])
            Vec_Tst_SP.append(Image_Tst_SP[x,y])
            Vec_VT.append(VT[x,y])
        
        Vec_Pixel=np.array(Vec_Pixel)/np.max(Vec_Pixel)
        Vec_VT=np.array(Vec_VT)
        l=len(Vec_Pixel)
        
        if np.average(Vec_Tst_SP)>200:
            
            Mean=np.sum(Vec_Pixel)/l

            for j in range(0,si[1]):
                SD=SD+(np.sum(np.power((Vec_Pixel[j]-Mean),2)))                
                Energy=Energy+(np.sum(np.power(Vec_Pixel[j],2)))
                Entropy=Entropy+np.sum(Vec_Pixel[j]*np.log2((Vec_Pixel[j]+1)))
            
            SD=np.power(SD/l,1/2)
            Smoothness=1-(1/(1+np.power(SD,2)))
            
            for j in range(0,si[1]):
                Skewness=Skewness+(np.sum(np.power((Vec_Pixel[j]-Mean)/SD,3)))
                Kurtosis=Kurtosis+np.sum(Vec_Pixel[j]-Mean/SD)
                
            Skewness=Skewness/l
            Kurtosis=Kurtosis-3
        
            Data_SP=[Mean,SD,Energy,Entropy,Smoothness,Skewness,Kurtosis]
            Data_SP = np.expand_dims(Data_SP, axis = 0)
        
            pred = rf.predict(Data_SP)
                 
            if pred[0,1]>0.7 :
                segmentation[mask==True]=1      

            else :
                segmentation[mask==True]=2
            
            
            
    Seg=(segmentation==1).astype(int)
    VT=(VT==255).astype(int)
    Eval.append(SEG_EVAL(Seg,VT))
    E=SEG_EVAL(Seg,VT)
    

    VIS_Seg,VIS_GT=VIS_EVAL(Image_Test*255, Seg, VT, option = 'region')
    cv2.imwrite('SP_Seg_'+Test_Im[i], VIS_Seg)
    cv2.imwrite('SP_GT_'+Test_Im[i], VIS_GT)
    
np.savetxt("SP_Results.csv", Eval, delimiter=",",fmt='%s')
            

