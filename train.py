#!/usr/bin/env python3

import numpy as np

import tensorflow as tf
import keras
import os

np.random.seed(1337)

from sklearn.utils import shuffle

from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *

from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D,Input, Dense
from keras.layers import Conv1D,AveragePooling1D,MaxPooling2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
from keras.models import Model
from keras.layers import Input,merge,LSTM,Concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras import optimizers
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from scipy import ndimage
import random
import cv2
#%%
# Data input for training
#%%

img_row=300

# Directory containing training data and label
file_path   = "./Data/"

# training data and labels
data=np.load(file_path+'/data_train.npy') 
labels=np.load(file_path+'/labels_train_updated.npy')

data=data.T

data,labels=shuffle(data,labels,random_state=3915)
train_data=data[0:5376]
test_data=data[6048:]
val_data=data[5376:6048]
train_labels=labels[0:5376]
test_labels=labels[6048:]
val_labels=labels[5376:6048]





train_data=np.reshape(train_data,(train_data.shape[0],img_row,img_row))
test_data=np.reshape(test_data,(test_data.shape[0],img_row,img_row))
val_data=np.reshape(val_data,(val_data.shape[0],img_row,img_row))


for i in range(train_data.shape[0]):
    train_data[i,0:150,0:150]=cv2.resize(train_data[i],(150,150))
    
train_data=train_data[:,0:150,0:150]
for i in range(test_data.shape[0]):
    test_data[i,0:150,0:150]=cv2.resize(test_data[i],(150,150))    
    
test_data=test_data[:,0:150,0:150]

for i in range(val_data.shape[0]):
    val_data[i,0:150,0:150]=cv2.resize(val_data[i],(150,150))    
    
val_data=val_data[:,0:150,0:150]

#%%
save_path = './split_data/'

# Check whether the specified path exists or not
isExist = os.path.exists(save_path)
if not isExist:  
  # Create a new directory because it does not exist 
  os.makedirs(save_path)

np.save(save_path+'train_images.npy',train_data)
np.save(save_path+'train_labels.npy',train_labels)

np.save(save_path+'test_images.npy',test_data)
np.save(save_path+'test_labels.npy',test_labels)

np.save(save_path+'val_images.npy',val_data)
np.save(save_path+'val_labels.npy',val_labels)

#%% Augmentation Functions

#1
def Dilation(image):
  
    kernel = np.ones((3,3), np.uint8)

    img_dilation = cv2.dilate(image, kernel, iterations=1)
    
    return img_dilation

#2
def Erosion(image):
    kernel = np.ones((5,5), np.uint8)
    img_erotion = cv2.erode(image, kernel, iterations=1)
    return img_erotion

#3
def rotate_image(image):
	#rotation angle in degree
    angles=[-90,-75,-60,-45,-30,30,45,60,75,90,180]
    angle=random.choice(angles)
    rotated = ndimage.rotate(image,angle,prefilter=False,reshape=False,mode='reflect')
    return rotated

#4
def width_shift(image):
    num_rows, num_cols = image.shape
    shift_pixels=[15,20,25,-15,-20,-25]
    shift_pixel=random.choice(shift_pixels)
    img_translation=ndimage.shift(image, (shift_pixel, 0),prefilter=False,mode='reflect')
    return img_translation

#5
def height_shift(image):
    num_rows, num_cols = image.shape
    shift_pixels=[15,20,25,-15,-20,-25]
    shift_pixel=random.choice(shift_pixels)
    img_translation=ndimage.shift(image, (0,shift_pixel),prefilter=False,mode='reflect')
    return img_translation

#6
def zoom_out(image):
    values=[0.3,0.4,.5,.6,0.55,.45]
    value=random.choice(values)
    a=ndimage.zoom(image, zoom=value, mode='reflect',grid_mode=True)
    out=np.pad(a, (150, 150), 'constant', constant_values=a[2,2])
    return out

#7
def draw_grid(img, line_color=(50, 50, 50), thickness=1, type_=cv2.LINE_AA, pxstep=random.choice([10,15,20,25,30,40])):
   
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep
        
    return img

#8
def draw_grid_horizontal(img, line_color=(50, 50, 50), thickness=1, type_=cv2.LINE_AA, pxstep=random.choice([10,15,20,25,30,40])):
   
    x = pxstep
    
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

        
    return img

#9
def draw_grid_vertical(img, line_color=(50, 50, 50), thickness=1, type_=cv2.LINE_AA, pxstep=random.choice([10,15,20,25,30,40])):
   
    
    y = pxstep
    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

        
    return img
   
def my_aug(image,aug_typ):

   
    if aug_typ==1:
        out=Dilation(image)
    
    if aug_typ==2:
        out=Erosion(image)
        
    if aug_typ==3:
         out=rotate_image(image)
    
    if aug_typ==4:
        out=width_shift(image)
    
    if aug_typ==5:
        out=height_shift(image)
    
    if aug_typ==6:
        out=zoom_out(image)
        
    if aug_typ==7:
        out=draw_grid(image)
        
    if aug_typ==8:
        out=draw_grid_horizontal(image)
        
    if aug_typ==9:
        out=draw_grid_vertical(image)

    return out


#%% Calling augmentation functions

count=0
aug_train_data=np.zeros((train_data.shape[0]*6,150,150))
aug_choice=np.array([1,2,4,5,6])
labels=[]
for i in range(train_data.shape[0]):
    
    aug_train_data[count]=train_data[i]
    count=count+1
    
    out=my_aug(train_data[i],3)
    aug_train_data[count]=cv2.resize(out,(150,150))
    count=count+1
    
    
    for j in range(3):
        temp=random.choice(aug_choice)
        out=my_aug(train_data[i],temp)
        aug_train_data[count]=cv2.resize(out,(150,150))
        count=count+1
      
    out=my_aug(train_data[i],random.choice([7,8,9]))
    aug_train_data[count]=cv2.resize(out,(150,150))
    count=count+1
        
aug_train_labels=[]

for i in range(train_data.shape[0]):
    for j in range(6):
        aug_train_labels.append(train_labels[i])

#%% Saving augmentation data

np.save(save_path+'aug_train_images2.npy',aug_train_data)
np.save(save_path+'aug_train_labels2.npy',aug_train_labels)

#%% Training with augmented data

img_row = 150

train_data = np.load(save_path+'aug_train_images2.npy')
val_data   = np.load(save_path+'val_images.npy')

train_labels = np.load(save_path+'aug_train_labels2.npy')
val_labels   = np.load(save_path+'val_labels.npy')


train_data = train_data/255
val_data   = val_data/255


train_data,train_labels = shuffle(train_data,train_labels,random_state=2461)

# Scaling

train_data = np.reshape(train_data,(train_data.shape[0],img_row,img_row,1))
val_data   = np.reshape(val_data,(val_data.shape[0],img_row,img_row,1))

train_labels = np_utils.to_categorical(train_labels, 10)
val_labels   = np_utils.to_categorical(val_labels, 10)


#%%
input_shape = (img_row,img_row,1)
img_input   = Input(shape=input_shape)

# First Convolutional Layer
x = Conv2D(32, (3,3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block1_conv1')(img_input)
x = BatchNormalization()(x)
x = MaxPooling2D((3,3), strides=(2,2), name='block1_pool')(x)
x=Dropout(0.2)(x)

# Second Convolutional Layer
x = Conv2D(64, (3,3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block2_conv1')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3,3), strides=(2,2), name='block2_pool')(x)
x=Dropout(0.2)(x)

# Third Convolutional Layer
x = Conv2D(128, (3,3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block3_conv1')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3,3), strides=(2,2), name='block3_pool')(x)
x=Dropout(0.2)(x)

# Fourth Convolutional Layer
x = Conv2D(256, (3,3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block4_conv1')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3,3), strides=(2,2), name='block4_pool')(x)
x=Dropout(0.2)(x)

# Fifth Convolutional Layer
x = Conv2D(512, (3,3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block5_conv1')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3,3), strides=(2,2), name='block5_pool')(x)
x=Dropout(0.2)(x)


x = Flatten(name='flatten')(x)

        # Classification part
# First FCN layer
x=Dense(256,activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(.2)(x)    

# Second FCN layer
x=Dense(256,activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(.2)(x)    

# Third FCN layer
x=Dense(256,activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(.2)(x)    

# Final dense layer
out=Dense(10, activation= "softmax")(x)
merged_model=Model(inputs=img_input ,outputs=out)

adam=optimizers.Adam(lr= 0.00010  ,beta_1=.9,beta_2=.999) 

merged_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

checkpoint=[ModelCheckpoint(filepath='./modelCNN.hdf5',  monitor='val_accuracy', save_best_only=True, mode='max',verbose=1)]

hist_train=merged_model.fit(train_data,train_labels,callbacks=checkpoint, batch_size=32,epochs=100, shuffle=True,verbose=1,validation_data=(val_data,val_labels))
