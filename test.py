#!/usr/bin/env python3

import numpy as np

import tensorflow as tf
import keras

np.random.seed(1337)
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
#from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D,Input, Dense
from keras.layers import Conv1D,AveragePooling1D,MaxPooling2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,merge,LSTM,Concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import BatchNormalization,Reshape
from keras import optimizers
from tensorflow.keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import cv2
from keras.models import load_model
#%%
# Reading test data and labels and modification for input in the model

# Directory containing test data and label
file_path   = "./Data/"
# Testing data and labels
test_data   = np.load(file_path+'my_test_images.npy')
test_labels = np.load(file_path+'my_test_labels.npy')

test_data = test_data/255

img_row = 150
img_col = 150

#reshape 1D data to 300*300 array
test_data=test_data.T
test_data=np.reshape(test_data,(test_data.shape[0],300,300))

test_data_resized = np.zeros((test_data.shape[0],img_row,img_col))
for i in range(test_data.shape[0]):
    test_data_resized[i] = cv2.resize(test_data[i],(img_row,img_col))    

test_data_resized = np.reshape(test_data_resized,(test_data_resized.shape[0],img_row,img_col,1))


test_labels = np_utils.to_categorical(test_labels, 10)

#%%
# Test results

merged_model1 = load_model('modelCNN.hdf5')
predicted_labels = merged_model1.predict(test_data_resized,batch_size=1,verbose=1)
accuracy = merged_model1.evaluate(test_data_resized,test_labels,batch_size=1,verbose=1)
#print('Test loss:', accuracy[0]) 
print('\nTest accuracy:', accuracy[1]*100, '%')

pred = np.argmax(predicted_labels, axis = 1)#[:10] 
label = np.argmax(test_labels,axis = 1)#[:10] 

print('\nPredicted labels: ',pred) 
print('\nActual labels: ',label)
#print("Accuracy on Testing Data: ",accuracy[1]*100, '%')