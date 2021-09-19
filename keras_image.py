# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 19:09:24 2021

@author: user
"""
#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten,\
BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
#%matplotlib.inline
import pandas as pd
import numpy as np

#%% split data into train,val,test
#os.chdir('C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/Keras/dogs_vs_cats_wie_in_yt')
#all_dog_file_names = glob.glob('dog*')
#all_cat_file_names = glob.glob('cat*')
##random.shuffle(all_dog_file_names)
##random.shuffle(all_cat_file_names)
#split_positions = [500,849,949]
#train_dog = all_dog_file_names[:split_positions[0]]
#train_cat = all_cat_file_names[:split_positions[0]]
#valid_dog = all_dog_file_names[split_positions[1]:split_positions[2]]
#valid_cat = all_cat_file_names[split_positions[1]:split_positions[2]]
#test_dog = all_dog_file_names[split_positions[2]:]
#test_cat = all_cat_file_names[split_positions[2]:]
#for c in train_dog:
#    shutil.move(c, 'train\dog')
#for c in train_cat:
#    shutil.move(c, 'train\cat')
#for c in valid_dog:
#    shutil.move(c, 'valid\dog')
#for c in valid_cat:
#    shutil.move(c, 'valid\cat')
#for c in test_dog:
#    shutil.move(c, 'test\dog')
#for c in test_cat:
#    shutil.move(c, 'test\cat')
#os.chdir('../../')

#%%     
train_path = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/Keras/dogs_vs_cats_wie_in_yt/train'
valid_path = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/Keras/dogs_vs_cats_wie_in_yt/valid'
test_path =  'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/Keras/dogs_vs_cats_wie_in_yt/test'

train_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
        .flow_from_directory(directory=train_path,target_size=(224,224),
        classes=['cat','dog'],batch_size=10)
valid_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
        .flow_from_directory(directory=valid_path,target_size=(224,224),
        classes=['cat','dog'],batch_size=10)
test_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
        .flow_from_directory(directory=test_path,target_size=(224,224),
        classes=['cat','dog'],batch_size=10, shuffle=False)


#%% 
model = Sequential([
        Conv2D(filters=32, kernel_size=(3,3),input_shape=(224,224,3), 
               activation='relu',padding='same'), #3 color channels
        MaxPool2D(pool_size=(2,2),strides=2),
#        Conv2D(filters=64, kernel_size=(3,3),activation='relu',padding='same'),
        Flatten(),
        Dense(units=2, activation='softmax')
              ])
        
print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', #not sparse_categorical...!!!
              metrics= ['accuracy'])

model.fit(x=train_batches,validation_data=valid_batches,batch_size=None, 
          epochs=8, verbose=1)


preds = model.predict(test_batches,verbose=1)

preds = np.argmax(preds,axis=1)
preds

print(test_batches.class_indices)
print(' tp   fn\n fp   tn')
print(confusion_matrix(test_batches.classes, preds))
print(' 11   01\n 10   00')







