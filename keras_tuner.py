# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:28:23 2021

@author: user
"""
#https://pythonprogramming.net/keras-tuner-optimizing-neural-network-tutorial/
WATCH THE GODDAM TUTORIAL!
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten,\
BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
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


import kerastuner
#https://www.youtube.com/watch?v=vvC15l4CY1Q&ab_channel=sentdex
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()



import matplotlib.pyplot as plt
#%matplotlib inline

#print(y_test[0])
#plt.imshow(x_test[0], cmap="gray")
#
#print(y_test[1])
#plt.imshow(x_test[1], cmap="gray")

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)




from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
#Next, we'll specify the name to our log directory. I am just going to give it a name that is the time. Feel free to name it something else.

import time
#Now, we will add a build_model function. For this, we're just going to start by copying and pasting our exact model above:
LOG_DIR = f"{int(time.time())}"
#
def build_model(hp):  # random search passes this hyperparameter() object 
    model = keras.models.Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) 

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model




tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=1,  # how many model variations to test?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
    directory=LOG_DIR)




tuner.search(x=x_train,
             y=y_train,
             verbose=2, # just slapping this here bc jupyter notebook. The console out was getting messy.
             epochs=1,
             batch_size=64,
             #callbacks=[tensorboard],  # if you have callbacks like tensorboard, they go here.
             validation_data=(x_test, y_test))





print(stop)



model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))

#and making the number of features, currently 32, dynamic. The way we do this is by converting 32 to:

hp.Int('input_units',
        min_value=32,
        max_value=256,
        step=32)

#What this says is we want our hyperparameter object to create an int for us, which we'll call input_units, randomly, between 32 and 256, with a step of 32. So basically pick a number of units from [32, 64, 96, ..., 256].

#So our new input line becomes:

model.add(Conv2D(hp.Int('input_units',
                         min_value=32,
                         max_value=256,
                         step=32), (3, 3), input_shape=x_train.shape[1:]))

#Making our build_model function:
#FINAL MODEL SKIPPED ALL BEFORE

def build_model(hp):  # random search passes this hyperparameter() object 
    model = keras.models.Sequential()
    
    model.add(Conv2D(hp.Int('input_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3), input_shape=x_train.shape[1:]))
    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('n_layers', 1, 4)):  # adding variation of layers.
        model.add(Conv2D(hp.Int(f'conv_{i}_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3)))
        model.add(Activation('relu'))

    model.add(Flatten()) 
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model
#With the above model, our input layer, and each of the dynamic 1-4 more convolutional layers will all get variable units/features per convolutional layer. Let's do a quick test to make sure everything works, then I'll create an example of a much longer test that can show us what to do after the tests are done.

#LOG_DIR = f"{int(time.time())}"
#
#tuner = RandomSearch(
#    build_model,
#    objective='val_accuracy',
#    max_trials=1,  # how many model variations to test?
#    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
#    directory=LOG_DIR)
#
#tuner.search(x=x_train,
#             y=y_train,
#             verbose=2, # just slapping this here bc jupyter notebook. The console out was getting messy.
#             epochs=1,
#             batch_size=64,
#             #callbacks=[tensorboard],  # if you have callbacks like tensorboard, they go here.
#             validation_data=(x_test, y_test))
#
#
#
#








