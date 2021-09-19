# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:04:08 2021

@author: user
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

model = Sequential([
        Dense(units=16, input_shape=(1,), activation='relu'),
        Dense(units=32,  activation='relu'),
        Dense(units=2, activation='softmax')
              ])
print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy',
              metrics= ['accuracy'])
              
data = pd.read_csv('C:/Users/user/Documents/B.A. Governance Sem.6/Plataforma5/ML 2019/labs/Lab02/wine.csv')
data = data[:3500]
data = data.sample(frac = 1) #shuffle
predict_data = data[3000:]
train_data = data[:3000]
    
x_train = [np.array(train_data['residual sugar']),
           np.array(train_data['chlorides'])]
y_train = train_data['type']
col_map = {'white':0,'red':1}
y_train = y_train.map(col_map)
y_train = np.array(y_train)

model.fit(x=x_train, y=y_train, epochs=60,batch_size=30, verbose=1, 
          validation_split=0.1,shuffle=True)


x_test = [np.array(predict_data['residual sugar']),
          np.array(predict_data['chlorides'])]
y_test = predict_data['type']
y_test = y_test.map(col_map)
y_test = np.array(y_test)

preds = model.predict(x_test,verbose=1,batch_size=10)

preds = np.argmax(preds,axis=1)
preds

print(' tp   fn\n fp   tn')
print(confusion_matrix(y_test, preds))
print(' 11   01\n 10   00')




