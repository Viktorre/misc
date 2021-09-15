# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:09:08 2020

@author: user
"""


import pandas as pd
import numpy as np



###### General Functions 
###
#

def print_r_squared_in_and_oos(predictor,X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test ):
    print(predictor.__class__.__name__,'r-squared:')
    print('in sample:',predictor.score(X_train, y_train))
    print('OOS:',predictor.score(X_test, y_test)) #this is the out of sample r-squared
    print('\n')

#
###
#####


######   create test dataset
###
#

cols = []
for i in range(1000):
    cols.append('col'+str(i))
#print(cols)
    
df = pd.DataFrame(np.random.randn(93972, 1000), columns=cols)
df['ints']= range(len(df))
print(df)

#
###
#### end create test dataset





######  split data
###
#
y = df['col0']
X = df.drop(['col0'],1)

from sklearn.model_selection import train_test_split

#we need 3 subsets: 1/3 for training, 1/3 for validation, 1/3 for testing
# first split testing away
X_train_and_validation, X_test, y_train_and_validation, y_test = train_test_split(
    X, y, test_size=1/3, random_state=42)

# then split last 2/3 into train and val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_and_validation, y_train_and_validation, test_size=0.5, random_state=42)

#for subset in X_train, X_val, X_test, y_train, y_val, y_test:
#    print(subset)
#    pass


#
###
#### end split data



######  linear model
###
#

from sklearn.linear_model import LinearRegression, HuberRegressor
reg = LinearRegression().fit(X_train, y_train)
print_r_squared_in_and_oos(reg)

#from sklearn.metrics import r2_score  ##oos-r2
#print(r2_score(y_test, reg.predict(X_test)))

#ols with huber loss fct
huber = HuberRegressor().fit(X_train, y_train)
print_r_squared_in_and_oos(huber)

#
###
#### end linear model


######   Enet (with huber loss)
###
#

#idea estimnate enet and get the thetas?
from sklearn.linear_model import ElasticNet
Enet = ElasticNet(random_state=0).fit(X, y)
print_r_squared_in_and_oos(Enet)


#no longer use this! all server

#
###
#####







