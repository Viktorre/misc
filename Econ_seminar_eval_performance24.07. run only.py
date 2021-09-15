# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:52:52 2020

@author: fx236
"""


##### import general libraries
###
#

import pandas as pd
import numpy as np

import warnings

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA

#
###
##### end libraries import

##### setup: path, surpress sklearn warnings, pandas
###
#

#path = 'C:/Users/fx236/Desktop/Spyder_code/'
path = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/Econ seminar/'

# warnings.filterwarnings("ignore", category=DeprecationWarning) 
def warn(*args, **kwargs):
    pass
warnings.warn = warn

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 2000)

#
###
##### end setup: path, surpress sklearn warnings



##### import dm test
###
#

# local import for dm_test.py does not work for some reason. long workaround:
#t his block substitutes like "import dm_test as dm"
import importlib.util
spec = importlib.util.spec_from_file_location("dm_test", path+"dm_test.py")
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)
# dm.dm_test()  

#
###
##### end dm import





###### import df as pkl to save time
###
#

df = pd.read_pickle(path+'final_dataset.pkl')

print(df,'\n')

df = df.reset_index()

#
###
##### end pkl import


##### use shorter test df
###
#

#df = pd.read_pickle(path+'final_dataset.pkl')
#
#print(df,'\n')
#
#df = df.reset_index()
#
#df = df.drop_duplicates(subset='DATE')
#
#print(df)

#
###
##### end shorter test df



######   Neural Networks (need to be defined before performance eval due to ensemble)
###
#

    
def return_list_of_10_estimator_tuples(estimator):
    
    list_of_estimators = []
    
    for w in range(1,11):
        
        list_of_estimators.append((str(w),estimator))
    
    return list_of_estimators



## no loops here for better oversight. Note the different hidden_layer_sizes
NN1 = MLPRegressor(hidden_layer_sizes= (32,),
                    alpha = 10e-4,  #tuneable
                    learning_rate_init = 0.005, #tune
                    learning_rate = 'constant',n_iter_no_change = 100,solver = 'adam',activation = 'relu',batch_size = 10000,
                    )

NN2 = MLPRegressor(hidden_layer_sizes= (32,16),
                    alpha = 10e-4,  #tuneable
                    learning_rate_init = 0.005, #tune
                    learning_rate = 'constant',n_iter_no_change = 100,solver = 'adam',activation = 'relu',batch_size = 10000,
                    )

NN3 = MLPRegressor(hidden_layer_sizes= (32,16,8),
                    alpha = 10e-4,  #tuneable
                    learning_rate_init = 0.005, #tune
                    learning_rate = 'constant',n_iter_no_change = 100,solver = 'adam',activation = 'relu',batch_size = 10000,
                    )

NN4 = MLPRegressor(hidden_layer_sizes= (32,16,8,4),
                    alpha = 10e-4,  #tuneable
                    learning_rate_init = 0.005, #tune
                    learning_rate = 'constant',n_iter_no_change = 100,solver = 'adam',activation = 'relu',batch_size = 10000,
                    )

NN5 = MLPRegressor(hidden_layer_sizes= (32,16,8,4,2),
                    alpha = 10e-4,  #tuneable
                    learning_rate_init = 0.005, #tune
                    learning_rate = 'constant',n_iter_no_change = 100,solver = 'adam',activation = 'relu',batch_size = 10000,
                    )

Ten_NN1s = return_list_of_10_estimator_tuples(NN1)
Ten_NN2s = return_list_of_10_estimator_tuples(NN2)
Ten_NN3s = return_list_of_10_estimator_tuples(NN3)
Ten_NN4s = return_list_of_10_estimator_tuples(NN4)
Ten_NN5s = return_list_of_10_estimator_tuples(NN5)

ensemble_of_NN1 = VotingRegressor(estimators=Ten_NN1s) #has no voting, only avg
ensemble_of_NN2 = VotingRegressor(estimators=Ten_NN2s)
ensemble_of_NN3 = VotingRegressor(estimators=Ten_NN3s)
ensemble_of_NN4 = VotingRegressor(estimators=Ten_NN4s)
ensemble_of_NN5 = VotingRegressor(estimators=Ten_NN5s)

#test
# get_avg_oos_r2_from_one_predictor(df,ensemble_of_NN1)  #needs tunig! #<0


#
###
##### NN1



######   tests for tune hyperparams (alpha, l1_ratio)
### 
#


# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.model_selection import GridSearchCV
# from sklearn.utils.fixes import loguniform
# import scipy.stats as stats


# tscv = TimeSeriesSplit(n_splits=20) #here idk how to get it yearwise, maybe manually
# CV = KFold(n_splits=10,
#           random_state=1234)

    
    



PLS_grid = [{'n_components': range(0,len(df.columns)-1)}]
PCR_grid = PLS_grid
# Enet_grid = [{'alpha': [loguniform(1e-4, 1e-3)] }]
# Enet_grid = [{'alpha': [1e-4, 1e-3] }]
Enet_grid = [{'alpha': np.linspace(1e-4, 1e-3, 100) }] #pseudo continous range
RF_grid = [{'max_depth':range(1,7),'min_samples_split':[3,5,10,20,30,50]}]
GBRT_grid = [{'max_depth':range(1,3),'n_estimators':range(1,1001),'learning_rate':np.linspace(0.01,0.1, 10)}]
NN_grid = [{'alpha': np.linspace(1e-5, 1e-3, 100),'learning_rate':np.linspace(0.01,0.1, 10)}]


grid_dict = {'LinearRegression':[],
    'HuberRegressor':[],
    'custom_ols3':[],
    'PLSRegression':PLS_grid,
    'custom_pcr': [],
    'ElasticNet': Enet_grid, 
    'RandomForestRegressor': RF_grid,
    'GradientBoostingRegressor': GBRT_grid,
    'VotingRegressor': NN_grid
    }   


#"Para ver los valores posibles y cuáles combinaciones son compatibles, ver la documentación"
#logreg = LogisticRegression(n_jobs=-1,random_state=42, max_iter = 500)

# GS = GridSearchCV(estimator = ElasticNet(random_state=0,alpha=0.01,l1_ratio=0.5),
#                   param_grid = Enet_grid,
#                   # cv=None,#tscv, i think gu et use a basic validation, no CV oä
#                   n_jobs= -1)

# GS.fit(X_val,y_val)      #must Estimator be trained or untrained in params? 

# print(GS.best_estimator_) #danach ka wie ich die optimalen params beuntzen kann, oder kann GS dann auch score ausgeben für oos r2? 
# # predictor = GS



#
###
##### tests tune params



##### def fct to split data
### first function only train and test; second train, val and test, third for ols3
#

def return_train_test_for_one_year(df,last_training_year):
    
    train = df[df['DATE'].dt.year <= last_training_year+1]
    test = df[df['DATE'].dt.year == last_training_year+2]
    
    train.set_index(['permno','DATE'],inplace=True)
    # val.set_index(['permno','DATE'],inplace=True)
    test.set_index(['permno','DATE'],inplace=True)
    
    # print('\n',last_training_year,'as last training year:',len(train),len(val),len(test))
    print(last_training_year,'as last training year:',len(train),len(test))
    
    X_train, y_train = train.drop(['ret_discrete'],1),train['ret_discrete']
    # X_val, y_val = val.drop(['ret_discrete'],1),val['ret_discrete']
    X_test, y_test = test.drop(['ret_discrete'],1),test['ret_discrete']


    return X_train, y_train, X_test, y_test 

##currently not used:
# def return_train_val_test_for_one_year(df,last_training_year):

#     train = df[df['DATE'].dt.year <= last_training_year]
#     val = df[df['DATE'].dt.year == last_training_year+1]
#     test = df[df['DATE'].dt.year == last_training_year+2]
    
#     train.set_index(['permno','DATE'],inplace=True)
#     val.set_index(['permno','DATE'],inplace=True)
#     test.set_index(['permno','DATE'],inplace=True)
    
#     print('\n',last_training_year,'as last training year:',len(train),len(val),len(test))
#     # print('\n',last_training_year,'as last training year:',len(train),len(test))
    
#     X_train, y_train = train.drop(['ret_discrete'],1),train['ret_discrete']
#     X_val, y_val = val.drop(['ret_discrete'],1),val['ret_discrete']
#     X_test, y_test = test.drop(['ret_discrete'],1),test['ret_discrete']


#     return X_train, y_train,  X_val, y_val, X_test, y_test 

def return_train_test_for_one_year_ols3(df,last_training_year):
    
    train = df[df['DATE'].dt.year <= last_training_year+1]
    test = df[df['DATE'].dt.year == last_training_year+2]
    
    train.set_index(['permno','DATE'],inplace=True)
    # val.set_index(['permno','DATE'],inplace=True)
    test.set_index(['permno','DATE'],inplace=True)
    
    # print('\n',last_training_year,'as last training year:',len(train),len(val),len(test))
    print('\n',last_training_year+1,'as last training year:',len(train),len(test))
    
    X_train, y_train = train[['mvel1','b/m','mom1m']],train['ret_discrete']
    # X_val, y_val = val.drop(['ret_discrete'],1),val['ret_discrete']
    X_test, y_test = test[['mvel1','b/m','mom1m']],test['ret_discrete']


    return X_train, y_train, X_test, y_test 

#
###
#### fct to split data
    

#####  custom predictor class definiton for ols3
###
#

class custom_ols3():
    
    def __init__(self, estimator):
        
        self.estimator = estimator

    def print_coefs(self):
        
        print(self.estimator.coef_)

    def fit(self,X_train,y_train):
        
        X_train = X_train[['mvel1','b/m','mom1m']]

        self.estimator.fit(X_train,y_train)
          
  # def score_with_altered_data(self,X_test,y_test,sample_weight=None):
    def score(self,X_test,y_test,sample_weight=None):
            

        X_test = X_test[['mvel1','b/m','mom1m']]

        return self.estimator.score(X_test,y_test)
      
    def predict(self,X_test):
        
        X_test = X_test[['mvel1','b/m','mom1m']]
        
        
        return self.estimator.predict(X_test)
              
#          
###
##### end custom predictor class definiton for ols3


#####  custom predictor class definiton for pcr
###
#

class custom_pcr():
    
    def __init__(self, estimator, k):
        
        self.estimator = estimator
        self.pca = PCA(n_components=k)
        # self.k = k # tuning parameter: how many principal components used in regression

    def print_coefs(self):
        
        print(self.estimator.coef_)

    def fit(self,X_train,y_train):
     
        self.pca.fit(X_train)
        
        # print(X_train)
        
        X_train = self.pca.fit_transform(X_train)
        
        # X_train = self.pca.singular_values_ #sklearn does this in some example... but I prefer transform/fit_transform

        self.estimator.fit(X_train,y_train)
        
        # print('new, pca params',self.pca.get_params())
          
  # def score_with_altered_data(self,X_test,y_test,sample_weight=None):
    def score(self,X_test,y_test,sample_weight=None):
            
        X_test = self.pca.transform(X_test)

        return self.estimator.score(X_test,y_test)
      
    def predict(self,X_test):
         
        X_test = self.pca.transform(X_test)
        
        # print(X_test.shape)

        return self.estimator.predict(X_test)
        
    def get_params():
        
        return 'no params'
    
    def set_params(self,params):
        
        print(params)
         
#
###
##### end custom predictor class definiton for pcr
        
    
    

######  define function to get r² from rolling window
###
#


def get_avg_oos_r2_from_one_predictor(df,predictor,grid):

    predictor_avg_r2 = []

    for last_training_year in [2009,2010,2011,2012,2013,2014]: #rolling time window
    
        X_train, y_train, X_test, y_test = return_train_test_for_one_year(df,last_training_year)
        
       
        if False:#grid[predictor.__class__.__name__]: #tunable estimators use grid, the rest only plain predictor # i do this condition by giving non tunable estimators FALSE in their grid
            
            print(predictor.get_params())    
            predictor.fit(X_train, y_train)
            print('#### OOS:',predictor.score(X_test, y_test),'to compare with tuning')
            # print(predictor.get_params())
            
            tuned_predictor = GridSearchCV(estimator = predictor, #grid takes untrained predictor
                      param_grid = grid[predictor.__class__.__name__],
                      # cv=None,#tscv, i think gu et use a basic validation, no CV oä
                      #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
                       # 'An iterable yielding (train, test) splits as arrays of indices.'
                       cv = TimeSeriesSplit(n_splits=2), # best option i have
                      # cv = test_index,
                      n_jobs= -1)
    
            tuned_predictor.fit(X_train,y_train)  #not sure if i should train here again...
            print('#### tuned oos:',tuned_predictor.score(X_test, y_test))      
            print(tuned_predictor.get_params())
            
            predictor_avg_r2.append(tuned_predictor.score(X_test, y_test))
            
        else:
             
            # print('no tuning')
            predictor.fit(X_train, y_train)
            print('#### OOS:',predictor.score(X_test, y_test))
            # print(predictor.get_params())

            predictor_avg_r2.append(predictor.score(X_test, y_test))

        print('')
        
    print('avg oos r2 ',predictor.__class__.__name__,' = ',sum(predictor_avg_r2)/len(predictor_avg_r2))

    return sum(predictor_avg_r2)/len(predictor_avg_r2)
#
###
#### define function to get r² from rolling window




###### put all predictors that do not need special attention into one array
###
#



predictors = [
## LinearRegression() # only here for testing,
#HuberRegressor(),
#custom_ols3(HuberRegressor()),
#PLSRegression(n_components=1000), #needs tuning # -2.28
#custom_pcr(LinearRegression(),1000),
ElasticNet(random_state=0,alpha=0.005,l1_ratio=0.5), # 0.099 #alpha tunable
RandomForestRegressor(max_depth=3,n_estimators=300,min_samples_split=20), #min_samples_split and max_depth tunable
GradientBoostingRegressor(learning_rate=0.05,n_estimators=500,max_depth=2), 
#ensemble_of_NN1,
#ensemble_of_NN2,
#ensemble_of_NN3,
#ensemble_of_NN4,
#ensemble_of_NN5
]


'''
vllt noch 1,2 oder 3 "leichtere" predictors mit rein als beitrag
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

'''

#
###
#####


######  execute r2 function on predictors array
###
#

#r2_result_dict = {}

#for one_predictor in predictors:
for one_predictor in predictors[::-1]:
    
    if one_predictor.__class__.__name__ != 'VotingRegressor':

        print(one_predictor.__class__.__name__)
        
        r2_result_dict[one_predictor.__class__.__name__] = get_avg_oos_r2_from_one_predictor(df,one_predictor,grid_dict)

    else:
        
        print(str('NN'+str(len(one_predictor.estimators[0][1].hidden_layer_sizes))))
    
        r2_result_dict[str('NN'+str(len(one_predictor.estimators[0][1].hidden_layer_sizes)))] = get_avg_oos_r2_from_one_predictor(df,one_predictor,grid_dict)

    print('\n')

print(r2_result_dict)

#
###
#####







######   definition get cross table of DM tests
###
# (dm test might lack cross sectional average of errors)



def return_dm_table(array_of_predictors,df):
    
    result_array = []
    
    for first_predictor in array_of_predictors:
        
        for second_predictor in array_of_predictors:
            
            # if first_predictor != second_predictor: #cond to avoid dm test of two equal forecasts, which throws error
            if array_of_predictors.index(first_predictor) < array_of_predictors.index(second_predictor):#condition to get all unique pairs of predictors

                if False:#grid[predictor.__class__.__name__]: #tunable estimators use grid, the rest only plain predictor # i do this condition by giving non tunable estimators FALSE in their grid
                
                    pass
                
                else: #no tuning
                
                    print(first_predictor.__class__.__name__,second_predictor.__class__.__name__,'#########')
                    
                    all_y_true = pd.DataFrame([])# i apply dm on sum of all predictions
                    all_pred1 = pd.DataFrame([])
                    all_pred2 = pd.DataFrame([])
                    
                    for last_training_year in [2006]:#[2009,2010,2011,2012,2013,2014]: #rolling time window
                
                       
                        X_train, y_train, X_test, y_test = return_train_test_for_one_year(df,last_training_year)
    
                    
                        first_predictor.fit(X_train, y_train)
                        second_predictor.fit(X_train, y_train)
        
                        all_y_true = pd.concat([all_y_true,y_test])
                        all_pred1  = pd.concat([all_pred1,pd.DataFrame(first_predictor.predict(X_test))])
                        all_pred2  = pd.concat([all_pred2,pd.DataFrame(second_predictor.predict(X_test))])
                            
                    
                    
                    # print statements for debuggibg:
                    # print(all_y_true,all_pred1,all_pred2)
                    # print(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list())
                    # print(dm.dm_test(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list(),h=1, crit="MSE"))
                    # print(dm.dm_test(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list(),h=1, crit="MSE")['DM'])
                    # print(dm.dm_test(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list(),h=1, crit="MSE")[0])
    
    
                    result_array.append(np.array(dm.dm_test(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list(),h=1, crit="MSE")))#,
    #                        dm.dm_test(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list(),h=1, crit="MSE")[0])                
            else:
                result_array.append('') #append empty space
    
    result_table = pd.DataFrame(np.array(result_array).reshape(len(array_of_predictors),len(array_of_predictors)))

    names_of_each_predictor = []
    for predictor in array_of_predictors:
        if predictor.__class__.__name__ == 'VotingRegressor': #cond to rename the 5 votingregrssors into NN1,...NN5
            names_of_each_predictor.append('NN'+str(len(predictor.get_params()['1__hidden_layer_sizes'])))
        else:
            names_of_each_predictor.append(predictor.__class__.__name__)
        
    result_table.columns = names_of_each_predictor
    
    result_table.index = names_of_each_predictor
    
    result_table.drop(names_of_each_predictor[0],axis=1, inplace=True)
    
    result_table.drop(names_of_each_predictor[-1],axis=0, inplace=True)
    
    return result_table



# 
###
######   get cross table of DM tests

    
#####    execute dm table function
###
#    
    

predictors = [
## LinearRegression() # only here for testing,
 HuberRegressor(),
custom_ols3(HuberRegressor()),
# PLSRegression(n_components=1000), #needs tuning # -2.28
# custom_pcr(LinearRegression(),1000),
#ElasticNet(random_state=0,alpha=0.005,l1_ratio=0.5), # 0.099 #alpha tunable
#RandomForestRegressor(max_depth=3,n_estimators=300,min_samples_split=20), #min_samples_split and max_depth tunable
#GradientBoostingRegressor(learning_rate=0.05,n_estimators=500,max_depth=2), 
ensemble_of_NN1,
ensemble_of_NN2,
#ensemble_of_NN3,
#ensemble_of_NN4,
#ensemble_of_NN5
]

#dm_table = return_dm_table(predictors,df)
#
#pd.set_option('display.max_rows', 20)
#pd.set_option('display.max_columns', 20)
#pd.set_option('display.width', 2000)
#
#print(dm_table)


# print('which criterin in test?')


#   
###
##### execute dm table function



















































##########################################################################
######     Graveyard for no longer used codes
###
#



# Enet_avg_r2 = []

# y = df['ret_discrete']
# X = df.drop(['ret_discrete','permno','DATE'],1)

# from sklearn.model_selection import train_test_split

# #we need 3 subsets: 1/3 for training, 1/3 for validation, 1/3 for testing
# # first split testing away
# X_train_and_validation, X_test, y_train_and_validation, y_test = train_test_split(
#     X, y, test_size=1/3, random_state=42)

# # then split last 2/3 into train and val
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_and_validation, y_train_and_validation, test_size=0.5, random_state=42)

# for subset in X_train, X_val, X_test, y_train, y_val, y_test:
#     print(subset)
#     # pass




# for last_training_year in [2009,2010,2011,2012,2013,2014]:
    
#     train = df[df['DATE'].dt.year <= last_training_year]
#     val = df[df['DATE'].dt.year == last_training_year+1]
#     test = df[df['DATE'].dt.year == last_training_year+2]
    
#     train.set_index(['permno','DATE'],inplace=True)
#     val.set_index(['permno','DATE'],inplace=True)
#     test.set_index(['permno','DATE'],inplace=True)
    
    
#     print(last_training_year,'as last training year:',len(train),len(val),len(test))
    
#     X_train, y_train = train.drop(['ret_discrete'],1),train['ret_discrete']
#     X_val, y_val = val.drop(['ret_discrete'],1),val['ret_discrete']
#     X_test, y_test = test.drop(['ret_discrete'],1),test['ret_discrete']

#     Enet = ElasticNet(random_state=0,alpha=0.01,l1_ratio=0.5)
#     Enet.fit(X_train, y_train)
#     print('OOS:',Enet.score(X_test, y_test))
#     Enet_avg_r2.append(Enet.score(X_test, y_test))
#     # print(Enet.get_params())

# print('avg oos r2 enet = ',sum(Enet_avg_r2)/len(Enet_avg_r2))





# def print_r_squared_in_and_oos(predictor,X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test ):
#     print(predictor.__class__.__name__,'r-squared:')
#     print('in sample:',predictor.score(X_train, y_train))
#     print('OOS:',predictor.score(X_test, y_test)) #this is the out of sample r-squared
#     print('\n')





#
###
#####
















##### define function that returns avg of dm test results
# ###
# #
# def return_avg_dm_statistic(df,predictor_1,predictor_2):

#     all_dm_stats = []
    
#     for last_training_year in [2009,2010,2011,2012,2013,2014]: #rolling time window
        
#         train = df[df['DATE'].dt.year <= last_training_year]
#         val = df[df['DATE'].dt.year == last_training_year+1]
#         test = df[df['DATE'].dt.year == last_training_year+2]
        
#         train.set_index(['permno','DATE'],inplace=True)
#         val.set_index(['permno','DATE'],inplace=True)
#         test.set_index(['permno','DATE'],inplace=True)
        
#         print(last_training_year,'as last training year:',len(train),len(val),len(test))
#         X_train, y_train = train.drop(['ret_discrete'],1),train['ret_discrete']
#         # X_val, y_val = val.drop(['ret_discrete'],1),val['ret_discrete']
#         X_test, y_test = test.drop(['ret_discrete'],1),test['ret_discrete']
    
#         # predictor = ElasticNet(random_state=0,alpha=0.01,l1_ratio=0.5)
#         predictor_1.fit(X_train, y_train)
#         predictor_2.fit(X_train, y_train)
        
#         print( dm.dm_test(y_test,
#                                         predictor_1.predict(X_test),
#                                         predictor_2.predict(X_test)))
            
#    #all_dm_stats.append
    
# return_avg_dm_statistic(df,
#                         ElasticNet(random_state=0,alpha=0.01,l1_ratio=0.5), 
#                         LinearRegression())   

# #
# ###
# #####










# ######  define function to get r² from rolling window
# ###
# #

# def get_avg_oos_r2_from_one_predictor(df,predictor,grid):

#     predictor_avg_r2 = []

#     for last_training_year in [2002,2003,2004]:#[2009,2010,2011,2012,2013,2014]: #rolling time window
        
#         # train = df[df['DATE'].dt.year <= last_training_year]
#         # val = df[df['DATE'].dt.year == last_training_year+1]
#         #Edit: we take train and val together
#         train = df[df['DATE'].dt.year <= last_training_year+1]
#         test = df[df['DATE'].dt.year == last_training_year+2]
        
#         train.set_index(['permno','DATE'],inplace=True)
#         # val.set_index(['permno','DATE'],inplace=True)
#         test.set_index(['permno','DATE'],inplace=True)
        
#         # print('\n',last_training_year,'as last training year:',len(train),len(val),len(test))
#         print('\n',last_training_year,'as last training year:',len(train),len(test))
        
#         X_train, y_train = train.drop(['ret_discrete'],1),train['ret_discrete']
#         # X_val, y_val = val.drop(['ret_discrete'],1),val['ret_discrete']
#         X_test, y_test = test.drop(['ret_discrete'],1),test['ret_discrete']

#         # print(predictor.get_params())    
#         # predictor.fit(X_train, y_train)
#         # print('#### OOS:',predictor.score(X_test, y_test))
#         # print(predictor.get_params())       
    
   
#         tuned_predictor = GridSearchCV(estimator = predictor, #grid takes untrained predictor
#                   param_grid = grid[predictor.__class__.__name__],
#                   # cv=None,#tscv, i think gu et use a basic validation, no CV oä
#                   #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#                    # 'An iterable yielding (train, test) splits as arrays of indices.'
#                   cv = TimeSeriesSplit(n_splits=2),
#                   n_jobs= -1)

#         tuned_predictor.fit(X_train,y_train)  #not sure if i should train here again...
#         print('#### tuned oos:',tuned_predictor.score(X_test, y_test))      
#         print(tuned_predictor.get_params())
        


#         predictor_avg_r2.append(predictor.score(X_test, y_test))
    
#     print('avg oos r2 ',predictor.__class__.__name__,' = ',sum(predictor_avg_r2)/len(predictor_avg_r2))


# #
# ###
# #### define function to get r² from rolling window









        # # train = df[df['DATE'].dt.year <= last_training_year]
        # # val = df[df['DATE'].dt.year == last_training_year+1]
        # #Edit: we take train and val together
        # train = df[df['DATE'].dt.year <= last_training_year+1]
        # test = df[df['DATE'].dt.year == last_training_year+2]
        
        # train.set_index(['permno','DATE'],inplace=True)
        # # val.set_index(['permno','DATE'],inplace=True)
        # test.set_index(['permno','DATE'],inplace=True)
        
        # # print('\n',last_training_year,'as last training year:',len(train),len(val),len(test))
        # print('\n',last_training_year,'as last training year:',len(train),len(test))
        
        # X_train, y_train = train.drop(['ret_discrete'],1),train['ret_discrete']
        # # X_val, y_val = val.drop(['ret_discrete'],1),val['ret_discrete']
        # X_test, y_test = test.drop(['ret_discrete'],1),test['ret_discrete']
















# #####  custom predictor class definiton for pcr
# ###
# #

# class custom_pcr():
    
#     def __init__(self, estimator):
        
#         self.estimator = estimator
#         self.pca = PCA()

#     def print_coefs(self):
        
#         print(self.estimator.coef_)

#     def fit(self,X_train,y_train):
     
#         # pca = PCA()  
#         self.pca.fit(X_train)
        
#         X_train = self.pca.fit_transform(X_train)

#         self.estimator.fit(X_train,y_train)
        
#         print('new, pca params',self.pca.get_params())
          
#   # def score_with_altered_data(self,X_test,y_test,sample_weight=None):
#     def score(self,X_test,y_test,sample_weight=None):
            
#         # pca = PCA()  
#         # self.pca.fit(X_train)
#         print(888,X_test)
#         X_test = self.pca.fit_transform(X_test)
#         print(999,X_test)

#         return self.estimator.score(X_test,y_test)
      
#     def predict(self,X_test):
         
#         # pca = PCA()  
#         # self.pca.fit(X_train)

#         X_test = self.pca.fit_transform(X_test)
        
        
#         return self.estimator.predict(X_test)
        
#     def get_params():
        
#         return 'no params'
              
          
# # X_train, y_train, X_test, y_test = return_train_test_for_one_year(df,2009)
# # pcr = custom_pcr(LinearRegression())   
# # print('hier fehler finden warum so großer r2')
# # pcr.fit(X_train, y_train)
# # # dir(pcr)  
# # # pcr.print_coefs()
# # print(pcr.score(X_test, y_test))
# # # print(pcr.predict(X_test))
# # # print(pcr.__class__.__name__)

# #
# ###
# ##### end custom predictor class definiton for pcr






# #####  custom predictor class definiton for pcr
# ###
# #

# class custom_pcr():
    
#     def __init__(self, estimator):
        
#         self.estimator = estimator
#         self.pca = PCA()

#     def print_coefs(self):
        
#         print(self.estimator.coef_)

#     def fit(self,X_train,y_train):
     
#         # pca = PCA()  
#         self.pca.fit(X_train)
        
#         X_train = self.pca.fit_transform(X_train)

#         self.estimator.fit(X_train,y_train)
        
#         # print('new, pca params',self.pca.get_params())
          
#   # def score_with_altered_data(self,X_test,y_test,sample_weight=None):
#     def score(self,X_test,y_test,sample_weight=None):
            
#         # pca = PCA()  
#         # self.pca.fit(X_train)
#         # print(888,X_test)
#         X_test = self.pca.transform(X_test)
#         # print(999,X_test)

#         return self.estimator.score(X_test,y_test)
      
#     def predict(self,X_test):
         
#         # pca = PCA()  
#         # self.pca.fit(X_train)

#         X_test = self.pca.transform(X_test)
        
        
#         return self.estimator.predict(X_test)
        
#     def get_params():
        
#         return 'no params'
    
#     def set_params(self, **params):
        
        
              
# # X_train, y_train, X_test, y_test = return_train_test_for_one_year(df,2009)
# # pcr = custom_pcr(LinearRegression())   
# # print('hier fehler finden warum so großer r2')
# # pcr.fit(X_train, y_train)
# # # dir(pcr)  
# # # pcr.print_coefs()
# # print(pcr.score(X_test, y_test))
# # # print(pcr.predict(X_test))
# # # print(pcr.__class__.__name__)
        
# #
# ###
# ##### end custom predictor class definiton for pcr









# ######   OLS 3: only 3 regressors (needs own oos fct due to different input data)
# ###  (seems as though not enough data to get good oosR2)
# #
    
# def OLS3_get_avg_oos_r2(df,predictor):

#     predictor_avg_r2 = []

#     for last_training_year in [2009,2010,2011,2012,2013,2014]: #rolling time window
        
#         X_train, y_train, X_test, y_test = return_train_test_for_one_year_ols3(df,last_training_year)
    
#         predictor.fit(X_train, y_train)
#         print('OOS:',predictor.score(X_test, y_test))
#         predictor_avg_r2.append(predictor.score(X_test, y_test))
#         # print(predictor.get_params())
    
#     print('avg oos r2 ','OLS-3 HuberRegressor',' = ',sum(predictor_avg_r2)/len(predictor_avg_r2))


# # OLS3_get_avg_oos_r2(df,HuberRegressor()) #-95!!!


# #
# ###
# ##### end ols3

# ######   PCR (looks like first do pca, second ols with singular vals)
# ###
# #
    


# def PCR_get_avg_oos_r2(df,predictor):

#     predictor_avg_r2 = []

#     for last_training_year in [2009]:#[2009,2010,2011,2012,2013,2014]: #rolling time window
        
#         X_train, y_train, X_test, y_test = return_train_test_for_one_year(df,last_training_year)

#         pca = PCA()  
#         pca.fit(X_train)
 
#         X_train = pca.fit_transform(X_train)  
#         X_test = pca.transform(X_test)
 
                
#         predictor.fit(X_train, y_train)
#         print('OOS:',predictor.score(X_test, y_test))
#         predictor_avg_r2.append(predictor.score(X_test, y_test))
    
#     print('avg oos r2 ','PCR',' = ',sum(predictor_avg_r2)/len(predictor_avg_r2))


# # PCR_get_avg_oos_r2(df,LinearRegression()) #

  
# #
# ###
# ##### end PCR










#######   Neural Networks (need to be defined before performance eval due to ensemble)
####
##
#
#    
#def return_list_of_10_estimator_tuples(estimator):
#    
#    list_of_estimators = []
#    
#    for w in range(1,11):
#        
#        print((str(w),estimator))
#        
#        list_of_estimators.append((str(w),estimator))
#    
#    return list_of_estimators
#
#
#
#NN1, NN2, NN3 ,NN4 ,NN5 = NN, NN , NN ,NN ,NN
#
#all_hidden_layer_sizes = [(32,),(32,16),(32,16,8),(32,16,8,4),(32,16,8,4,2)] 
#
#print(NN2)
#
#NN1.set_params(hidden_layer_sizes=all_hidden_layer_sizes[0])#no loops to better see what is happening
#NN2.set_params(hidden_layer_sizes=all_hidden_layer_sizes[1])
#NN3.set_params(hidden_layer_sizes=all_hidden_layer_sizes[2])
#NN4.set_params(hidden_layer_sizes=all_hidden_layer_sizes[3])
#NN5.set_params(hidden_layer_sizes=all_hidden_layer_sizes[4])
#
#print(NN2)
#
#
#
#Ten_NN1s = return_list_of_10_estimator_tuples(NN1)
#Ten_NN2s = return_list_of_10_estimator_tuples(NN2)
#Ten_NN3s = return_list_of_10_estimator_tuples(NN3)
#Ten_NN4s = return_list_of_10_estimator_tuples(NN4)
#Ten_NN5s = return_list_of_10_estimator_tuples(NN5)
#
#
#print(Ten_NN2s[0])
#                    
#ensemble_of_NN1 = VotingRegressor(estimators=Ten_NN1s) #has no voting, only avg
#ensemble_of_NN2 = VotingRegressor(estimators=Ten_NN2s)
#ensemble_of_NN3 = VotingRegressor(estimators=Ten_NN3s)
#ensemble_of_NN4 = VotingRegressor(estimators=Ten_NN4s)
#ensemble_of_NN5 = VotingRegressor(estimators=Ten_NN5s)
#
##test
## get_avg_oos_r2_from_one_predictor(df,ensemble_of_NN1)  #needs tunig! #<0
#
#
##
####
###### NN1



#######   definition get cross table of DM tests
####
## (dm test might lack cross sectional average of errors)
#
#
#
#def return_dm_table(array_of_predictors,df):
#    
#    result_array = []
#    
#    for first_predictor in array_of_predictors:
#        
#        for second_predictor in array_of_predictors:
#            
#            # if first_predictor != second_predictor: #cond to avoid dm test of two equal forecasts, which throws error
#            if array_of_predictors.index(first_predictor) < array_of_predictors.index(second_predictor):#condition to get all unique pairs of predictors
#
#                print(first_predictor.__class__.__name__,second_predictor.__class__.__name__,'#########')
#                
#                all_y_true = pd.DataFrame([])# i apply dm on sum of all predictions
#                all_pred1 = pd.DataFrame([])
#                all_pred2 = pd.DataFrame([])
#                
#                for last_training_year in [2006]:#[2009,2010,2011,2012,2013,2014]: #rolling time window
#            
#                   
#                    X_train, y_train, X_test, y_test = return_train_test_for_one_year(df,last_training_year)
#
#                
#                    first_predictor.fit(X_train, y_train)
#                    second_predictor.fit(X_train, y_train)
#    
#                    all_y_true = pd.concat([all_y_true,y_test])
#                    all_pred1  = pd.concat([all_pred1,pd.DataFrame(first_predictor.predict(X_test))])
#                    all_pred2  = pd.concat([all_pred2,pd.DataFrame(second_predictor.predict(X_test))])
#                        
#                
#                
#                # print statements for debuggibg:
#                # print(all_y_true,all_pred1,all_pred2)
#                # print(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list())
#                # print(dm.dm_test(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list(),h=1, crit="MSE"))
#                # print(dm.dm_test(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list(),h=1, crit="MSE")['DM'])
#                # print(dm.dm_test(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list(),h=1, crit="MSE")[0])
#
#
#                result_array.append(dm.dm_test(all_y_true[0].to_list(), all_pred1[0].to_list(), all_pred2[0].to_list(),h=1, crit="MSE")[0])
#                
#            else:
#                result_array.append('') #append empty space
#    
#    result_table = pd.DataFrame(np.array(result_array).reshape(len(array_of_predictors),len(array_of_predictors)))
#
#    names_of_each_predictor = []
#    for predictor in array_of_predictors:
#        if predictor.__class__.__name__ == 'VotingRegressor': #cond to rename the 5 votingregrssors into NN1,...NN5
#            names_of_each_predictor.append('NN'+str(len(predictor.get_params()['1__hidden_layer_sizes'])))
#        else:
#            names_of_each_predictor.append(predictor.__class__.__name__)
#        
#    result_table.columns = names_of_each_predictor
#    
#    result_table.index = names_of_each_predictor
#    
#    result_table.drop(names_of_each_predictor[0],axis=1, inplace=True)
#    
#    result_table.drop(names_of_each_predictor[-1],axis=0, inplace=True)
#    
#    return result_table
#
#
#
## 
####
#######   get cross table of DM tests
        
        
        
        
        
#        
#   backup results:     
#{'NN2': -0.20036736451104206,
# 'NN3': -1.3608658650398215,
# 'NN1': -116.23900273352893,
# 'NN5': -70.62060615698236,
# 'NN4': -21.43143060966175,
# 'custom_pcr': 0.1903220531356545,
# 'PLSRegression': -0.029887715991306185,
# 'custom_ols3': -51.02041647162148,
# 'HuberRegressor': -0.01004719813345171,
# 'GradientBoostingRegressor': 0.1742516652043459,
# 'RandomForestRegressor': -0.3442537719849615,
# 'ElasticNet': 0.20857866989293572}        