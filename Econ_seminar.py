# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:59:56 2020

@author: user
"""

import pandas as pd
#from sklearn.preprocessing import StandardScaler

path = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/Econ seminar/'

file_and_path_for_big_dataset = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/Econ seminar/data shihou gu/datashare.csv'

################################################################################################
### read data ###
################################################################################################

data_victor = pd.read_csv(path+'data_victor.csv')

#PredictorData2019 = pd.read_csv(path+'PredictorData2019.csv')

chunked_df = pd.read_csv(file_and_path_for_big_dataset , chunksize=10**5) #chunksize is always 1?

for chunk in chunked_df:
    
#    print(chunk)
    
   ### parse entire csv
    try: #parse all
        
        df = pd.concat([df,chunk])
        
    except:
        
        df = chunk
    
#   ## parse only first chunk of df
#    df = chunk  #parse only first chunk
#    break

 
    
#df = pd.read_pickle("C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/Econ seminar/dataset_econ.pkl")

################################################################################################
### do something with data ###
################################################################################################

### get r as csv form data_victor:

print(data_victor)

data_victor['year_month'] = pd.to_datetime(data_victor['year_month'])

data_victor.rename(columns={'year_month':'DATE'}, inplace=True)

print(data_victor.dtypes)

data_victor = data_victor.sort_values(by=['DATE', 'permno'])



y = data_victor['ret_discrete']
#
#print(y)
#
### mean normalization (goes beyond 1 and -1)
##y = (y-y.mean())/y.std()
##
### min max normaization (only range [0,1])
y = (y-y.min())/(y.max()-y.min())
#
#print(y)
#
#y.to_csv(path+'r1.csv',index=False,header=False)

data_victor['ret_discrete'] = y

data_victor = data_victor.drop(['close','mktcap'], 1)

print(data_victor)

### end r as csv



### get macro data

#print(PredictorData2019)
    
### end macro data



### get firm characteristics ###

print(df)

#def slice_object(object):
#    
#    return object[]

#df['DATE'] = str(df['DATE'][0:7])

dates = []

for row in df.itertuples():
    
    dates.append(str(row.DATE)[0:7])
    
df['DATE'] = dates


print(df.dtypes)

df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d')) #very slow

permno_and_date = df[['permno','DATE']]

print(permno_and_date)

df.drop(['permno','DATE'],axis=1,inplace=True)

df = df.reindex(sorted(df.columns), axis=1)

df = pd.concat([permno_and_date, df], axis=1, sort=False)

print(df)

#df[0:50000].to_csv('50000samples_alphabetically.csv')

##c_it = df.loc[:, df.columns not in ['permno','DATE']]

#df[0:15000].to_csv('sample_larger.csv')

#df.to_pickle("dataset_econ.pkl")

### end firm charactericts



### merge firm characteristics and risk premium

#r_and_x = pd.merge(df, data_victor, how='outer', on=['DATE', 'Week', 'Colour'])

#print(r_and_x)


### end merge firm char and r










