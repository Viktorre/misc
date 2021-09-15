# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:59:56 2020

@author: user
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing


#path = 'C:/Users/fx236/Desktop/Spyder_code/'
path = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/Econ seminar/'

file_and_path_for_big_dataset = path+'data shihou gu/datashare.csv'






##### get returns r as csv form data_victor and make date editings etc
###
#

data_victor = pd.read_csv(path+'data_victor.csv')

print(data_victor)

data_victor['year_month'] = pd.to_datetime(data_victor['year_month'])

data_victor.rename(columns={'year_month':'DATE'}, inplace=True)

print(data_victor.dtypes)

data_victor = data_victor.sort_values(by=['DATE', 'permno'])

#y = data_victor['ret_discrete'] # normaliatzion is done later with sklearn
##
##print(y)
##
#### mean normalization (goes beyond 1 and -1)
###y = (y-y.mean())/y.std()
###
#### min max normaization (only range [0,1])
#y = (y-y.min())/(y.max()-y.min())
##
##print(y)
##
##y.to_csv(path+'r1.csv',index=False,header=False)
#
#data_victor['ret_discrete'] = y

data_victor = data_victor.drop(['close','mktcap'], 1)

print(data_victor)
#
###
##### end r as csv



##### get macro data
###
#
PredictorData2019 = pd.read_csv(path+'PredictorData2019.csv',thousands=',')

PredictorData2019['yyyymm'] = PredictorData2019.yyyymm.astype(int)*100+1

PredictorData2019['yyyymm'] = PredictorData2019['yyyymm'].astype(str)

PredictorData2019['DATE'] = pd.to_datetime(PredictorData2019['yyyymm'])#,format='%Y/%m/d')

PredictorData2019.drop(['yyyymm'],axis=1,inplace=True)

print(PredictorData2019)

print(PredictorData2019.dtypes)

#
###
##### end get macro data



##### get firm characteristics ### old
###
#


##def slice_object(object):
##    
##    return object[]
#
##df['DATE'] = str(df['DATE'][0:7])


#df = pd.read_csv(file_and_path_for_big_dataset)
#
#dates = []
#
#for row in df.itertuples():
#    
#    dates.append(str(row.DATE)[0:7])
#    
#df['DATE'] = dates
#
#
#print(df.dtypes)
#
#
##edit data:
#df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d')) #very slow
#
##df['DATE'] = df['DATE'].apply(lambda dt: dt.replace(day=1)) #faster
#
#print(df.dtypes)
#
#permno_and_date = df[['permno','DATE']]
#
#print(permno_and_date)
#
#df.drop(['permno','DATE'],axis=1,inplace=True)
#
#df = df.reindex(sorted(df.columns), axis=1)
#
#df = pd.concat([permno_and_date, df], axis=1, sort=False)


##df.to_pickle(path+"dataset_with_fixed_pandas_date_and_sorted_columns.pkl")


##shortcut (all steps above already done because slow):
#df = pd.read_pickle(path+'dataset_with_fixed_pandas_date_and_sorted_columns.pkl')


#
###
#### get firm characteristics old


##### get firm characteristics
###
#

#gu et al's datashare csv is problematic and pandas cannot parse it properly. this solves the problem without killing performance

df = pd.read_csv(path+'datashare_shortened.csv')

df['DATE'] = pd.to_datetime(df['DATE'])

print(df)

#
###
##### end firm charactericts ##


#
###
##### end firm charactericts ##

##### merge firm characteristics and returns
###
#
print(data_victor)

print(df)


r_and_x = pd.merge(df, data_victor, how='inner', on=['DATE', 'permno']) #jetzt nur join wo beide sets daten haben, al y udn x daten haben

#
###
##### merge firm characteristics and returns ##




##### create ind4stry dummies and save them to be added later
###
#
industry_dummies = (pd.get_dummies(r_and_x['sic2']))

r_and_x = r_and_x.drop( ['sic2'], axis=1) # drop for now because they dont need scaling
#
###
##### end indsstury dummies ###


#####  merge firm characteristics+risk premium and macro data
###
#
final_dataset = pd.merge(r_and_x,PredictorData2019, how='inner', on=['DATE']) #jetzt nur join wo beide sets daten haben, al y udn x daten haben

print(final_dataset)
#
###
##### end merge firm characteristics+risk premium and macro data #



##### create interactions between firm characteristics and macro factors
###
#
print(df.drop( ['DATE','sic2','permno'], axis=1).columns)

print(PredictorData2019.drop( ['DATE'], axis=1).columns)



for column_from_x in df.drop( ['DATE','sic2','permno'], axis=1).columns:
    
#    print(final_dataset[column_from_x].dtypes)
    
    for column_from_c in PredictorData2019.drop( ['DATE'], axis=1).columns:
        
        print(column_from_x,'*',column_from_c,final_dataset[column_from_x].dtypes,final_dataset[column_from_c].dtypes)

        final_dataset[column_from_x+'*'+column_from_c] = final_dataset[column_from_x] * final_dataset[column_from_c]


print(final_dataset)
#
###
##### end create interactions ##






##### scaler ###
###
#
columns = final_dataset.columns

permno_and_date = final_dataset[['permno','DATE']]

#x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#x_scaled = min_max_scaler.fit_transform(x)
#df = pd.DataFrame(x_scaled)

final_dataset = min_max_scaler.fit_transform(final_dataset.drop( ['permno','DATE'],axis=1)) # i dont see why asix=1 is needed, but it does not work without. for test df with two columns it works without

### bei ret_discrete 2. wert kommt ein negavitver wert raus, obwhol davor positiv... passt das? ich dneke schon...

final_dataset = pd.DataFrame(final_dataset)

final_dataset.columns = columns[2:]

final_dataset = pd.concat([permno_and_date,final_dataset, ], axis=1, sort=False)


print(final_dataset)
#
###
##### end scaler ##



##### get volatility as csv export 
###
#

# i do this before subtracting the tbill rate from the returns
df = final_dataset.set_index('DATE')

monthly_avg_ret_discrete = []
# monthly_avg_ret_discrete = pd.DataFrame([])
dates = []

for date in df.index.unique():

    
    print(df.loc[date]['tbl'].mean())
    
    dates.append(str(date)[:10])
    
    monthly_avg_ret_discrete.append(df.loc[date]['ret_discrete'].mean())
    # monthly_avg_ret_discrete[str(date)[:10] ] = float(df.loc[date]['ret_discrete'].mean())
    
# monthly_avg_ret_discrete = pd.DataFrame(monthly_avg_ret_discrete)

monthly_avg_ret_discrete = pd.DataFrame(monthly_avg_ret_discrete)
monthly_avg_ret_discrete.index = dates

print(monthly_avg_ret_discrete)

monthly_avg_ret_discrete.to_csv(path+'monthly_avg_ret_discrete.csv')

#
###
##### end get volatility


##### get volatility as csv export 
###
#

# i do this before subtracting the tbill rate from the returns
df = final_dataset.set_index('DATE')

monthly_avg_ret_discrete = []
# monthly_avg_ret_discrete = pd.DataFrame([])
dates = []

for date in df.index.unique():

    
    print(df.loc[date]['tbl'].mean())
    
    dates.append(str(date)[:10])
    
    monthly_avg_ret_discrete.append(df.loc[date]['ret_discrete'].mean())
    # monthly_avg_ret_discrete[str(date)[:10] ] = float(df.loc[date]['ret_discrete'].mean())
    
# monthly_avg_ret_discrete = pd.DataFrame(monthly_avg_ret_discrete)

monthly_avg_ret_discrete = pd.DataFrame(monthly_avg_ret_discrete)
monthly_avg_ret_discrete.index = dates

print(monthly_avg_ret_discrete)

monthly_avg_ret_discrete.to_csv('monthly_avg_ret_discrete.csv')

#
###
##### end get volatility


##### subtract tbill rate from returns
###
#
final_dataset['ret_discrete'] -= final_dataset['tbl']


#
###
##### end tbill subtract form returns





##### add industry dummies
###
#
final_dataset = pd.concat([final_dataset,industry_dummies ], axis=1, sort=False)
# print(final_dataset)
#
###
##### end add dummies ##


##### turn NA values into 0
###
#
print(sum(final_dataset.isna().sum()),'NA values')
final_dataset.fillna(float(0), inplace=True)
print(sum(final_dataset.isna().sum()),'NA values afteer fillna()')
#
###
##### end turn NA vals into 0


##### create multiindex
###
#

# df
print(final_dataset)



#
###
##### end mutliindex




##### export pkl
###
#

#final_dataset.to_pickle(path+"final_dataset.pkl")



#
###
##### end export pkl





















############################
### graveyard
####



#for column in r_and_x.columns: #normmalize all columns [0,1]
#    
#    if column not in ['DATE', 'permno']:
#        
##        r_and_x[column] = (r_and_x[column]-r_and_x[column].min())/(r_and_x[column].max()-r_and_x[column].min())
#        


