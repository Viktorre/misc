# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:50:37 2020

@author: cb2rtor
"""





##### imports and general settings
###
#

#import mdds 
import numpy as np
import pandas as pd
#import mdds_online as mdds
import matplotlib.pyplot as plt
pd.set_option('display.max_rows',15)
pd.set_option('display.max_columns', 4)
pd.set_option('display.width', 100000000)



#
###
##### end


##### define fct to full print dataframe
###
#
def fprint(df):
    pd.set_option('display.max_rows',1000)
#    pd.set_option('display.max_columns', 8)
    print(df)
    pd.set_option('display.max_rows',10)
#    pd.set_option('display.max_columns', 6)

#
###
##### end fprint def


##### impport excels
###
#
files = ['C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-25_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-26_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-27_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-28_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-02_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-03_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-04_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-05_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-06_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-09_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-10_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-11_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-12_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-13_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-16_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-17_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-18_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-19_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-19_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-20_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-21_0300_LON.xls', 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-24_0300_LON.xls']
#files = [
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-02-19_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-02-20_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-02-21_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-02-24_0300_LON.xls',       
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-02-25_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-02-26_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-02-27_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-02-28_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-02_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-03_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-04_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-05_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-06_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-09_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-10_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-11_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-12_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-13_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-16_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-17_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-18_0300_LON.xls',
# 'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdaten/EQ Files/EQ_Volatility_2020-03-19_0300_LON.xls'
#        ]

dfs = []

for file in files: #i already drop some columns that have no data
    dfs.append(pd.read_excel(file).drop(['Unnamed: 1','Unnamed: 13'], 1))
    
#
###
##### end import excels


##### define ados as dict
###
#

ados = {
'NIKKEI 225':'D#000000369318#JPY#PCE',
'DAX 30':'D#000000369306#EUR#PCE',
'E.ON SE':'D#000000199120#EUR#PCA',
'Hochtief AG':'D#000000077640#EUR#PCA',
'EURO STOXX 50':'D#000000369310#EUR#PCE',
'TUI AG':'D#000000078067#EUR#PCA',
'Amazon.com Inc.':'D#000000046446#USD#PCA',
'Citigroup Inc.':'D#000009605047#USD#PCA',
'Microsoft Corp.':'D#000000062120#USD#PCA',
'S&P 500':'D#000000369322#USD#PCE'
        }

#
###
##### end define ados as dict




##### def ine return_mat_days_as_mdds_unit
###
#

#mdds_mats = ['002W','001M','002M','003M','006M','009M', 
#             '001Y','002Y','003Y','005Y,'010Y']


def return_mat_days_as_mdds_unit(mat):
    
    if mat <= 22:
        return '002W'
    if 22 < mat <= 45:
        return '001M'
    if 45 < mat <= 75:
        return '002M'
    if 75 < mat <= 135:
        return '003M'
    if 135 < mat <= 225:
        return '006M'
    if 225 < mat <= 318: #here i did (365+270) / 2
        return '009M'
    if 318 < mat <= 548: #365 * 1.5
        return '001Y'
    if 548 < mat <= 913:
        return '002Y'
    if 913 < mat <= 1460: #365 * 4
        return '003Y'
    if 1460 < mat <= 2738: 
        return '005Y'
    if 2338 < mat:
        return '010Y'



#
###
##### end return_mat_days_as_mdds_unit


##### define fct to turn mdds objects back into floats and ints
###
#

def return_mdds_value(datapoint):
    
    return datapoint.value
    
return_mdds_value = np.vectorize(return_mdds_value) #performance increase
#
###
##### end

##### rearrange dataframes to get ipvols structure: main loop over instruments
###
#


length = 34
gap = 3
mm = ['050' ,'060' ,'070' ,'080' ,'090' , '100' ,'110' ,'120' ,'130' ,'140' ,'150' , ]
#ice = {}

start_rows_of_new_instrument_index_excel = [0,37,59,81,103,125,147,170,193,216]

##this is the main loop. eahc iteration is one instrument. loop goes through several code blocks
for index in start_rows_of_new_instrument_index_excel:
    
    ##### rearrange dataframes to get ipvols structure: rearrange one instrument
    ###
    #
        
    one_instrument_array = []
#    index = 37#start_rows_of_new_instrument_index_excel[1]
    
    for df in dfs: #each df contains maturity and moneyness for ONE DAY, dfs contain time dimension
    #    fprint(df)
    
        'vllt besser jeden df slice hier loopen um code zu sparen...'
        '''BUG: SOME TIME DIFFERENCES BETWEEN DATE AND MATURITY DATE ARE MAPPED INTO THE SAME 
            MDDS TIME SCHEME, LIKE 001M, WHICH CREATES DUPLICATES 
        '''
    #    print(df[index+0:index+1]['Unnamed: 2'].values[0],ados[df[index+0:index+1]['Unnamed: 2'].values[0]])
    #    print(df[index+0:index+1]['Unnamed: 14'].values[0])
        if index==0:#first instrument has 37 rows, all others have 20
            matrix = df[index+3:index+35].drop('Unnamed: 14', 1)
#            print(matrix)
        elif index>146: #citigroup and all instruments after that are one row longer
            matrix = df[index+3:index+21].drop('Unnamed: 14', 1)    
#            print(matrix)
        else:
            matrix = df[index+3:index+20].drop('Unnamed: 14', 1)        
        matrix = matrix.set_index('Unnamed: 0')
#        print(matrix)
    #    matrix = matrix.to_numpy()
        cols = mm #cols are unnames. cols must be moneyness, defined above
        rows = matrix.index.values #dates
    #    print('''bei rows defnintiv noch umrechnung zeitdiff zu df[0:1]['Unnamed: 14'].values[0] ''')
        new_cols = [] #column names for each row of flattened matrix
        for mat in rows:
            for moneyness in mm:
                new_cols.append(str(mat)+moneyness)
        
#        print(df[index+0:index+1])
#        print(df[index+0:index+1]['Unnamed: 2'].values[0])
        symbol_and_date_for_index = ados[df[index+0:index+1]['Unnamed: 2'].values[0]]+str(df[index+0:index+1]['Unnamed: 14'].values[0])[:10]
        pd_row = pd.DataFrame(dict(zip(new_cols, matrix.to_numpy().flatten())),index=[symbol_and_date_for_index])
#        print(pd_row)
    #    instrument = pd.concat([instrument, pd_row])
        one_instrument_array.append(pd_row)

    
    #    break
    #    instrument[df[0:1]['Unnamed: 14'].values[0]] = df[3:35].drop('Unnamed: 14', 1)
    #    print(df[37:38][['Unnamed: 2','Unnamed: 14']])
    #    print(df[40:57])
    #    print(df[59:60][['Unnamed: 2','Unnamed: 14']])
    #    print(df[62:79])
    #    print(df[81:82][['Unnamed: 2','Unnamed: 14']])
    #    print(df[84:101])
    #    print(df[103:104][['Unnamed: 2','Unnamed: 14']])
    #    print(df[106:123])
    #    print(df[125:126][['Unnamed: 2','Unnamed: 14']])
    #    print(df[128:145])
    #    print(df[147:148][['Unnamed: 2','Unnamed: 14']])
    #    print(df[150:168])
    #    print(df[170:171][['Unnamed: 2','Unnamed: 14']])
    #    print(df[173:191])
    #    print(df[193:194][['Unnamed: 2','Unnamed: 14']])
    #    print(df[196:214])
    #    print(df[216:217][['Unnamed: 2','Unnamed: 14']])
    #    print(df[219:237])
    #    break
        
        
    #
    ###
    ##### end rearrange ice data
    
    
    
    
    ##### use one_instrument_array to turn date diffs into days for maturity
    ###
    #
    
    #row = pd.read_pickle('row.pkl')
    #
    #date =  instrument.index.values[0][-10:]
    #
    #date =  pd.to_datetime(date)#, format='%Y%m%d')
    #
    #date
    #
    new_cols = []
    
    mats = []
    #
    one_instrument_array
    
    new_cols_for_row = []
    
    for row in one_instrument_array:
        
        date = row.index.values[0][-10:]
        
    
        date =  pd.to_datetime(date)#, format='%Y%m%d')
        
    #    new_cols_for_row = [] #moved on indent higher to give all rows same column of first row
        
    #    print(row)
        
        if len(new_cols_for_row) < 2:  #cond to give all rows same column of first row
            
            for col in row.columns:
        
            #    print(date,pd.to_datetime(col[:11]))
            ##    print(col[11:])
            #    print((date - pd.to_datetime(col[:11])).days)
            #    
#                print(row)
#                print(col[:11],'#############')
                maturity = str(abs((date - pd.to_datetime(col[:11])).days))
                
    #            print(maturity, return_mat_days_as_mdds_unit(int(maturity)))
                maturity = return_mat_days_as_mdds_unit(int(maturity))
                
    #            mats.append(int(maturity))
                
                new_cols_for_row.append('ICE_SN_'+maturity+col[11:])  
                    
                    
                    
            
        row.columns = new_cols_for_row
    
    'problem ist dass du viele mats und immer unterschiedliche tage. ich weiß nicht wie mdds das macht'
    
    #mats = pd.DataFrame(mats)
    
    #print(mats[0].sort_values().unique())
    
    #2 optionen: entweder mapping anhand funktion, oder einfach alles gleich behandeln mit mappin
    #von 1. row. letzteres machen!!!
    #
    ###
    ##### end  one_instrument_array to turn date diffs into days for maturity
    
    
    MORGEN MAL NOCH CHECKEN WARUM SNAPTIME UNSORTIERT WENN EINFACH F5 HIER; UND
    OB DAS EIN PROBLEM MACHT
    
    
    
    
    ##### turn maturities dates into day-intervals
    ###
    #
    
    #row = pd.read_pickle('row.pkl')
    #
    #date =  instrument.index.values[0][-10:]
    #
    #date =  pd.to_datetime(date)#, format='%Y%m%d')
    #
    #date
    #
    #new_cols = []
    #
    #mats = []
    #
    #for row in instrument.iterrows():
    #    
    ##    print(row)
    #    
    #    for col in instrument.columns:
    #        
    #    #    print(date,pd.to_datetime(col[:11]))
    #    ##    print(col[11:])
    #    #    print((date - pd.to_datetime(col[:11])).days)
    #    #    
    #        maturity = str(abs((date - pd.to_datetime(col[:11])).days))
    #        
    #    #    print(maturity)
    #        
    #        mats.append(maturity)
    #        
    #    #    print('SN'+maturity+col[11:])
    #    
    #
    #mats = pd.DataFrame(mats)
    #
    ##print(mats[0].unique())
    #
    #mdds_mats = ['002W','001M','002M','003M','006M','009M', 
    #             '001Y','002Y','003Y',]
    ##
    ###
    ##### end turn maturities dates into day-intervals
    
    
    
    
    
    
    
    
    
    
    
    ##### merge all rows in one instrument array into one dataframe
    ###
    #
    instrument = pd.DataFrame(None)
    
    for pd_row in one_instrument_array:
        
        instrument = pd.concat([instrument, pd_row])
    
        
    #print(instrument)
    #
    ###
    ##### end merge all rows in kikkei array into one dataframe
    
    
    
    ##### instrument df separate date from index to get structure like mdds data
    ###
    #
    def return_last_10_chars(val):
        return val[:-10]
    return_last_10_chars = np.vectorize(return_last_10_chars) #performance increase
    def return_all_but_10_last_chars(val):
        return val[-10:]
    return_all_but_10_last_chars = np.vectorize(return_all_but_10_last_chars) #performance increase
    
    
    instrument = instrument.reset_index()
    
    instrument['Symbol'] = return_last_10_chars(instrument['index'])
    
    instrument['SnapTime'] = return_all_but_10_last_chars(instrument['index'])
    
    instrument['SnapTime'] = pd.to_datetime(instrument['SnapTime'])
    
    instrument = instrument.drop(['index'],1)
    
    #instrument = instrument.set_index(['Symbol','SnapTime'])
    
    #print(instrument.dtypes)    
    
    #
    ###
    ##### end
    
    ##### short term bugfix: drop all duplicate columns, as same maturites are mapped several times into mdds maturities
    ###
    #
    
    instrument = instrument.loc[:,~instrument.columns.duplicated()]
        
    
    #
    ###
    ##### end


    print(instrument)

#
###
##### end: rearrange dataframes to get ipvols structure: main loop over instruments



print(stop)




###### download mdds data
####
##
#symbols = [
#'D#000000369318#JPY#PCE',
#'D#000000369306#EUR#PCE',
#'D#000000199120#EUR#PCA',
#'D#000000077640#EUR#PCA',
#'D#000000369310#EUR#PCE',
#'D#000000078067#EUR#PCA',
#'D#000000046446#USD#PCA',
#'D#000009605047#USD#PCA',
#'D#000000062120#USD#PCA',
#'D#000000369322#USD#PCE'
#        ]
#
#m = mdds.getIPVOLS(symbols, '20200201', '20200401', source="General")
#
#m = m.drop('ChangeComment', 1)
#
#print(m.columns)
#
#for column in m.columns:
#    try:
#        m[column] = return_mdds_value(m[column])
#    except:
#        pass
#
#def slice_str(val):
#    return val[:9]
#slice_str = np.vectorize(slice_str) #performance increase
#
#m['SnapTime'] = slice_str(m['SnapTime'])
#
#m['SnapTime'] = pd.to_datetime(m['SnapTime'])
#
##m = m.set_index(['Symbol','SnapTime'])
#
#
#
##
####
###### end




##### join m and instrument together without mulitindex 
###
#

instrument = instrument.set_index(['Symbol','SnapTime'])
m = m.set_index(['Symbol','SnapTime'])

inner = pd.concat([m, instrument], axis=1, join='inner')
    
print(inner)
##
###
##### end

##### loop through columns and get correlation
###
#

corrs = {}

for col in m.columns:
    
#    print('-----',col,'ICE_'+col)
    try:
        print(col,np.corrcoef(inner[col],inner['ICE_'+col])[0,1],'correlation')
        corrs[col] = np.corrcoef(inner[col],inner['ICE_'+col])[0,1] 
        
#        fig, ax = plt.subplots(1)
#        inner[col].plot(ax=ax)
#        inner['ICE_'+col].plot(ax=ax)
#        break
    except:
        print('################################################# failed')
              
#    print('correlation:',row.Index,np.corrcoef(plain_array_of_one_maturity_mdds,plain_array_of_one_maturity_ice)[0,1])
#
        
corrs = pd.DataFrame(corrs,index=['instrument 225'])

#print(corrs.T.values)

print('avg',corrs.T.mean())
    

#
###
##### end


#####
###
#


    

#
###
##### end


#####
###
#


    

#
###
##### end


#####
###
#


    

#
###
##### end    



##### print all cols for mdds data
###
#

for w in m.columns:
    
#    print(w)
    pass
    

#
###
##### end
    



################################################################################################





#
#
#
##markit_ipvol_cm = pd.read_excel('O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdatei/CM_Volatility_2019-11-11_0300_LON (2).xls').dropna(how='all')
#
##ice_ipvol_eq = pd.read_excel('O:\DE-O-01\GRM-MR_MT.MarketData\09_Home\ViktorReif\Python Scripts/EQ_Volatility_2020-01-10_0300_LON - Copy.xls')
##ice_ipvol_eq = pd.read_excel('O:\DE-O-01\GRM-MR_MT.MarketData\09_Home\ViktorReif\Python Scripts/ice.xlsx')
#
##        'O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdatei/EQ_Volatility_2020-01-10_0300_LON - Copy.xls').dropna(how='all')
#
##df_eq = pd.read_csv('O:/DE-O-01/GRM-MR_MT.MarketData/04_Analyse/ImpliedVols_Hagans_new_Vendors/ICE/Testdatei/xlsx_to_test_data.xls', encoding= 'unicode_escape',error_bad_lines=False)
# 
#print(ice_ipvol_eq)
#
#
##from mpl_toolkits.mplot3d import Axes3D
##import matplotlib.pyplot as plt
##from matplotlib import cm
#
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#
#
#
##ice_ipvol_eq.plot_surface()
#
##turn this into dataframe with a lot of emtpy, but sames as mdds_df
#
##for row in markit_ipvol_cm.itertuples():
##    
##    for column in markit_ipvol_cm.columns:
##        
##        print(row.column)
#
#
#print('')
#
#
#
#
###pd.set_option('display.max_rows',10)
##
####df = mdds.getIPVOLS('D#000003276909#XWH#PCA', '20191111', '20191111', source="General")
#df = mdds.getIPVOLS('D#000000196934#EUR#PCA', '20200111', '20200111', source="General")
#
#df = df.drop(['Symbol', 'SnapTime','ChangeComment'],1)
#
#df = df.T
#
#
#
#
#
#
#'''
#get this into same maturity moneyness format as in csv! und dann corr entlang beider axen( also mit .T)
#'''
#
#df.columns= ['values']
#
#new_df = pd.DataFrame([])
#
#for mat in ['06M','01Y','02Y','03Y','04Y','05Y','06Y','07Y','08Y','09Y','10Y']:
#    
#    one_column = []
#
#    for row in df.itertuples():        
#    
#        if row.Index[4:7] == mat:            
#            
#            one_column.append(row.values)
#    try:
#        new_df[mat] = one_column
#
#    except:
#        pass
#
#
#new_df = new_df.T   
#
#new_df.columns = ['070','080','90','100','110','120','140']
#
#
#ice_ipvol_eq = ice_ipvol_eq.drop(ice_ipvol_eq.columns[0:2],1)
#ice_ipvol_eq = ice_ipvol_eq.drop(ice_ipvol_eq.columns[-3],1)
#ice_ipvol_eq = ice_ipvol_eq.drop(ice_ipvol_eq.columns[-1],1)
#
#
#
#ice_ipvol_eq = ice_ipvol_eq.drop(['1W','6W','2M 10D','8M','1Y 2M 9D','1Y 5M','4Y','6Y','7Y','8Y','9Y'],0)
#
#ice_ipvol_eq.index = ['06M', '01Y', '02Y', '03Y', '05Y', '10Y']
#
#ice_ipvol_eq.columns = new_df.columns 
#
#print(ice_ipvol_eq,'ice_ipvol_eq has now columns and index as mdds dataframe')
#
#
##print(new_df.corrwith( ice_ipvol_eq, axis=0))
#
#
#
#print(new_df)
#
#print('\n')
#
#print('along moneyness:#################')
#for row in new_df.itertuples():
#    
##    print(row.Index)
#    
#    plain_array_of_one_moneyness_mdds = []
#    plain_array_of_one_moneyness_ice = []
#
#    
#    for column in new_df.columns:
#        
#        print(column, new_df[column][row.Index], ice_ipvol_eq[column][row.Index])
#
#        plain_array_of_one_moneyness_mdds.append( new_df[column][row.Index])
#
#        plain_array_of_one_moneyness_ice.append( ice_ipvol_eq[column][row.Index])
#
#    print('correlation:',row.Index,np.corrcoef(plain_array_of_one_moneyness_mdds,plain_array_of_one_moneyness_ice)[0,1])
#
#
#
#print('along maturity:#################')
#for column in new_df.columns:
#        
#    print(column)
#        
#    plain_array_of_one_maturity_mdds = []
#    plain_array_of_one_maturity_ice = []        
#        
#    for row in new_df.itertuples():
#            
#        print(row.Index,new_df[column][row.Index], ice_ipvol_eq[column][row.Index])
#        
#        plain_array_of_one_maturity_mdds.append( new_df[column][row.Index])
#
#        plain_array_of_one_maturity_ice.append( ice_ipvol_eq[column][row.Index])
#            
#    print('correlation:',row.Index,np.corrcoef(plain_array_of_one_maturity_mdds,plain_array_of_one_maturity_ice)[0,1])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
################ not relevant
##'fast test with hard copy/paste because corrwith does not work:'
##
##
##a1 = [ 0.309313,  0.287817 , 0.254358 , 0.230601,  0.208208 , 0.202530  ,0.204595]
##a2 = [34.87 , 29.94 , 26.07 , 23.17 , 22.31 , 23.12 , 26.57] 
##
##import numpy as np
##print(np.corrcoef(a1,a2), 'result corr along moneyness')
##
##a1 = [ 0.309313,  0.279456 , 0.260893 , 0.240517,  0.234009 , 0.238621 ]
##a2 = [34.87 , 30.90 , 27.97 , 26.72 , 25.57 ,  24.50 ] 
##
##import numpy as np
##print(np.corrcoef(a1,a2), 'result corr along maturities')
#
#
#
#
#
#
#
#
