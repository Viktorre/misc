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
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width', 100000000)



#
###
##### end


##### impport excels
###
#

#path = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files'

#files = ['C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-10_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-11_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-12_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-13_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-16_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-17_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-18_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-19_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-19_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-20_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-21_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-24_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-25_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-26_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-27_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-02-28_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-02_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-03_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-04_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-05_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-06_0300_LON.xls',
# 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/spyder_code/EQ Files/EQ_Volatility_2020-03-09_0300_LON.xls']
#
#
#pd.read_excel(files[0])




#
###
##### end




#####
###
#

row = pd.read_pickle('row.pkl')

date =  row.index.values[0][-10:]

date =  pd.to_datetime(date)#, format='%Y%m%d')

date

new_cols = []

for col in row.columns:
    
#    print(date,pd.to_datetime(col[:11]))
    print(col[11:])
#    print((date - pd.to_datetime(col[:11])).days)
    
    

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




#####
###
#



#
###
##### end













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
