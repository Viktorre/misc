# -*- coding: utf-8 -*-
"""
# *************************************************************************
 M   M DDDD  DDDD   SSSS
 MM MM D   D D   D S
 M M M D   D D   D  SSS
 M   M D   D D   D     S
 M   M DDDD  DDDD  SSSS


The Outlier_GUI Module:
This Module provides access to the MDDS timeseries data via
the COFI application and downloads the timeseries into Python DataFrame
objects. The data can then be interactively analysed. The results
can be visualized and exported.

USAGE:
When excecuted, the main window as well as a small login window are opened.
The login takes the user's normal AC client login credentials. After the login,
the usual steps of a data analysis are as followed:
    
1. Load data into python:
    
In the first segement of the left panel, the user can specify which instruments
shall be loaded. This is done either firstly by copy/pasting the wanted symbols 
(ado) from a csv, where all symbols are in a column, and thusly one column containing 
the symbols is pasted into the first text field next to "Commit Symbols", where
already one example symbol is set as the default text. Note, that the default
text in any entry field will disappear when clicked. Secondly, via 
"Commit Symbols Via CSV" the user can specify an input csv, where the program
will parse Symbols from its first column.
After the Commit, the user can specify dates and fields in the second line of
the first segment and then load the data via "import data"

2. Analyse data:
    
This step done in the second segment of the left panel. Here, each line gives
one analysis approach. The first is gives the n (selectable via integer input)
largest jumps in the data. Whether percent jumps, absolute jumps or difflog jumps
shall be regarded, can be specified via "Analysis Settings" under "change 
analysis method". The second line filters the data and returns all jumps that
are larger than are a certain threshold. However, the function will not regard
data points as outliers, if withing the next n days (specified by the entry
"how many jumps back") there is no strong jump (strongness specified by
"autoregressive factor") back to the original level. Only if the jump is larger
than threshold * "super suspect factor", then the jump is always treated as
outlier. The third method works similarly, but without the jump back functionality.

3. Visualize and Export results:
    
in the third and fourth segment the results can be visualized and exported. Plots
are shown in different tabs. There plots can be navigated and zoomend by the icons
underneath and also saved.
    
"""











# ---- this line blocks execution of script ----



import matplotlib
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg #toolbar und background um plots im fenster zu zeigen
except:
    #win10 needs different import
    from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
    NavigationToolbar2TkAgg = NavigationToolbar2Tk # in win10 names are slightly different
    
#import matplotlib
#matplotlib.use("Qt4agg")
    
import matplotlib.pyplot as plt

from matplotlib import style
#import matplotlib
#%matplotlib qt #to open plots in console and not as new 

import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as tkst
from tkinter.filedialog import asksaveasfilename
#from tkinter.ttk import Separator, Style
#from tkinter import *


import pandas as pd
import numpy as np

import sys 

import os

#import mdds as mdds
try:
    import mdds_online as mdds #edit: we use the web cofi interface to avoid the cofi.exe
except:
    pass
import warnings



#from threading import Thread
#from PIL import ImageTk, Image


warnings.filterwarnings('ignore')
#matplotlib.use("TkAgg") #backend of matplotlib
pd.set_option('display.width', 100000000) 
pd.set_option('display.max_rows',20)    
pd.set_option('display.max_columns', 6)

LARGE_FONT= ("Consolas", 42, 'bold') #fonts that are used to display text
BIG_FONT= ("Consolas", 18, 'bold') #fonts that are used to display text
THIS_FONT= ("Consolas", 16)
NORM_FONT = ("Consolas", 12)
SMALL_FONT = ("Helvetica", 10)
CONSOLE_FONT = ('Consolas',10)

style.use('ggplot') #style for plots   #ggplot, grayscale or default (reset kernel)

file_path = os.path.dirname(os.path.abspath(__file__))


#style = ttk.Style()
#style.configure("BW.TLabel", foreground="black", background="white")





real_console = sys.stdout  #we store the true console here to direct outputs back to it when gui is done. otherwisely console remains buggy as outputs are printed onto non existing widget
    
global_symbols = 'no symbols given'   #global variable that gets filled by retrieve_symbols_from_text_from_input(). 
    
global_show_inputs = False   #global bool to show or not show input symbols
 

#gdf ='no data loaded'  #global data frame
gdf = pd.read_pickle('gdf.pkl')[:4000]

#cdf = pd.DataFrame(None) #corrected dataframe

global_template_for_request = None

global_fields_for_request = None

global_selected_fields_for_request = [None]

global_snapshot_for_request = None

global_result =pd.DataFrame(None)

global_absolute_or_percent_or_logidff_setting_for_computing_change = 'pct_change'  #pct_change,abs_change,log_change are the 3 options available

global_serive_for_request = 'General' #other option is RiskGeneral

#global_result_n_largest_jumps = pd.DataFrame([None])

###global_result_jumps_larger_than = [None] #no longer in use

mb = None #placeholder for menubox widget

fig = None  #placeholder for plot_reset()

canvas = None #placeholder for plot_reset()


default_text_1 = True  #for functionality of disappearing text in input if clicked, but only first time

default_text_2 = True  #for functionality of disappearing text ...

default_text_3 = True

default_text_4 = True

default_text_5 = True

default_text_6 = True

default_text_entry_field_find_n_largest_abs_jumps = True

default_text_entry_field_for_threshold = True

default_text_entry_field_for_how_many_obs_back = True

default_text_entry_field_for_autoregressive_factor = True

default_text_entry_field_for_always_outlier_factor = True

default_text_entry_field_for_n_times_standart_deviation_threshold = True

default_text_entry_field_for_quantiles = True

default_text_entry_field_for_min_window_size = True

default_text_8 = True

default_text_9 = True

default_text_10 = True

default_text_11 = True

#coba_png_for_window_background = out = Image.open("coba.png").resize((128, 128))



def end():  #help function for "quit"-button. a direct command=quit , =break or =app.destroy did not work
    
    app.destroy()  #if main variable that stores the class that inherited tk.Tk is changed, then this line will no work
        

def re_login(): #pops up new window with button to close it
    
    mdds.login()



def popupmsg_old(msg): #pops up new window with button to close it
    
    popup = tk.Tk()  #define new tkinter object to get seperate window
    popup.wm_title('!')
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text='Ok', command = popup.destroy)
    B1.pack()
    popup.mainloop()
    


#def retrieve_symbols_from_text(label):  #curretntly not in use
#    
#    var = label.get("1.0",'end')  #get from first to end charactr
#    
#    all_symbols = []
#    
#    one_symbol = []
#    
#    for char in var:   #loop to parse symbols out of input field
#        
#        if char == '\n':
#            
#            one_symbol =  ''.join( one_symbol )
#            
#            if len( one_symbol ) > 2: #condition to aviod issues with double indents. symbols should be longer than two characters
#                all_symbols.append( one_symbol )
#            
#            one_symbol = [] 
#            
#        else:
#            one_symbol.append( char )
#            
#            
#    print(all_symbols)
#    
#    global global_symbols   #allows function to rewrite variable 
#    
#    global_symbols = all_symbols   #return statement does not work in button's command=... . global variable is needed as workaround




def show_inputs( global_symbols ):   #input_label

        try:
            label_text = '\n'.join(global_symbols[0:3])
            
        except:
            label_text = '\n'.join(global_symbols)
            
        label_text += ( '\n... \n' +  str( len(global_symbols) )+' Symbols in total' )

#        print( str( len(global_symbols) )+' Symbols in toal' )        

        if len( global_symbols ) > 0:
            
#            input_label.config(text=str(label_text) )
            
            print('')
            print( str(label_text) )

        


#def data_importer( global_symbols):
#    
#    try:
#            
#        import pandas as pd
#        
#        global_symbols = pd.DataFrame( global_symbols)
#        global_symbols.columns=['symbol']   #dfs column name must be named Symbol for get_cado()-call
#        
#        
#        global gdf
#        
#        try:
#            import mdds_ircurve_gui as mdds
#            
#            gdf = mdds.get_cado(global_symbols, 'test',1000,2,"H:\Cofi",'20100620', '20181114','cb2rtor', 'Alexander1!')  
#            
#        except:
#            print('import failed! loading test data')
#            
#    #        df = pd.DataFrame( {'one':[1,2,3,4,5,6,7,8,9],'two':[3,4,3,4,3,5,2,4,3] } )
#            
#    #        import numpy as np
#            
#            np.random.seed(42)
#            
#            gdf = pd.DataFrame(np.random.rand(10000,100))   #{'one':np.random.rand(1,1000), 'two':np.random.rand(1,1000)})
#            
#            gdf['const'] = 0.5
#            
#        print(gdf)
#
#    except:
#        
#        print( '' )
#        print( 'data import failed' )
#
#    print('\n \n \n')
#    
def multi_indexer(df):
###get index list for multiindex###
 
    meta_data=[]
    df['Index']=df.index
    df=df.set_index(['Symbol','Index'])  #creates multiindex as step to get index_list
#    print(df)
    for symbol, part_df in df.groupby(level=0,sort=False):
#        print(symbol,len(part_df))
        meta_data.append(symbol)
        meta_data.append(len(part_df))
#    print(meta_data) 
    index_list=[]
    for w in meta_data:
        if isinstance(w, int):
#            print(range(w))
            index_list.append(range(w))   #index_list=index_list+list(range(w)) does not work for some reason and needs next line to work
    flat_list = [item for sublist in index_list for item in sublist]
#    print(flat_list)
    df['index_list']=flat_list
    df.reset_index(inplace=True)  
####create_multiindex()###
    df=df.set_index(['Symbol','index_list'])
    df = df.drop('Index', 1)  #drops index column
    return df


def return_mdds_value(datapoint):
    
    return datapoint.value
    
return_mdds_value = np.vectorize(return_mdds_value) #performance increase


def data_importer( global_symbols,start_date,end_date,fields,all_available_fields,template, global_snapshot_for_request,button_make_boxplot_of_all_pct_jumps_per_day,max_number_of_ados_per_request=1000,number_of_sub_queries=10,service='General'):
    
    if fields == [None]:
        
        fields = all_available_fields
        
    print((global_symbols,template, start_date,end_date,service,fields))
    
    try:
        
        global gdf
        
        try:
#            print(global_symbols,template, start_date,end_date,"General",global_snapshot_for_request, fields)
            
            if len(global_symbols) <= max_number_of_ados_per_request:
                
#                print('full query')
                
                gdf = mdds.getInstruments(global_symbols,template, start_date,end_date,source=service,snapshot=global_snapshot_for_request, fields=fields)
                
            else:
                
                print('\n chained query \n')
        
                instruments_splitted=np.array_split(global_symbols, number_of_sub_queries)
                
                container_for_all_parts_of_gdf = pd.DataFrame([])
                
                for sub_list_of_global_symbols in instruments_splitted:
                    
#                    print(sub_list_of_global_symbols.tolist() )

                    
                    one_part_of_gdf = mdds.getInstruments(sub_list_of_global_symbols.tolist() ,template , start_date,end_date,source=service,snapshot=global_snapshot_for_request, fields=fields)
                    
#                    print(one_part_of_gdf)
                    
                    container_for_all_parts_of_gdf = pd.concat([container_for_all_parts_of_gdf, one_part_of_gdf])

                gdf = container_for_all_parts_of_gdf

#                    print(11111111111,gdf)

#            print(gdf.dtypes)
            #each value is actually a custom mdds_object with status etc. we only want the real value instead of unwieldy object
            for column in gdf.columns:
                try:
                    gdf[column] = return_mdds_value(gdf[column])
                except:
#                    print(column)
                    pass
#            print(gdf.dtypes)

            gdf = multi_indexer(gdf)  #viktor config
            
#            gdf = gdf.fillna(value=np.NaN)
            gdf = gdf.dropna()
            
            
            #turn datetime into pandas date without hours:
            gdf['SnapTime'] = gdf.apply(lambda x: x['SnapTime'][:8], axis = 1)
            gdf['SnapTime'] = pd.to_datetime(gdf['SnapTime'], format='%Y%m%d')
            
            
            
            try:  #many occurences of empty column called quotetime, see if this workaround has conflicts with somehtin else
                
                gdf = gdf.drop('QuoteTime', 1)
                
            except:
                
                pass
            
        except:#
            print('data import failed!')
            print( '')
            
    #        df = pd.DataFrame( {'one':[1,2,3,4,5,6,7,8,9],'two':[3,4,3,4,3,5,2,4,3] } )
            
    #        import numpy as np
            
#            np.random.seed(42)
#            
#            gdf = pd.DataFrame(np.random.rand(10000,100))   #{'one':np.random.rand(1,1000), 'two':np.random.rand(1,1000)})
#            
#            gdf['const'] = 0.5
            
        print(gdf)

    except:
        
        print( '' )
        print( 'data import failed' )

    print('\n \n \n')
    
#    button_make_boxplot_of_all_pct_jumps_per_day.pack(padx=10,pady=10,side='right') 
    
    
#def call_data_importer(start_date,end_date,thread_get_template, template):     #workaround for issue that calling thread earlier does not allow for passing arguemnts dynamically 
##    thread_get_template.join()
##    print('thread_get_template.join() evtl hier wieder raus') # avoid crashing, which occurs when second thread starts while fist is still running. join closes first thread  
#    global thread_data_importer  #global args are used freely. normal args need to be passed into function
#    thread_data_importer = t.Thread(target = data_importer, args = (global_symbols,start_date,end_date,global_fields_for_request, template))      
#    #entry field maybe global



def show_gdf_in_widget(gdf,  input_label):
            
            input_label.config(text=str(gdf) )



def open_popup_msg():
    None
#    print('fehlt, oder subprocess')


def close_popup_msg():
    None
#    print('fehlt, oder subprocess')

def first_day_deleter(value,index):  
    
#    print(index,value)
 
    if index == 0:
        
        return float(0.0)
        
    else:
        
        return value

first_day_deleter = np.vectorize(first_day_deleter)  #for better performance


def delete_all_computed_columns_from_df(df)     :
    try:
        
        for column in df.columns:
            
            if  column not in ['SnapTime','QuoteTime','ChangeComment','SOURCE_VADO']:
                
                for column_name_ending in ['_std', '_change', '_sdev_outlier']:
                    
                    if column_name_ending in column:
                
                        df = df.drop(column,1)
                
    except:
    
        pass       

    return df    

def calculate_pct_or_abs_change_of_df(df,which_kind_of_change='pct_change',abs_or_plus_minus_change='plus_minus',delete_computed_columns_from_before=True):
    
    if delete_computed_columns_from_before == True:
        
        df = delete_all_computed_columns_from_df(df)       

    computed_columns = []

    for column in df.columns:
        
        if 'change' not in column  and 'std' not in column and column not in ['SnapTime','QuoteTime','ChangeComment','SOURCE_VADO']:
            
            if which_kind_of_change == 'pct_change': #pct_change
                
#                print('pct change computed for',column)
        
                df[column+'_change'] = df[column].pct_change()
                
            if which_kind_of_change == 'abs_change': #absolute change
                
#                print('abs change')
                
                df[column+'_change']= df[column] - df[column].shift(1)
            
            if which_kind_of_change == 'log_change':
                
                plain_return=(df[column]) /(df[column].shift(1))
                
                df[column+'_change'] = np.log(plain_return)   #log_return ignores that there is no change for first value of new instrument
                
            if abs_or_plus_minus_change == 'plus_minus':
                
                pass  #leave negative changes in change column
        
            else:
                
                df[column+'_change'] =  df[column+'_change'].abs()
            
            
            df[column+'_change'] = first_day_deleter(df[column+'_change'],df.index.get_level_values(1))

            computed_columns.append( column + '_change')  # collect column names that change_column() computes and adds to dataframe
            
#        change_df= df[computed_columns]   
    
#        df = df.drop(computed_columns,1)   
    
#    else:
#        
#        print( [column for column in df.columns if "change" in column])
#        
#        print('re-calculation avoided to save time')
#        
#        computed_columns = [column for column in df.columns if "change" in column]

    return df


def function_that_takes_df_and_threshold_and_returns_result_with_outliers(df, threshold):

    change_columns = [ column for column in df.columns if 'change' in column ]
    
    largest_rows = df.nlargest( len(df) , columns=change_columns) #df[change_columns].abs().nlargest( len(df) , columns=change_columns)
    
    
#    print('largest_rows',largest_rows)
    
#    print(largest_rows.loc[largest_rows.ge(0.01, axis='index').any(1)])  #test... gets every row in largest_rows that has a value greater than 0.01
    
    largest_rows['max_value'] = largest_rows[change_columns].max(axis=1,skipna=False)  #10.7. added [computed_columns]

#        print( largest_rows[ computed_columns ].idxmax( axis=1 ) )
    largest_rows['max_column'] = largest_rows[ change_columns ].idxmax( axis=1 )

    largest_rows = largest_rows.sort_values(by=['max_value'], ascending=False)
    
#    print('largest_rows',largest_rows)
    
    if len(largest_rows.columns) < 255: #swvols have to many columns! itertuples caps at 255 columns
    
        result = pd.DataFrame([{'Symbol':row.Index[0], 'SnapTime':row.SnapTime,'index_list':row.Index[1] ,'Field':row.max_column[:-7], 'Change':row.max_value, 'Value':largest_rows.loc[row.Index, row.max_column[:-7]] } for row in largest_rows.itertuples()])
    else:
        
        print('too many columns! swvols need re programming.')
        
    result = result.reindex(columns=['Symbol','SnapTime','index_list','Field','Change','Value'] )

    result = result.loc[ (result['Change'].abs() > threshold ) & (result['Change'].abs() < np.inf)]
    
    return result


def function_that_eliminates_outliers_from_result_that_have_no_jump_back(df,result,threshold, wait_how_many_obs_for_jump_back, auto_regressive_factor=0.66, always_outlier_factor = 10):
    
    if wait_how_many_obs_for_jump_back == 0:  #if 0 jumps back is chosen, entire functin can be skipped which greatly enhances performance
        
        return result
    
    result.loc[result['index_list'] > 1]  #remove jumps from first to second day, ie from 0.00 to some value

    for one_row_of_result in result.itertuples():
        
        try: #here try is needed as indexing with obs_back can go beyond range of selected time series. try statement will skip those
        
            if one_row_of_result.Change > threshold * always_outlier_factor:  #if jump is very large, it will be flagged as otlier no matter how time series behaves the days after
                
                continue
            
    #        print('')
    
    #        print(one_row_of_result.Value ,one_row_of_result.Change,one_row_of_result.SnapTime, one_row_of_result.Symbol,one_row_of_result.Field ) #df[one_row_of_result.Field].loc[one_row_of_result.Symbol,one_row_of_result.index_list] 
            
            
    #        print('###',df[one_row_of_result.Field+'_change'].loc[one_row_of_result.Symbol,one_row_of_result.index_list-obs_back)
            
            
            for obs_back in range(1,wait_how_many_obs_for_jump_back+1):  #loop to check if any of the n next days has a similar jump back
                
    #           # print(obs_back)            
                
                if one_row_of_result.Value - df[one_row_of_result.Field].loc[one_row_of_result.Symbol,one_row_of_result.index_list - 1] > 0: #if positive jump, checked by comparing with day before, as result has only abs changes. df has positive and negative!
    
    #                
    #                print('positive jump',one_row_of_result.Symbol , one_row_of_result.Field, df['SnapTime'].loc[one_row_of_result.Symbol,one_row_of_result.index_list + obs_back],df[one_row_of_result.Field+'_change'].loc[one_row_of_result.Symbol,one_row_of_result.index_list + obs_back], one_row_of_result.Change * (-auto_regressive_factor))
    
    ##                
                    if not df[one_row_of_result.Field+'_change'].loc[one_row_of_result.Symbol,one_row_of_result.index_list + obs_back] <  abs(one_row_of_result.Change * auto_regressive_factor):
    #                    
    #                    print('there is no negative jump back, hence no outlier  #######################################################################################################')
                              
                        try:
                              
                            result = result.drop(index=one_row_of_result.Index)
    
                        except:
                            
                            pass
    
                    else:
                        
                        pass
    #                    print('remains outlier +++')
                        
                        
                else:
                    
    #                print('negative jump',one_row_of_result.Symbol , one_row_of_result.Field, df['SnapTime'].loc[one_row_of_result.Symbol,one_row_of_result.index_list + obs_back],df[one_row_of_result.Field+'_change'].loc[one_row_of_result.Symbol,one_row_of_result.index_list + obs_back], one_row_of_result.Change * (-auto_regressive_factor))
                    
                    if not df[one_row_of_result.Field+'_change'].loc[one_row_of_result.Symbol,one_row_of_result.index_list + obs_back] > abs(one_row_of_result.Change * auto_regressive_factor):
                        
    #                    print('there is no positive jump back,hence no outlier  #######################################################################################################')
                              
                        try:
                              
                            result = result.drop(index=one_row_of_result.Index)
    
                        except:
                            
                            pass
                    
                    else:
                        
                        pass
    #                    print('remains outlier +++')
    

        except:
            
            pass
        
    return result
    


def function_that_finds_all_outlier_that_are_larger_than_a_threshold_and_also_looks_for_jump_back(df,what_kind_of_change='pct_change',threshold=0.01,wait_how_many_obs_for_jump_back=2,auto_regressive_factor=0.66,always_outlier_factor=10):
    
    df = calculate_pct_or_abs_change_of_df(df,what_kind_of_change)
    
    result = function_that_takes_df_and_threshold_and_returns_result_with_outliers(df, threshold)
    
    print('result')
    len_of_result_before_jump_correction = len(result)
    
    result = function_that_eliminates_outliers_from_result_that_have_no_jump_back(df,result,threshold, wait_how_many_obs_for_jump_back,auto_regressive_factor,always_outlier_factor)
   
    print(len_of_result_before_jump_correction -  len(result),'outlier removed as time series no return to prior level')

#    print(result)
    
    global global_result #global_result_jumps_larger_than  #for export and plot
    
#    global_result = result #i concat current result with new ones. thus i can add found outliers from different methods into one result
    global_result = pd.concat([global_result,result]).drop_duplicates()
    
    print(global_result)    
    
    
#def change_column(df,field):  #maybe faster with NotImplementedError: 'first' only supports a DatetimeIndex index (using date as index...)
#    
#    
#    df[field+'_change']= abs(df[field] - df[field].shift(1))
#    
##    for isin, part_df in df.groupby(level=0,sort=False):
##        
##        df[field+'_change'].loc[part_df.index[0]] = 0.0    #first value of new instrument is treated as 0    
#    
#    return df


#def first_day_deleter(value,index):   #for better performance
#    
##    print(index,value)
# 
#    if index == 0:
#        
#        return float(0.0)
#        
#    else:
#        
#        return value
#
#first_day_deleter = np.vectorize(first_day_deleter)


        
def find_n_largest_abs_jumps(n, df,what_kind_of_change, column='LAST_PRICE'): 
    
    column = df.columns[1]
    
#    if any('change' in column for column in df.column)
    if not any('change' in column for column in df.columns): #avoid double computation of change columns
        
#        print('new')

        computed_columns = []
        
#        change_df = []
        
        for column in df.columns:
            
            if 'change' not in column and column not in ['SnapTime','QuoteTime','ChangeComment','SOURCE_VADO']:
            
    #            df[column+'_change']= abs(df[column] - df[column].shift(1))
                df = calculate_pct_or_abs_change_of_df(df,which_kind_of_change=what_kind_of_change  ,abs_or_plus_minus_change='abs',delete_computed_columns_from_before=False)
#                df[column+'_change']=df[column].pct_change().abs()
                
                
                df[column+'_change'] = first_day_deleter(df[column+'_change'],df.index.get_level_values(1))
    
                computed_columns.append( column + '_change')  # collect column names that change_column() computes and adds to dataframe
                
#        change_df= df[computed_columns]   
        
#        df = df.drop(computed_columns,1)   
#    else: #No longer used, as we added the option to change what kind of change is computed. Hence, re-calculation is always needed in
#        
#        print( [column for column in df.columns if "change" in column])
#        
#        print('re-calculation avoided to save time')
#        
#        computed_columns = [column for column in df.columns if "change" in column]
    
    
#    print(df)
#    print(pd.concat([df, change_df], axis=1, join='inner'))
#    change_df.loc[:,'SnapTime'] = df['SnapTime']  #df = df.astype(int)   #snaptime double
    
#    change_df = pd.concat([df, change_df], axis=1, join='inner')

#    print(change_df)
#    print(df)
#    largest_rows = df.nlargest( n , columns=computed_columns)
    largest_rows = df.nlargest( n , columns=column+'_change')#edit: only max vals of one column (chanes several lines afterwards. see older verion of gui)

    largest_rows = largest_rows.rename(columns={column:'max_value'})

#    largest_rows['max_value'] = largest_rows[column].max(axis=1,skipna=False)  #10.7. added [computed_columns]
    
#        print( largest_rows[ computed_columns ].idxmax( axis=1 ) )
#    largest_rows['max_column'] = largest_rows[ computed_columns ].idxmax( axis=1 )
    
#    largest_rows = largest_rows.sort_values(by=['max_value'], ascending=False)
    
#    print(largest_rows)

    result = []
    
    if len(largest_rows.columns) < 255: #swvols have to many columns! itertuples caps at 255 columns
    
        result = pd.DataFrame([{'Symbol':row.Index[0], 'SnapTime':row.SnapTime, 'Value':row.max_value} for row in largest_rows.itertuples()])
#        result = pd.DataFrame([{'Symbol':row.Index[0], 'SnapTime':row.SnapTime, 'Field':row.max_column[:-7], 'Change':row.max_value, 'Value':largest_rows.loc[row.Index, row.max_column[:-7]] } for row in largest_rows.itertuples()])
      
    else:
        
        print('too many columns! swvols and other instruments need re programming.')
        
    result = result.reindex(columns=['Symbol','SnapTime','Value'] )
    
#    print(result)       
    
    global global_result#global_result_n_largest_jumps  #for export
    
#    global_result = result #i concat current result with new ones. thus i can add found outliers from different methods into one result
    global_result = pd.concat([global_result,result]).drop_duplicates()
    
    print(global_result)    
    
    
# 
#def find_all_jumps_larger_than(threshold, df, gui=None):
#
#### change columns
#    
#    if not any('change' in column for column in df.columns): #avoid double computation of change columns
#        
##        print('new')
#
#        computed_columns = []
#        
##        change_df = []
#        
#        for column in df.columns:
#            
#            if 'change' not in column and column not in ['SnapTime','QuoteTime','ChangeComment','SOURCE_VADO']:
#            
#    #            df[column+'_change']= abs(df[column] - df[column].shift(1))
#                df[column+'_change']=df[column].pct_change().abs()
#                
#                
#                df[column+'_change'] = first_day_deleter(df[column+'_change'],df.index.get_level_values(1))
#    
#                computed_columns.append( column + '_change')  # collect column names that change_column() computes and adds to dataframe
#                
##        change_df= df[computed_columns]   
#        
##        df = df.drop(computed_columns,1)   
#    
#    else:
#        
#        print( [column for column in df.columns if "change" in column])
#        
#        print('re-calculation avoided to save time')
#        
#        computed_columns = [column for column in df.columns if "change" in column]
#          
#### end change columns     
#    
#    largest_rows = df.nlargest( len(df) , columns=computed_columns)
#
##    print(largest_rows)
#    
#    largest_rows['max_value'] = largest_rows[computed_columns].max(axis=1,skipna=False)  #10.7. added [computed_columns]
#
##        print( largest_rows[ computed_columns ].idxmax( axis=1 ) )
#    largest_rows['max_column'] = largest_rows[ computed_columns ].idxmax( axis=1 )
#
#    largest_rows = largest_rows.sort_values(by=['max_value'], ascending=False)
#    
#    if len(largest_rows.columns) < 255: #swvols have to many columns! itertuples caps at 255 columns
#    
#        result = pd.DataFrame([{'Symbol':row.Index[0], 'SnapTime':row.SnapTime, 'Field':row.max_column[:-7], 'Change':row.max_value, 'Value':largest_rows.loc[row.Index, row.max_column[:-7]] } for row in largest_rows.itertuples()])
#    else:
#        
#        print('too many columns! swvols need re programming.')
#
##    for row in largest_rows.itertuples():
##        
###        print(row.max_column[:-7])
##        print(row.Index[0])
##        print(largest_rows.loc[row.Index, row.max_column[:-7]])
##        print(largest_rows.loc[row.Index])
##    
##    for row in largest_rows.iterrows(): #slower iteration to use row['columnname']
##        
##        print(row[row['max_column'][:-7]])
#        
#        
#    result = result.reindex(columns=['Symbol','SnapTime','Field','Change','Value'] )
#
##    result = alle rows bei denen result['Change'] > threshold ist
#    result = result.loc[result['Change'] > threshold]   
#   
#
#
#    print(result)
#    
#    global global_result #global_result_jumps_larger_than  #for export and plot
#    
#    global_result = result
        
   
#        
def visualize_commited_symbols(all_symbols): 
    try:  #nice way to visualize input symbols
        
        for w in all_symbols[:3]:
            
            print(w)
            
        print('...\n')
        
        if len(all_symbols) > 6:
        
            for w in all_symbols[-3:]:
            
                print(w)
            
    except:
        
        for w in all_symbols:
            
            print(w)
        
    print('\n' +  str( len(all_symbols) )+' Symbols in total')
    
    print('\n')
    

def return_symbols_from_input(entry):
        
    var = str(entry.get())
    
    all_symbols = []
    
    one_symbol = []
    
    for char in var:   #loop to parse symbols out of input field

        if char in ['\n']:

            one_symbol =  ''.join( one_symbol )
            
            if len( one_symbol ) > 2: #condition to aviod issues with double indents. symbols should be longer than two characters
                all_symbols.append( one_symbol )
            
            one_symbol = [] 
            
        else:
            one_symbol.append( char )
    
    one_symbol =  ''.join( one_symbol )

    if len( one_symbol ) > 3:#to append last symbol when loop is over (entire string does not end with another \n)
                all_symbols.append( one_symbol )
    
    visualize_commited_symbols(all_symbols)            
    
    global global_symbols   #allows function to rewrite variable 
    
    global_symbols = all_symbols   #return statement does not work in button's command=... . global variable is needed as workaround

    print('\n')



def return_int_from_input(entry):
    
    var = entry.get()
    
    try:
        
        try:
            
            return int(var)     
    
        except:
            
            return float(var)
        
        
    except:
        
        print('')
        print('input must be a number')    


def return_string_from_input(entry):
    
    var = entry.get()
    
    try:
        
        return str(var)
        
    except:
        
        print('')
        print('input must be str')    

def return_dates_from_input(entry):
    
    var = entry.get()
    
    try:
        
        return str(var[0:8])     
    
    except:
        
        print('')
        print('input must be integer')    





def return_fields_from_input(entry):
    
    var = entry.get()
    
    try:
    
        var = "".join(var.split())
        
        return var.split(",")   
    
    except:
        
        print('')
        print('return_fields_from_input failed')    



def return_template_from_ados(ado_list):
    
#    import mdds_templates as templates
    
    global global_template_for_request
    
#    ados_and_templates = templates.get_templates(ado_list,'template_viktor','H:\Cofi', 'cb2rtor', 'Alexander1!')
    
    try: # workaround for issue that wrong login is not checked when app is started. if this import fails, most likely due to a wrong login
        
        ados_and_templates = mdds.getTemplatesBySymbol(ado_list)

        if len( ados_and_templates['InstrumentType'].unique() ) > 1:
            
            raise print("Error: Inputs contain different templates! New input requiered.")
        
    
        else:
            
            global_template_for_request = ados_and_templates['InstrumentType'].unique()[0]
    
            global_template_for_request = global_template_for_request[3:]  #delete mdds prefix. eg ot_ircvure --> ircurve
            
            print(global_template_for_request)

    except:
        
        print('')
        print('Re-enter login and commit again!')
        print('')        
        
        mdds.login()  # in the long run, the first login should use its own query to the database to check the password

        
#def call_thread_get_template():     #workaround for issue that calling thread earlier does not allow for passing arguemnts dynamically 
#    global thread_get_template
#    thread_get_template = t.Thread(target = return_template_from_ados, args = (global_symbols, )) 
#    

#    print(dir(t.current_thread))
#    assert app_thread is t.main_thread() #tests


##new call:
#thread_get_template.start()  #start thread
#thread_get_template.join()   #call for script to wait if thread is still running (not really needed in this script)









    


def return_fields_from_template(template):  #needs get_template thread to finish
    
    'it seems as iic and fts are missing in mdds_peter'
    
    global global_fields_for_request
    
    global global_snapshot_for_request
    
    if template == "IRCURVE":
        
        global_fields_for_request = ["CN_1D", "CN_2D", "CN_1W", "CN_1M", "CN_2M", "CN_3M", "CN_4M", "CN_5M", "CN_6M", "CN_7M", "CN_8M", "CN_9M", "CN_10M", "CN_11M", "CN_1Y", "CN_15M", "CN_18M", "CN_21M", "CN_2Y", "CN_3Y", "CN_4Y", "CN_5Y", "CN_6Y", "CN_7Y", "CN_8Y", "CN_9Y", "CN_10Y", "CN_11Y", "CN_12Y", "CN_13Y", "CN_14Y", "CN_15Y", "CN_16Y", "CN_17Y", "CN_18Y", "CN_19Y", "CN_20Y", "CN_21Y", "CN_22Y", "CN_23Y", "CN_24Y", "CN_25Y", "CN_26Y", "CN_27Y", "CN_28Y", "CN_29Y", "CN_30Y", "CN_35Y", "CN_40Y", "CN_45Y", "CN_50Y", "CN_55Y", "CN_60Y"]

    elif template == "SWVOLS":  #>255 fields! causes problem in comp core
        
        global_fields_for_request = ["SN_001M001Y", "SN_001M002Y", "SN_001M003Y", "SN_001M004Y", "SN_001M005Y", "SN_001M006Y", "SN_001M007Y", "SN_001M008Y", "SN_001M009Y", "SN_001M010Y", "SN_001M015Y", "SN_001M020Y", "SN_001M025Y", "SN_001M030Y", "SN_001Y001Y", "SN_001Y002Y", "SN_001Y003Y", "SN_001Y004Y", "SN_001Y005Y", "SN_001Y006Y", "SN_001Y007Y", "SN_001Y008Y", "SN_001Y009Y", "SN_001Y010Y", "SN_001Y015Y", "SN_001Y020Y", "SN_001Y025Y", "SN_001Y030Y", "SN_002Y001Y", "SN_002Y002Y", "SN_002Y003Y", "SN_002Y004Y", "SN_002Y005Y", "SN_002Y006Y", "SN_002Y007Y", "SN_002Y008Y", "SN_002Y009Y", "SN_002Y010Y", "SN_002Y015Y", "SN_002Y020Y", "SN_002Y025Y", "SN_002Y030Y", "SN_003M001Y", "SN_003M002Y", "SN_003M003Y", "SN_003M004Y", "SN_003M005Y", "SN_003M006Y", "SN_003M007Y", "SN_003M008Y", "SN_003M009Y", "SN_003M010Y", "SN_003M015Y", "SN_003M020Y", "SN_003M025Y", "SN_003M030Y", "SN_003Y001Y", "SN_003Y002Y", "SN_003Y003Y", "SN_003Y004Y", "SN_003Y005Y", "SN_003Y006Y", "SN_003Y007Y", "SN_003Y008Y", "SN_003Y009Y", "SN_003Y010Y", "SN_003Y015Y", "SN_003Y020Y", "SN_003Y025Y", "SN_003Y030Y", "SN_004Y001Y", "SN_004Y002Y", "SN_004Y003Y", "SN_004Y004Y", "SN_004Y005Y", "SN_004Y006Y", "SN_004Y007Y", "SN_004Y008Y", "SN_004Y009Y", "SN_004Y010Y", "SN_004Y015Y", "SN_004Y020Y", "SN_004Y025Y", "SN_004Y030Y", "SN_005Y001Y", "SN_005Y002Y", "SN_005Y003Y", "SN_005Y004Y", "SN_005Y005Y", "SN_005Y006Y", "SN_005Y007Y", "SN_005Y008Y", "SN_005Y009Y", "SN_005Y010Y", "SN_005Y015Y", "SN_005Y020Y", "SN_005Y025Y", "SN_005Y030Y", "SN_006M001Y", "SN_006M002Y", "SN_006M003Y", "SN_006M004Y", "SN_006M005Y", "SN_006M006Y", "SN_006M007Y", "SN_006M008Y", "SN_006M009Y", "SN_006M010Y", "SN_006M015Y", "SN_006M020Y", "SN_006M025Y", "SN_006M030Y", "SN_007Y001Y", "SN_007Y002Y", "SN_007Y003Y", "SN_007Y004Y", "SN_007Y005Y", "SN_007Y006Y", "SN_007Y007Y", "SN_007Y008Y", "SN_007Y009Y", "SN_007Y010Y", "SN_007Y015Y", "SN_007Y020Y", "SN_007Y025Y", "SN_007Y030Y", "SN_010Y001Y", "SN_010Y002Y", "SN_010Y003Y", "SN_010Y004Y", "SN_010Y005Y", "SN_010Y006Y", "SN_010Y007Y", "SN_010Y008Y", "SN_010Y009Y", "SN_010Y010Y", "SN_010Y015Y", "SN_010Y020Y", "SN_010Y025Y", "SN_010Y030Y", "SN_015Y001Y", "SN_015Y002Y", "SN_015Y003Y", "SN_015Y004Y", "SN_015Y005Y", "SN_015Y006Y", "SN_015Y007Y", "SN_015Y008Y", "SN_015Y009Y", "SN_015Y010Y", "SN_015Y015Y", "SN_015Y020Y", "SN_015Y025Y", "SN_015Y030Y", "SN_020Y001Y", "SN_020Y002Y", "SN_020Y003Y", "SN_020Y004Y", "SN_020Y005Y", "SN_020Y006Y", "SN_020Y007Y", "SN_020Y008Y", "SN_020Y009Y", "SN_020Y010Y", "SN_020Y015Y", "SN_020Y020Y", "SN_020Y025Y", "SN_020Y030Y", "SN_025Y001Y", "SN_025Y002Y", "SN_025Y003Y", "SN_025Y004Y", "SN_025Y005Y", "SN_025Y006Y", "SN_025Y007Y", "SN_025Y008Y", "SN_025Y009Y", "SN_025Y010Y", "SN_025Y015Y", "SN_025Y020Y", "SN_025Y025Y", "SN_025Y030Y", "SN_030Y001Y", "SN_030Y002Y", "SN_030Y003Y", "SN_030Y004Y", "SN_030Y005Y", "SN_030Y006Y", "SN_030Y007Y", "SN_030Y008Y", "SN_030Y009Y", "SN_030Y010Y", "SN_030Y015Y", "SN_030Y020Y", "SN_030Y025Y", "SN_030Y030Y"]

    elif template == "IPVOLS":
        
        global_fields_for_request = ["SN_002W070", "SN_002W080", "SN_002W090", "SN_002W100", "SN_002W110", "SN_002W120", "SN_002W140", "SN_001M070", "SN_001M080", "SN_001M090", "SN_001M100", "SN_001M110", "SN_001M120", "SN_001M140", "SN_002M070", "SN_002M080", "SN_002M090", "SN_002M100", "SN_002M110", "SN_002M120", "SN_002M140", "SN_003M070", "SN_003M080", "SN_003M090", "SN_003M100", "SN_003M110", "SN_003M120", "SN_003M140", "SN_006M070", "SN_006M080", "SN_006M090", "SN_006M100", "SN_006M110", "SN_006M120", "SN_006M140", "SN_009M070", "SN_009M080", "SN_009M090", "SN_009M100", "SN_009M110", "SN_009M120", "SN_009M140", "SN_001Y070", "SN_001Y080", "SN_001Y090", "SN_001Y100", "SN_001Y110", "SN_001Y120", "SN_001Y140", "SN_002Y070", "SN_002Y080", "SN_002Y090", "SN_002Y100", "SN_002Y110", "SN_002Y120", "SN_002Y140", "SN_003Y070", "SN_003Y080", "SN_003Y090", "SN_003Y100", "SN_003Y110", "SN_003Y120", "SN_003Y140", "SN_005Y070", "SN_005Y080", "SN_005Y090", "SN_005Y100", "SN_005Y110", "SN_005Y120", "SN_005Y140", "SN_010Y070", "SN_010Y080", "SN_010Y090", "SN_010Y100", "SN_010Y110", "SN_010Y120", "SN_010Y140"]
        
        global_snapshot_for_request = 'Europe3'  #take this out again, only for ad hoc analysis
        
    elif template == "FXVOLS":
        
        global_fields_for_request = ["SN_001D070","SN_001D080","SN_001D090","SN_001D100","SN_001D110","SN_001D120","SN_001D140","SN_001M070","SN_001M080","SN_001M090","SN_001M100","SN_001M110","SN_001M120","SN_001M140","SN_001W070","SN_001W080","SN_001W090","SN_001W100","SN_001W110","SN_001W120","SN_001W140","SN_001Y070","SN_001Y080","SN_001Y090","SN_001Y100","SN_001Y110","SN_001Y120","SN_001Y140","SN_002M070","SN_002M080","SN_002M090","SN_002M100","SN_002M110","SN_002M120","SN_002M140","SN_002Y070","SN_002Y080","SN_002Y090","SN_002Y100","SN_002Y110","SN_002Y120","SN_002Y140","SN_003M070","SN_003M080","SN_003M090","SN_003M100","SN_003M110","SN_003M120","SN_003M140","SN_003Y070","SN_003Y080","SN_003Y090","SN_003Y100","SN_003Y110","SN_003Y120","SN_003Y140","SN_004Y070","SN_004Y080","SN_004Y090","SN_004Y100","SN_004Y110","SN_004Y120","SN_004Y140","SN_005Y070","SN_005Y080","SN_005Y090","SN_005Y100","SN_005Y110","SN_005Y120","SN_005Y140","SN_006M070","SN_006M080","SN_006M090","SN_006M100","SN_006M110","SN_006M120","SN_006M140","SN_009M070","SN_009M080","SN_009M090","SN_009M100","SN_009M110","SN_009M120","SN_009M140","SN_010Y070","SN_010Y080","SN_010Y090","SN_010Y100","SN_010Y110","SN_010Y120","SN_010Y140"]

    elif template == "CFVOLS":
        
        global_fields_for_request = ["CN_1Y","CN_2Y","CN_3Y","CN_4Y","CN_5Y","CN_6Y","CN_7Y","CN_8Y","CN_9Y","CN_10Y"]

    elif template == "CERT":
        
        global_fields_for_request = ["LAST_PRICE", "SOURCE_VADO"]

    elif template == "BD":
        
        global_fields_for_request = ["LAST_PRICE", "SOURCE_VADO"]
        
    elif template == "FND":
        
        global_fields_for_request = ["LAST_PRICE", "SOURCE_VADO"]

    elif template == "FUT":
        
        global_fields_for_request = ["LAST_PRICE", "SOURCE_VADO"]

    elif template == "ILS":
        
        global_fields_for_request = ["ASK", "BID", "SOURCE_VADO"]

    elif template == "IRS":
        
        global_fields_for_request = ["ASK", "BID", "SOURCE_VADO"]
        
    elif template == "SWAPVOL":  #what is this?
        
        global_fields_for_request = ["ASK", "BID", "SOURCE_VADO"]

    elif template == "CAPFLRVOL":
        
        global_fields_for_request = ["LAST_PRICE", "SOURCE_VADO"]

    elif template == "FX":
        
        global_fields_for_request = ["ASK", "BID", "SOURCE_VADO"]#, "HIGH", "LOW"]? fields crash?
        
    elif template == "IROPTSKEW":
#        
        global_fields_for_request = ["Volatility","ATMNormVol"]  #if you request a field that the datafile of the wanted ado does not have, you get a zero dataframe!!!
        
        global_snapshot_for_request = 'America'   #we need this to be selectable in gui. here, this is only a workaround
        
    elif template == "STK":
        
        global_fields_for_request = ["LAST_PRICE", "SOURCE_VADO"]
#
#    elif template == "IRCURVE":
#        
#        global_fields_for_request = ["LAST_PRICE", "SOURCE_VADO"]
#    
    else:
        print('no fields implemented for this template!')
        
        
        
def get_fields_from_dropdown(widget): #should be placed as argument into get_data coll in order not to overwrite global_fields_for_request
    
    return None 


#def join_all_threads():
#    
#    for w in t.enumerate():
#        
#        if str(w) not in ['<_MainThread(MainThread, started 17548)>','<Thread(Thread-4, started daemon 15344)>','<Heartbeat(Thread-5, started daemon 4472)>','<HistorySavingThread(IPythonHistorySavingThread, started 14988)>','<ParentPollerWindows(Thread-3, started daemon 396)>']:
#            
#            print(w.join())


#def print_all_threads():
#    
#    for w in t.enumerate():
#        
#        print(w)


def export_results(df, directory_and_filename_in_one_array):
    
    
    if len(directory_and_filename_in_one_array[0]) > 0:
        
        print('')
        print('results exported to:',directory_and_filename_in_one_array[0])
        print('')
    
    directory_earlier = os.path.dirname(os.path.abspath(__file__))
    
    os.chdir(directory_and_filename_in_one_array[0])
    
    df.to_csv(directory_and_filename_in_one_array[1])

    os.chdir(directory_earlier)



#def thread_buffer_simple(n=5):  #tkinter crashes if thread.join() is used. this a simple workaround that assumes that the 6 default threads (see below) always run. If this fails,  for w in t.enumerate(): should be used!
#    '''
#<_MainThread(MainThread, started 3232)>
#<Thread(Thread-4, started daemon 12084)>
#<Heartbeat(Thread-5, started daemon 12460)>
#<HistorySavingThread(IPythonHistorySavingThread, started 12484)>
#<ParentPollerWindows(Thread-3, started daemon 12456)>
#<GarbageCollectorThread(Thread-6, started daemon 3976)>
#    '''
#    
#    while t.active_count() > n:
#        
#        print('buffering threads')
#        
#        time.sleep(0.5)


#def update_menu_box(container_dataimport,global_fields_for_request):
#    
#    global mb
#    
#    mb = ttk.Menubutton(container_dataimport, text='choose fields' )
#    mb.pack(side='left',padx=10,pady=10)
#    
#    mb.menu  = tk.Menu( mb, tearoff = 0 )
#    mb["menu"]  =  mb.menu
#    
#    if global_fields_for_request:
#        
#        for field in global_fields_for_request:
#            
##            exec(field+' = IntVar()')
#
#            mb.menu.add_checkbutton ( label=str(field), variable=tk.IntVar())
            






    
def get_selection_from_lstbox(lstbox):   #listboxes for field selection dialogue

    reslist = list()

    selection = lstbox.curselection()

    for i in selection:
        
        entry = lstbox.get(i)
        
        reslist.append(entry)
        
    return reslist


def get_listbox_selection_and_update_listbox_2(lstbox, lstbox_2):
    
    selected_fields = get_selection_from_lstbox(lstbox)
        
    for field in selected_fields:
        
        if field not in global_selected_fields_for_request and field not in list(lstbox_2.get(0, 'end')):  #field not selected earlier
            
            lstbox_2.insert('end', field)
        
        
def delete_selection_from_lstbox_2(lstbox_2):
    
    global global_selected_fields_for_request
    
    selection = lstbox_2.curselection()

    for i in selection:
        
        lstbox_2.delete(i)
        
    global_selected_fields_for_request = get_selection_from_lstbox(lstbox_2)  #update fields for input


def select_all_from_listbox_and_update(lstbox,lstbox_2):

    selected_fields = list(lstbox.get(0, 'end'))
        
    for field in selected_fields:
        
        if field not in global_selected_fields_for_request:
            
            lstbox_2.insert('end', field)
            
            

def clear_lstbox_2(lstbox_2):

    lstbox_2.delete(0,'end')
    



def assign_global_selected_fields_from_lstbox_2(lstbox_2):
    
    global global_selected_fields_for_request
    
    global_selected_fields_for_request = list(lstbox_2.get(0, 'end')) #tuple changed to list
    
    print('')
    
    print('selected fields:')
    
    print(global_selected_fields_for_request)



    
def popup_field_selection(global_fields_for_request, global_selected_fields_for_request): #pops up new window with button to close it

    window = tk.Tk()
    
    
#    print('hier noch icon rein')
#    window.iconbitmap(file)
    window.wm_title('Field Selection')
    
    container_top = tk.Frame(window)#, height=100, width=200)  
    container_bottom = tk.Frame(window, height=40)
        
    container_left = tk.Frame(container_top)#, height=100, width=200)
    container_right = tk.Frame(container_top)#, height=100, width=200)
    
    container_left_left = tk.Frame(container_left)#, height=100, width=200)
    container_left_right = tk.Frame(container_left)#, height=100, width=200)  
    
    container_right_left = tk.Frame(container_right)#, height=100, width=200)
    container_right_right = tk.Frame(container_right)#, height=100, width=200)  
    

#    container_left.pack_propagate(False)  #packing things inside this container will not change its size
    
    container_top.pack(side="top", fill="both", expand=True)
    container_bottom.pack(side="top", fill="both", expand=False)    

    container_left.pack(side="left", fill="both", expand=True)#, padx=10,pady=10) #poadding to counter console padding left
    container_right.pack(side="right", fill="both", expand=True)#, padx=10,pady=10)

    container_left_left.pack(side="left", fill="both", expand=True)#, padx=10,pady=10)
    container_left_right.pack(side="right", fill="both", expand=False)#, padx=10,pady=10)

    container_right_left.pack(side="left", fill="both", expand=True)#, padx=10,pady=10)
    container_right_right.pack(side="right", fill="both", expand=False)#, padx=10,pady=10)    
        
    
    lstbox = tk.Listbox(container_left_left,selectmode='multiple')
    lstbox_2 = tk.Listbox(container_right_left,selectmode='multiple')
    
    create_window = True
    
    try:
    
        for field in global_fields_for_request:
            
            lstbox.insert('end', field)
        
    except:
        
        create_window = False
        
    add_button = ttk.Button(container_left_right, text="add", command = lambda: [get_listbox_selection_and_update_listbox_2( lstbox, lstbox_2 )] )
    
    add_all_button = ttk.Button(container_left_right, text="add all", command = lambda: [ select_all_from_listbox_and_update(lstbox,lstbox_2)] ) 
    
    delete_button = ttk.Button(container_right_right, text="remove", command = lambda: [delete_selection_from_lstbox_2( lstbox_2 )] )
    
    delete_all_button = ttk.Button(container_right_right, text="remove all", command = lambda: [clear_lstbox_2( lstbox_2 )] )
    
    lstbox.pack(fill="both",expand=True,padx=5,pady=5)

    add_button.pack(fill="both",pady=20)    

    add_all_button.pack(fill="both",pady=20)

    lstbox_2.pack(fill="both",expand=True,padx=5,pady=5)
    
    delete_button.pack(fill="both",pady=20,padx=5) 
    
    delete_all_button.pack(fill="both",pady=20,padx=5)
    
    
    for field in global_selected_fields_for_request:
    
        lstbox_2.insert('end', field)

    
    close_button = ttk.Button(container_bottom, text="accept and close", command = lambda: [assign_global_selected_fields_from_lstbox_2(lstbox_2),window.destroy()] )
    close_button.pack(pady=20)
    
    
    if create_window == False:

        window.destroy()
        
        print('no fields available!')
        
    window.mainloop()
    

#def mdds_login_if_necessary(): #hier weiter gui bei commit nach passwort fragen, und wenn passowrt falsch in mdds_peter, dann passwort resetten
#    
#    if mdds.mdds_pass == None :
#        
#        print('start login()')
#        
#        mdds.login()
#        
#        print('logged in')



#### idea to put plots into seperate window
    
#def show_plot_window_global_result(global_result):  #creates new window, no longe in use
#    
#    try:
#        
#        dlg = PlotWindow(global_result)
#        
#        dlg.iconbitmap('graph.ico')
#        
#        dlg.lift()
#        
##        dlg.attributes("-topmost", True) #window will ignore other windows if clicked. bad practive for tkinter
#        
##        dlg.focus_force()
#        
#        dlg.mainloop()
#        
#    except:
#        
#        print('Plot failed')
    
def show_plot_in_other_tab(global_result,container_middle_tab_2):
    
    try:
        

        dist = pd.DataFrame(global_result['Symbol'].value_counts())
        
        dist['ados'] = dist.index
        
        dist.columns = ['Number of outliers','ADO']

        fig, ax = plt.subplots(1)
        
        canvas = FigureCanvasTkAgg(fig, container_middle_tab_2)  #tie plot to canvas, not console
        canvas.draw()  # make canvas visible
            
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  #canvas.show() needs to be packed to be shown
         
        
#        print(dist)
        
        dist.plot(x='ADO',y='Number of outliers',kind='bar',ax=ax) #plots suspect dist over instruments
        
        if len(dist)>35:  #if there are too many instruments, x-axis is cramped by too many symbols. in this case, axis will be hidden
                ax.get_xaxis().set_visible(False)
                
        toolbar = NavigationToolbar2TkAgg(canvas, container_middle_tab_2)
        
        toolbar.update()
    
        
    except:
        
        print('Plot failed')
        

def clear_canvas_in_plot_tab(canvas):
    
    try:
        
        print(canvas)
        canvas.delete("all")

    except:
        
        print('canvas delete failed')
        
        
def create_new_tab_for_each_show_plot_call(global_result,tab_bar_middle):
    
    try:
        
        container_middle_tab_3 = tk.Frame(tab_bar_middle)   #it is no problem to redefine same container
        container_middle_tab_3.pack(side="top", fill="both", expand=True)
        tab_bar_middle.add(container_middle_tab_3, text='  Plot-'+str(len(tab_bar_middle.tabs()))+'  ' )  

        button_close_tab = ttk.Button(container_middle_tab_3, text='close tab', command= lambda:  delete_tab_that_is_currently_selected(tab_bar_middle) )
        button_close_tab.pack(pady=0,padx=10,side='top')#,anchor=tk.NE)   

        dist = pd.DataFrame(global_result['Symbol'].value_counts())
        
        dist['ados'] = dist.index
        dist.columns = ['Number of outliers','ADO']

        fig, ax = plt.subplots(1)
        
        canvas = FigureCanvasTkAgg(fig, container_middle_tab_3)  #tie plot to canvas, not console
        canvas.draw()  # make canvas visible
            
#        print(dist)
        
        dist.plot(x='ADO',y='Number of outliers',kind='bar',ax=ax) #plots suspect dist over instruments
        
        if len(dist)>35:  #if there are too many instruments, x-axis is cramped by too many symbols. in this case, axis will be hidden
                ax.get_xaxis().set_visible(False)
                
        toolbar = NavigationToolbar2TkAgg(canvas, container_middle_tab_3)

        
        toolbar.update()
        
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  #canvas.show() needs to be packed to be shown
        
    except:
        
        print('Plot failed 3')       



def delete_tab_that_is_currently_selected(tab_bar):
    
    tab_bar.forget(tab_bar.select())


def show_time_series(df,symbol, date, field, n_entries_of_times_series_to_be_shown=20):  #allgmeine funktion, die von gdf anhand symbol und datum die n umliegenden tage als neuer df returnt
   
    instrument = df.xs(symbol, level='Symbol') #is always unique

    column_to_show_where_outlier_is = [item for sublist in [[' ']*n_entries_of_times_series_to_be_shown,['<--- outlier'],[' ']*n_entries_of_times_series_to_be_shown] for item in sublist]
    
    for row in instrument.itertuples():  #inefficient way to to match SnapTime field with wanted date
        
        if date in row.SnapTime:

            outlier_snippet = instrument[['SnapTime' , field ]].loc[row.Index-n_entries_of_times_series_to_be_shown : row.Index+n_entries_of_times_series_to_be_shown ].set_index('SnapTime')
            
#            print(len(outlier_snippet),len(column_to_show_where_outlier_is))
            
            try: 
                    
                outlier_snippet[' '] = column_to_show_where_outlier_is  #help column to visualize outlier
                
                print(outlier_snippet)
                
                outlier_snippet = outlier_snippet.drop(columns=[' '])
                
            except: #adding help column might fail
                
                print(outlier_snippet)
            
            return outlier_snippet
        

#            return (instrument[['SnapTime' , field ]].loc[row.Index-n : row.Index+n ] ).set_index('SnapTime') #if outlier date matches instrument date, n days around outlier are returned
                
#a = show_time_series(gdf,'O#000015642623', '20060706', field='CN_1D',n=5 )                  
#a.plot()
            
def show_rows_of_global_result(global_result, df, k):  #show first k results
    
    print('''available data: global result has all outlier, dist hast outlier quantity across instruments.
         for general it is useful to see date/instrument of n largest jumps (and height of jump)                 --> take n first rows from global result, still has issue
         for tullett it is more useful to see n largest jumps per instrument, or even within a certain timerange --> take symbol from dist and take n first matches from global_result, develop later
          ''')
    
    dist = pd.DataFrame(global_result['Symbol'].value_counts())
    
    dist['ados'] = dist.index
    
#    dist.columns = ['Number of outliers','ADO']
    
    print(dist)
    
    print('')
    
    
    for row in global_result.loc[0:k-1].itertuples(): # repsektiert hier nicht das k! loc[k] geht vllt nicht wiel index symbole und nicht zahlen???
        
        print(row.Symbol) #,row.SnapTime, row.Field)
        
#        print((show_time_series(df, row.Symbol,row.SnapTime, row.Field )))
        
#        for row in show_time_series(df, row.Symbol,row.SnapTime, row.Field).itertuples():
#           
#            print(row)
        
        show_time_series(df, row.Symbol,row.SnapTime, row.Field).plot()   #plots what called functions returns. called function also prints modified snippet without extra column which blocks plot
        
        print('')
    
    
#show_rows_of_global_result(global_result, gdf,3)           


def function_that_opens_save_file_dialog_window_and_returns_path_where_to_save_csv():
    
    path_and_file_name = tk.filedialog.asksaveasfilename()

#    print(path_and_file_name)

#    print(path_and_file_name[::-1].find('/'))
    
    position_of_last_slash = path_and_file_name[::-1].find('/')
    
    return [path_and_file_name[0:len(path_and_file_name)-position_of_last_slash],path_and_file_name[-(position_of_last_slash):]+'.csv']


def function_that_calls_functions_to_plot_suspect_dist_per_instrument_that_all_show_outlier_button_use(global_result,tab_bar_middle, button_export_as_csv):
    
    create_new_tab_for_each_show_plot_call(global_result,tab_bar_middle)    
    
    button_export_as_csv.pack(side='right',padx=10,pady=10)
    
    
def give_warning_if_gdf_has_no_data(gdf):

    if type(gdf) == str:  #for gdf
        
        print('No Data Available!')
    
    if gdf is None:  #for global result
        
        print('No Data Available!')


def turn_value_into_datetime_and_return_date(value):
    
    return pd.to_datetime(str(value.value)[:8], format='%Y%m%d')

turn_value_into_datetime_and_return_date = np.vectorize(turn_value_into_datetime_and_return_date)  #for better performance


def create_time_beam_that_shows_how_many_outliers_per_day(global_result, df):
    
    ''' function has issues pandas datetime, but works. difficult to hide hours in timestamp of date'''
        
    how_many_outliers_per_date = pd.DataFrame(global_result['SnapTime'].value_counts())
    
    all_unique_dates_in_gdf = pd.DataFrame(df['SnapTime'].unique())
        
    all_unique_dates_in_gdf.set_index(0, inplace=True)
    
    time_beam_that_shows_how_many_outliers_per_day = pd.concat([how_many_outliers_per_date, all_unique_dates_in_gdf], axis=1) #concat is outer join
    
#    time_beam_that_shows_how_many_outliers_per_day = time_beam_that_shows_how_many_outliers_per_day.reset_index()
    
#    print(time_beam_that_shows_how_many_outliers_per_day.dtypes)
    
#    print(pd.to_datetime((time_beam_that_shows_how_many_outliers_per_day['index'])))
    
#    print(time_beam_that_shows_how_many_outliers_per_day)
    
#    dates = []
#    
#    for date in time_beam_that_shows_how_many_outliers_per_day.index:
#        
##        dates.append(turn_value_into_datetime_and_return_date(date))
#        dates.append(pd.to_datetime(str(date.value)[:8]).to_pydatetime().date())
#
    
#    
#    time_beam_that_shows_how_many_outliers_per_day['pandas_date'] = dates    
    
#    time_beam_that_shows_how_many_outliers_per_day['pandas_date'] = time_beam_that_shows_how_many_outliers_per_day['pandas_date'].dt.date
    
#    time_beam_that_shows_how_many_outliers_per_day['pandas_date'] = time_beam_that_shows_how_many_outliers_per_day.index.astype('str')
# 
#    time_beam_that_shows_how_many_outliers_per_day['pandas_date'].astype()
    ### several try/except statements to give more robustness against date format issues. last try/except is most recent
    try:
        time_beam_that_shows_how_many_outliers_per_day['pandas_date'] = pd.to_datetime(time_beam_that_shows_how_many_outliers_per_day.index.str[0:8], format='%Y%m%d')#.dt.date
    except:
        pass
    try:
        time_beam_that_shows_how_many_outliers_per_day['pandas_date'] = pd.to_datetime(time_beam_that_shows_how_many_outliers_per_day.index.str[0:8], format='%Y%m%d').dt.date
    except:
        pass
    try:
        time_beam_that_shows_how_many_outliers_per_day['pandas_date'] = (time_beam_that_shows_how_many_outliers_per_day.index)
    except:
        pass


    time_beam_that_shows_how_many_outliers_per_day = time_beam_that_shows_how_many_outliers_per_day.fillna(value=0)
            
    time_beam_that_shows_how_many_outliers_per_day = time_beam_that_shows_how_many_outliers_per_day.set_index('pandas_date')
    
    time_beam_that_shows_how_many_outliers_per_day = time_beam_that_shows_how_many_outliers_per_day.rename({'SnapTime': 'Outlier'}, axis=1)
    
#    print(time_beam_that_shows_how_many_outliers_per_day) 
    
    return time_beam_that_shows_how_many_outliers_per_day




    
    
def plot_time_beam(time_beam_that_shows_how_many_outliers_per_day, tab_bar):
    
    fig, ax = plt.subplots(1)
    
    define_and_pack_frame_with_canvas_into_tab_bar(fig,tab_bar)  #missing fct, canvas neeed fig
    
    time_beam_that_shows_how_many_outliers_per_day.plot(style='-',ax=ax)
    
    
    
def define_and_pack_frame_with_canvas_into_tab_bar(fig,tab_bar):
    
    try:
        
        container_for_canvas = tk.Frame(tab_bar)   #it is no problem to redefine same container
        container_for_canvas.pack(side="top", fill="both", expand=True)
   
        tab_bar.add(container_for_canvas, text='  Plot-'+str(len(tab_bar.tabs()))+'  ' )  

        button_close_tab = ttk.Button(container_for_canvas, text='close tab', command= lambda:  delete_tab_that_is_currently_selected(tab_bar) )
        
        button_close_tab.pack(pady=0,padx=10,side='top')#,anchor=tk.NE)   
        
        canvas = FigureCanvasTkAgg(fig, container_for_canvas)  #tie plot to canvas, not console
        canvas.draw()  # make canvas visible
                
        toolbar = NavigationToolbar2TkAgg(canvas, container_for_canvas)
        
        toolbar.update()
    
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  #canvas.show() needs to be packed to be shown
        
        
    except:
        
        print('Plot failed 4')       


def scroll_to_bottom_in_console(console_widget):
        
    console_widget.see(tk.END)    




#def function_that_takes_gdf_and_returns_dataframe_with_dates_and_daily_change_of_each_instrument(df):
#    
#    print('maybe here i should recompute changes, but allow for negative changes')
#    
#    df = df.reset_index()
#
#    df['pandas_date'] = pd.to_datetime(df['SnapTime'].str[0:8], format='%Y%m%d')
#    
#    grouped_df = df.groupby(['pandas_date']).describe()
#    
#    print(grouped_df)
#
#    return grouped_df


#function_that_takes_gdf_and_returns_dataframe_with_dates_and_daily_change_of_each_instrument(gdf)
    
    
def function_that_takes_gdf_and_calculates_daily_pct_changes_without_abs_and_creates_box_plot_by_date(df,tab_bar):    
    
    print(df)

    for column in df.columns: 
        
        if 'change' in column:
            
            df = df.drop(column,1)
        
    for column in df.columns:
        
        if 'change' not in column and column not in ['SnapTime','QuoteTime','ChangeComment','SOURCE_VADO']:
        
            df[column+'_change']=df[column].pct_change() #no abs()!
            
            df[column+'_change'] = first_day_deleter(df[column+'_change'],df.index.get_level_values(1))

    
    df = df.reset_index()

#    df['pandas_date'] = pd.to_datetime(df['SnapTime'].str[0:8], format='%Y%m%d').dt.date  #.dt.date removes hours
#    dates = []
#    
#    for date in df['SnapTime']: #note that pydatetime turn pandas timestamp into python datetime object!!! (plot seems to work with python datetime object)
#        
##        dates.append(turn_value_into_datetime_and_return_date(date))
#        
#        dates.append(pd.to_datetime(str(date.value)[:8]).to_pydatetime().date())
#
#    
#    df['pandas_date'] = dates


    
#    print(df.groupby(['pandas_date']).describe()) #see boxplot as table more or less. only useful as interactive table
    
    
    fig, ax = plt.subplots(1)

#    fig.title(text='')
    
    column_name = 'LAST_PRICE_change'
    
    df.rename(columns={column_name:''}, inplace=True)
    
    df.boxplot(column='', by='SnapTime',ax=ax)  #creates boxplot  
    
    df.rename(columns={'':column_name}, inplace=True)
    
#    ax_copy = ax.twinx()
#    
#    ax_copy.tick_params(axis='x')
    
    
#    ax_x_2 = ax.twinx()
#    ax_y_2 = ax.twiny()


#    new_tick_locations = np.array([.2, .5, .9])
    

    
#    ax_2.set_xlim(ax.get_xlim())
#    ax2.set_xticks(new_tick_locations)
#    ax2.set_xticklabels(tick_function(new_tick_locations))
#    ax_2.set_xlabel(r"Modified x-axis: $1/(1+X)$")
    
    
#    for label in ax_copy.xaxis.get_ticklabels():
#        label.set_visible(False)
#
#    for label in ax_copy.xaxis.get_ticklabels()[:: round((df['pandas_date'].max() -df['pandas_date'].min()) /100 ) ]:
#        label.set_visible(True)
    
    
    define_and_pack_frame_with_canvas_into_tab_bar(fig,tab_bar)  #missing fct, canvas neeed fig
    



def function_that_opens_open_file_dialog_window_and_returns_path_where_to_open_csv():
    
    path_and_file_name = tk.filedialog.askopenfilename()

#    print(path_and_file_name)

    return path_and_file_name


def function_that_takes_path_and_file_name_and_parses_csv_to_rewrite_global_symbols(directory_and_file_name):
    
#    input_csv_only_first_column = pd.read_csv(directory_and_file_name,usecols=[0])
#    input_csv_only_first_column = pd.read_csv(directory_and_file_name,usecols=[0]).values.tolist()
    
    symbol_array = pd.read_csv(directory_and_file_name,usecols=[0]).values.tolist()
    
    symbol_array = [item for sublist in symbol_array for item in sublist]
    
    visualize_commited_symbols(symbol_array)
    
    global global_symbols
    
    global_symbols = symbol_array


def calculate_rolling_standard_deviation(df,rolling_window_size=255, min_window_size=1,delete_computed_columns_from_before= True):
    
    if delete_computed_columns_from_before == True:
        
        df = delete_all_computed_columns_from_df(df)
    
    for column in df.columns:
        
        if 'change' not in column  and 'std' not in column and '_sdev_outlier' not in column and column not in  ['Symbol','SnapTime','QuoteTime','ChangeComment','SOURCE_VADO']:
            
            print(column)
            
            df[column+'_std'] = df.groupby('Symbol')[column].apply(lambda x : pd.rolling_std(x,window=255,min_periods=min_window_size))            

    return df




def check_if_pct_change_is_larger_than_standard_devation_times_threshold(pct_change,standard_deviation, threshold):  
    
#    print(index,value)
 
    if abs(pct_change) > standard_deviation * threshold:
        
        return True
        
    else:
        
        return False

check_if_pct_change_is_larger_than_standard_devation_times_threshold = np.vectorize(check_if_pct_change_is_larger_than_standard_devation_times_threshold)  #for better performance

from sklearn.neighbors import LocalOutlierFactor

def find_outlier_by_lof():

    column = df.columns[1]
    
    X = gdf[column].values
    
    clf = LocalOutlierFactor(n_neighbors=2)
    clf.fit_predict(X)



def function_that_takes_df_with_outlier_columns_and_returns_result(df):
    
    
    df.reset_index(inplace=True)  

    result = pd.DataFrame([])
    
#    print(df)
    
    for column in df.columns:

#        print(column)
        
        if '_sdev_outlier' in column:
            
#            print(df.ix[(df[column]==True)])
            
            result = pd.concat([result,df.ix[(df[column]==True)]] )
    
    
#    result = df.ix[(df['type'] == True)]
#    result = pd.DataFrame([{'Symbol':row.Index[0], 'SnapTime':row.SnapTime,'index_list':row.Index[1] ,'Field':row.max_column[:-7], 'Change':row.max_value, 'Value':largest_rows.loc[row.Index, row.max_column[:-7]] } for row in largest_rows.itertuples()])    

    df=df.set_index(['Symbol','index_list'])    
    
    df = delete_all_computed_columns_from_df(df)
    
    
    return result


def function_that_finds_all_outliers_that_have_larger_pct_change_than_n_times_their_standard_deviation(df,which_kind_of_change='pct_change',n_times_standart_deviation_threshold=5,rolling_window_size=255,min_window_size=1):

    try:
        df=df.set_index(['Symbol','index_list']) 
    except:
        pass

    df = calculate_rolling_standard_deviation(df,rolling_window_size=255,min_window_size=min_window_size,delete_computed_columns_from_before=True)
    
    df = calculate_pct_or_abs_change_of_df(df,which_kind_of_change=which_kind_of_change,abs_or_plus_minus_change='plus_minus',delete_computed_columns_from_before=False)

    for column in df.columns:
        
        if 'change' not in column  and 'std' not in column and '_sdev_outlier' not in column and column not in  ['Symbol','SnapTime','QuoteTime','ChangeComment','SOURCE_VADO']:
            
#            print(column)
            
            df[column+'_sdev_outlier'] = check_if_pct_change_is_larger_than_standard_devation_times_threshold(df[column+'_change'],df[column+'_std'],n_times_standart_deviation_threshold)
    
#    print(df)
    
    for column in df.columns:

#        print(column)
        
        if '_sdev_outlier' in column:
            
            print(column,df[df[column] == True].shape[0])
        
        
    result = function_that_takes_df_with_outlier_columns_and_returns_result(df)
    
    result = result.drop_duplicates()
    
#    print(result)
    
    try:
        df=df.set_index(['Symbol','index_list']) 
    except:
        pass    

    global global_result #global_result_jumps_larger_than  #for export and plot
    
#    global_result = result #i concat current result with new ones. thus i can add found outliers from different methods into one result
    global_result = pd.concat([global_result,result]).drop_duplicates()

    print(global_result)

def uncheck_selections_of_other_options_in_menu_dropdown(one_string_var_to_stay_at_true, first_one_set_to_false,second_one_set_to_false):
    
    first_one_set_to_false.set(False)
    second_one_set_to_false.set(False)

def change_global_change_setting_according_to_what_menubox_selected(pct_change_selection,abs_change_selection,log_change_selection):

    if [pct_change_selection.get(),abs_change_selection.get(),log_change_selection.get()].count(True) > 1:
        
        print('Error')
    
    global global_absolute_or_percent_or_logidff_setting_for_computing_change   #pct_change,abs_change,log_change are the 3 options available
        
    if pct_change_selection.get():
        
        global_absolute_or_percent_or_logidff_setting_for_computing_change = 'pct_change'

    if abs_change_selection.get():
        
        global_absolute_or_percent_or_logidff_setting_for_computing_change = 'abs_change'

    if log_change_selection.get():
        
        global_absolute_or_percent_or_logidff_setting_for_computing_change = 'log_change'        
        
        
        
def change_global_serive_for_request_according_to_what_menubox_selected(general_service_selection,riskgeneral_service_selection):

    if [general_service_selection.get(),riskgeneral_service_selection.get()].count(True) > 1:
        
        print('Error')
    
    global global_serive_for_request   #two options available
        
    if general_service_selection.get():
        
        global_serive_for_request = 'General'

    if riskgeneral_service_selection.get():
        
        global_serive_for_request = 'RiskGeneral'


#def onclick(event): #function for saving clicks in graph
#    
#    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#          ('double' if event.dblclick else 'single', event.button,
#           event.x, event.y, event.xdata, event.ydata))



global_xy = [0,0]


def add_xy_coordinates(xy,value):
    
    return [xy[0]+value,xy[1]+value]

def callback(event): #function for saving clicks in graph
    global global_xy
#    print(global_xy)
    if global_xy in [[event.x, event.y],
                     add_xy_coordinates([event.x, event.y],1),
                     add_xy_coordinates([event.x, event.y],-1),    
                     add_xy_coordinates([event.x, event.y],2),
                     add_xy_coordinates([event.x, event.y],-2)
                     ]:
        print('doubleclick')
        print(event.x,event.y)
        print(event.xdata,event.ydata)
        global gdf
        gdf.at[('C#000000139629#JPY',1), 'LAST_PRICE'] = np.NaN
        
    else:
        print('click',global_xy)
    
    global_xy = [event.x, event.y]
    
    print(event.x,event.y)

    print(event.xdata,event.ydata)
#    for d in [ 'button', 'canvas', 'dblclick', 'guiEvent', 'inaxes', 'key', 'lastevent', 'name', 'step', 'x', 'xdata', 'y', 'ydata']:
#        print('print(event.',d,')')


def to_strftime(value):
    
    return str(value)[:10]

to_strftime = np.vectorize(to_strftime)


def  plot_entire_column(df,tab_bar_middle,column_name='LAST_PRICE',color='blue'):
        
    try:

        data = df 
        column_name = data.columns[1] #make this selectable later! here only takes first numeric col
        
        container_middle_tab_3 = tk.Frame(tab_bar_middle)   #it is no problem to redefine same container
        container_middle_tab_3.pack(side="top", fill="both", expand=True)
        tab_bar_middle.add(container_middle_tab_3, text='  Plot-'+str(len(tab_bar_middle.tabs()))+'  ' )  

        button_close_tab = ttk.Button(container_middle_tab_3, text='close tab', command= lambda:  delete_tab_that_is_currently_selected(tab_bar_middle) )
        button_close_tab.pack(pady=0,padx=10,side='top')#,anchor=tk.NE)   

        fig, ax = plt.subplots(1)
        
        canvas = FigureCanvasTkAgg(fig, container_middle_tab_3)  #tie plot to canvas, not console
        canvas.draw()  # make canvas visible
        
        
        
#        zeros.plot(x='SnapTime',y=column_name,ax=ax,color='red',kind='area')
#        data.plot(x='SnapTime',y=column_name,ax=ax,color=color ) #plots suspect dist over instruments
        data = data.reset_index()
#        data['SnapTime'] = to_strftime(data['SnapTime']) these two lines seem to do nothing
#        data['symb+date']=data['Symbol']+data['SnapTime']
#        pd.DataFrame(data[column_name]).plot(ax=ax,color='yellow')
        '1. plot mit ax, da kann man dann werte markieren'
        y=data[column_name]
        ax.plot(y,color=color,lw=0.75)#,markevery=0.425100)


#       #create frame with nans and outlier values in right position
        try: #this block of code will fail if global_result is empty, ie when no filter was applied prior
            zeros = pd.DataFrame(df['SnapTime'])
            zeros[column_name] = np.NaN
            zeros = zeros.reset_index()
            zeros['SnapTime'] = to_strftime(zeros['SnapTime'])
            zeros
            zeros = zeros.set_index(['Symbol','SnapTime'])
            zeros
            copy_result = global_result
            copy_result
            copy_result['SnapTime'] = to_strftime(copy_result['SnapTime'])#.apply(lambda x: x['SnapTime'][:8], axis = 1)
    #        zeros.at[row.Index, 'Volatility'] = 1000
            for row in global_result.itertuples():
                zeros.at[(row.Symbol,row.SnapTime), column_name] = row.Value
    #                break
    #        zeros = zeros.reset_index()
            zeros = zeros.reset_index()
            y_zeros = zeros[column_name]
    #        all_outliers_array = list(y_zeros.dropna().values)
            ax.plot(y_zeros,color='red',lw=0.75,marker='o',markevery=None)#,markevery=0.425100)        
        except:
            pass 

        '''
        noch extra button subplot(1,2) neue/alte ts übereinander!!!

        sdev debuggen
        
        pct change vllt 

        
        '''


#       ##plot vertical lines to separate instuments:
        index = 0
        
        data = data.set_index(['Symbol','index_list'])
        
        for symbol, part_df in data.groupby(level=0,sort=False):  
            #mark where symbols start by red lines
            ax.axvline(x=index,color='lightblue')
            ax.text(index, 0,symbol)
            index += len(part_df)
            
        
        ## plot corrected dataframe multiplied by 2 

        cdf = calcualte_corrected_dataframe(column_name)
        
        if len(cdf) > 2: #condition to not plot cdf if no corrections done yet
            y_cdf= cdf.reset_index()[column_name]
##            y_cdf
            y_cdf -= gdf[column_name].mean()
            'hier besser subtraktion noch rein anstatt faktor'
            ax.plot(y_cdf,color='green',lw=0.75)#,markevery=0.425100)  
            
                
        toolbar = NavigationToolbar2TkAgg(canvas, container_middle_tab_3)

        toolbar.update()
        
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  #canvas.show() needs to be packed to be shown
        
    except:
        
        print(' plot_entire_column() failed')  
        
        
def calcualte_corrected_dataframe(column_name):
    
    global global_result
    
    global gdf
    
#    col = pd.DataFrame(gdf['SnapTime'])
#    col[column_name] = np.NaN
#    col = col.reset_index()
#    col['SnapTime'] = to_strftime(col['SnapTime'])
    cdf = gdf.reset_index()
#    cdf
    cdf = cdf.set_index(['Symbol','SnapTime'])
#    cdf
    copy_result = global_result
    copy_result
    copy_result['SnapTime'] = to_strftime(copy_result['SnapTime'])#.apply(lambda x: x['SnapTime'][:8], axis = 1)
#        zeros.at[row.Index, 'Volatility'] = 1000
    for row in global_result.itertuples():
        cdf.at[(row.Symbol,row.SnapTime), column_name] = np.NaN   
    
    cdf[column_name] = cdf[column_name].interpolate()
    
    return cdf


#def plot_each_instrument_seperately(plot_fct, df,tab_bar_middle,column_name='LAST_PRICE'):
#
#    for symbol, part_df in df.groupby(level=0,sort=False):  
#
#        plot_fct(part_df,tab_bar_middle,column_name=column_name)
#        
#        
        
#def plot_all_instruments_as_subplots(data, tab_bar_middle, column_name='LAST_PRICE'):
#    
#        column_name = data.columns[1] #make this selectable later! here only takes first numeric col
#    
#        container_middle_tab_3 = tk.Frame(tab_bar_middle)   #it is no problem to redefine same container
#        container_middle_tab_3.pack(side="top", fill="both", expand=True)
#        tab_bar_middle.add(container_middle_tab_3, text='  Plot-'+str(len(tab_bar_middle.tabs()))+'  ' )  
#
#        button_close_tab = ttk.Button(container_middle_tab_3, text='close tab', command= lambda:  delete_tab_that_is_currently_selected(tab_bar_middle) )
#        button_close_tab.pack(pady=0,padx=10,side='top')#,anchor=tk.NE)   
#
#        number_of_instruments = len(data.index.get_level_values(0).unique())
##        fig, tuple_of_axes = plt.subplot(1, number_of_instruments) 
#        fig, axes = plt.subplots(nrows=number_of_instruments, ncols=1)
#        
#        canvas = FigureCanvasTkAgg(fig, container_middle_tab_3)  #tie plot to canvas, not console
#        canvas.draw()  # make canvas visible
#        
#        counter = 0
#
#        for symbol, part_df in data.groupby(level=0,sort=False):  
#            
#            part_df.plot(x='SnapTime',y=column_name,ax=axes[counter])  
#            counter += 1



#fig1 = plt.figure()
#plt.plot(x,y)
#plt.show(fig1)

                
#        toolbar = NavigationToolbar2TkAgg(canvas, container_middle_tab_3)
#
#        
#        toolbar.update()
#        
#        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  #canvas.show() needs to be packed to be shown
#        
        
        
def return_gdf_with_all_outliers_dropped(df):
    
    return df


def delete_all_found_outliers():
    
    global global_result
    
    global_result = pd.DataFrame(None)
    
    print('all prior outliers removed','\n')


def find_all_outlier_by_quantile(df,quantile,column_name = 'LAST_PRICE'):
    
    column_name = df.columns[1]
    
    result = pd.DataFrame(None)
    
    for symbol, part_df in df.groupby(level=0,sort=False):
        
#        print(part_df[column_name].quantile(quantile))
#        print(part_df[column_name].quantile(1-quantile))
 
        outlier = part_df.loc[(part_df[column_name] < part_df[column_name].quantile(quantile)) | (part_df[column_name] > part_df[column_name].quantile(1-quantile))]
        outlier = outlier.reset_index()[['Symbol','SnapTime',column_name]]
        outlier = outlier.rename(columns={column_name: 'Value'})
#        print(outlier)

        result = pd.concat([result,outlier])    
        
    
    global global_result #global_result_jumps_larger_than  #for export and plot
#    
#    global_result = result #i concat current result with new ones. thus i can add found outliers from different methods into one result
    global_result = pd.concat([global_result,result]).drop_duplicates()
    
    print(global_result)   
    
    
#    
#    
##    ##z score
##    print(part_df)
###    part_df.plot(x='SnapTime' , y='Volatility')
##    print(part_df[(np.abs(stats.zscore(part_df['Volatility'])) < 1)]  )
###    print(len(part_df)-len(part_df[
###            (np.abs(stats.zscore(part_df['Volatility'])) < 3)]),'outlier found')
##
##    break
###        
##        
#        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
###########################################################################################################################################################################################################################################

#class PlotWindow(tk.Tk):
#    def __init__(self,global_result):
#        global mdds_user
#        global mdds_pass
#        tk.Tk.__init__(self)
#
#        self.title("Plots")
#
#        fig, ax = plt.subplots(1)
#        
#        canvas = FigureCanvasTkAgg(fig, self)  #tie plot to canvas, not console
#        canvas.draw()  # make canvas visible
#            
#        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  #canvas.show() needs to be packed to be shown
#        
#        button4 = ttk.Button(self, text='quit', command=self.destroy) #lambda: end does not work
#        button4.pack(pady=10,padx=10)
#         
#        dist = pd.DataFrame(global_result['Symbol'].value_counts())
#        
#        dist['ados'] = dist.index
#        
#        dist.columns = ['Number of outliers','ADO']
#        
##        print(dist)
#        
#        dist.plot(x='ADO',y='Number of outliers',kind='bar',ax=ax) #plots suspect dist over instruments
#        
#        if len(dist)>35:  #if there are too many instruments, x-axis is cramped by too many symbols. in this case, axis will be hidden
#                ax.get_xaxis().set_visible(False)
#                




class ViktorsApp(tk.Tk): #alles von tk.Tk erben
    
    def __init__(self, *args, **kwargs): # definiere methode die immer am anfang ausgeführt wird
        
        tk.Tk.__init__(self, *args, **kwargs)  # init tk.Tk to get all in its __init__()
        
#        self.lift() # assign screen to be in foreground of screen within object definition. did not help issue
#        tk.Tk.iconbitmap(self,default='graph.ico') 
        tk.Tk.wm_title(self, 'Viktor´s Application')
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)  #seems to act like a "deploy"
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)
        
        
        menubar = tk.Menu(container)  #define menubar variable as tk object
        menu_for_change_login = tk.Menu(menubar, tearoff=0) #tearoff=able to turn dropdown into own window
        menu_for_change_login.add_command(label='change login', command= lambda : re_login())
        menu_for_change_login.add_separator()
        menu_for_change_login.add_command(label='quit', command=end)
        menubar.add_cascade(label='User',menu=menu_for_change_login) #string menubar and filemenu together
        
        menu_for_data_anaylsis_settings = tk.Menu(menubar, tearoff=0)
#        menu_for_data_anaylsis_settings.add_command(label='change analysis method', command= lambda : change_global_var_what_kind_of_change())
        menubar.add_cascade(label='Analysis Settings',menu=menu_for_data_anaylsis_settings)
        
        
        pct_change_selection = tk.BooleanVar()
        pct_change_selection.set(True)
        abs_change_selection = tk.BooleanVar()
        log_change_selection = tk.BooleanVar()
        
        sub_menu_for_choosing_what_kind_of_change = tk.Menu(menu_for_data_anaylsis_settings)
        
        sub_menu_for_choosing_what_kind_of_change.add_checkbutton(label="percent change", onvalue=1, offvalue=0, variable=pct_change_selection,command= lambda :  [
                                                                    uncheck_selections_of_other_options_in_menu_dropdown(pct_change_selection,abs_change_selection,log_change_selection),
                                                                    change_global_change_setting_according_to_what_menubox_selected(pct_change_selection,abs_change_selection,log_change_selection)
                                                                    ])
        sub_menu_for_choosing_what_kind_of_change.add_checkbutton(label="absolute change", onvalue=1, offvalue=0, variable=abs_change_selection,command= lambda :  [   
                                                                    uncheck_selections_of_other_options_in_menu_dropdown(abs_change_selection,log_change_selection,pct_change_selection),
                                                                    change_global_change_setting_according_to_what_menubox_selected(pct_change_selection,abs_change_selection,log_change_selection)
                                                                    ])
        sub_menu_for_choosing_what_kind_of_change.add_checkbutton(label="logdiff change", onvalue=1, offvalue=0, variable=log_change_selection,command= lambda :  [
                                                                    uncheck_selections_of_other_options_in_menu_dropdown(log_change_selection,pct_change_selection,abs_change_selection),
                                                                    change_global_change_setting_according_to_what_menubox_selected(pct_change_selection,abs_change_selection,log_change_selection)
                                                                    ])        
        menu_for_data_anaylsis_settings.add_cascade(label='change analysis method',menu=sub_menu_for_choosing_what_kind_of_change)


        general_service_selection = tk.BooleanVar()
        general_service_selection.set(True)
        riskgeneral_service_selection = tk.BooleanVar()
        
        sub_menu_for_choosing_which_service_for_request = tk.Menu(menu_for_data_anaylsis_settings)
        
        sub_menu_for_choosing_which_service_for_request.add_checkbutton(label="General", onvalue=1, offvalue=0, variable=general_service_selection,command= lambda :  [
                                                                    uncheck_selections_of_other_options_in_menu_dropdown(general_service_selection,riskgeneral_service_selection,riskgeneral_service_selection),
                                                                    change_global_serive_for_request_according_to_what_menubox_selected(general_service_selection,riskgeneral_service_selection)
                                                                    ])
        sub_menu_for_choosing_which_service_for_request.add_checkbutton(label="RiskGeneral", onvalue=1, offvalue=0, variable=riskgeneral_service_selection,command= lambda :  [   
                                                                    uncheck_selections_of_other_options_in_menu_dropdown(riskgeneral_service_selection,general_service_selection,general_service_selection),
                                                                    change_global_serive_for_request_according_to_what_menubox_selected(general_service_selection,riskgeneral_service_selection)
                                                                    ]) 
        menu_for_data_anaylsis_settings.add_cascade(label='change service for request',menu=sub_menu_for_choosing_which_service_for_request)       
        
        
        
        
        
        tk.Tk.config(self,menu=menubar)  #deploy menubar
        
        
        self.frames = {}  #dictionary that contains pages
        
        for F in (PageOne, PageTwo):#StartPage,PageThree is kicked out   #iterations that saves page-classes into dictionary
        
            frame = F(container, self)
            
            self.frames[F] = frame
            
            frame.grid(row=0, column=0, sticky="nsew")
        

        self.show_frame(PageOne)   #shows wanted frame as default, ie page class#
        
        

        
        
# #   ######
# #   def refresh(self):   #idea mutlithreading
# #       self.root.update()
#  ##      self.root.after(1000,self.refresh)
##
##    def start(self):
##        self.refresh()
###        t.Thread(target=test).start()        
#    ######    
        
        
    def show_frame(self, cont):   #takes one page of frames dict and raises it to front of window
        
        frame = self.frames[cont]
        frame.tkraise()
        
    def full_screen(self):
        
        self.wm_state('zoomed')
    
    



class PageOne(tk.Frame):
        
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
##### main structure with tab bars ########
        
#        container_top = tk.Frame(self, height=30)#,bg='lightgrey')#, height=100, width=200)
# replaced by notebook directly       container_middle = tk.Frame(self)#, bg='lightblue')#, height=100, width=200)        
        container_bottom = tk.Frame(self)#,height=50)#,bg='lightgrey')#, height=100, width=200)        
        
        tab_bar_middle = ttk.Notebook(self) #notebook acts like a frame. hence needs to take the spot of one of the main frames
        
        tab_bar_middle.enable_traversal()
        
        
        container_middle_tab = tk.Frame(tab_bar_middle)#, height=100, width=200)   
        container_middle_tab.pack(side="top", fill="both", expand=True)
        
#        container_middle_tab_2 = tk.Frame(tab_bar_middle)#, height=100, width=200)   
#        container_middle_tab_2.pack(side="top", fill="both", expand=True)
   
        tab_bar_middle.add(container_middle_tab, text='  Data Analysis  ')
#        tab_bar_middle.add(container_middle_tab_2, text='Test')    
        
        
#        button1 = ttk.Button(self, text='go to Outlier Detection', 
#                            command= lambda:  controller.show_frame(PageOne) )
        
        
        
        
        tab_bar_middle.pack(expand=True, fill="both")  


        container_left = tk.Frame(container_middle_tab)#, height=100, width=200)
        container_right = tk.Frame(container_middle_tab ,borderwidth = 1,relief='groove')#, height=100, width=200)



#        container_top.pack(side="top", fill="both", expand=False)  #not needed here, as we use tabs
        
#        container_middle.pack(side="top", fill="both", expand=True)  #contains two more containers. this should not be used as root for packing more things into it. better the left/right container
        
        container_bottom.pack(side="top", fill="both", expand=False)
        
        container_left.pack(side="left", fill="both", expand=True)#, padx=10,pady=10) #poadding to counter console padding left
        container_left.pack_propagate(False)  #packing things inside this container will not change its size
                
        container_right.pack(side="right", fill="both", expand=True)#, padx=10,pady=10)
        container_right.pack_propagate(False)  #packing things inside this container will not change its size
        
        
        
        container_left_top = tk.Frame(container_left,borderwidth = 1,relief='groove')  #we take this container as root for both input field and commit button
        container_left_top.pack(side="top", fill="both", expand=True)
        
        container_left_middle = tk.Frame(container_left,borderwidth = 1,relief='groove')  
        container_left_middle.pack(side="top", fill="both", expand=True)
        
        container_left_bottom = tk.Frame(container_left,borderwidth = 1,relief='groove')  
        container_left_bottom.pack(side="top", fill="both", expand=True)

#        tab_bar_right = ttk.Notebook(container_right) #notebook acts like a frame that fills all of container_right. hence needs to take the spot of one of the main frames
#        console_widget = tkst.ScrolledText(container_right, bg='black', fg='lightblue', font=CONSOLE_FONT)  #scrollbar is added here with tkst
   
        
##### end main structure ######
    
#        tab_bar_right.add(console_widget, text='Console')
#        tab_bar_right.add(plot_widget, text='Plots')    
#        
#        
#        tab_bar_right.pack(expand=True, fill="both") 
        
#        ###tree test
#        tree=ttk.Treeview(container_right)
#        
#        tree["columns"]=("one","two","three")
#        
#        tree.column("#0", width=270, minwidth=270, stretch=tk.NO)
#        tree.column("one", width=150, minwidth=150, stretch=tk.NO)
#        tree.column("two", width=400, minwidth=200)
#        tree.column("three", width=80, minwidth=50, stretch=tk.NO)
#        
#        tree.heading("#0",text="Name",anchor=tk.W)
#                     
#        tree.heading("one", text="Date modified",anchor=tk.W)
#        tree.heading("two", text="Type",anchor=tk.W)
#        tree.heading("three", text="Size",anchor=tk.W)
#        
#        
#        
#        folder1=tree.insert("", 1, "", text="Folder 1", values=("23-Jun-17 11:05","File folder",""))
#        tree.insert("", 2, "", text="text_file.txt", values=("23-Jun-17 11:25","TXT file","1 KB"))
#        # Level 2
#        tree.insert(folder1, "end", "", text="photo1.png", values=("23-Jun-17 11:28","PNG file","2.6 KB"))
#        tree.insert(folder1, "end", "", text="photo2.png", values=("23-Jun-17 11:29","PNG file","3.2 KB"))
#        tree.insert(folder1, "end", "", text="photo3.png", values=("23-Jun-17 11:30","PNG file","3.1 KB"))
#        tree.pack(side="top", fill="both", expand=True)        
#        ###end tree test
#        
        
        
        
#        hier weiter, aber alles rauskommentieren und kopoieren nicht überschriebern
        
        ##### end tab bar
        
        console_widget = tkst.ScrolledText(container_right, bg='black', fg='lightblue', border=8, font=CONSOLE_FONT, relief='sunken')  #scrollbar is added here with tkst
        console_widget.pack(fill='both', expand=True)


        output = PrintLogger(console_widget)        # create instance of file like object

        sys.stdout = output        # replace sys.stdout with our object   

        print('Console set')   
        print('')
        
#        HeaderTop = tk.Label(container_top, text="Outlier Detection", font=BIG_FONT)#, bg='lightgrey')
#        HeaderTop.pack(pady=0) 

        
        
#        HeaderImport = tk.Label(container_left_top, text="Data Import", font=THIS_FONT)
#        HeaderImport.pack(pady=50)     
        
        container_inputs = tk.Frame(container_left_top)  #we take this container as root for both input field and commit button
        container_inputs.pack(pady=20)

        container_dataimport = tk.Frame(container_left_top)  #we take this container as root for both input field and commit button
        container_dataimport.pack(pady=20)

        container_boxplot_button = tk.Frame(container_left_bottom)
        container_boxplot_button.pack(pady=20)
        
        container_select_what_kind_of_change = tk.Frame(container_left_middle)  #we take this container as root for both input field and commit button
        container_select_what_kind_of_change.pack(pady=20)

        container_find_n_largest_abs_jumps = tk.Frame(container_left_middle)  #we take this container as root for both input field and commit button
        container_find_n_largest_abs_jumps.pack(pady=20)      
        
        container_jumps_larger_than_threshold = tk.Frame(container_left_middle)  #we take this container as root for both input field and commit button
        container_jumps_larger_than_threshold.pack(pady=20)      

#        container_jumps_larger_than_threshold_second_row = tk.Frame(container_left_middle)  #we take this container as root for both input field and commit button
#        container_jumps_larger_than_threshold_second_row.pack(pady=20) 

        container_quantiles = tk.Frame(container_left_middle)  #we take this container as root for both input field and commit button
        container_quantiles.pack(pady=20) 
        
        container_jumps_larger_than_standard_deviation = tk.Frame(container_left_middle)  #we take this container as root for both input field and commit button
        container_jumps_larger_than_standard_deviation.pack(pady=20)    

        container_export = tk.Frame(container_left)  #we take this container as root for both input field and commit button
        container_export.pack(pady=20)  

#        container_show_outlier = tk.Frame(container_left_top)  #we take this container as root for both input field and commit button
#        container_show_outlier.pack(pady=20)       



        
        entry_field_1 = ttk.Entry(container_inputs)
        entry_field_1.insert('end', "O#000015642623")
#        l1 = ttk.Label(text="Test", style="BW.TLabel")
        entry_field_1.pack(side='left',padx=10,pady=10)
#        default_text_1 = True
        
        def delete_text_1(event):
            
            global default_text_1  #draw global variable into function, as bind() does not allow to pass parameters
            
            if default_text_1:
                
                entry_field_1.delete(0, 'end')
                
                default_text_1 = False        

                
        entry_field_1.bind("<Button-1>", delete_text_1) 

            
#        text = tk.Text(container_inputs)   #text field for symbols input
#        text.config(width=8,height=4,pady=5,padx=5, bg='black', fg='lightblue', border=4, font=CONSOLE_FONT, relief='sunken')
#        text.insert('insert','Asdfasdgfafdg') #hier weiter das dann als even wieder rauslöschen
#        text.pack(side='left',padx=10,pady=10)
        
        
#        here dlg for upload csv
        
#        function_that_opens_open_file_dialog_window_and_returns_path_where_to_open_csv
        
        button_import_csv = ttk.Button(container_inputs, text='Commit Symbols Via CSV', 
                                                                      command= lambda: [
#                                                                                    print( function_that_opens_open_file_dialog_window_and_returns_path_where_to_open_csv() ) ,
                                                                                    function_that_takes_path_and_file_name_and_parses_csv_to_rewrite_global_symbols( function_that_opens_open_file_dialog_window_and_returns_path_where_to_open_csv() ) ,
                                                                                    return_template_from_ados(global_symbols),
                                                                                    return_fields_from_template(global_template_for_request),
                                                                                    scroll_to_bottom_in_console(console_widget),
                                                                                    print('\n')
                                                                                    ])  #return_string_from_input( entry_field_9) #retrieve_integer_from_label(n_jumps_input_field) , gdf )
        button_import_csv.pack(side='right',padx=10,pady=10)
        
        
        buttonC = ttk.Button(container_inputs, text='Commit Symbols', command= lambda :   [return_symbols_from_input(entry_field_1) ,
#                                                                                   mdds_login_if_necessary(),
                                                                                   return_template_from_ados(global_symbols),
#                                                                                   call_thread_get_template(),
#                                                                                   print(t.current_thread().name), 
#                                                                                   thread_get_template.start(), 
#                                                                                   print(t.current_thread().name) 
                                                                                   return_fields_from_template(global_template_for_request),
                                                                                   scroll_to_bottom_in_console(console_widget),
                                                                                   print('\n')
#                                                                                   delete_old_menu_box(mb,container_inputs),
#                                                                                   update_menu_box(container_dataimport,global_fields_for_request)
                                                                                   ] )   #two functions passed for command  
    
        buttonC.pack(side='right',padx=10,pady=10)          #old return_template_from_ados(global_symbols) # older:  retrieve_symbols_from_text(text) , show_inputs( global_symbols))  
        
        
        
        


        
        entry_field_2 = ttk.Entry(container_dataimport)
        entry_field_2.insert('end', "20060626")  #would like to make this text disappear when clicked
#        l1 = ttk.Label(text="Test", style="BW.TLabel")
        entry_field_2.pack(side='left',padx=10,pady=10)
#        default_text_1 = True
        
        def delete_text_2(event):
            
            global default_text_2  #draw global variable into function, as bind() does not allow to pass parameters
            
            if default_text_2:
                
                entry_field_2.delete(0, 'end')
                
                default_text_2 = False        

                
        entry_field_2.bind("<Button-1>", delete_text_2)        
        
        
        entry_field_3 = ttk.Entry(container_dataimport)
        entry_field_3.insert('end', "20181028")  #would like to make this text disappear when clicked
#        l1 = ttk.Label(text="Test", style="BW.TLabel")
        entry_field_3.pack(side='left',padx=10,pady=10)
#        default_text_1 = True
        
        def delete_text_3(event):
            
            global default_text_3  #draw global variable into function, as bind() does not allow to pass parameters
            
            if default_text_3:
                
                entry_field_3.delete(0, 'end')
                
                default_text_3 = False        

                
        entry_field_3.bind("<Button-1>", delete_text_3)        
        



        button_fields = ttk.Button(container_dataimport, text='specify fields if not all wanted', command= lambda: [popup_field_selection(global_fields_for_request, global_selected_fields_for_request)] )
        button_fields.pack(side='left',padx=10)  
        
        
#        services = ['General','RiskGeneral']  no londer used!
#        
#        string_var = tk.StringVar()
#        
#        dropdown_to_chose_service = ttk.OptionMenu(container_dataimport, string_var,services[0], *services)
#
#        dropdown_to_chose_service.pack(side='left',padx=10)

    
        button_import_data = ttk.Button(container_dataimport, text='import data', command= lambda: [
#                                                                                         return_fields_from_template(global_template_for_request),
#                                                                                         print(global_fields_for_request),
                                                                                         data_importer( global_symbols,return_dates_from_input(entry_field_2),return_dates_from_input(entry_field_3),global_selected_fields_for_request,global_fields_for_request,global_template_for_request, global_snapshot_for_request,button_make_boxplot_of_all_pct_jumps_per_day,service=global_serive_for_request ),
                                                                                         scroll_to_bottom_in_console(console_widget),
#                                                                                         call_data_importer(return_dates_from_input(entry_field_2),return_dates_from_input(entry_field_3),thread_get_template,global_template_for_request),
#                                                                                         thread_data_importer.setDaemon(True),
#                                                                                         thread_data_importer.start()
#                                                                                         print(t.active_count())
#                                                                                         print(t.current_thread().name),
#                                                                                         thread_data_importer.join(),
#                                                                                         print(t.current_thread().name)                                                                                         
                                                                                        ] ) #old: data_importer( global_symbols,return_dates_from_input(entry_field_2),return_dates_from_input(entry_field_3),global_fields_for_request )
        button_import_data.pack(side='right',padx=10)        
                
        entry_field_find_n_largest_abs_jumps = ttk.Entry(container_find_n_largest_abs_jumps)
        entry_field_find_n_largest_abs_jumps.insert('end', "n largest jumps")  #would like to make this text disappear when clicked
        
        def delete_text_entry_field_find_n_largest_abs_jumps(event):
            
            global default_text_entry_field_find_n_largest_abs_jumps  #draw global variable into function, as bind() does not allow to pass parameters
            
            if default_text_entry_field_find_n_largest_abs_jumps:
                
                entry_field_find_n_largest_abs_jumps.delete(0, 'end')
                
                default_text_entry_field_find_n_largest_abs_jumps = False        

        entry_field_find_n_largest_abs_jumps.bind("<Button-1>", delete_text_entry_field_find_n_largest_abs_jumps) 
        

#        options_of_different_kinds_of_changes = ['pct_change','abs_change','log_change']
#        
#        string_var_change_options_for_button_find_n_largest_abs_jumps = tk.StringVar()
#        
#        dropdown_to_chose_what_kind_of_change_for_button_find_n_largest_abs_jumps = ttk.OptionMenu(container_select_what_kind_of_change, string_var_change_options_for_button_find_n_largest_abs_jumps,options_of_different_kinds_of_changes[0], *options_of_different_kinds_of_changes)
#
#        dropdown_to_chose_what_kind_of_change_for_button_find_n_largest_abs_jumps.pack(side='left',padx=10)      


        button_find_n_largest_abs_jumps = ttk.Button(container_find_n_largest_abs_jumps, text='find_n_largest_abs_jumps', 
                            command= lambda: [
                                    give_warning_if_gdf_has_no_data(gdf),
                                    find_n_largest_abs_jumps( return_int_from_input( entry_field_find_n_largest_abs_jumps), gdf,what_kind_of_change=global_absolute_or_percent_or_logidff_setting_for_computing_change ) ,  #retrieve_integer_from_label(n_jumps_input_field) , gdf )
                                    function_that_calls_functions_to_plot_suspect_dist_per_instrument_that_all_show_outlier_button_use(global_result,tab_bar_middle,button_export_as_csv),
                                    scroll_to_bottom_in_console(console_widget),
#                                    clear_canvas_in_plot_tab(canvas),
#                                    show_plot_in_other_tab(global_result,container_middle_tab_2),
#                                   reset_plot_if_necessary(fig, canvas),
#                                    plot_global_result() 
                                    ] )
    
        button_find_n_largest_abs_jumps.pack(side='right',padx=10,pady=10)    

        entry_field_find_n_largest_abs_jumps.pack(side='right',padx=10,pady=10)
        
        



        entry_field_for_threshold = ttk.Entry(container_jumps_larger_than_threshold)
        entry_field_for_threshold.insert('end', "value for threshold")  #would like to make this text disappear when clicked
#        entry_field_for_threshold.pack(side='left',padx=10,pady=10)
        
        def delete_text_entry_field_for_threshold(event):
            
            global default_text_entry_field_for_threshold  #draw global variable into function, as bind() does not allow to pass parameters
            
            if default_text_entry_field_for_threshold:
                
                entry_field_for_threshold.delete(0, 'end')
                
                default_text_entry_field_for_threshold = False       
                
        entry_field_for_threshold.bind("<Button-1>", delete_text_entry_field_for_threshold) 
        
        
        entry_field_for_how_many_obs_back = ttk.Entry(container_jumps_larger_than_threshold)
        entry_field_for_how_many_obs_back.insert('end', "how many jumps back")  #would like to make this text disappear when clicked
#        entry_field_for_how_many_obs_back.pack(side='left',padx=10,pady=10)
        
        def delete_text_entry_field_for_how_many_obs_back(event):
            
            global default_text_entry_field_for_how_many_obs_back  #draw global variable into function, as bind() does not allow to pass parameters
            
            if default_text_entry_field_for_how_many_obs_back:
                
                entry_field_for_how_many_obs_back.delete(0, 'end')
                
                default_text_entry_field_for_how_many_obs_back = False       
                
        entry_field_for_how_many_obs_back.bind("<Button-1>", delete_text_entry_field_for_how_many_obs_back)         
        

        entry_field_for_autoregressive_factor = ttk.Entry(container_jumps_larger_than_threshold)
        entry_field_for_autoregressive_factor.insert('end', "autoregressive factor")  #would like to make this text disappear when clicked
#        entry_field_for_autoregressive_factor.pack(side='left',padx=10,pady=10)
        
        def delete_text_entry_field_for_autoregressive_factor(event):
            
            global default_text_entry_field_for_autoregressive_factor  #draw global variable into function, as bind() does not allow to pass parameters
            
            if default_text_entry_field_for_autoregressive_factor:
                
                entry_field_for_autoregressive_factor.delete(0, 'end')
                
                default_text_entry_field_for_autoregressive_factor = False       
                
        entry_field_for_autoregressive_factor.bind("<Button-1>", delete_text_entry_field_for_autoregressive_factor)         
      

        entry_field_for_always_outlier_factor = ttk.Entry(container_jumps_larger_than_threshold)
        entry_field_for_always_outlier_factor.insert('end', "super suspect factor")  #would like to make this text disappear when clicked
#        entry_field_for_always_outlier_factor.pack(side='left',padx=10,pady=10)
        
        def delete_text_entry_field_for_always_outlier_factor(event):
            
            global default_text_entry_field_for_always_outlier_factor  #draw global variable into function, as bind() does not allow to pass parameters
            
            if default_text_entry_field_for_always_outlier_factor:
                
                entry_field_for_always_outlier_factor.delete(0, 'end')
                
                default_text_entry_field_for_always_outlier_factor = False       
                
        entry_field_for_always_outlier_factor.bind("<Button-1>", delete_text_entry_field_for_always_outlier_factor) 
        

#        buttonFind = ttk.Button(container_jumps_larger_than_threshold, text='find outliers larger than x', 
#                            command= lambda: [
#                                    give_warning_if_gdf_has_no_data(gdf),
#                                    function_that_finds_all_outlier_that_are_larger_than_a_threshold_and_also_looks_for_jump_back(gdf,what_kind_of_change=global_absolute_or_percent_or_logidff_setting_for_computing_change,threshold=return_int_from_input( entry_field_for_threshold),wait_how_many_obs_for_jump_back=return_int_from_input( entry_field_for_how_many_obs_back),auto_regressive_factor=return_int_from_input( entry_field_for_autoregressive_factor),always_outlier_factor=return_int_from_input( entry_field_for_always_outlier_factor)),
##                                    find_all_jumps_larger_than( return_int_from_input( entry_field_for_threshold), gdf, parent ),
#                                    function_that_calls_functions_to_plot_suspect_dist_per_instrument_that_all_show_outlier_button_use(global_result,tab_bar_middle,button_export_as_csv),
#                                    scroll_to_bottom_in_console(console_widget)
#        #                            show_plot_in_other_tab(global_result,container_middle_tab_2),
#        #                            create_new_tab_for_each_show_plot_call(global_result,tab_bar_middle)
#        #                            plot_global_result(axes, 0, global_result)
#                                    ] )  #retrieve_integer_from_label(n_jumps_input_field) , gdf )
##        buttonFind.pack(side='left',padx=10,pady=10)    




        button_interactive_plot = ttk.Button(container_boxplot_button, text='Graphical Outlier Search', 
                            command= lambda: [
                                    give_warning_if_gdf_has_no_data(gdf),
#                                    function_that_takes_gdf_and_calculates_daily_pct_changes_without_abs_and_creates_box_plot_by_date(gdf, tab_bar_middle),
#                                    plot_entire_column(gdf,tab_bar_middle),
#                                    plot_all_instruments_as_subplots(gdf, tab_bar_middle),
                                    plot_entire_column( gdf,tab_bar_middle),
#                                    plot_each_instrument_seperately(plot_entire_column, gdf,tab_bar_middle),
                                    scroll_to_bottom_in_console(console_widget)

                                    ] )
    
        button_interactive_plot.pack(padx=10,pady=10) 
        
        


        button_make_boxplot_of_all_pct_jumps_per_day = ttk.Button(container_boxplot_button, text='make boxplot of all pct jumps per day (only stk)', 
                            command= lambda: [
                                    give_warning_if_gdf_has_no_data(gdf),
                                    function_that_takes_gdf_and_calculates_daily_pct_changes_without_abs_and_creates_box_plot_by_date(gdf, tab_bar_middle),
                                    scroll_to_bottom_in_console(console_widget)

                                    ] )
    
#        button_make_boxplot_of_all_pct_jumps_per_day.pack(padx=10,pady=10) 




        button_show_and_plot_outlier_dist_over_time = ttk.Button(container_left_bottom, text='plot outlier dist over time', 
                            command= lambda: [
                                    give_warning_if_gdf_has_no_data(global_result),
                                    plot_time_beam( create_time_beam_that_shows_how_many_outliers_per_day(global_result, gdf) , tab_bar_middle),
                                    scroll_to_bottom_in_console(console_widget),
                                    

                                    ] )
    
        button_show_and_plot_outlier_dist_over_time.pack(padx=10,pady=10) 
        
        
        
        
        
        
        entry_field_for_quantiles = ttk.Entry(container_quantiles)
        entry_field_for_quantiles.insert('end', "quantile, eg 0.05 = 5%")  #would like to make this text disappear when clicked
        entry_field_for_quantiles.pack(side='left',padx=10,pady=10)
        
        def delete_text_entry_field_for_quantiles(event):
            
            global default_text_entry_field_for_quantiles  #draw global variable into function, as bind() does not allow to pass parameters
            
            if default_text_entry_field_for_quantiles:
                
                entry_field_for_quantiles.delete(0, 'end')
                
                default_text_entry_field_for_quantiles = False       
                
        entry_field_for_quantiles.bind("<Button-1>", delete_text_entry_field_for_quantiles) 

    
    
        button_quantiles = ttk.Button(container_quantiles, text='find outlier by sdev threshold', 
                            command= lambda: [
                                    give_warning_if_gdf_has_no_data(gdf),
                                    find_all_outlier_by_quantile(gdf,return_int_from_input( entry_field_for_quantiles)),
                                    scroll_to_bottom_in_console(console_widget),
                                    ] )
        button_quantiles.pack(side='left',padx=10,pady=10)              



#        entry_field_for_n_times_standart_deviation_threshold = ttk.Entry(container_jumps_larger_than_standard_deviation)
#        entry_field_for_n_times_standart_deviation_threshold.insert('end', "n times s. deviation threshold")  #would like to make this text disappear when clicked
#        entry_field_for_n_times_standart_deviation_threshold.pack(side='left',padx=10,pady=10)
#        
#        def delete_text_entry_field_for_n_times_standart_deviation_threshold(event):
#            
#            global default_text_entry_field_for_n_times_standart_deviation_threshold  #draw global variable into function, as bind() does not allow to pass parameters
#            
#            if default_text_entry_field_for_n_times_standart_deviation_threshold:
#                
#                entry_field_for_n_times_standart_deviation_threshold.delete(0, 'end')
#                
#                default_text_entry_field_for_n_times_standart_deviation_threshold = False       
#                
#        entry_field_for_n_times_standart_deviation_threshold.bind("<Button-1>", delete_text_entry_field_for_n_times_standart_deviation_threshold) 
#
#        entry_field_for_min_window_size = ttk.Entry(container_jumps_larger_than_standard_deviation)
#        entry_field_for_min_window_size.insert('end', "min window size s. dev")  #would like to make this text disappear when clicked
#        entry_field_for_min_window_size.pack(side='left',padx=10,pady=10)
#        
#        def delete_text_entry_field_for_min_window_size(event):
#            
#            global default_text_entry_field_for_min_window_size  #draw global variable into function, as bind() does not allow to pass parameters
#            
#            if default_text_entry_field_for_min_window_size:
#                
#                entry_field_for_min_window_size.delete(0, 'end')
#                
#                default_text_entry_field_for_min_window_size = False       
#                
#        entry_field_for_min_window_size.bind("<Button-1>", delete_text_entry_field_for_min_window_size) 
#    
#    
#        button_find_outlier_by_sdev_threshold = ttk.Button(container_jumps_larger_than_standard_deviation, text='find outlier by sdev threshold', 
#                            command= lambda: [
#                                    give_warning_if_gdf_has_no_data(gdf),
#                                    pd.set_option('display.max_columns', 5),
##                                    function_that_finds_all_outlier_that_are_larger_than_a_threshold_and_also_looks_for_jump_back(gdf,what_kind_of_change='pct_change',threshold=return_int_from_input( entry_field_for_threshold),wait_how_many_obs_for_jump_back=return_int_from_input( entry_field_for_how_many_obs_back),auto_regressive_factor=return_int_from_input( entry_field_for_autoregressive_factor),always_outlier_factor=return_int_from_input( entry_field_for_always_outlier_factor)),
#                                    function_that_finds_all_outliers_that_have_larger_pct_change_than_n_times_their_standard_deviation(gdf,which_kind_of_change=global_absolute_or_percent_or_logidff_setting_for_computing_change,n_times_standart_deviation_threshold=return_int_from_input( entry_field_for_n_times_standart_deviation_threshold),min_window_size=return_int_from_input( entry_field_for_min_window_size)),
#                                    function_that_calls_functions_to_plot_suspect_dist_per_instrument_that_all_show_outlier_button_use(global_result,tab_bar_middle,button_export_as_csv),
#                                    scroll_to_bottom_in_console(console_widget),
#                                    pd.set_option('display.max_columns', 6)
#        #                            plot_global_result(axes, 0, global_result)
#                                    ] )  #retrieve_integer_from_label(n_jumps_input_field) , gdf )
#        button_find_outlier_by_sdev_threshold.pack(side='left',padx=10,pady=10)      
    
        button_delete_all_found_outliers = ttk.Button(container_left_middle, text='delete all found outliers', 
                            command= lambda: [
                                    give_warning_if_gdf_has_no_data(gdf),
                                    delete_all_found_outliers(),
                                    scroll_to_bottom_in_console(console_widget)
                                    ] ) 
        button_delete_all_found_outliers.pack(padx=10,pady=10)          


        button_export_as_csv = ttk.Button(container_export, text='export as csv', 
                            command= lambda: [
                                    export_results( global_result, function_that_opens_save_file_dialog_window_and_returns_path_where_to_save_csv() ) ,
                                    scroll_to_bottom_in_console(console_widget)
#                                    console_widget.see(tk.END)
                                    ])  #return_string_from_input( entry_field_9) #retrieve_integer_from_label(n_jumps_input_field) , gdf )
#        button_export_as_csv.pack(side='right',padx=10,pady=10)   #only packed if there is something to export 
        
        
#        entry_field_9.pack(side='right',padx=10,pady=10)
        
        
  




#        entry_field_5 = ttk.Entry(container_show_outlier)
#        entry_field_5.insert('end', '5' )  #would like to make this text disappear when clicked
#
#        def delete_text_5(event):
#            
#            global default_text_5  #draw global variable into function, as bind() does not allow to pass parameters
#            
#            if default_text_5:
#                
#                entry_field_5.delete(0, 'end')
#                
#                default_text_5 = False        
#
#                
#        entry_field_5.bind("<Button-1>", delete_text_5) 


#        button1 = ttk.Button(container_bottom, text='Back to home', command= lambda:  controller.show_frame(StartPage) )
#        
#        button1.pack(pady=10,padx=10)        
    


if 'PrintLogger' not in dir():  #bug hat eher was mit damit zu tun dass davor die konsole nicht leer war...
    
    class PrintLogger(): # create file like object
        
        def __init__(self, textbox): # pass reference to text widget
            self.textbox = textbox # keep ref
    
        def write(self, text):
            self.textbox.insert(tk.END, text) # write text to textbox
                # could also scroll to end of textbox here to make sure always visible
    
        def flush(self): # needed for file like object
            pass        




class PageTwo(tk.Frame):
        
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)        
        '''
        
        ###### ######
        ###### ######
        ###### ######
        ------ ######
        ###### ######
        ###### ######
        ###### ######
        
        
        '''        
        
#        container_top = tk.Frame(self,height=25)#,bg='lightgrey')#, height=100, width=200)
        container_middle = tk.Frame(self)#, bg='lightblue')#, height=100, width=200)        
#        container_bottom = tk.Frame(self,height=50)#,bg='lightgrey')#, height=100, width=200)        
        
#        container_top.pack(side="top", fill="both", expand=False)
        
        container_middle.pack(side="top", fill="both", expand=True)  #contains two more containers. this should not be used as root for packing more things into it. better the left/right container
        
#        container_bottom.pack(side="top", fill="both", expand=False)

#        HeaderTop = tk.Label(container_top, text="Outlier Detection", font=BIG_FONT)#, bg='lightgrey')
#        HeaderTop.pack(pady=0)   

        
        tab_bar_middle = ttk.Notebook(container_middle) #notebook acts like a frame. hence needs to take the spot of one of the main frames

        container_middle_tab = tk.Frame(tab_bar_middle, bg='lightblue')#, height=100, width=200)   
        container_middle_tab.pack(side="top", fill="both", expand=True)
#        container_left_bottom = tk.Frame(tab_bar_middle,borderwidth = 1,relief='groove',bg='black')  #we take this container as root for both input field and commit button
#        container_left_bottom.pack(side="top", fill="both", expand=True)

#        page1 = ttk.Frame(nb)
#        page2 = ttk.Frame(nb)
#        text = ScrolledText(page2)
#        text.pack(expand=1, fill="both")
#        tab_bar_middle = ttk.Notebook(container_left) #notebook acts like a frame. hence needs to take the spot of one of the main frames
#   
        tab_bar_middle.add(container_middle_tab, text='One')
#        tab_bar_middle.add(container_left_bottom, text='Two')    
        
        tab_bar_middle.pack(expand=True, fill="both")  

        
        container_left = tk.Frame(container_middle_tab, bg='lightgreen')#, height=100, width=200)
        container_right = tk.Frame(container_middle_tab,bg='black',borderwidth = 1,relief='groove')#, height=100, width=200)

        container_left.pack(side="left", fill="both", expand=True)#, padx=10,pady=10) #poadding to counter console padding left
        container_left.pack_propagate(False)  #packing things inside this container will not change its size
                
        container_right.pack(side="right", fill="both", expand=True)#, padx=10,pady=10)
        container_right.pack_propagate(False)  #packing things inside this container will not change its size
        
        
        
        container_left_top = tk.Frame(container_left,borderwidth = 1,relief='groove', bg='red')  #we take this container as root for both input field and commit button
        container_left_top.pack(side="top", fill="both", expand=True)

        
        container_left_bottom = tk.Frame(container_left,borderwidth = 1,relief='groove', bg='brown')  #we take this container as root for both input field and commit button
        container_left_bottom.pack(side="top", fill="both", expand=True)
        
        
      
        
        
        
        button1 = ttk.Button(container_left_bottom, text='Back to home', command= lambda:  controller.show_frame(PageOne) )
        button1.pack(pady=10,padx=10)    


##        scrollbar = ttk.Scrollbar(self)
##        scrollbar.grid( column=2,rowspan=3,pady=20,padx=10)
#
#        
#        label = tk.Label(self, text="Test Page", font=LARGE_FONT)
#        label.grid(row=1, column=0,pady=20,padx=1000)#.pack(pady=20,padx=20)
#        
##        console_width = 50
##        console_height = 50        
#        
#
##        self.configure(background='grey')
#        
#        ###Right side
#        
##        label = tk.Label(self, text="Console", font=LARGE_FONT)
##        label.grid(row=1,column=2,pady=0,padx=0, columnspan=5  )
##        
##        console_widget = tk.Text(self, width=console_width, height = console_height, bg='black', fg='lightblue', border=4, font=CONSOLE_FONT, relief='sunken')
##        console_widget.grid(row=2,column=2,pady=20,padx=20, rowspan=50,  columnspan=5)
###
###
##        output = PrintLogger(console_widget)        # create instance of file like object
###
##        sys.stdout = output        # replace sys.stdout with our object   
##        
##        print('Console set')
#        
#        ###Right side end
#        
#        
#        
#        
#        sep = Separator(self, orient="vertical")
#        sep.grid(column=1, row=0,rowspan=100, sticky="nse")
#
#        sty = Style(self)
#        sty.configure("TSeparator", background="light_blue")          
#        
#        
#        


   
    
    






if __name__ == '__main__':
    
    app = ViktorsApp()
    
    #app.full_screen()
    
#    app.iconbitmap('graph.ico')  #redefine icon to avoid bug that icon becomes blurry after first run
    
    #app.geometry('900x500')  #define size of windows
    
    
    
    #app.configure(bg='lightblue')  #does not work!
    
    
    #app.full_screen()   
    
    
    app.lift()  #seems do the same as focus force()
    
    
    #app.attributes("-topmost", True)  #supresses other windows, not recommended
    
    #app.focus_force()  #puts window upfront
    
    #app.full_screen()  #oop way to get full screen view, will be later put into button #old: app.wm_state('zoomed')
    
    app.wm_state('zoomed')
    
    #app.grid_rowconfigure(1, weight=1)
    #app.grid_columnconfigure(0, weight=1)
    
    
    
    #app.pack(fill="both", expand=True)
    #app.content_area.grid(row=1, column=0, sticky="nsew")
    #app.grid_rowconfigure(1, weight=1)
    #app.grid_columnconfigure(0, weight=1)
    
    
    
#    mdds.login()  #bugfix that global variables from mdds.py have problemos as they are not from the main script 
    
    app.mainloop()

#Tk.update() and a Tk.update_idletasks(). 




#    assert t.current_thread() is t.main_thread()
#    https://stackoverflow.com/questions/23206787/check-if-current-thread-is-main-thread-in-python
   
    
    
    

    
    
    
    #gdf = 'no data loaded'  #just to be sure frame is empty if script is reloaded
    
    sys.stdout = real_console #fix console print bug
    print('done')








######################################################################################################################################################################################################################################################
### not relevant: ###


#class StartPage(tk.Frame):
#        
#    def __init__(self, parent, controller):
#        tk.Frame.__init__(self, parent)
#        
#        label = tk.Label(self, text="Home", font=LARGE_FONT)
#        label.pack(pady=20,padx=40)
#        
#
#        
##        background_canvas = tk.Canvas(self)
##        background_canvas.pack(fill="both", expand=True)
##        
##        coba_png_for_window_background = tk.PhotoImage(file = "coba.png")
##        windows_background_label = tk.Label(background_canvas, image=coba_png_for_window_background)
##        windows_background_label.pack()
##        windows_background_label.image=coba_png_for_window_background #avoid garbage collection
###        background_canvas.image=coba_png_for_window_background #avoid garbage collection
##        'das hier noch ordnelich rescalen und buttons in vordergrund'
##        
#
#
#        button1 = ttk.Button(self, text='go to Outlier Detection', 
#                            command= lambda:  controller.show_frame(PageOne) )
#        button1.pack(pady=10,padx=40)
#        
#        button2 = ttk.Button(self, text='go to Layout test page', 
#                            command= lambda:  controller.show_frame(PageTwo) )
#        button2.pack(pady=10,padx=10)
#        
#        button3 = ttk.Button(self, text='go to Graph test page', 
#                            command= lambda:  controller.show_frame(PageThree) )
#        button3.pack(pady=10,padx=10)
#        
#        button4 = ttk.Button(self, text='quit', command=end) #lambda: end does not work
#        button4.pack(pady=10,padx=20)
#        
##        image2 =Image.open('C:\\Users\\adminp\\Desktop\\titlepage\\front.gif')
##        image1 = ImageTk.PhotoImage(image2) 
##        labelText = StringVar()
##        labelText.set("Welcome !!!!")
##        #labelText.fontsize('10')
##        
##        label1 = tk.Label(self, image=image1,
##                       font=("Times New Roman", 24), height=4, fg="blue")
##        label1.pack()


#class PageThree(tk.Frame):
#        
#    def __init__(self, parent, controller):
#        tk.Frame.__init__(self, parent)
#        label = tk.Label(self, text="Page Three- Graph Page", font=LARGE_FONT)
#        label.pack(pady=20,padx=20)
#    
##        ### coba plots
#        import matplotlib.pyplot as plt
#        import pandas as pd
#        
#        fig, axes = plt.subplots(nrows=2, ncols=2)  #creates 2x2 window for the 3 treshold plots
#
#        data = pd.DataFrame([1,2,3,4,5,4,5,1,3,2,1,0])
#        
#        data.plot(ax=axes[0,0])
##        ###
#        
#        ### other way to get subplots
##        fig = Figure(figsize=(5,5), dpi=100)
##        a = fig.add_subplot(221) # example argument: "234" means "2x3 grid, 4th subplot"
#        
###        a.clear()
##        a.plot([1,2,3,4,5,6],[3,2,3,4,5,1])
#        ###
#                
#        canvas = FigureCanvasTkAgg(fig, self)  #tie plot to canvas, not console
#        
#        canvas.show()  # make canvas visible
#        
#        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  #canvas.show() needs to be packed to be shown
#        
#        ### toolbar
#        toolbar = NavigationToolbar2TkAgg(canvas, self)
#        
#        toolbar.update()
#        
#        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True) #side= determines position in window
#        ###
#
#
#
#        button1 = ttk.Button(self, text='Back to home', 
#                            command= lambda:  controller.show_frame(StartPage) )
#        button1.pack(pady=10,padx=10)
