### This script regresses umeployment data


##### module imports
###
#
import pandas as pd
#
###
##### end module iports


##### 2. 
###
#
path = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/macropy/assignment 5/'
ue = pd.read_csv(path+'Mean_UNEMP_Level.csv')
fc = pd.read_csv(path+'UNRATE.csv')
#
###
##### 2.


##### 3. make unemployment data quarterly
###
#
ue['QUARTER']=ue['QUARTER'].astype(str)
ue['YEAR']=ue['YEAR'].astype(str)
ue['year+quarter']= ue['YEAR']+'Q'+ue['QUARTER']
ue = ue.set_index(['year+quarter'],drop=True)
#
###
##### 3.


##### 4. prepare fc dataframe
###
#
fc_new_index = []
date_mapping = {'03':'1','06':'2','09':'3','12':'4',}
for date in fc['DATE'].values[2:][::3]: #sorry for poor programming
        fc_new_index.append(date[:4]+'Q'+date_mapping[date[5:7]]) 
fc_new_index = pd.DataFrame({'year+quarter':fc_new_index},index=range(len(fc_new_index)))
fc_new_index = fc_new_index.set_index('year+quarter')
fc_new_index['UNRATE']= fc.rolling(3).mean().values[2:][::3]
fc = fc_new_index 
fc = fc[83:]
#
###
##### 


##### 5.
###
#
merge = fc.join(ue, how='outer')
#
###
##### 5.


##### 6.
###
#
#xt+3-Ftxt+3
merge['forecast_error'] = merge['UNRATE'].shift(periods=-3) - merge['UNEMP5']
# FtXt+3 - Ft-1Xt+3
merge['forecast_revision'] = merge['UNEMP5'] - merge['UNEMP5'].shift(periods=1)
#
###
##### 6.


##### 7.
###
#
from statsmodels.formula.api import ols
mod = ols(formula='forecast_error ~ forecast_revision', data=merge)
fit = mod.fit(cov_type = "HAC", cov_kwds = {'maxlags' : 3})
print(fit.summary())
#
###
##### 7.


##### 8.
###
#
print('''\n
Interpretation: A positive forecast revision leads to an increase in the 
forecast error (significant at 5%). Beta quantifies how large this increase 
per unit of increase in forecast_revision is. Hence, this result is in line with the 
findings of Coibion and Gorodnychenko.
    ''')
#
###
##### 8.