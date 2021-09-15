# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:36:07 2020

@author: user
"""



t = pd.DataFrame({'integers':[1,3,2,3,1,2]})

t['two'] =['01/12/2000', '01/12/2000', '01/12/2000', '02/12/2000','02/12/2000', '02/12/2000']

t['Date'] = pd.to_datetime(t['two'], errors='coerce')


t['two'] = [2345,2345,654,23,2536,12435]

print(t)

print(t.sort_values(by=['Date', 'integers']))









