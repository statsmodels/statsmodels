# -*- coding: utf-8 -*-
"""
Created on Wed May 07 15:10:00 2014

@author: Frank
"""
#import scipy
#import patsy
#import cython
#import statsmodels.api
import sys
import pandas as pd
#import numpy as np
sys.path.insert(0,"C:/Users/Frank/Documents/GitHub/statsmodels/statsmodels/")
from statsmodels.sandbox.mice import mice
#print statsmodels.__file__

#data = pd.DataFrame.from_csv('missingdata.csv')
data = pd.read_csv('C:/Users/Frank/Documents/GitHub/statsmodels/statsmodels/sandbox/mice/tests/results/missingdata.csv')
#data = np.genfromtxt('missingdata.csv',delimiter = ',')
data.columns = ['x2','x3']
impdata = mice.ImputedData(data)
print(impdata.values)
impdata.mean_fill()
print(impdata.values)
print(impdata.data)
#print(impdata.data.fillna(impdata.data.mean()))
m1 = mice.Imputer(impdata,"x2~x3","OLS")
m1.impute_asymptotic_bayes()
print(impdata.data)
m2 = mice.Imputer(impdata,"x3~x2","OLS")
m2.impute_asymptotic_bayes()
print(impdata.data)
