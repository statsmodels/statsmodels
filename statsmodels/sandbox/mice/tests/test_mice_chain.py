# -*- coding: utf-8 -*-
"""
Created on Thu May 08 04:39:24 2014

@author: Frank
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 07 15:10:00 2014

@author: Frank
"""

import sys
import pandas as pd
import numpy as np
sys.path.insert(0,"C:/Users/Frank/Documents/GitHub/statsmodels/statsmodels/")
from statsmodels.sandbox.mice import mice
import matplotlib.pyplot as plt
#print statsmodels.__file__

#data = pd.DataFrame.from_csv('missingdata.csv')
data = pd.read_csv('C:/Users/Frank/Documents/GitHub/statsmodels/statsmodels/sandbox/mice/tests/results/missingdata.csv')
#data = np.genfromtxt('missingdata.csv',delimiter = ',')
data.columns = ['x2','x3']
impdata = mice.ImputedData(data)

#print(impdata.data.fillna(impdata.data.mean()))
m1 = mice.Imputer(impdata,"x2~x3","OLS")

m2 = mice.Imputer(impdata,"x3~x2","OLS")

impchain = mice.ImputerChain([m1,m2])
impchain.generate_data(3,5,'ftest')

f0 = pd.read_csv('ftest_0.csv')
f1 = pd.read_csv('ftest_1.csv')
f2 = pd.read_csv('ftest_2.csv')

s1 = set(impdata.values['x2'][0])
s2 = set(impdata.values['x3'][0])
mval = s1.union(s2)
mvalb = s1.intersection(s2)

f0c = []
f0both = []
f0a = []
f0b = []
for i in range(len(f0.x2)):
    f0c.append(i not in mval)
    f0both.append(i in mvalb)
    f0a.append(i in s1)
    f0b.append(i in s2)

colors = np.where(f0c,'r','k')

plt.scatter(f0.x2, f0.x3, s=120, c=colors)

f0['subset'] = np.select([f0a, f0b, f0c],
                         ['x2 missing','x3 missing', 'not missing'])
for color, label in zip('bgr',  ['x2 missing','x3 missing', 'not missing']):
    subset = f0[f0.subset == label]
    plt.scatter(subset.x2, subset.x3, s=120, c=color, label=str(label))
#plt.legend()