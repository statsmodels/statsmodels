# -*- coding: utf-8 -*-
"""

Created on Sun May 26 13:23:40 2013

Author: Josef Perktold, based on Enrico Giampieri's multiOLS
"""

#import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.sandbox.multilinear import multiOLS, multigroup

data = sm.datasets.longley.load_pandas()
df = data.exog
df['TOTEMP'] = data.endog

#This will perform the specified linear model on all the
#other columns of the dataframe
res0 = multiOLS('GNP + 1', df)

#This select only a certain subset of the columns
res = multiOLS('GNP + 0', df, ['GNPDEFL', 'TOTEMP', 'POP'])
print(res.to_string())


url = "http://vincentarelbundock.github.com/"
url = url + "Rdatasets/csv/HistData/Guerry.csv"
df = pd.read_csv(url, index_col=1) #'dept')

#evaluate the relationship between the various parameters whith the Wealth
pvals = multiOLS('Wealth', df)['adj_pvals', '_f_test']

#define the groups
groups = {}
groups['crime'] = ['Crime_prop', 'Infanticide',
                   'Crime_parents', 'Desertion', 'Crime_pers']
groups['religion'] = ['Donation_clergy', 'Clergy', 'Donations']
groups['wealth'] = ['Commerce', 'Lottery', 'Instruction', 'Literacy']

#do the analysis of the significance
res3 = multigroup(pvals < 0.05, groups)
print(res3)
