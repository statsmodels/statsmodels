# -*- coding: utf-8 -*-
"""
Created on Sat May 15 19:59:42 2010
Author: josef-pktd
"""

import numpy as np

from statsmodels.sandbox import formula
import statsmodels.sandbox.contrast_old as contrast



#define a categorical variable - factor


f0 = ['a','b','c']*4
f = ['a']*4 + ['b']*3 + ['c']*4
fac = formula.Factor('ff', f)
fac.namespace = {'ff':f}
list(fac.values())
[f for f in dir(fac) if f[0] != '_']

#create dummy variable

fac.get_columns().shape
fac.get_columns().T

#this is a way of encoding effects from a categorical variable
#different from using dummy variables
#I never seen a reference for this.

fac.main_effect(reference=1)
#dir(fac.main_effect(reference=1))
fac.main_effect(reference=1)()
#fac.main_effect(reference=1).func
fac.main_effect(reference=1).names()
fac.main_effect(reference=2).names()
fac.main_effect(reference=2)().shape

#columns for the design matrix

fac.main_effect(reference=2)().T
fac.names()




