import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.mice import mice
#import os
#import numpy as np
#
#cur_dir = os.getcwd()
#fn = os.path.join(cur_dir,"missingdata.csv")
data = pd.read_csv("missingfull.csv")
data.columns = ['x1','x2','x3']
#data = data.drop('x1',1)
impdata = mice.ImputedData(data, go=False)
impdata.new_imputer("x2", method="pmm", k_pmm=20)
#impdata.new_imputer("x3", model_class=sm.Poisson, method="pmm", k_pmm=20)
impdata.new_imputer("x1", method="pmm", model_class=sm.Logit, k_pmm=20)
#impdata.new_imputer("x2")
impdata.new_imputer("x3", model_class=sm.Poisson, method="pmm", k_pmm=20)
#impdata.new_imputer("x1", model_class=sm.Logit)
impcomb = mice.MICE("x1 ~ x2 + x3", sm.Logit,impdata)
impcomb.run(2, 1, burnin=0)
p1 = impcomb.combine()
print p1.summary()
