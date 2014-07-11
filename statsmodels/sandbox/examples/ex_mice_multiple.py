import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.mice import mice
import os
import numpy as np
#
#cur_dir = os.getcwd()
#fn = os.path.join(cur_dir,"missingdata.csv")
np.random.seed(1325)
data = pd.read_csv("missingfull.csv")
data.columns = ['x1','x2','x3']
impdata = mice.ImputedData(data)
impdata.new_imputer("x2", method="pmm")
impdata.new_imputer("x3", model_class=sm.Poisson, method="pmm")
impdata.new_imputer("x1", model_class=sm.Logit, method="pmm")
impcomb = mice.MICE("x1 ~ x2 + x3", sm.Logit,impdata)
impcomb.run(20, 10)
p1 = impcomb.combine()
print p1.summary()
