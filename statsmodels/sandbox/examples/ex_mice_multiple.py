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
m1 = impdata.new_imputer("x2", method="pmm")
m2 = impdata.new_imputer("x3", model_class=sm.Poisson, method="pmm")
m3 = impdata.new_imputer("x1", model_class=sm.Logit, method="pmm")
impcomb = mice.MICE("x1 ~ x2 + x3", sm.Logit,[m3,m1,m2])
impcomb.run(2, 5)
p1 = impcomb.combine()
print p1.summary()
