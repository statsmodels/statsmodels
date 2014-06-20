import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.mice import mice
import os

cur_dir = os.getcwd()
fn = os.path.join(cur_dir,"missingdata.csv")
data = pd.read_csv(fn)
data.columns = ['x1','x2','x3']
impdata = mice.ImputedData(data)
m1 = impdata.new_imputer("x2", scale_method="perturb_chi2")
m2 = impdata.new_imputer("x3", scale_method="perturb_chi2")
m3 = impdata.new_imputer("x1", model_class=sm.Logit, scale_method="perturb_chi2")
impcomb = mice.MICE("x1 ~ x2 + x3", sm.Logit,[m1,m2,m3])
implist = impcomb.run(method="pmm")
p1 = impcomb.combine(implist)
#print p1.summary()
