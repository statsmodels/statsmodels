import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/fork4/statsmodels")

import numpy as np
from statsmodels.regression.lme import LME
import pandas as pd


data = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/MASS/Sitka.csv")
endog = data["size"]
exog = np.ones((data.shape[0], 2), dtype=np.float64)
exog[:,1] = data["Time"]
exog = pd.DataFrame(data=exog)
exog.columns = ["Intercept", "Time"]
exog_re = exog.copy()

exog_re = np.asarray(exog.copy())[:,0]
md1 = LME(endog, exog, exog_re, data["tree"])
mdf1 = md1.fit(num_sd=10, reml=False, do_cg=True, pen=1)
print mdf1.summary(xname=["Intercept", "Time", "Tree (RE)"])

exog_re = np.asarray(exog.copy())
md2 = LME(endog, exog, exog_re, data["tree"])
start_re = np.zeros((2,2), dtype=np.float64)
start_re[0,0] = mdf1.params[2]**2
start_re[1,1] = 0.000001
mdf2 = md2.fit(num_sd=10, start_fe=mdf1.params_fe,
               start_re=start_re, reml=False, do_cg=True, pen=0)
print mdf2.summary(xname=["Intercept", "Time", "Tree (RE)",
                          "Time (RE)"])


1/0

data = pd.read_csv("Dyestuff.csv")

endog = data["Yield"].astype(np.float64)
exog = np.ones(data.shape[0], dtype=np.float64)
exog_re = np.ones(data.shape[0], dtype=np.float64)
groups = data["Batch"]

md = LME(endog, exog, exog_re, groups)
mdf = md.fit(reml=True)




data = pd.read_csv("sleepstudy.csv")

endog = data["Reaction"].astype(np.float64)
exog = np.ones((data.shape[0],2), dtype=np.float64)
exog[:,1] = data["Days"].astype(np.float64)
exog_re = np.ones((data.shape[0],2), dtype=np.float64)
exog_re[:,1] = exog[:,1]
groups = data["Subject"]
exog = pd.DataFrame(data=exog)

md = LME(endog, exog, exog_re, groups)
mdf = md.fit(reml=True)
