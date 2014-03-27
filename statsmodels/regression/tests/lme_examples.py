import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/fork4/statsmodels")

import numpy as np
from statsmodels.regression.lme import MixedLM
import pandas as pd
import matplotlib.pyplot as plt

exog = np.random.normal(size=(300,10))
groups = np.kron(np.arange(100), np.ones(3))
group_effects = np.kron(np.random.normal(size=100), np.ones(3))
endog = exog[:,0] + exog[:,3] - exog[:,2] + group_effects + np.random.normal(size=300)
md1 = MixedLM(endog, exog, groups)
#mdf1 = md1.fit()
mdf1 = md1.fit_regularized(maxvar=5, xtol=1)
plt.clf()
for k in range(1,11):
    plt.plot(mdf1.iloc[:,0], mdf1.iloc[:,k], '-', color='grey')
plt.savefig("tt.pdf")
1/0

data = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/MASS/Sitka.csv")
endog = data["size"]
exog = np.ones((data.shape[0], 2), dtype=np.float64)
exog[:,1] = data["Time"]
exog_re = np.asarray(exog.copy())
md1 = MixedLM(endog, exog, data["tree"])
#mdf1 = md1.fit()
mdf1 = md1.fit_regularized()
1/0
print mdf1.summary() #xname=["Intercept", "Time", "Tree (RE)", "Time (RE)"])
likev = mdf1.profile_re(1, dist_low=0.01, dist_high=0.01)

data = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv")
data = data.dropna()
md = LME.from_formula("Weight ~ Time", data, groups=data["Pig"])
mdf = md.fit(reml=True)
print mdf.summary()
1/0

data = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv")
data = data.dropna()
md = LME.from_formula("Weight ~ Time", data, groups=data["Pig"])
mdf = md.fit()
print mdf.summary()
1/0

data = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/MASS/Sitka.csv")
endog = data["size"]
exog = np.ones((data.shape[0], 2), dtype=np.float64)
exog[:,1] = data["Time"]
exog = pd.DataFrame(data=exog)
exog.columns = ["Intercept", "Time"]
exog_re = exog.copy()

1/0

exog_re = np.asarray(exog.copy())
md2 = LME(endog, exog, exog_re, data["tree"])
start_re = np.zeros((2,2), dtype=np.float64)
start_re[0,0] = mdf1.params[2]**2
start_re[1,1] = 0.000001
mdf2 = md2.fit(num_sd=10, start_fe=mdf1.params_fe,
               start_re=start_re, reml=True, do_cg=True, pen=0)
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
