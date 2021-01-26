import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/fork4/statsmodels")

import numpy as np
from statsmodels.sandbox.phreg import PHreg
from statsmodels.datasets import ovarian_cancer

method = 'efron'

dta = ovarian_cancer.load()

time = np.asarray(dta.endog.time.values)
event = np.asarray(dta.endog.event.values)
exog = np.asarray(dta.exog)

nrep = 10
time = np.kron(time, np.ones(nrep))
event = np.kron(event, np.ones(nrep))
exog = np.kron(exog, np.ones((nrep,1)))

# this version doesn't use the Survival object
# it has its own internal format

mod = PHreg(time, exog, event, ties=method)
res = mod.fit()

# assume you don't have a Survial object in dta.endog already
from statsmodels.sandbox.survival2 import CoxPH, Survival
surv = Survival(time1=time, event=event)
mod_ss = CoxPH(surv, exog, ties=method)
res_ss = mod_ss.fit()

# let's profile
import cProfile, pstats
params = res_ss.params
cProfile.run("mod_ss.hessian(params)", "ss.prof")

params = res.params
cProfile.run("mod.hessian(params)", "ks.prof")

p1 = pstats.Stats("ss.prof")
p1.strip_dirs().sort_stats('time').print_stats()

p2 = pstats.Stats("ks.prof")
p2.strip_dirs().sort_stats('time').print_stats()
