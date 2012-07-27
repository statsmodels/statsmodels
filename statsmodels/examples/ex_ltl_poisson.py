# -*- coding: utf-8 -*-
"""

Created on Fri Jul 27 03:57:30 2012

Author: Josef Perktold
"""

import numpy as np
from statsmodels.tools.tools import add_constant
from statsmodels.robust.least_trimmed_squares import LTLikelihood
from statsmodels.discrete.discrete_model import Poisson

from statsmodels.datasets import randhie

rand_data = randhie.load()
endog_randhie = rand_data.endog
exog_randhie = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
exog_randhie = add_constant(exog_randhie, prepend=False)

#randhie raises numpy.linalg.linalg.LinAlgError: Singular matrix in LTL

np.random.seed(98765678)
nobs = 200
rvs = np.random.randn(nobs,6)
data_exog = rvs
data_exog = add_constant(data_exog)
xbeta = 1 + 0.1*rvs.sum(1)
data_endog = np.random.poisson(np.exp(xbeta))

data_endog[:5*5:5] += 10  #add outliers

endog, exog = data_endog, data_exog

res_poisson = Poisson(endog, exog).fit()
print res_poisson

mod_poisson_ltl = LTLikelihood(endog, exog)
best = mod_poisson_ltl.fit(k_trimmed=5)
print np.nonzero(~best[-1])[0]

