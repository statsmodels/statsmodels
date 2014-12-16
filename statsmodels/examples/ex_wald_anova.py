# -*- coding: utf-8 -*-
"""Example for wald_test for terms - `wald_anova`

Created on Mon Dec 15 11:19:23 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np

from statsmodels.formula.api import ols, glm, poisson
from statsmodels.discrete.discrete_model import Poisson

import statsmodels.stats.tests.test_anova as ttmod

test = ttmod.TestAnova3()
test.setupClass()

data = test.data.drop([0,1,2])
res_ols = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit(use_t=False)

res_glm = glm("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)",
                        data).fit()

res_poi = Poisson.from_formula("Days ~ C(Weight) * C(Duration)", data).fit(cov_type='HC0')
res_poi_2 = poisson("Days ~ C(Weight) + C(Duration)", data).fit(cov_type='HC0')

print('\nOLS')
print(res_ols.wald_test_terms())
print('\nGLM')
print(res_glm.wald_test_terms(skip_single=False, combine_terms=['Duration', 'Weight']))
print('\nPoisson 1')
print(res_poi.wald_test_terms(skip_single=False, combine_terms=['Duration', 'Weight']))
print('\nPoisson 2')
print(res_poi_2.wald_test_terms(skip_single=False))

from statsmodels.discrete.discrete_model import NegativeBinomial
res_nb2 = NegativeBinomial.from_formula("Days ~ C(Weight) * C(Duration)", data).fit()
print('\nNegative Binomial nb2')
print(res_nb2.wald_test_terms(skip_single=False))

res_nb1 = NegativeBinomial.from_formula("Days ~ C(Weight) * C(Duration)", data,
                                        loglike_method='nb1').fit(cov_type='HC0')
print('\nNegative Binomial nb2')
print(res_nb1.wald_test_terms(skip_single=False))

