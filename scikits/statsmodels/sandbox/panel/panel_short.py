# -*- coding: utf-8 -*-
"""Panel data analysis for short T and large N

Created on Sat Dec 17 19:32:00 2011

Author: Josef Perktold
License: BSD-3


starting from scratch before looking at references again
just a stub to get the basic structure for group handling
target outsource as much as possible for reuse

not run yet

"""

import numpy as np
from scikits.statsmodels.regression.linear_model import OLS
from scikits.statsmodels.tools.grouputils import Group

#not used
class Unit(object):

    def __init__(endog, exog):
        self.endog = endog
        self.exog = exog


def sum_outer_product_loop(x, group_iter):
    '''sum outerproduct dot(x_i, x_i.T) over individuals

    loop version

    '''

    mom = 0
    for g in group_iter:
        x_g = x[g]
        mom += np.outer(x_g, x_g)

    return mom

def whiten_individuals_loop(x, transform, group_iter):
    '''apply linear transform for each individual

    loop version
    '''

    #Note: figure out dimension of transformed variable
    #so we can pre-allocate
    x_new = []
    for g in group_iter:
        x_g = x[g]
        x_new.append(np.dot(transform, x_g))

    return np.vstack(x_new)  #or np.array(x_new) #check shape



class ShortPanelGLS(object):
    '''Short Panel with general intertemporal within correlation

    assumes data is stacked by individuals, panel is balanced and
    within correlation structure is identical across individuals.


    '''

    def __init__(self, endog, exog, group):
        self.endog = endog
        self.exog = exog
        self.group = Group(group)
        self.n_groups = self.group.n_groups
        self.nobs_group

    def fit_ols(self):
        self.res_pooled = OLS(self.endog, self.exog).fit()
        return self.res_pooled  #return or not

    def get_within_cov(self, resid):
        #central moment or not?
        mom = sum_outer_product_loop(resid, self.group.group_iter)
        return mom / self.n_groups   #df correction ?

    def whiten_groups(self, x, cholsigmainv_i):
        #from scipy import sparse #use sparse
        wx = whiten_individuals_loop(x, cholsigmainv_i, self.group.group_iter)
        return wx

    def fit(self):
        res_pooled = self.fit_ols() #get starting estimate
        sigma_i = self.get_within_cov(res_pooled.resid)
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        wendog = self.whiten_groups(self.endog, self.cholsigmainv_i)
        wexog = self.whiten_groups(self.exog, self.cholsigmainv_i)
        self.res1 = OLS(wendog, wexog).fit()
        return self.res1



if __name__ == '__main__':

    pass


