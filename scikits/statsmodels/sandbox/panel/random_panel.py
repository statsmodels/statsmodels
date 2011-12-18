# -*- coding: utf-8 -*-
"""Generate a random process with panel structure

Created on Sat Dec 17 22:15:27 2011

Author: Josef Perktold


Notes
-----
* written with unbalanced panels in mind, but not flexible enough yet
* need more shortcuts and options for balanced panel
* need to add random intercept or coefficients
* only one-way (repeated measures) so far

"""

import numpy as np
import correlation_structures as cs


class PanelSample(object):
    '''data generating process for panel with within correlation

    allows various within correlation structures, but no random intercept yet

    '''

    def __init__(self, nobs, k_vars, n_groups, exog=None,
                 corr_structure=np.eye, corr_args=(), scale=1, seed=None):


        nobs_i = nobs//n_groups
        nobs = nobs_i * n_groups  #make balanced
        self.nobs = nobs
        self.nobs_i = nobs_i
        self.n_groups = n_groups
        self.k_vars = k_vars
        self.corr_structure = corr_structure
        self.groups = np.repeat(np.arange(n_groups), nobs_i)

        self.group_indices = np.arange(n_groups+1) * nobs_i #check +1

        if exog is None:
            t = np.repeat(np.linspace(-1,1,nobs_i), n_groups)
            exog = t[:,None]**np.arange(k_vars)

        self.exog = exog
        self.y_true = exog.sum(1)  #all coefficients equal 1
        if seed is None:
            seed = np.random.randint(0, 999999)

        self.seed = seed
        self.random_state = np.random.RandomState(seed)

        #this makes overwriting difficult, move to method?
        self.std = scale * np.ones(nobs_i)
        corr = self.corr_structure(nobs_i, *corr_args)
        self.cov = cs.corr2cov(corr, self.std)
        self.group_means = np.zeros(n_groups)



    def generate_panel(self):
        '''
        generate endog for a random panel dataset with within correlation

        '''

        random = self.random_state

        use_balanced = True
        if use_balanced: #much faster for balanced case
            noise = np.random.multivariate_normal(np.zeros(nobs_i),
                                                  self.cov,
                                                  size=n_groups).ravel()
            #need to add self.group_means
            noise += np.repeat(self.group_means, nobs_i)
        else:
            noise = np.empty(self.nobs, np.float64)
            noise.fill(np.nan)
            for ii in range(self.n_groups):
                #print ii,
                idx, idxupp = self.group_indices[ii:ii+2]
                #print idx, idxupp
                mean_i = self.group_means[ii]
                noise[idx:idxupp] = random.multivariate_normal(
                                        mean_i * np.ones(self.nobs_i), self.cov)

        endog = self.y_true + noise
        return endog


if __name__ == '__main__':

    nobs = 1000
    nobs_i = 5
    n_groups = nobs // nobs_i
    k_vars = 3

#    dgp = PanelSample(nobs, k_vars, n_groups, corr_structure=cs.corr_equi,
#                      corr_args=(0.6,))
#    dgp = PanelSample(nobs, k_vars, n_groups, corr_structure=cs.corr_ar,
#                      corr_args=([1, -0.95],))
    dgp = PanelSample(nobs, k_vars, n_groups, corr_structure=cs.corr_arma,
                      corr_args=([1], [1., -0.9],))
    print 'seed', dgp.seed
    y = dgp.generate_panel()
    noise = y - dgp.y_true
    print np.corrcoef(y.reshape(-1,n_groups, order='F'))
    print np.corrcoef(noise.reshape(-1,n_groups, order='F'))

    from panel_short import ShortPanelGLS
    mod = ShortPanelGLS(y, dgp.exog, dgp.groups)
    res = mod.fit()
    print res.params
    print res.bse
    #Now what?
    #res.resid is of transformed model
    #np.corrcoef(res.resid.reshape(-1,n_groups, order='F'))
    y_pred = np.dot(mod.exog, res.params)
    resid = y - y_pred
    print np.corrcoef(resid.reshape(-1,n_groups, order='F'))
    print resid.std()
    err = y_pred - dgp.y_true
    print err.std()
    #OLS standard errors are too small
    mod.res_pooled.params
    mod.res_pooled.bse
    #heteroscedasticity robust doesn't help
    mod.res_pooled.HC1_se
    #compare with cluster robust se
    import scikits.statsmodels.sandbox.panel.sandwich_covariance as sw
    print sw.cov_cluster(mod.res_pooled, dgp.groups.astype(int))[1]
    #not bad, pretty close to panel estimator
    #and with Newey-West Hac
    print sw.se_cov(sw.cov_nw_panel(mod.res_pooled, 5, mod.group.groupidx))
    #too small, assuming no bugs,
    #see Peterson assuming it refers to same kind of model
    print dgp.cov
