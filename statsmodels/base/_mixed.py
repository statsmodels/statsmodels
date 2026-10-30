# -*- coding: utf-8 -*-
"""
Created on Sun May  3 09:21:53 2015

Author: Josef Perktold
License: BSD-3
"""

from collections import OrderedDict
import time

import numpy as np
from scipy import stats

# local copy, original in dcm branch
from statsmodels.discrete.halton_sequence_bak import halton

DEBUG = False


class MixedMixin(object):


    def __init__(self, endog, exog, random_idx=None, n_points=64, **kwds):

        self.random_idx = random_idx

        if random_idx is not None:
            self.n_rparam = len(random_idx)
            self.n_points = n_points
        else:
            self.random_idx = []
            self.n_rparam = 0
            self.n_points = 1

        self.group_idx = kwds.pop('group_idx', None)

        super(MixedMixin, self).__init__(endog, exog, **kwds)

        # super is already calling _initialize, don't call it twice
        self._initialize()

    def _initialize(self):
        """
        Preprocesses the data for MXLogit
        """
        # Need a private Model Class for preprocesses the data for this model?

        #super(MixedMixin, self)._initialize()

        self.values_random = self.random_idx
        self.names_random = ['x%d' % d for d in range(self.n_rparam)]

        # TODO: name parsing
#        for name in self.NORMAL:
#            random = re.compile("[ ]\b*" + name)
#            for ii, jj in enumerate(self.paramsnames):
#                if random.search(jj) is not None:
#                    self.values_ramdon.append(ii)
#                    self.n_ramdon.append(jj)

        if DEBUG:
            print(self.values_ramdon)
            print(self.n_ramdon)

        ms_params = []

        for param in self.names_random:
            #ms_params.append('mean_%s' % param)
            ms_params.append('sd_%s' % param)

        self.ms_params = ms_params
        self.param_names = np.r_[self.exog_names, ms_params]
        self.k_params = len(self.param_names)
        # TODO: we should be able to remove the following after rebase to master
        self.data.xnames.extend(ms_params)

        #mapping coefficient names to indices to unique/parameter array
        self.paramsidx = OrderedDict((name, idx) for (idx, name) in
                              enumerate(self.param_names))

        if DEBUG:
            print(self.param_names)
            print(self.paramsidx)
            print(self.nparams)

        # TODO : implement test
        # from math import sqrt
        # sqrt(210)/2 # the ratio should be small (See Train, 2001)
        self.haltonsequence = halton(len(self.names_random), self.n_points)

    def drawndistri(self, params):
        """
        """
        dv = np.empty((len(self.haltonsequence), len(self.names_random)))

        for ii, name in enumerate(self.names_random):
            #mean = params[self.paramsidx['mean_' + name]]
            std = params[self.paramsidx['sd_' + name]]
            hs = self.haltonsequence[:, ii]
            mean = 0
            std = np.exp(std)
            #std = np.abs(std)
            if std < 0:
                raise RuntimeError('negative scale for random effect')
            dv[:, ii] = stats.norm.ppf(hs, mean, std)

        self.dv = dv

        return self.dv

    def cdf_(self, params):
        """
        Mixed Logit cumulative distribution function.

        Parameters
        ----------
        X : array (nobs, J)
            the linear predictor of the model.

        Returns
        --------
        cdf : ndarray
            the cdf evaluated at `X`.

        Notes
        -----
        .. math:: P_{ni} = \int L_{ni} (\beta) f(\beta) d\beta = \int \frac{e^{\beta x_{ni}}} {\sum_J e^{\beta x_{nj}}} f(\beta) d \beta

        if :math:` \phi(\beta | b,W)' is the normal distribution density with
        mean 'b' and covariance 'W', it can be re-written:

        .. math:: P_{ni} = \int \frac{e^{\beta x_{ni}}} {\sum_J e^{\beta x_{nj}}} \phi(\beta | b,W) d \beta

        However, it can be applied with different distribution types, such as
        lognormal or uniform distributions.

        """
        # index should handle empty random
        return np.exp(self.loglikeobs(params[:len(params)-len(self.names_random)]))

    def cdf_average(self, params):
        """
        """
        if DEBUG:
            print("___working in average")

        # special case for no random effects
        if len(self.names_random) == 0:
            return self.cdf_(params)

        self.drawndistri(params)

        pdf_sum = None
        for jj in range(self.dv.shape[0]):
            if DEBUG:
                print(self.values_ramdon)
                print(self.dv[jj])

            params_ = params.copy()
            params_[self.values_random] += self.dv[jj]  # numpy.ndarray

            if DEBUG:
                print(params_)

            if pdf_sum is None:
                # initialize
                pdf_sum = self.cdf_(params_)
            else:
                pdf_sum += self.cdf_(params_)

        if DEBUG:
            print("___returning to average")

        return pdf_sum / self.n_points

    def cdf_average_groups(self, params):
        """
        """
        if DEBUG:
            print("___working in average")

        # special case for no random effects
        if len(self.names_random) == 0:
            return self.cdf_(params)

        self.drawndistri(params)

        pdf_sum = None
        for jj in range(self.dv.shape[0]):
            if DEBUG:
                print(self.values_ramdon)
                print(self.dv[jj])

            params_ = params.copy()
            params_[self.values_random] += self.dv[jj]  # numpy.ndarray

            if DEBUG:
                print(params_)

            params_reduced = params_[:len(params_)-len(self.names_random)]
            if pdf_sum is None:
                # initialize
                #pdf_sum = np.exp(self.loglikeobs(params_reduced))
                llfs = np.add.reduceat(self.loglikeobs(params_reduced),
                                       self.group_idx)
                pdf_sum = np.exp(llfs)
            else:
                #pdf_sum += np.exp(self.loglikeobs(params_reduced))
                llfs = np.add.reduceat(self.loglikeobs(params_reduced),
                                       self.group_idx)
                pdf_sum += np.exp(llfs)

        if DEBUG:
            print("___returning to average")

        return pdf_sum / self.n_points

    def loglike(self, params):

        """
        Log-likelihood of the mixed logit model.

        Parameters
        ----------
        params : array
            the parameters of the model.

        Returns
        -------
        loglike : float
            the log-likelihood function of the model evaluated at `params`.

        Notes
        ------
        Since mixed logit probability is not a close form, it needs to be
        approximated through simulation.

        Assume :math:`\beta^r;' is generated from the 'r'-th random draw.
        There are totally R times of draws in the simulation.
        The average simulated probability can be expressed as:

        .. math:: \bar{P}_{ni} = \frac{1}{R} \sum_{r=0}^{R} L_{ni} (\beta^r)

        so, the simulated log-likelihood:

        .. math:: \LL = \sum_{n=1}^{N} \sum_{j=0}^{J} d_{ij} \ln \bar{P}_{nj}

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """
        if DEBUG:
            print("___working in loglike")
            print(params)

        for name in self.names_random:
            std = params[self.paramsidx['sd_' + name]]
            #std = 1e-8 if std < 1e-8 else std
            params[self.paramsidx['sd_' + name]] = std

        if self.group_idx is None:
            loglike = np.log(self.cdf_average(params))
        else:
            loglike = np.log(self.cdf_average_groups(params))
        if DEBUG:
            print("___returning to loglike")

        return loglike.sum()

    def score(self, params):
        """
        """
        from statsmodels.tools.numdiff import approx_fprime
        return approx_fprime(params, self.loglike, epsilon=1e-8)

    def hessian(self, params, **kwds):
        """
        """
        from statsmodels.tools.numdiff import approx_hess
        return approx_hess(params, self.loglike)

    def fit(self, start_params=None, maxiter=5000, maxfun=5000,
            method='bfgs', full_output=1, disp=None, callback=None, **kwds):
        """
        Fits MXLogit() model using maximum likelihood.

        Returns
        -------
        Fit object for likelihood based models
        See: GenericLikelihoodModelResults

        """
        start_time = time.time()

        if start_params is None:
            start_params = 0.1 * np.ones(self.k_params)

            # TODO: add better default later
#            if DEBUG:
#                print("__working on start params")
#
#            Logit_res = Logit(self.endog, self.exog_matrix).fit(disp=0)
#
#            # Initial values of:
#            # means -> coeficients from conditional logit
#            # standart deviations -> 0.1
#            func_params = []
#
#            for rand in self.values_ramdon:
#                mean = Logit_res.params[rand]  # loc
#                #func_params.append(mean)
#                sd = 0.1   # loc
#                #sd = np.log(sd)
#                func_params.append(sd)
#
#            start_params = np.r_[Logit_res.params, func_params]
#
#            if DEBUG:
#                print("start_params", start_params)
#
#            self.satpar = start_params

        else:
            start_params = np.asarray(start_params)

        if DEBUG:
            print("___working on fit")

        model_fit = super(MixedMixin, self).fit(disp = disp,
                                            start_params = start_params,
                                            method=method, maxiter=maxiter,
                                            maxfun=maxfun, **kwds)

        self.params = model_fit.params

        if DEBUG:
            print(self.params)
            print("___returning to fit")

        end_time = time.time()
        self.elapsed_time = end_time - start_time

        return model_fit


#    def loglike(self, params):
#        llf = super(MixedMixin, self).loglike(params)
#        return llf - self.pen_weight * self.penal.func(params)
#
#    def score(self, params):
#        sc = super(MixedMixin, self).score(params)
#        return sc - self.pen_weight * self.penal.grad(params)
#
#    def hessian_(self, params):
#        sc = super(MixedMixin, self).hessian(params)
#        return sc - np.diag(self.pen_weight * self.penal.grad(params))

    def predict(self, params, exog=None, **kwds):
        # we need to strip the extra params
        # we might also or instead have to adjust the results predict
        # or agree on a convention
        # TODO temporary solution see issue #2387
        try:
            res = super(MixedMixin, self).predict(params, exog=None, **kwds)
        except ValueError:
            params = params[:len(params)-len(self.names_random)]
            res = super(MixedMixin, self).predict(params, exog=exog, **kwds)

        return res

