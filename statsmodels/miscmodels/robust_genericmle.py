# -*- coding: utf-8 -*-
"""

Created on Thu Feb 27 13:32:36 2014

Author: Josef Perktold
"""

import numpy as np
from statsmodels.base.model import GenericLikelihoodModel

import statsmodels.robust.norms as rnorms
import statsmodels.robust.scale as rscale
import statsmodels.robust.robust_linear_model as rlm


class MEstimator(GenericLikelihoodModel):

    def __init__(self, *args, **kwds):
        self.norm = rnorms.HuberT()
        super(MEstimator, self).__init__(*args, **kwds)
        self.scale_fixed = None

    def transform_params(self, params):

        if self.scale_fixed is None:
            params_loc, scale = params[:-1], params[-1]
            scale = np.abs(scale)
        else:
            params_loc = params
            scale = self.scale_fixed

        return params_loc, scale


    def nloglikeobs(self, params):
        # Note: norms is a loss function, so this is nloglikeobs

        params_loc, scale = self.transform_params(params)

        #nobs = len(self.endog)
        resid_scaled = (self.endog - np.dot(self.exog, params_loc)) / scale
        scale_fac = 0.491
        rho_fac = scale * scale_fac
        objective = (rho_fac * self.norm(resid_scaled / scale_fac) +
                      rho_fac * np.log(scale * scale_fac))
        return objective

    def score(self, params):
        return self.jac(params).sum(0)

    def jac(self, params): #, **kwds):
        # Todo   verify signs,  derivative of loglike or nloglike ?
        params_loc, scale = self.transform_params(params)

        resid_scaled = (self.endog - np.dot(self.exog, params_loc)) / scale / 0.491
        score_obs = -self.norm.psi(resid_scaled)[:,None] * self.exog #/ scale
        # note: extra scale cancels between nominator and denominator

        if self.scale_fixed is None:
            def func(s):
                s = np.atleast_1d(s)
                #for derivative wrt. scale
                p = np.concatenate((params_loc, s))
                return self.loglikeobs(p)

            import statsmodels.tools.numdiff as nd
            grad_scale = nd.approx_fprime([scale], func)
            return -np.column_stack((score_obs, -grad_scale))

        else:
            return -score_obs

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(exog, params[:self.exog.shape[1]])


class MEstimatorHD(GenericLikelihoodModel):

    def __init__(self, *args, **kwds):
        self.norm = kwds.get('norm', rnorms.HuberT())
        super(MEstimatorHD, self).__init__(*args, **kwds)
        self.scale_fixed = None
        # TODO verify scale_fac
        # stats.norm.expect(lambda t: t*norm.psi(t) - norm.rho(t))
        self.scale_fac = 0.35508
        if isinstance(self.norm, rnorms.TukeyBiweight):
            # hardcoded for default shape/tuning parameter
            self.scale_fac = 3.97913

    def transform_params(self, params):

        if self.scale_fixed is None:
            params_loc, scale = params[:-1], params[-1]
            scale = np.abs(scale)
        else:
            params_loc = params
            scale = self.scale_fixed

        return params_loc, scale


    def nloglikeobs(self, params):
        # Note: norms is a loss function, so this is nloglikeobs

        params_loc, scale = self.transform_params(params)

        #nobs = len(self.endog)
        resid_scaled = (self.endog - self.predict(params_loc)) / scale
        scale_fac = self.scale_fac
        objective = (scale * self.norm(resid_scaled) + scale_fac * scale)
        return objective

    def score(self, params):
        return self.jac(params).sum(0)

    def jac(self, params): #, **kwds):
        # Todo   verify signs,  derivative of loglike or nloglike ?
        params_loc, scale = self.transform_params(params)

        resid_scaled = (self.endog - self.predict(params_loc)) / scale
        score_obs = -self.norm.psi(resid_scaled)[:,None] * self.predict_jac(params_loc) #/ scale
        # note: extra scale cancels between nominator and denominator

        if self.scale_fixed is None:
            def func(s):
                s = np.atleast_1d(s)
                #for derivative wrt. scale
                p = np.concatenate((params_loc, s))
                return self.loglikeobs(p)

            import statsmodels.tools.numdiff as nd
            grad_scale = nd.approx_fprime(np.atleast_1d(scale), func)
            return -np.column_stack((score_obs, -grad_scale))

        else:
            return -score_obs

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(exog, params[:self.exog.shape[1]])

    def predict_jac(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return exog

    def fit_iterative(self, start_params=None, **kwds):
        """
        iterative fit with MAD scale updating

        """
        # TODO add fit keywords, method is hardcoded

        # TODO: this should only be needed in `fit`
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params(self.endog, self.exog)
            else:
                start_params = 0.1 * np.ones(self.exog.shape[1])

        res_it = self.fit(start_params, method='nm')
        history = []
        #self.scale_fixed = rscale.mad(res_it.resid, center=0)
        for i in range(10):
            fittedvalues = res_it.predict()
            self.scale_fixed = rscale.mad(self.endog - fittedvalues, center=0)
            res_it = self.fit(res_it.params, method='bfgs')
            history.append(np.concatenate((res_it.params, [self.scale_fixed])))
            if (i > 0) and np.allclose(history[-1], history[-2], atol=1e-6, rtol=1e-6):
                # Note: we check when we have at least two fit iterations
                converged = True
                break
        else:
            converged = False

        #TODO: update res_it with iteration results
        res_it.converged = converged
        res_it.history = history

        return res_it



class RLMIterative(rlm.RLM):

    def __init__(self, *args, **kwds):
        self.weights_prior = kwds.pop('weights', None)
        self.scale_bias = kwds.pop('weights', 1)   # TODO
        self.meef_scale = kwds.pop('meef_scale', None)
        self.update_scale = kwds.pop('update_scale', True)

        super(RLMIterative, self).__init__(*args, **kwds)
        # linear to get started
        from statsmodels.regression.linear_model import WLS
        self.fit_loc = WLS
        self.norm_loc = self.M



    def update_location(self, weight, *args):
        pass

    def fit(self, start_params=None, start_scale=None, update_scale=True,
            optim_options=None):
        # note scale is not squared
        #nobs = self.endog.shape[0]

        if optim_options is None:
            optim_options = {}
        maxiter = optim_options.get('maxiter', 10)
        res_loc = self.fit_loc(self.endog, self.exog).fit()#, weights=None)
        if start_params is None:
            resid = res_loc.resid
            params = res_loc.params
        else:
            params = start_params
            resid = self.endog - res_loc.model.predict(params)
        if start_scale is None:
            # use MAD for now, TODO: use same as in iteration
            scale = rscale.mad(resid, center=0)
        else:
            scale = start_scale

        for i in range(maxiter):
            # update location
            weights = self.norm_loc.weights(resid / scale)
            if self.weights_prior is not None:
                weights *= self.weights_prior
            res_loc = self.fit_loc(self.endog, self.exog, weights=weights).fit()
            params_old = params
            params = res_loc.params
            resid = res_loc.resid
            resid_scaled = resid / scale

            if update_scale:
                # update scale
                # Note: use old residuals or new ?
                weights_scale = self.meef_scale(resid_scaled) / resid_scaled**2
                #scale_old = scale
                scale2 = (weights_scale * resid**2).mean()
                #scale *= scale_old**2
                scale2 /= self.scale_bias
                scale = np.sqrt(scale2)

            #check convergence
            if np.allclose(params, params_old, rtol=1e-6, atol=1e-10):
                converged = True
                break
        else:
            converged = False

        # Force estimated scale for WLS results
        res_loc._cache['scale'] = scale**2

        # TODO return Results
        return res_loc, scale, converged


# examples: special cases of non-linear functions

def func_menten(params, x):
    a, b = params
    return a * x / (np.exp(b) + x)

class MentenNL(MEstimatorHD):

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.squeeze(func_menten(params[:2], exog))

    def predict_jac(self, params, exog=None):
        if exog is None:
            exog = self.exog
        from statsmodels.tools.numdiff import approx_fprime
        return approx_fprime(params[:2], self.predict)


class ExpNL(MEstimatorHD):

    def predict(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        xb = np.dot(exog, params[:self.exog.shape[1]])
        if linear:
            return xb
        else:
            return np.exp(xb)

    def _predict_jac(self, params, exog=None):
        if exog is None:
            exog = self.exog
        from statsmodels.tools.numdiff import approx_fprime
        return approx_fprime(params[:2], self.predict)

    def predict_jac(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        if linear:
            return exog
        else:
            xb = np.dot(exog, params[:self.exog.shape[1]])
            return np.exp(xb)[:, None] * exog


def sigmoid(params, x):
    x0, y0, c, k = params
    y = c / (1. + np.exp(-k * (x - x0))) + y0
    return y


def sigmoid_deriv(params, x):
    x0, y0, c, k = params
    term = np.exp(-k * (x - x0))
    denom = 1. / (1 + term)
    denom2 = denom**2
    dx0 =  - c * denom2 * term * k
    dy0 = np.ones(x.shape[0])
    dc = denom
    dk =  c * denom2 * term * (x - x0)

    return np.column_stack([dx0, dy0, dc, dk])

def sig_start(y, x):
    #return x.min(), y.min(), y.max(), np.corrcoef(x, y)[0, 1]
    return np.median(x), np.median(y), y.max(), np.corrcoef(x, y)[0, 1]


class SigmoidNL(MEstimatorHD):

    def predict(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        xb = np.dot(exog, params[:self.exog.shape[1]])
        if linear:
            return xb
        else:
            return np.squeeze(sigmoid(params[:4], exog))

    def _predict_jac(self, params, exog=None):
        if exog is None:
            exog = self.exog
        from statsmodels.tools.numdiff import approx_fprime
        return approx_fprime(params[:4], self.predict)

    def predict_jac(self, params, exog=None, linear=False):
        if exog is None:
            exog = self.exog
        if linear:
            # doesn't make sense in this case
            return exog
        else:
            #xb = np.dot(exog, params[:self.exog.shape[1]])
            return sigmoid_deriv(params[:4], exog)
