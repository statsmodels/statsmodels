
from __future__ import division

__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '08/07/15'

import numpy as np
import scipy as sp
from statsmodels.discrete.discrete_model import Logit
from scipy.stats import chi2
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults, GLMResultsWrapper, lm, _check_convergence
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.sandbox.gam_gsoc2015.gam_penalties import UnivariateGamPenalty, MultivariateGamPenalty

## this class will be later removed and taken from another push
class PenalizedMixin(object):
    """Mixin class for Maximum Penalized Likelihood
    TODO: missing **kwds or explicit keywords
    TODO: do we really need `pen_weight` keyword in likelihood methods?
    """

    def __init__(self, *args, **kwds):
        super(PenalizedMixin, self).__init__(*args, **kwds)
        print(kwds)
        penal = kwds.pop('penal', None)
        # I keep the following instead of adding default in pop for future changes
        if penal is None:
           print("Define a penalty")

        else:
            self.penal = penal

        # TODO: define pen_weight as average pen_weight? i.e. per observation
        # I would have prefered len(self.endog) * kwds.get('pen_weight', 1)
        # or use pen_weight_factor in signature
        self.pen_weight =  kwds.get('pen_weight', len(self.endog))

        self._init_keys.extend(['penal', 'pen_weight'])

    def loglike(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        llf = super(PenalizedMixin, self).loglike(params)
        if pen_weight != 0:
            llf -= pen_weight * self.penal.func(params)

        return llf

    def loglikeobs(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        llf = super(PenalizedMixin, self).loglikeobs(params)
        nobs_llf = float(llf.shape[0])

        if pen_weight != 0:
            llf -= pen_weight / nobs_llf * self.penal.func(params)

        return llf

    def score(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        sc = super(PenalizedMixin, self).score(params)
        if pen_weight != 0:
            sc -= pen_weight * self.penal.grad(params)

        return sc

    def scoreobs(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        sc = super(PenalizedMixin, self).scoreobs(params)
        nobs_sc = float(sc.shape[0])
        if pen_weight != 0:
            sc -= pen_weight / nobs_sc  * self.penal.grad(params)

        return sc

    def hessian_(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight
            loglike = self.loglike
        else:
            loglike = lambda p: self.loglike(p, pen_weight=pen_weight)

        from statsmodels.tools.numdiff import approx_hess
        return approx_hess(params, loglike)

    def hessian(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        hess = super(PenalizedMixin, self).hessian(params)
        if pen_weight != 0:
            h = self.penal.deriv2(params)
            if h.ndim == 1:
                hess -= np.diag(pen_weight * h)
            else:
                hess -= pen_weight * h

        return hess

    def fit(self, method=None, trim=None, **kwds):
        # If method is None, then we choose a default method ourselves

        # TODO: temporary hack, need extra fit kwds
        # we need to rule out fit methods in a model that will not work with
        # penalization
        if hasattr(self, 'family'):  # assume this identifies GLM
            kwds.update({'max_start_irls' : 0})

        # currently we use `bfgs` by default
        if method is None:
            method = 'bfgs'

        if trim is None:
            trim = False  # see below infinite recursion in `fit_constrained

        res = super(PenalizedMixin, self).fit(method=method, **kwds)

        if trim is False:
            # note boolean check for "is False" not evaluates to False
            return res
        else:
            # TODO: make it penal function dependent
            # temporary standin, only works for Poisson and GLM,
            # and is computationally inefficient
            drop_index = np.nonzero(np.abs(res.params) < 1e-4) [0]
            keep_index = np.nonzero(np.abs(res.params) > 1e-4) [0]
            rmat = np.eye(len(res.params))[drop_index]

            # calling fit_constrained raise
            # "RuntimeError: maximum recursion depth exceeded in __instancecheck__"
            # fit_constrained is calling fit, recursive endless loop
            if drop_index.any():
                # todo : trim kwyword doesn't work, why not?
                #res_aux = self.fit_constrained(rmat, trim=False)
                res_aux = self._fit_zeros(keep_index, **kwds)
                return res_aux
            else:
                return res


class LogitGam(PenalizedMixin, Logit):
    pass


class GLMGAMResults(GLMResults):

    def partial_values(self, smoother, mask):

        y = np.dot(smoother.basis_, self.params[mask])
        # select the submatrix corresponding to a single variable
        partial_normalized_cov_params = self.normalized_cov_params[mask, :]
        partial_normalized_cov_params = partial_normalized_cov_params[:, mask]

        var = np.diag(smoother.basis_.dot(partial_normalized_cov_params).dot(smoother.basis_.T))
        se = np.sqrt(var)
        return y, se

    def plot_partial(self, multivariate_smoother, plot_se=True):
        """just to try a method in overridden Results class
        """
        import matplotlib.pyplot as plt
        # TODO: This function will be available when we will have the self.model.x variable
        # if x_values is None:
        #     plt.plot(self.model.x, self.model.endog, '.')
        #     plt.plot(self.model.x, self.predict())
        # else:

        for i, smoother in enumerate(multivariate_smoother.smoothers_):
            y_est, se = self.partial_values(smoother, multivariate_smoother.mask[i])

            plt.figure()
            plt.plot(smoother.x, y_est, '.')
            if plot_se:
                plt.plot(smoother.x, y_est + 1.96 * se, '.')
                plt.plot(smoother.x, y_est - 1.96 * se, '.')
            plt.xlabel(smoother.variable_name)
            plt.show()

        return

    def significance_test(self, basis=None, y=None, alpha=None):
        # v = basis.dot(self.normalized_cov_params).dot(basis.T)
        # p_inv_v = pinv(v)
        # hat_y = self.predict(basis)
        # tr = hat_y.T.dot(p_inv_v).dot(hat_y)
        #
        # # TODO: FIRST WAY TO COMPUTE DF
        # lin_pred = self.predict(basis)
        # mu = self.family.fitted(lin_pred)
        # mu = self.family.link(mu)
        #
        # weights = self._data_weights*self.family.weights(mu)
        # weights /= len(weights) # A normalization is probably required
        #
        #
        # f = self.normalized_cov_params.dot(basis.T * weights).dot(basis) / self.scale
        # rank = np.trace(2 * f - np.dot(f, f))

        # TODO: Second way to estimate the significance
        n_samples, k_var = basis.shape
        r = np.linalg.qr(basis, 'r')

        vf = r.dot(self.normalized_cov_params).dot(r.T)
        vf_inv = np.linalg.pinv(vf)
        tr = self.params.T.dot(r.T).dot(vf_inv).dot(r).dot(self.params)

        rank = 1
        p_val = 1 - chi2.cdf(tr, df=rank)
        print('tr=', tr, 'pval=', p_val, 'rank=', rank, "scale=", self.scale)
        print('expected values: tr=', 8.141, 'pval=', 0.0861, 'rank=', 3.997)

        return tr, p_val, rank


# TODO: This is an old version of GLMGAM and is used only for testing. This will soon be removed
class GLMGam(PenalizedMixin, GLM):

    _results_class = GLMGAMResults

    # pag 165 4.3 # pag 136 PIRLS
    def _fit_pirls(self, y, spl_x, spl_s, alpha, start_params=None, maxiter=100, tol=1e-8,
                   scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, weights=None):

        endog = y
        wlsexog = spl_x

        n_samples, n_columns = wlsexog.shape

        # TODO what are these values?
        if weights is None:
            self.data_weights = np.array([1.] * n_samples)
        else:
            self.data_weights = weights

        self._offset_exposure = np.array([.1] * n_samples)
        self.scaletype = 'dev'

        if start_params is None:
            mu = self.family.starting_mu(endog)
            lin_pred = self.family.predict(mu)
        else:
            lin_pred = np.dot(wlsexog, start_params) + self._offset_exposure
            mu = self.family.fitted(lin_pred)
        dev = self.family.deviance(endog, mu)

        history = dict(params=[None, start_params], deviance=[np.inf, dev])
        converged = False
        criterion = history['deviance']
        # This special case is used to get the likelihood for a specific
        # params vector.
        if maxiter == 0:
            mu = self.family.fitted(lin_pred)
            self.scale = self.estimate_scale(mu)
            wls_results = lm.RegressionResults(self, start_params, None)
            iteration = 0

        for iteration in range(maxiter):

            # TODO: is this equivalent to point 1 of page 136: w = 1 / (V(mu) * g'(mu))  ?
            self.weights = self.data_weights * self.family.weights(mu)

            #TODO: is this equivalent to point 1 of page 136:  z = g(mu)(y - mu) + X beta  ?
            wlsendog = (lin_pred + self.family.link.deriv(mu) * (endog-mu)
                        - self._offset_exposure)

            # this defines the augmented matrix point 2a on page 136
            wls_results = penalized_wls(wlsexog, wlsendog, spl_s, self.weights, alpha)
            lin_pred = np.dot(wlsexog, wls_results.params).ravel() + self._offset_exposure
            mu = self.family.fitted(lin_pred)

            history = self._update_history(wls_results, mu, history)

            self.scale = self.estimate_scale(mu)
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)
            converged = _check_convergence(criterion, iteration, tol)
            if converged:
                break
        self.mu = mu

        glm_results = GLMResults(self, wls_results.params,
                                 wls_results.normalized_cov_params,
                                 self.scale,
                                 cov_type=cov_type, cov_kwds=cov_kwds,
                                 use_t=use_t)

        glm_results.method = "PIRLS"
        history['iteration'] = iteration + 1
        glm_results.fit_history = history
        glm_results.converged = converged
        return GLMResultsWrapper(glm_results)


def penalized_wls(x, y, s, weights, alpha):

    aug_x, aug_y, aug_weights = make_augmented_matrix(x, y, s, weights, alpha)
    wls_results = lm.WLS(aug_y, aug_x, aug_weights).fit()
    wls_results.params = wls_results.params.ravel()

    return wls_results


def make_augmented_matrix(x, y, s, w, alphas):

    n_samples, n_columns = x.shape
    import collections

    if isinstance(alphas, collections.Iterable):
        alpha_s = sp.linalg.block_diag(*[s[i] * alphas[i] for i in range(len(alphas))])
    else:
        alpha_s = alphas * s

    rs = get_sqrt(alpha_s)
    x1 = np.vstack([x, rs])  # augmented x
    n_samp1es_x1 = x1.shape[0]

    y1 = np.array([0.] * n_samp1es_x1)  # augmented y
    y1[:n_samples] = y

    id1 = np.array([1.] * n_columns)
    w1 = np.concatenate([w, id1])
    w1 = np.sqrt(w1)

    return x1, y1, w1


def get_sqrt(x):
    u, s, v = np.linalg.svd(x)
    s[s < 0] = 0

    sqrt_s = np.sqrt(s)

    b = np.dot(u, np.dot(np.diag(sqrt_s), v))
    return b
