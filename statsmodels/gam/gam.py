from __future__ import division

__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '08/07/15'

import numpy as np
import scipy as sp
from statsmodels.discrete.discrete_model import Logit
from scipy.stats import chi2
from statsmodels.genmod.generalized_linear_model import (GLM, GLMResults,
    GLMResultsWrapper, lm, _check_convergence)
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty


## this class will be later removed and taken from another push
class PenalizedMixin2(object):
    """Mixin class for Maximum Penalized Likelihood
    TODO: missing **kwds or explicit keywords
    TODO: do we really need `pen_weight` keyword in likelihood methods?
    """

    def __init__(self, *args, **kwds):
        super(PenalizedMixin, self).__init__(*args, **kwds)
        penal = kwds.pop('penal', None)
        # I keep the following instead of adding default in pop for future changes
        if penal is None:
            print("Define a penalty")

        else:
            self.penal = penal

        # TODO: define pen_weight as average pen_weight? i.e. per observation
        # I would have prefered len(self.endog) * kwds.get('pen_weight', 1)
        # or use pen_weight_factor in signature
        self.pen_weight = kwds.get('pen_weight', len(self.endog))

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
            sc -= pen_weight / nobs_sc * self.penal.grad(params)

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
            kwds.update({'max_start_irls': 0})

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
            drop_index = np.nonzero(np.abs(res.params) < 1e-4)[0]
            keep_index = np.nonzero(np.abs(res.params) > 1e-4)[0]
            rmat = np.eye(len(res.params))[drop_index]

            # calling fit_constrained raise
            # "RuntimeError: maximum recursion depth exceeded in __instancecheck__"
            # fit_constrained is calling fit, recursive endless loop
            if drop_index.any():
                # todo : trim kwyword doesn't work, why not?
                # res_aux = self.fit_constrained(rmat, trim=False)
                res_aux = self._fit_zeros(keep_index, **kwds)
                return res_aux
            else:
                return res


class GLMGAMResults(GLMResults):

    def partial_values(self, smoother, variable):
        mask = smoother.mask[variable]
        y = np.dot(smoother.basis_[:, mask], self.params[mask])
        # select the submatrix corresponding to a single variable
        partial_cov_params = self.cov_params()[mask, :]
        partial_cov_params = partial_cov_params[:, mask]

        # var = np.diag(smoother.basis_[:, mask].dot(partial_cov_params).dot(smoother.basis_[:, mask].T))
        exog = smoother.basis_[:, mask]
        covb = partial_cov_params
        var = (exog * np.dot(covb, exog.T).T).sum(1)
        se = np.sqrt(var)

        return y, se

    def plot_partial(self, smoother, variable, plot_se=True):
        """just to try a method in overridden Results class
        """
        import matplotlib.pyplot as plt
        y_est, se = self.partial_values(smoother, variable)

        x = smoother.smoothers_[variable].x
        sort_index = np.argsort(x)
        x = x[sort_index]
        y_est = y_est[sort_index]
        se = se[sort_index]

        plt.figure()
        plt.plot(x, y_est, c='blue')
        if plot_se:
            plt.plot(smoother.x, y_est + 1.96 * se, '.', c='blue')
            plt.plot(smoother.x, y_est - 1.96 * se, '.', c='blue')

        plt.xlabel(smoother.smoothers_[variable].variable_name)

        return

    def significance_test(self, basis=None, y=None, alpha=None):
        # TODO: this is not working
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


class GLMGam(PenalizedMixin, GLM):
    _results_class = GLMGAMResults

    def __init__(self, endog, exog=None, smoother=None, alpha=0, family=None,
                 offset=None, exposure=None, missing='none', **kwargs):

        if exog is not None:
            exog_linear = np.asarray(exog)
            k_exog_linear = exog_linear.shape[1]
        else:
            exog_linear = None
            k_exog_linear = 0
        self.k_exog_linear = k_exog_linear

        import collections
        if not isinstance(alpha, collections.Iterable):
            alpha = np.array([alpha] * len(smoother.smoothers_))

        self.smoother = smoother
        self.alpha = alpha
        penal = MultivariateGamPenalty(smoother, alpha=alpha,
                                       start_idx=k_exog_linear,
                                       )
        kwargs.pop('penal', None)
        if exog_linear is not None:
            exog = np.column_stack((exog_linear, smoother.basis_))
        else:
            exog = smoother.basis_
        super(GLMGam, self).__init__(endog, exog=exog, family=family,
                                     offset=offset, exposure=exposure,
                                     penal=penal, missing=missing, **kwargs)


    def fit(self, start_params=None, maxiter=1000, method='PIRLS', tol=1e-8,
            scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None,
            full_output=True, disp=False, max_start_irls=3, **kwargs):

        if method.lower() == 'pirls':
            return self._fit_pirls(self.alpha,
                                   cov_type=cov_type, cov_kwds=cov_kwds,
                                   **kwargs)
        else:
            return super(GLMGam, self).fit(start_params=start_params, maxiter=maxiter, method=method, tol=tol,
                                           scale=scale, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
                                           full_output=full_output, disp=disp, max_start_irls=max_start_irls, **kwargs)
        return

    # pag 165 4.3 # pag 136 PIRLS
    def _fit_pirls(self, alpha, start_params=None, maxiter=100, tol=1e-8,
                   scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, weights=None):

        # alpha = alpha * len(y) * self.scale / 100 # TODO: we need to rescale alpha
        endog = self.endog
        wlsexog = self.exog #smoother.basis_
        spl_s = self.smoother.penalty_matrices_
        if self.k_exog_linear > 0:
            if not isinstance(spl_s, list):
                spl_s = [spl_s]
            if not isinstance(alpha, list):
                alpha = [alpha]
            # assumes spl_s and alpha are lists
            spl_s = [np.zeros((self.k_exog_linear, self.k_exog_linear))] + spl_s
            alpha = [0] + alpha

        n_samples, n_columns = wlsexog.shape

        # TODO what are these values?
        if weights is None:
            self.data_weights = np.array([1.] * n_samples)
        else:
            self.data_weights = weights

        if not hasattr(self, '_offset_exposure'):
            self._offset_exposure = 0

        self.scaletype = 'dev'
        # during iteration
        self.scale = 1

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

            # TODO: is this equivalent to point 1 of page 136:  z = g(mu)(y - mu) + X beta  ?
            wlsendog = (lin_pred + self.family.link.deriv(mu) * (endog - mu)
                        - self._offset_exposure)

            # this defines the augmented matrix point 2a on page 136
            wls_results = penalized_wls(wlsexog, wlsendog, spl_s, self.weights,
                                        np.array(2.) * alpha)
            lin_pred = np.dot(wlsexog, wls_results.params).ravel() + self._offset_exposure
            mu = self.family.fitted(lin_pred)

            #self.scale = self.estimate_scale(mu)
            history = self._update_history(wls_results, mu, history)


            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)

            # TODO need atol, rtol
            converged = _check_convergence(criterion, iteration, tol, 0)
            if converged:
                break
        self.mu = mu
        self.scale = self.estimate_scale(mu)
        glm_results = GLMGAMResults(self, wls_results.params,
                                    wls_results.normalized_cov_params,
                                    self.scale,
                                    cov_type=cov_type, cov_kwds=cov_kwds,
                                    use_t=use_t)

        glm_results.method = "PIRLS"
        history['iteration'] = iteration + 1
        glm_results.fit_history = history
        glm_results.converged = converged

        return GLMResultsWrapper(glm_results)


class LogitGam(PenalizedMixin, Logit):
    def __init__(self, endog, smoother, alpha, *args, **kwargs):
        import collections
        if not isinstance(alpha, collections.Iterable):
            alpha = np.array([alpha] * len(smoother.smoothers_))

        self.smoother = smoother
        self.alpha = alpha
        self.pen_weight = 1  # TODO: pen weight should not be defined here!!
        penal = MultivariateGamPenalty(smoother, alpha=alpha)

        super(LogitGam, self).__init__(endog, smoother.basis_, penal=penal, *args, **kwargs)

    pass


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
