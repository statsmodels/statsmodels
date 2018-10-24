# -*- coding: utf-8 -*-
"""
Generalized Additive Models

Author: Luca Puggini
Author: Josef Perktold

created on 08/07/2015
"""

from __future__ import division
import numpy as np
import scipy as sp
from statsmodels.discrete.discrete_model import Logit
from scipy.stats import chi2
from statsmodels.genmod.generalized_linear_model import (GLM, GLMResults,
    GLMResultsWrapper, lm, _check_convergence)
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.tools.linalg import matrix_sqrt


class GLMGAMResults(GLMResults):
    """Results class for generalized additive models, GAM.

    This inherits from GLMResults.

    Warning: not all inherited methods might take correctly account of the
    penalization

    """

    def predict(self, exog=None, x=None, transform=True, **kwds):
        if transform is False:
            ex = exog
        else:
            if x is not None:
                exog_smooth = self.model.smoother.transform(x)
                if exog is None:
                    ex = exog_smooth
                else:
                    ex = np.column_stack((exog, exog_smooth))
            else:
                ex = exog
        return super(GLMGAMResults, self).predict(ex, **kwds)

    def partial_values(self, smoother, variable):
        """contribution of a smooth term to the linear prediction

        Returns
        -------
        predicted : nd_array
            predicted value of linear term.
            This is not the expected response if the link function is not
            linear.
        se_pred : nd_array
            standard error of linear prediction

        """
        mask = smoother.mask[variable]
        y = np.dot(smoother.basis_[:, mask], self.params[mask])
        # select the submatrix corresponding to a single variable
        partial_cov_params = self.cov_params()[mask, :]
        partial_cov_params = partial_cov_params[:, mask]

        # var = np.diag(smoother.basis_[:, mask].dot(
        #       partial_cov_params).dot(smoother.basis_[:, mask].T))
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

        return  # TODO

    def significance_test(self, basis=None, y=None, alpha=None):
        """hypothesis test that a smooth component is zero.

        not verified and not yet correct.

        """
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
    """Model class for generalized additive models, GAM.

    This inherits from `GLM`.

    Warning: not all inherited methods might take correctly account of the
    penalization

    """

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

        self.smoother = smoother
        self.alpha = self._check_alpha(alpha)
        penal = MultivariateGamPenalty(smoother, alpha=self.alpha,
                                       start_idx=k_exog_linear)
        kwargs.pop('penal', None)
        if exog_linear is not None:
            exog = np.column_stack((exog_linear, smoother.basis_))
        else:
            exog = smoother.basis_
        super(GLMGam, self).__init__(endog, exog=exog, family=family,
                                     offset=offset, exposure=exposure,
                                     penal=penal, missing=missing, **kwargs)

    def _check_alpha(self, alpha):
        import collections
        if not isinstance(alpha, collections.Iterable):
            alpha = [alpha] * len(self.smoother.smoothers_)
        elif not isinstance(alpha, list):
            # we want alpha to be a list
            alpha = list(alpha)
        return alpha

    def fit(self, start_params=None, maxiter=1000, method='PIRLS', tol=1e-8,
            scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None,
            full_output=True, disp=False, max_start_irls=3, **kwargs):
        """estimate parameters and create instance of GLMGAMResults class

        Parameters
        ----------
        most parameters are the same as for GLM
        method : optimization method
            The special optimization method is "pirls" which uses a penalized
            version of IRLS. Other methods are gradient optimizers as used in
            base.model.LikelihoodModel.

        Returns
        -------
        res : instance of GLMGAMResults
        """
        # TODO: alpha not allowed yet, but is in `_fit_pirls`
        # alpha = self._check_alpha()

        if method.lower() == 'pirls':
            res = self._fit_pirls(self.alpha,
                                  cov_type=cov_type, cov_kwds=cov_kwds,
                                  **kwargs)
        else:
            res = super(GLMGam, self).fit(start_params=start_params,
                                          maxiter=maxiter, method=method,
                                          tol=tol, scale=scale,
                                          cov_type=cov_type, cov_kwds=cov_kwds,
                                          use_t=use_t,
                                          full_output=full_output, disp=disp,
                                          max_start_irls=max_start_irls,
                                          **kwargs)
        return res

    # pag 165 4.3 # pag 136 PIRLS
    def _fit_pirls(self, alpha, start_params=None, maxiter=100, tol=1e-8,
                   scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None,
                   weights=None):

        # alpha = alpha * len(y) * self.scale / 100
        # TODO: we need to rescale alpha
        endog = self.endog
        k_exog_linear = self.k_exog_linear
        wlsexog = self.exog #smoother.basis_
        spl_s = self.penal.penalty_matrix(alpha=alpha)

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

            # TODO: is this equivalent to point 1 of page 136:
            # w = 1 / (V(mu) * g'(mu))  ?
            self.weights = self.data_weights * self.family.weights(mu)

            # TODO: is this equivalent to point 1 of page 136:
            # z = g(mu)(y - mu) + X beta  ?
            wlsendog = (lin_pred + self.family.link.deriv(mu) * (endog - mu)
                        - self._offset_exposure)

            # this defines the augmented matrix point 2a on page 136
            wls_results = penalized_wls(wlsexog, wlsendog, spl_s, self.weights)
                                        #np.array(2.) * alpha)
            lin_pred = np.dot(wlsexog, wls_results.params).ravel()
            lin_pred += self._offset_exposure
            mu = self.family.fitted(lin_pred)

            # self.scale = self.estimate_scale(mu)
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
    """Generalized Additive model for discrete Logit

    This subclasses discrete_model Logit.

    Warning: not all inherited methods might take correctly account of the
    penalization

    not verified yet.

    """
    def __init__(self, endog, smoother, alpha, *args, **kwargs):
        import collections
        if not isinstance(alpha, collections.Iterable):
            alpha = np.array([alpha] * len(smoother.smoothers_))

        self.smoother = smoother
        self.alpha = alpha
        self.pen_weight = 1  # TODO: pen weight should not be defined here!!
        penal = MultivariateGamPenalty(smoother, alpha=alpha)

        super(LogitGam, self).__init__(endog, smoother.basis_, penal=penal,
                                       *args, **kwargs)


def penalized_wls(x, y, s, weights):
    """weighted least squares with quadratic penalty
    """
    # TODO: I don't understand why I need 2 * s
    aug_x, aug_y, aug_weights = make_augmented_matrix(x, y, 2 * s, weights)
    wls_results = lm.WLS(aug_y, aug_x, aug_weights).fit()
    wls_results.params = wls_results.params.ravel()

    return wls_results


def make_augmented_matrix(x, y, s, w):
    n_samples, n_columns = x.shape

    # TODO: needs full because of broadcasting with weights
    # check what weights should be doing
    rs = matrix_sqrt(s)#, full=True)
    x1 = np.vstack([x, rs])  # augmented x
    n_samp1es_x1 = x1.shape[0]

    y1 = np.array([0.] * n_samp1es_x1)  # augmented y
    y1[:n_samples] = y

    id1 = np.array([1.] * rs.shape[0])
    w1 = np.concatenate([w, id1])
    w1 = np.sqrt(w1)

    return x1, y1, w1
