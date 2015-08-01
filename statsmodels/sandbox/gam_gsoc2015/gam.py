
from __future__ import division

__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '08/07/15'

import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from statsmodels.discrete.discrete_model import Logit
from scipy.stats import chi2
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults, GLMResultsWrapper, lm


## this class will be later removed and taken from another push
class PenalizedMixin(object):
    """Mixin class for Maximum Penalized Likelihood
    TODO: missing **kwds or explicit keywords
    TODO: do we really need `pen_weight` keyword in likelihood methods?
    """

    def __init__(self, *args, **kwds):
        super(PenalizedMixin, self).__init__(*args, **kwds)

        penal = kwds.pop('penal', None)
        # I keep the following instead of adding default in pop for future changes
        if penal is None:
            # TODO: switch to unpenalized by default
            self.penal = SCADSmoothed(0.1, c0=0.0001)
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


## this class will be later removed and taken from another push
class Penalty(object):
    """
    A class for representing a scalar-value penalty.
    Parameters
    wts : array-like
        A vector of weights that determines the weight of the penalty
        for each parameter.
    Notes
    -----
    The class has a member called `alpha` that scales the weights.
    """

    def __init__(self, wts):
        self.wts = wts
        self.alpha = 1.

    def func(self, params):
        """
        A penalty function on a vector of parameters.
        Parameters
        ----------
        params : array-like
            A vector of parameters.
        Returns
        -------
        A scalar penaty value; greater values imply greater
        penalization.
        """
        raise NotImplementedError

    def grad(self, params):
        """
        The gradient of a penalty function.
        Parameters
        ----------
        params : array-like
            A vector of parameters
        Returns
        -------
        The gradient of the penalty with respect to each element in
        `params`.
        """
        raise NotImplementedError


class UnivariateGamPenalty(Penalty):
    __doc__ = """
    Penalty for Generalized Additive Models class

    Parameters
    -----------
    alpha : float
        the penalty term

    wts: TODO: I do not know!

    cov_der2: the covariance matrix of the second derivative of the basis matrix

    der2: The second derivative of the basis function

    Attributes
    -----------
    alpha : float
        the penalty term

    wts: TODO: I do not know!

    cov_der2: the covariance matrix of the second derivative of the basis matrix

    der2: The second derivative of the basis function

    n_samples: The number of samples used during the estimation



    """

    def __init__(self, univariate_smoother, wts=1, alpha=1):

        self.wts = wts #should we keep wts????
        self.alpha = alpha
        self.univariate_smoother = univariate_smoother
        self.n_samples, self.n_columns = self.univariate_smoother.n_samples, self.univariate_smoother.dim_basis

    def func(self, params):
        '''
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        '''

        # The second derivative of the estimated regression function
        f = np.dot(self.univariate_smoother.der2_basis_, params)

        return self.alpha * np.sum(f**2) / self.n_samples

    def grad(self, params):
        '''
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        3) cov_der2 is obtained as np.dot(der2.T, der2)
        '''

        return 2 * self.alpha * np.dot(self.univariate_smoother.cov_der2_, params) / self.n_samples

    def deriv2(self, params):
        return 2 * self.alpha * self.univariate_smoother.cov_der2_ / self.n_samples


class MultivariateGamPenalty(Penalty):
    __doc__ = """
    GAM penalty for multivariate regression

    Parameters
    -----------
    cov_der2: list of matrices
     is a list of squared matrix of shape (size_base, size_base)

    der2: list of matrices
     is a list of matrix of shape (n_samples, size_base)

    alpha: array-like
     list of doubles. Each one representing the penalty
          for each function

    wts: array-like
     is a list of doubles of the same length of alpha

    """

    def __init__(self, multivariate_smoother, wts=None, alphas=None):

        if len(multivariate_smoother.smoothers_) != len(alphas):
            raise ValueError('all the input values should be list of the same length')

        self.multivariate_smoother = multivariate_smoother
        self.k_columns = self.multivariate_smoother.k_columns
        self.k_variables = self.multivariate_smoother.k_variables
        self.n_samples = self.multivariate_smoother.n_samples
        self.alphas = alphas

        # TODO: Review this
        if wts is None:
            self.wts = [1] * len(alphas)
        else:
            self.wts = wts

        self.mask = [np.array([False]*self.k_columns)
                     for _ in range(self.k_variables)]
        param_count = 0
        for i, smoother in enumerate(self.multivariate_smoother.smoothers_):

            # the mask[i] contains a vector of length k_columns. The index
            # corresponding to the i-th input variable are set to True.
            self.mask[i][param_count: param_count + smoother.dim_basis] = True
            param_count += smoother.dim_basis

        self.gp = []
        for i in range(self.k_variables):
            gp = UnivariateGamPenalty(wts=self.wts[i], alpha=self.alphas[i],
                                      univariate_smoother=self.multivariate_smoother.smoothers_[i])
            self.gp.append(gp)

        return

    def func(self, params):
        cost = 0
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            cost += self.gp[i].func(params_i)

        return cost

    def grad(self, params):
        grad = []
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            grad.append(self.gp[i].grad(params_i))

        return np.concatenate(grad)

    def deriv2(self, params):
        deriv2 = np.empty(shape=(0,0))
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            deriv2 = block_diag(deriv2, self.gp[i].deriv2(params_i))

        return deriv2


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


def get_sqrt(x):
     """
     :param x:
     :return: b the sqrt of the matrix x. np.dot(b.T, b) = x
     """
     u, s, v = np.linalg.svd(x)
     sqrt_s = np.sqrt(s)
     b = np.dot(u, np.dot(np.diag(sqrt_s), v))
     return b


class GLMGam(PenalizedMixin, GLM):

    _results_class = GLMGAMResults

    def _fit_pirls(self, y, spl_x, spl_s, alpha, start_params=None, maxiter=100, tol=1e-8,
                   scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, weights=None):

        n_samples, n_columns = spl_x.shape

        if isinstance(spl_s, list):
            alpha_spl_s = [spl_s[i] * alpha[i] for i in range(len(spl_s))]
            alpha_spl_s = sp.linalg.block_diag(*alpha_spl_s)
        else:
            alpha_spl_s = alpha * spl_s

        rs = get_sqrt(alpha_spl_s)

        x1 = np.vstack([spl_x, rs])  # augmented x
        n_samp1es_x1 = x1.shape[0]
        y1 = np.array([0] * n_samp1es_x1)  # augmented y
        y1[:n_samples] = y

        if weights is None:
            self.weights = np.array([1./n_samp1es_x1] * n_samp1es_x1)  # TODO: should the weight be of size n_samples_x1?
                                                                       # Probably we have to use equation 4.22 from the Wood's
                                                                       # book and replace WLS with OLS

        if start_params is None:
            params = np.zeros(shape=(n_columns,))
            params[0] = 1
        else:
            params = start_params

        lin_pred = np.dot(x1, params)[:n_samples]
        self.mu = self.family.fitted(lin_pred)
        dev = self.family.deviance(self.endog, self.mu)

        history = dict(params=[None, params], deviance=[np.inf, dev])
        converged = False
        criterion = history['deviance']

        new_norm = 0
        old_norm = 1
        iteration = 0
        for iteration in range(maxiter):
            if np.abs(new_norm - old_norm) < tol:
                break

            lin_pred = np.dot(x1, params)[:n_samples]
            self.mu = self.family.fitted(lin_pred)
            z = np.zeros(shape=(n_samp1es_x1,))
            z[:n_samples] = (y - self.mu) / self.mu + lin_pred # TODO: review this and the following line
            wls_results = lm.WLS(z, x1, self.weights).fit() # TODO: should weights be used?
            lin_pred = np.dot(spl_x, wls_results.params)

            new_mu = self.family.fitted(lin_pred)

            old_norm = new_norm
            new_norm = np.linalg.norm((z[:n_samples] - new_mu))

        self.scale = 0 # TODO: check the right value scale
        self.data_weights = 0 # TODO: add support for weights
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
