import numpy as np
from statsmodels.discrete.discrete_model import Logit
from statsmodels.api import GLM
from smooth_basis import BS
from patsy.state import stateful_transform



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


class GamPenalty(Penalty):

    def __init__(self, wts=1, alpha=1, cov_der2=None, der2=None):

        self.wts = wts #should we keep wts????
        self.alpha = alpha
        self.cov_der2 = cov_der2
        self.der2 = der2

    def func(self, params):
        '''
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        '''

        # The second derivative of the estimated regression function
        f = np.dot(self.der2, params)

        return self.alpha * np.sum(f**2)

    def grad(self, params):
        '''
        1) params are the coefficients in the regression model
        2) der2  is the second derivative of the splines basis
        3) cov_der2 is obtained as np.dot(der2.T, der2)
        '''

        return 2 * self.alpha * np.dot(self.cov_der2, params)

    def deriv2(self, params):

        return 2 * self.alpha * self.cov_der2





class MultivariateGamPenalty(Penalty):

    def __init__(self, wts=None, alpha=None, cov_der2=None, der2=None):
        '''
        GAM penalty for multivariate regression
        - cov_der2 is a list of squared matrix of shape (size_base, size_base)
        - der2 is a list of matrix of shape (n_samples, size_base)
        - alpha is a list of doubles. Each one representing the penalty
          for each function
        - wts is a list of doubles of the same length of alpha
        '''

        assert(len(cov_der2) == len(der2))

        # the total number of columns in der2 i.e. the len of the params vector
        self.n_columns = np.sum(d2.shape[1] for d2 in der2)

        # the number of variables in the GAM model
        self.n_variables = len(cov_der2)

        # if wts and alpha are not a list then each function has the same penalty
        # TODO: Review this
        self.alpha = alpha
        self.wts = wts

        n_samples = der2[0].shape[0]
        self.mask = [np.array([False]*self.n_columns)
                     for i in range(self.n_variables)]
        param_count = 0
        for i, d2 in enumerate(der2):
            n, dim_base = d2.shape
            #check that all the basis have the same number of samples
            assert(n_samples == n)
            self.mask[i][param_count: param_count + dim_base] = True
            param_count += dim_base

        self.gp = []
        for i in range(self.n_variables):
            gp = GamPenalty(wts=self.wts[i], alpha=self.alpha[i],
                            cov_der2=cov_der2[i], der2=der2[i])
            self.gp.append(gp)

        return


    def func(self, params):

        cost = 0
        for i in range(self.n_variables):
            params_i = params[self.mask[i]]
            cost += self.gp[i].func(params_i)

        return  cost


    def grad(self, params):
        grad = []
        for i in range(self.n_variables):
            params_i = params[self.mask[i]]
            grad.append(self.gp[i].grad(params_i))

        return np.concatenate(grad)


    def deriv2(self, params):
        deriv2 = []
        for i in range(self.n_variables):
            params_i = params[self.mask[i]]
            deriv2.append(self.gp[i].grad(params_i))
        return np.concatenate(deriv2)



class LogitGam(PenalizedMixin, Logit):
    pass


class GLMGam(PenalizedMixin, GLM):
    pass

