"""
Limited dependent variable and qualitative variables.

Includes binary outcomes, count data, (ordered) ordinal data and limited
dependent variables.

General References
--------------------

A.C. Cameron and P.K. Trivedi.  `Regression Analysis of Count Data`.
    Cambridge, 1998

G.S. Madalla. `Limited-Dependent and Qualitative Variables in Econometrics`.
    Cambridge, 1983.

W. Greene. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
"""
from __future__ import division

__all__ = ["Poisson", "Logit", "Probit", "MNLogit", "NegativeBinomial"]

from statsmodels.compat.python import lmap, lzip, range
import numpy as np
from scipy.special import gammaln
from scipy import stats, special, optimize  # opt just for nbin
import statsmodels.tools.tools as tools
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import (resettable_cache,
        cache_readonly)
from statsmodels.regression.linear_model import OLS
from scipy import stats, special, optimize  # opt just for nbin
from scipy.stats import nbinom
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.tools.numdiff import (approx_fprime, approx_hess,
                                       approx_hess_cs, approx_fprime_cs)
import statsmodels.base.model as base
from statsmodels.base.data import handle_data  # for mnlogit
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.compat.numpy import np_matrix_rank
from pandas.core.api import get_dummies

from statsmodels.base.l1_slsqp import fit_l1_slsqp
try:
    import cvxopt
    have_cvxopt = True
except ImportError:
    have_cvxopt = False

#TODO: When we eventually get user-settable precision, we need to change
#      this
FLOAT_EPS = np.finfo(float).eps

#TODO: add options for the parameter covariance/variance
# ie., OIM, EIM, and BHHH see Green 21.4

_discrete_models_docs = """
"""

_discrete_results_docs = """
    %(one_line_description)s

    Parameters
    ----------
    model : A DiscreteModel instance
    params : array-like
        The parameters of a fitted model.
    hessian : array-like
        The hessian of the fitted model.
    scale : float
        A scale parameter for the covariance matrix.

    Returns
    -------
    *Attributes*

    aic : float
        Akaike information criterion.  `-2*(llf - p)` where `p` is the number
        of regressors including the intercept.
    bic : float
        Bayesian information criterion. `-2*llf + ln(nobs)*p` where `p` is the
        number of regressors including the intercept.
    bse : array
        The standard errors of the coefficients.
    df_resid : float
        See model definition.
    df_model : float
        See model definition.
    fitted_values : array
        Linear predictor XB.
    llf : float
        Value of the loglikelihood
    llnull : float
        Value of the constant-only loglikelihood
    llr : float
        Likelihood ratio chi-squared statistic; `-2*(llnull - llf)`
    llr_pvalue : float
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
    prsquared : float
        McFadden's pseudo-R-squared. `1 - (llf / llnull)`
%(extra_attr)s"""

_l1_results_attr = """    nnz_params : Integer
        The number of nonzero parameters in the model.  Train with
        trim_params == True or else numerical error will distort this.
    trimmed : Boolean array
        trimmed[i] == True if the ith parameter was trimmed from the model."""


# helper for MNLogit (will be generally useful later)

def _numpy_to_dummies(endog):
    if endog.dtype.kind in ['S', 'O']:
        endog_dummies, ynames = tools.categorical(endog, drop=True,
                                                  dictnames=True)
    elif endog.ndim == 2:
        endog_dummies = endog
        ynames = range(endog.shape[1])
    else:
        endog_dummies, ynames = tools.categorical(endog, drop=True,
                                                  dictnames=True)
    return endog_dummies, ynames


def _pandas_to_dummies(endog):
    if endog.ndim == 2:
        if endog.shape[1] == 1:
            yname = endog.columns[0]
            endog_dummies = get_dummies(endog.iloc[:, 0])
        else:  # series
            yname = 'y'
            endog_dummies = endog
    else:
        yname = endog.name
        endog_dummies = get_dummies(endog)
    ynames = endog_dummies.columns.tolist()

    return endog_dummies, ynames, yname


#### Private Model Classes ####


class DiscreteModel(base.LikelihoodModel):
    """
    Abstract class for discrete choice models.

    This class does not do anything itself but lays out the methods and
    call signature expected of child classes in addition to those of
    statsmodels.model.LikelihoodModel.
    """
    def __init__(self, endog, exog, **kwargs):
        super(DiscreteModel, self).__init__(endog, exog, **kwargs)
        self.raise_on_perfect_prediction = True

    def initialize(self):
        """
        Initialize is called by
        statsmodels.model.LikelihoodModel.__init__
        and should contain any preprocessing that needs to be done for a model.
        """
        # assumes constant
        self.df_model = float(np_matrix_rank(self.exog) - 1)
        self.df_resid = (float(self.exog.shape[0] -
                         np_matrix_rank(self.exog)))

    def cdf(self, X):
        """
        The cumulative distribution function of the model.
        """
        raise NotImplementedError

    def pdf(self, X):
        """
        The probability density (mass) function of the model.
        """
        raise NotImplementedError

    def _check_perfect_pred(self, params, *args):
        endog = self.endog
        fittedvalues = self.cdf(np.dot(self.exog, params[:self.exog.shape[1]]))
        if (self.raise_on_perfect_prediction and
                np.allclose(fittedvalues - endog, 0)):
            msg = "Perfect separation detected, results not available"
            raise PerfectSeparationError(msg)

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        """
        Fit the model using maximum likelihood.

        The rest of the docstring is from
        statsmodels.base.model.LikelihoodModel.fit
        """
        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass # make a function factory to have multiple call-backs

        mlefit = super(DiscreteModel, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)

        return mlefit # up to subclasses to wrap results

    fit.__doc__ += base.LikelihoodModel.fit.__doc__

    def fit_regularized(self, start_params=None, method='l1',
                        maxiter='defined_by_method', full_output=1, disp=True,
                        callback=None, alpha=0, trim_mode='auto',
                        auto_trim_tol=0.01, size_trim_tol=1e-4, qc_tol=0.03,
                        qc_verbose=False, **kwargs):
        """
        Fit the model using a regularized maximum likelihood.
        The regularization method AND the solver used is determined by the
        argument method.

        Parameters
        ----------
        start_params : array-like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : 'l1' or 'l1_cvxopt_cp'
            See notes for details.
        maxiter : Integer or 'defined_by_method'
            Maximum number of iterations to perform.
            If 'defined_by_method', then use method defaults (see notes).
        full_output : bool
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool
            Set to True to print convergence messages.
        fargs : tuple
            Extra arguments passed to the likelihood function, i.e.,
            loglike(x,*args)
        callback : callable callback(xk)
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.
        alpha : non-negative scalar or numpy array (same size as parameters)
            The weight multiplying the l1 penalty term
        trim_mode : 'auto, 'size', or 'off'
            If not 'off', trim (set to zero) parameters that would have been
            zero if the solver reached the theoretical minimum.
            If 'auto', trim params using the Theory above.
            If 'size', trim params if they have very small absolute value
        size_trim_tol : float or 'auto' (default = 'auto')
            For use when trim_mode == 'size'
        auto_trim_tol : float
            For sue when trim_mode == 'auto'.  Use
        qc_tol : float
            Print warning and don't allow auto trim when (ii) (above) is
            violated by this much.
        qc_verbose : Boolean
            If true, print out a full QC report upon failure

        Notes
        -----
        Extra parameters are not penalized if alpha is given as a scalar.
        An example is the shape parameter in NegativeBinomial `nb1` and `nb2`.

        Optional arguments for the solvers (available in Results.mle_settings)::

            'l1'
                acc : float (default 1e-6)
                    Requested accuracy as used by slsqp
            'l1_cvxopt_cp'
                abstol : float
                    absolute accuracy (default: 1e-7).
                reltol : float
                    relative accuracy (default: 1e-6).
                feastol : float
                    tolerance for feasibility conditions (default: 1e-7).
                refinement : int
                    number of iterative refinement steps when solving KKT
                    equations (default: 1).


        Optimization methodology

        With :math:`L` the negative log likelihood, we solve the convex but
        non-smooth problem

        .. math:: \\min_\\beta L(\\beta) + \\sum_k\\alpha_k |\\beta_k|

        via the transformation to the smooth, convex, constrained problem
        in twice as many variables (adding the "added variables" :math:`u_k`)

        .. math:: \\min_{\\beta,u} L(\\beta) + \\sum_k\\alpha_k u_k,

        subject to

        .. math:: -u_k \\leq \\beta_k \\leq u_k.

        With :math:`\\partial_k L` the derivative of :math:`L` in the
        :math:`k^{th}` parameter direction, theory dictates that, at the
        minimum, exactly one of two conditions holds:

        (i) :math:`|\\partial_k L| = \\alpha_k`  and  :math:`\\beta_k \\neq 0`
        (ii) :math:`|\\partial_k L| \\leq \\alpha_k`  and  :math:`\\beta_k = 0`

        """
        ### Set attributes based on method
        if method in ['l1', 'l1_cvxopt_cp']:
            cov_params_func = self.cov_params_func_l1
        else:
            raise Exception("argument method == %s, which is not handled"
                            % method)

        ### Bundle up extra kwargs for the dictionary kwargs.  These are
        ### passed through super(...).fit() as kwargs and unpacked at
        ### appropriate times
        alpha = np.array(alpha)
        assert alpha.min() >= 0
        try:
            kwargs['alpha'] = alpha
        except TypeError:
            kwargs = dict(alpha=alpha)
        kwargs['alpha_rescaled'] = kwargs['alpha'] / float(self.endog.shape[0])
        kwargs['trim_mode'] = trim_mode
        kwargs['size_trim_tol'] = size_trim_tol
        kwargs['auto_trim_tol'] = auto_trim_tol
        kwargs['qc_tol'] = qc_tol
        kwargs['qc_verbose'] = qc_verbose

        ### Define default keyword arguments to be passed to super(...).fit()
        if maxiter == 'defined_by_method':
            if method == 'l1':
                maxiter = 1000
            elif method == 'l1_cvxopt_cp':
                maxiter = 70

        ## Parameters to pass to super(...).fit()
        # For the 'extra' parameters, pass all that are available,
        # even if we know (at this point) we will only use one.
        extra_fit_funcs = {'l1': fit_l1_slsqp}
        if have_cvxopt and method == 'l1_cvxopt_cp':
            from statsmodels.base.l1_cvxopt import fit_l1_cvxopt_cp
            extra_fit_funcs['l1_cvxopt_cp'] = fit_l1_cvxopt_cp
        elif method.lower() == 'l1_cvxopt_cp':
            message = ("Attempt to use l1_cvxopt_cp failed since cvxopt "
                        "could not be imported")

        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass # make a function factory to have multiple call-backs

        mlefit = super(DiscreteModel, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, extra_fit_funcs=extra_fit_funcs,
                cov_params_func=cov_params_func, **kwargs)

        return mlefit # up to subclasses to wrap results

    def cov_params_func_l1(self, likelihood_model, xopt, retvals):
        """
        Computes cov_params on a reduced parameter space
        corresponding to the nonzero parameters resulting from the
        l1 regularized fit.

        Returns a full cov_params matrix, with entries corresponding
        to zero'd values set to np.nan.
        """
        H = likelihood_model.hessian(xopt)
        trimmed = retvals['trimmed']
        nz_idx = np.nonzero(trimmed == False)[0]
        nnz_params = (trimmed == False).sum()
        if nnz_params > 0:
            H_restricted = H[nz_idx[:, None], nz_idx]
            # Covariance estimate for the nonzero params
            H_restricted_inv = np.linalg.inv(-H_restricted)
        else:
            H_restricted_inv = np.zeros(0)

        cov_params = np.nan * np.ones(H.shape)
        cov_params[nz_idx[:, None], nz_idx] = H_restricted_inv

        return cov_params

    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.
        """
        raise NotImplementedError

    def _derivative_exog(self, params, exog=None, dummy_idx=None,
            count_idx=None):
        """
        This should implement the derivative of the non-linear function
        """
        raise NotImplementedError

class BinaryModel(DiscreteModel):

    def __init__(self, endog, exog, **kwargs):
        super(BinaryModel, self).__init__(endog, exog, **kwargs)
        if (not issubclass(self.__class__, MultinomialModel) and
                not np.all((self.endog >= 0) & (self.endog <= 1))):
            raise ValueError("endog must be in the unit interval.")


    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array-like
            Fitted parameters of the model.
        exog : array-like
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used.
        linear : bool, optional
            If True, returns the linear predictor dot(exog,params).  Else,
            returns the value of the cdf at the linear predictor.

        Returns
        -------
        array
            Fitted values at exog.
        """
        if exog is None:
            exog = self.exog
        if not linear:
            return self.cdf(np.dot(exog, params))
        else:
            return np.dot(exog, params)

    def fit_regularized(self, start_params=None, method='l1',
            maxiter='defined_by_method', full_output=1, disp=1, callback=None,
            alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):
        bnryfit = super(BinaryModel, self).fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        if method in ['l1', 'l1_cvxopt_cp']:
            discretefit = L1BinaryResults(self, bnryfit)
        else:
            raise Exception(
                    "argument method == %s, which is not handled" % method)
        return L1BinaryResultsWrapper(discretefit)
    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__

    def _derivative_predict(self, params, exog=None, transform='dydx'):
        """
        For computing marginal effects standard errors.

        This is used only in the case of discrete and count regressors to
        get the variance-covariance of the marginal effects. It returns
        [d F / d params] where F is the predict.

        Transform can be 'dydx' or 'eydx'. Checking is done in margeff
        computations for appropriate transform.
        """
        if exog is None:
            exog = self.exog
        dF = self.pdf(np.dot(exog, params))[:,None] * exog
        if 'ey' in transform:
            dF /= self.predict(params, exog)[:,None]
        return dF

    def _derivative_exog(self, params, exog=None, transform='dydx',
            dummy_idx=None, count_idx=None):
        """
        For computing marginal effects returns dF(XB) / dX where F(.) is
        the predicted probabilities

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        #note, this form should be appropriate for
        ## group 1 probit, logit, logistic, cloglog, heckprob, xtprobit
        if exog is None:
            exog = self.exog
        margeff = np.dot(self.pdf(np.dot(exog, params))[:,None],
                                                          params[None,:])
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:,None]
        if count_idx is not None:
            from statsmodels.discrete.discrete_margins import (
                    _get_count_effects)
            margeff = _get_count_effects(margeff, exog, count_idx, transform,
                    self, params)
        if dummy_idx is not None:
            from statsmodels.discrete.discrete_margins import (
                    _get_dummy_effects)
            margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform,
                    self, params)
        return margeff

class MultinomialModel(BinaryModel):

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        if data_tools._is_using_ndarray_type(endog, None):
            endog_dummies, ynames = _numpy_to_dummies(endog)
            yname = 'y'
        elif data_tools._is_using_pandas(endog, None):
            endog_dummies, ynames, yname = _pandas_to_dummies(endog)
        else:
            endog = np.asarray(endog)
            endog_dummies, ynames = _numpy_to_dummies(endog)
            yname = 'y'

        if not isinstance(ynames, dict):
            ynames = dict(zip(range(endog_dummies.shape[1]), ynames))

        self._ynames_map = ynames
        data = handle_data(endog_dummies, exog, missing, hasconst, **kwargs)
        data.ynames = yname  # overwrite this to single endog name
        data.orig_endog = endog
        self.wendog = data.endog

        # repeating from upstream...
        for key in kwargs:
            try:
                setattr(self, key, data.__dict__.pop(key))
            except KeyError:
                pass
        return data

    def initialize(self):
        """
        Preprocesses the data for MNLogit.
        """
        super(MultinomialModel, self).initialize()
        # This is also a "whiten" method in other models (eg regression)
        self.endog = self.endog.argmax(1)  # turn it into an array of col idx
        self.J = self.wendog.shape[1]
        self.K = self.exog.shape[1]
        self.df_model *= (self.J-1)  # for each J - 1 equation.
        self.df_resid = self.exog.shape[0] - self.df_model - (self.J-1)

    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array-like
            2d array of fitted parameters of the model. Should be in the
            order returned from the model.
        exog : array-like
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used. If a 1d array is given
            it assumed to be 1 row of exogenous variables. If you only have
            one regressor and would like to do prediction, you must provide
            a 2d array with shape[1] == 1.
        linear : bool, optional
            If True, returns the linear predictor dot(exog,params).  Else,
            returns the value of the cdf at the linear predictor.

        Notes
        -----
        Column 0 is the base case, the rest conform to the rows of params
        shifted up one for the base case.
        """
        if exog is None: # do here to accomodate user-given exog
            exog = self.exog
        if exog.ndim == 1:
            exog = exog[None]
        pred = super(MultinomialModel, self).predict(params, exog, linear)
        if linear:
            pred = np.column_stack((np.zeros(len(exog)), pred))
        return pred

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        if start_params is None:
            start_params = np.zeros((self.K * (self.J-1)))
        else:
            start_params = np.asarray(start_params)
        callback = lambda x : None # placeholder until check_perfect_pred
        # skip calling super to handle results from LikelihoodModel
        mnfit = base.LikelihoodModel.fit(self, start_params = start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)
        mnfit.params = mnfit.params.reshape(self.K, -1, order='F')
        mnfit = MultinomialResults(self, mnfit)
        return MultinomialResultsWrapper(mnfit)
    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(self, start_params=None, method='l1',
            maxiter='defined_by_method', full_output=1, disp=1, callback=None,
            alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):
        if start_params is None:
            start_params = np.zeros((self.K * (self.J-1)))
        else:
            start_params = np.asarray(start_params)
        mnfit = DiscreteModel.fit_regularized(
                self, start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        mnfit.params = mnfit.params.reshape(self.K, -1, order='F')
        mnfit = L1MultinomialResults(self, mnfit)
        return L1MultinomialResultsWrapper(mnfit)
    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__


    def _derivative_predict(self, params, exog=None, transform='dydx'):
        """
        For computing marginal effects standard errors.

        This is used only in the case of discrete and count regressors to
        get the variance-covariance of the marginal effects. It returns
        [d F / d params] where F is the predicted probabilities for each
        choice. dFdparams is of shape nobs x (J*K) x (J-1)*K.
        The zero derivatives for the base category are not included.

        Transform can be 'dydx' or 'eydx'. Checking is done in margeff
        computations for appropriate transform.
        """
        if exog is None:
            exog = self.exog
        if params.ndim == 1: # will get flatted from approx_fprime
            params = params.reshape(self.K, self.J-1, order='F')

        eXB = np.exp(np.dot(exog, params))
        sum_eXB = (1 + eXB.sum(1))[:,None]
        J, K = lmap(int, [self.J, self.K])
        repeat_eXB = np.repeat(eXB, J, axis=1)
        X = np.tile(exog, J-1)
        # this is the derivative wrt the base level
        F0 = -repeat_eXB * X / sum_eXB ** 2
        # this is the derivative wrt the other levels when
        # dF_j / dParams_j (ie., own equation)
        #NOTE: this computes too much, any easy way to cut down?
        F1 = eXB.T[:,:,None]*X * (sum_eXB - repeat_eXB) / (sum_eXB**2)
        F1 = F1.transpose((1,0,2)) # put the nobs index first

        # other equation index
        other_idx = ~np.kron(np.eye(J-1), np.ones(K)).astype(bool)
        F1[:, other_idx] = (-eXB.T[:,:,None]*X*repeat_eXB / \
                           (sum_eXB**2)).transpose((1,0,2))[:, other_idx]
        dFdX = np.concatenate((F0[:, None,:], F1), axis=1)

        if 'ey' in transform:
            dFdX /= self.predict(params, exog)[:, :, None]
        return dFdX

    def _derivative_exog(self, params, exog=None, transform='dydx',
            dummy_idx=None, count_idx=None):
        """
        For computing marginal effects returns dF(XB) / dX where F(.) is
        the predicted probabilities

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.

        For Multinomial models the marginal effects are

        P[j] * (params[j] - sum_k P[k]*params[k])

        It is returned unshaped, so that each row contains each of the J
        equations. This makes it easier to take derivatives of this for
        standard errors. If you want average marginal effects you can do
        margeff.reshape(nobs, K, J, order='F).mean(0) and the marginal effects
        for choice J are in column J
        """
        J = int(self.J) # number of alternative choices
        K = int(self.K) # number of variables
        #note, this form should be appropriate for
        ## group 1 probit, logit, logistic, cloglog, heckprob, xtprobit
        if exog is None:
            exog = self.exog
        if params.ndim == 1: # will get flatted from approx_fprime
            params = params.reshape(K, J-1, order='F')
        zeroparams = np.c_[np.zeros(K), params] # add base in

        cdf = self.cdf(np.dot(exog, params))
        margeff = np.array([cdf[:,[j]]* (zeroparams[:,j]-np.array([cdf[:,[i]]*
            zeroparams[:,i] for i in range(int(J))]).sum(0))
                          for j in range(J)])
        margeff = np.transpose(margeff, (1,2,0))
        # swap the axes to make sure margeff are in order nobs, K, J
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:,None,:]

        if count_idx is not None:
            from statsmodels.discrete.discrete_margins import (
                    _get_count_effects)
            margeff = _get_count_effects(margeff, exog, count_idx, transform,
                    self, params)
        if dummy_idx is not None:
            from statsmodels.discrete.discrete_margins import (
                    _get_dummy_effects)
            margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform,
                    self, params)
        return margeff.reshape(len(exog), -1, order='F')

class CountModel(DiscreteModel):
    def __init__(self, endog, exog, offset=None, exposure=None, missing='none',
                 **kwargs):
        super(CountModel, self).__init__(endog, exog, missing=missing,
                                         offset=offset,
                                         exposure=exposure, **kwargs)
        if exposure is not None:
            self.exposure = np.log(self.exposure)
        self._check_inputs(self.offset, self.exposure, self.endog)
        if offset is None:
            delattr(self, 'offset')
        if exposure is None:
            delattr(self, 'exposure')

    def _check_inputs(self, offset, exposure, endog):
        if offset is not None and offset.shape[0] != endog.shape[0]:
            raise ValueError("offset is not the same length as endog")

        if exposure is not None and exposure.shape[0] != endog.shape[0]:
            raise ValueError("exposure is not the same length as endog")

    def _get_init_kwds(self):
        # this is a temporary fixup because exposure has been transformed
        # see #1609
        kwds = super(CountModel, self)._get_init_kwds()
        if 'exposure' in kwds and kwds['exposure'] is not None:
            kwds['exposure'] = np.exp(kwds['exposure'])
        return kwds

    def predict(self, params, exog=None, exposure=None, offset=None,
                linear=False):
        """
        Predict response variable of a count model given exogenous variables.

        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
        """
        #TODO: add offset tp
        if exog is None:
            exog = self.exog
            offset = getattr(self, 'offset', 0)
            exposure = getattr(self, 'exposure', 0)

        else:
            if exposure is None:
                exposure = 0
            else:
                exposure = np.log(exposure)
            if offset is None:
                offset = 0

        if not linear:
            return np.exp(np.dot(exog, params[:exog.shape[1]]) + exposure + offset) # not cdf
        else:
            return np.dot(exog, params[:exog.shape[1]]) + exposure + offset

    def _derivative_predict(self, params, exog=None, transform='dydx'):
        """
        For computing marginal effects standard errors.

        This is used only in the case of discrete and count regressors to
        get the variance-covariance of the marginal effects. It returns
        [d F / d params] where F is the predict.

        Transform can be 'dydx' or 'eydx'. Checking is done in margeff
        computations for appropriate transform.
        """
        if exog is None:
            exog = self.exog
        #NOTE: this handles offset and exposure
        dF = self.predict(params, exog)[:,None] * exog
        if 'ey' in transform:
            dF /= self.predict(params, exog)[:,None]
        return dF

    def _derivative_exog(self, params, exog=None, transform="dydx",
            dummy_idx=None, count_idx=None):
        """
        For computing marginal effects. These are the marginal effects
        d F(XB) / dX
        For the Poisson model F(XB) is the predicted counts rather than
        the probabilities.

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        # group 3 poisson, nbreg, zip, zinb
        if exog is None:
            exog = self.exog
        margeff = self.predict(params, exog)[:,None] * params[None,:]
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:,None]

        if count_idx is not None:
            from statsmodels.discrete.discrete_margins import (
                    _get_count_effects)
            margeff = _get_count_effects(margeff, exog, count_idx, transform,
                    self, params)
        if dummy_idx is not None:
            from statsmodels.discrete.discrete_margins import (
                    _get_dummy_effects)
            margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform,
                    self, params)
        return margeff

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        cntfit = super(CountModel, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)
        discretefit = CountResults(self, cntfit)
        return CountResultsWrapper(discretefit)
    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(self, start_params=None, method='l1',
            maxiter='defined_by_method', full_output=1, disp=1, callback=None,
            alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):
        cntfit = super(CountModel, self).fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        if method in ['l1', 'l1_cvxopt_cp']:
            discretefit = L1CountResults(self, cntfit)
        else:
            raise Exception(
                    "argument method == %s, which is not handled" % method)
        return L1CountResultsWrapper(discretefit)
    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__


class OrderedModel(DiscreteModel):
    pass

#### Public Model Classes ####

class Poisson(CountModel):
    __doc__ = """
    Poisson model for count data

%(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}


    def cdf(self, X):
        """
        Poisson model cumulative distribution function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the model.  See notes.

        Returns
        -------
        The value of the Poisson CDF at each point.

        Notes
        -----
        The CDF is defined as

        .. math:: \\exp\left(-\\lambda\\right)\\sum_{i=0}^{y}\\frac{\\lambda^{i}}{i!}

        where :math:`\\lambda` assumes the loglinear model. I.e.,

        .. math:: \\ln\\lambda_{i}=X\\beta

        The parameter `X` is :math:`X\\beta` in the above formula.
        """
        y = self.endog
        return stats.poisson.cdf(y, np.exp(X))

    def pdf(self, X):
        """
        Poisson model probability mass function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the model.  See notes.

        Returns
        -------
        pdf : ndarray
            The value of the Poisson probability mass function, PMF, for each
            point of X.

        Notes
        --------
        The PMF is defined as

        .. math:: \\frac{e^{-\\lambda_{i}}\\lambda_{i}^{y_{i}}}{y_{i}!}

        where :math:`\\lambda` assumes the loglinear model. I.e.,

        .. math:: \\ln\\lambda_{i}=x_{i}\\beta

        The parameter `X` is :math:`x_{i}\\beta` in the above formula.
        """
        y = self.endog
        return np.exp(stats.poisson.logpmf(y, np.exp(X)))

    def loglike(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        --------
        .. math :: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog
        return np.sum(-np.exp(XB) +  endog*XB - gammaln(endog+1))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Poisson model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        --------
        .. math :: \\ln L_{i}=\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]

        for observations :math:`i=1,...,n`

        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog
        #np.sum(stats.poisson.logpmf(endog, np.exp(XB)))
        return -np.exp(XB) +  endog*XB - gammaln(endog+1)

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        cntfit = super(CountModel, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)

        if 'cov_type' in kwargs:
            cov_kwds = kwargs.get('cov_kwds', {})
            kwds = {'cov_type':kwargs['cov_type'], 'cov_kwds':cov_kwds}
        else:
            kwds = {}
        discretefit = PoissonResults(self, cntfit, **kwds)
        return PoissonResultsWrapper(discretefit)
    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(self, start_params=None, method='l1',
            maxiter='defined_by_method', full_output=1, disp=1, callback=None,
            alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):
        cntfit = super(CountModel, self).fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        if method in ['l1', 'l1_cvxopt_cp']:
            discretefit = L1PoissonResults(self, cntfit)
        else:
            raise Exception(
                    "argument method == %s, which is not handled" % method)
        return L1PoissonResultsWrapper(discretefit)

    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__


    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        """fit the model subject to linear equality constraints

        The constraints are of the form   `R params = q`
        where R is the constraint_matrix and q is the vector of
        constraint_values.

        The estimation creates a new model with transformed design matrix,
        exog, and converts the results back to the original parameterization.

        Parameters
        ----------
        constraints : formula expression or tuple
            If it is a tuple, then the constraint needs to be given by two
            arrays (constraint_matrix, constraint_value), i.e. (R, q).
            Otherwise, the constraints can be given as strings or list of
            strings.
            see t_test for details
        start_params : None or array_like
            starting values for the optimization. `start_params` needs to be
            given in the original parameter space and are internally
            transformed.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the transformed model.

        Returns
        -------
        results : Results instance

        """

        #constraints = (R, q)
        # TODO: temporary trailing underscore to not overwrite the monkey
        #       patched version
        # TODO: decide whether to move the imports
        from patsy import DesignInfo
        from statsmodels.base._constraints import fit_constrained

        # same pattern as in base.LikelihoodModel.t_test
        lc = DesignInfo(self.exog_names).linear_constraint(constraints)
        R, q = lc.coefs, lc.constants

        # TODO: add start_params option, need access to tranformation
        #       fit_constrained needs to do the transformation
        params, cov, res_constr = fit_constrained(self, R, q,
                                                  start_params=start_params,
                                                  fit_kwds=fit_kwds)
        #create dummy results Instance, TODO: wire up properly
        res = self.fit(maxiter=0, method='nm', disp=0,
                       warn_convergence=False) # we get a wrapper back
        res.mle_retvals['fcall'] = res_constr.mle_retvals.get('fcall', np.nan)
        res.mle_retvals['iterations'] = res_constr.mle_retvals.get(
                                                        'iterations', np.nan)
        res.mle_retvals['converged'] = res_constr.mle_retvals['converged']
        res._results.params = params
        res._results.normalized_cov_params = cov
        k_constr = len(q)
        res._results.df_resid += k_constr
        res._results.df_model -= k_constr
        res._results.constraints = lc
        res._results.k_constr = k_constr
        res._results.results_constrained = res_constr
        return res


    def score(self, params):
        """
        Poisson model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\lambda_{i}\\right)x_{i}

        where the loglinear model is assumed

        .. math:: \\ln\\lambda_{i}=x_{i}\\beta
        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X,params) + offset + exposure)
        return np.dot(self.endog - L, X)

    def score_obs(self, params):
        """
        Poisson model Jacobian of the log-likelihood for each observation

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray (nobs, k_vars)
            The score vector of the model evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\lambda_{i}\\right)x_{i}

        for observations :math:`i=1,...,n`

        where the loglinear model is assumed

        .. math:: \\ln\\lambda_{i}=x_{i}\\beta
        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X,params) + offset + exposure)
        return (self.endog - L)[:,None] * X

    jac = np.deprecate(score_obs, 'jac', 'score_obs', "Use score_obs method."
                       " jac will be removed in 0.7")

    def hessian(self, params):
        """
        Poisson model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i=1}^{n}\\lambda_{i}x_{i}x_{i}^{\\prime}

        where the loglinear model is assumed

        .. math:: \\ln\\lambda_{i}=x_{i}\\beta

        """
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        X = self.exog
        L = np.exp(np.dot(X,params) + exposure + offset)
        return -np.dot(L*X.T, X)

class Logit(BinaryModel):
    __doc__ = """
    Binary choice logit model

%(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' : base._missing_param_doc}

    def cdf(self, X):
        """
        The logistic cumulative distribution function

        Parameters
        ----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.

        Returns
        -------
        1/(1 + exp(-X))

        Notes
        ------
        In the logit model,

        .. math:: \\Lambda\\left(x^{\\prime}\\beta\\right)=\\text{Prob}\\left(Y=1|x\\right)=\\frac{e^{x^{\\prime}\\beta}}{1+e^{x^{\\prime}\\beta}}
        """
        X = np.asarray(X)
        return 1/(1+np.exp(-X))

    def pdf(self, X):
        """
        The logistic probability density function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.

        Returns
        -------
        pdf : ndarray
            The value of the Logit probability mass function, PMF, for each
            point of X. ``np.exp(-x)/(1+np.exp(-X))**2``

        Notes
        -----
        In the logit model,

        .. math:: \\lambda\\left(x^{\\prime}\\beta\\right)=\\frac{e^{-x^{\\prime}\\beta}}{\\left(1+e^{-x^{\\prime}\\beta}\\right)^{2}}
        """
        X = np.asarray(X)
        return np.exp(-X)/(1+np.exp(-X))**2

    def loglike(self, params):
        """
        Log-likelihood of logit model.

        Parameters
        -----------
        params : array-like
            The parameters of the logit model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        ------
        .. math:: \\ln L=\\sum_{i}\\ln\\Lambda\\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        q = 2*self.endog - 1
        X = self.exog
        return np.sum(np.log(self.cdf(q*np.dot(X,params))))

    def loglikeobs(self, params):
        """
        Log-likelihood of logit model for each observation.

        Parameters
        -----------
        params : array-like
            The parameters of the logit model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        ------
        .. math:: \\ln L=\\sum_{i}\\ln\\Lambda\\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        for observations :math:`i=1,...,n`

        where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        q = 2*self.endog - 1
        X = self.exog
        return np.log(self.cdf(q*np.dot(X,params)))

    def score(self, params):
        """
        Logit model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params: array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\Lambda_{i}\\right)x_{i}
        """

        y = self.endog
        X = self.exog
        L = self.cdf(np.dot(X,params))
        return np.dot(y - L,X)

    def score_obs(self, params):
        """
        Logit model Jacobian of the log-likelihood for each observation

        Parameters
        ----------
        params: array-like
            The parameters of the model

        Returns
        -------
        jac : ndarray, (nobs, k_vars)
            The derivative of the loglikelihood for each observation evaluated
            at `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\Lambda_{i}\\right)x_{i}

        for observations :math:`i=1,...,n`

        """

        y = self.endog
        X = self.exog
        L = self.cdf(np.dot(X, params))
        return (y - L)[:,None] * X

    jac = np.deprecate(score_obs, 'jac', 'score_obs', "Use score_obs method."
                       " jac will be removed in 0.7")

    def hessian(self, params):
        """
        Logit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i}\\Lambda_{i}\\left(1-\\Lambda_{i}\\right)x_{i}x_{i}^{\\prime}
        """
        X = self.exog
        L = self.cdf(np.dot(X,params))
        return -np.dot(L*(1-L)*X.T,X)

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        bnryfit = super(Logit, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)

        discretefit = LogitResults(self, bnryfit)
        return BinaryResultsWrapper(discretefit)
    fit.__doc__ = DiscreteModel.fit.__doc__

class Probit(BinaryModel):
    __doc__ = """
    Binary choice Probit model

%(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' : base._missing_param_doc}

    def cdf(self, X):
        """
        Probit (Normal) cumulative distribution function

        Parameters
        ----------
        X : array-like
            The linear predictor of the model (XB).

        Returns
        --------
        cdf : ndarray
            The cdf evaluated at `X`.

        Notes
        -----
        This function is just an alias for scipy.stats.norm.cdf
        """
        return stats.norm._cdf(X)

    def pdf(self, X):
        """
        Probit (Normal) probability density function

        Parameters
        ----------
        X : array-like
            The linear predictor of the model (XB).

        Returns
        --------
        pdf : ndarray
            The value of the normal density function for each point of X.

        Notes
        -----
        This function is just an alias for scipy.stats.norm.pdf

        """
        X = np.asarray(X)
        return stats.norm._pdf(X)


    def loglike(self, params):
        """
        Log-likelihood of probit model (i.e., the normal distribution).

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i}\\ln\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        normal distribution is symmetric.
        """

        q = 2*self.endog - 1
        X = self.exog
        return np.sum(np.log(np.clip(self.cdf(q*np.dot(X,params)),
            FLOAT_EPS, 1)))

    def loglikeobs(self, params):
        """
        Log-likelihood of probit model for each observation

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----
        .. math:: \\ln L_{i}=\\ln\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        for observations :math:`i=1,...,n`

        where :math:`q=2y-1`. This simplification comes from the fact that the
        normal distribution is symmetric.
        """

        q = 2*self.endog - 1
        X = self.exog
        return np.log(np.clip(self.cdf(q*np.dot(X,params)), FLOAT_EPS, 1))


    def score(self, params):
        """
        Probit model score (gradient) vector

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left[\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}\\right]x_{i}

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        normal distribution is symmetric.
        """
        y = self.endog
        X = self.exog
        XB = np.dot(X,params)
        q = 2*y - 1
        # clip to get rid of invalid divide complaint
        L = q*self.pdf(q*XB)/np.clip(self.cdf(q*XB), FLOAT_EPS, 1 - FLOAT_EPS)
        return np.dot(L,X)

    def score_obs(self, params):
        """
        Probit model Jacobian for each observation

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        jac : ndarray, (nobs, k_vars)
            The derivative of the loglikelihood for each observation evaluated
            at `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left[\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}\\right]x_{i}

        for observations :math:`i=1,...,n`

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        normal distribution is symmetric.
        """
        y = self.endog
        X = self.exog
        XB = np.dot(X,params)
        q = 2*y - 1
        # clip to get rid of invalid divide complaint
        L = q*self.pdf(q*XB)/np.clip(self.cdf(q*XB), FLOAT_EPS, 1 - FLOAT_EPS)
        return L[:,None] * X

    jac = np.deprecate(score_obs, 'jac', 'score_obs', "Use score_obs method."
                       " jac will be removed in 0.7")

    def hessian(self, params):
        """
        Probit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\lambda_{i}\\left(\\lambda_{i}+x_{i}^{\\prime}\\beta\\right)x_{i}x_{i}^{\\prime}

        where

        .. math:: \\lambda_{i}=\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}

        and :math:`q=2y-1`
        """
        X = self.exog
        XB = np.dot(X,params)
        q = 2*self.endog - 1
        L = q*self.pdf(q*XB)/self.cdf(q*XB)
        return np.dot(-L*(L+XB)*X.T,X)

    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        bnryfit = super(Probit, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)
        discretefit = ProbitResults(self, bnryfit)
        return BinaryResultsWrapper(discretefit)
    fit.__doc__ = DiscreteModel.fit.__doc__

class MNLogit(MultinomialModel):
    __doc__ = """
    Multinomial logit model

    Parameters
    ----------
    endog : array-like
        `endog` is an 1-d vector of the endogenous response.  `endog` can
        contain strings, ints, or floats.  Note that if it contains strings,
        every distinct string will be a category.  No stripping of whitespace
        is done.
    exog : array-like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See `statsmodels.tools.add_constant`.
    %(extra_params)s

    Attributes
    ----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    J : float
        The number of choices for the endogenous variable. Note that this
        is zero-indexed.
    K : float
        The actual number of parameters for the exogenous design.  Includes
        the constant if the design has one.
    names : dict
        A dictionary mapping the column number in `wendog` to the variables
        in `endog`.
    wendog : array
        An n x j array where j is the number of unique categories in `endog`.
        Each column of j is a dummy variable indicating the category of
        each observation. See `names` for a dictionary mapping each column to
        its category.

    Notes
    -----
    See developer notes for further information on `MNLogit` internals.
    """ % {'extra_params' : base._missing_param_doc}

    def pdf(self, eXB):
        """
        NotImplemented
        """
        raise NotImplementedError

    def cdf(self, X):
        """
        Multinomial logit cumulative distribution function.

        Parameters
        ----------
        X : array
            The linear predictor of the model XB.

        Returns
        --------
        cdf : ndarray
            The cdf evaluated at `X`.

        Notes
        -----
        In the multinomial logit model.
        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}
        """
        eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
        return eXB/eXB.sum(1)[:,None]

    def loglike(self, params):
        """
        Log-likelihood of the multinomial logit model.

        Parameters
        ----------
        params : array-like
            The parameters of the multinomial logit model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        ------
        .. math:: \\ln L=\\sum_{i=1}^{n}\\sum_{j=0}^{J}d_{ij}\\ln\\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """
        params = params.reshape(self.K, -1, order='F')
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog,params)))
        return np.sum(d * logprob)

    def loglikeobs(self, params):
        """
        Log-likelihood of the multinomial logit model for each observation.

        Parameters
        ----------
        params : array-like
            The parameters of the multinomial logit model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        ------
        .. math:: \\ln L_{i}=\\sum_{j=0}^{J}d_{ij}\\ln\\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)

        for observations :math:`i=1,...,n`

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """
        params = params.reshape(self.K, -1, order='F')
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog,params)))
        return d * logprob

    def score(self, params):
        """
        Score matrix for multinomial logit model log-likelihood

        Parameters
        ----------
        params : array
            The parameters of the multinomial logit model.

        Returns
        --------
        score : ndarray, (K * (J-1),)
            The 2-d score vector, i.e. the first derivative of the
            loglikelihood function, of the multinomial logit model evaluated at
            `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta_{j}}=\\sum_{i}\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

        for :math:`j=1,...,J`

        In the multinomial model the score matrix is K x J-1 but is returned
        as a flattened array to work with the solvers.
        """
        params = params.reshape(self.K, -1, order='F')
        firstterm = self.wendog[:,1:] - self.cdf(np.dot(self.exog,
                                                  params))[:,1:]
        #NOTE: might need to switch terms if params is reshaped
        return np.dot(firstterm.T, self.exog).flatten()

    def loglike_and_score(self, params):
        """
        Returns log likelihood and score, efficiently reusing calculations.

        Note that both of these returned quantities will need to be negated
        before being minimized by the maximum likelihood fitting machinery.

        """
        params = params.reshape(self.K, -1, order='F')
        cdf_dot_exog_params = self.cdf(np.dot(self.exog, params))
        loglike_value = np.sum(self.wendog * np.log(cdf_dot_exog_params))
        firstterm = self.wendog[:, 1:] - cdf_dot_exog_params[:, 1:]
        score_array = np.dot(firstterm.T, self.exog).flatten()
        return loglike_value, score_array

    def score_obs(self, params):
        """
        Jacobian matrix for multinomial logit model log-likelihood

        Parameters
        ----------
        params : array
            The parameters of the multinomial logit model.

        Returns
        --------
        jac : ndarray, (nobs, k_vars*(J-1))
            The derivative of the loglikelihood for each observation evaluated
            at `params` .

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta_{j}}=\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}

        for :math:`j=1,...,J`, for observations :math:`i=1,...,n`

        In the multinomial model the score vector is K x (J-1) but is returned
        as a flattened array. The Jacobian has the observations in rows and
        the flatteded array of derivatives in columns.
        """
        params = params.reshape(self.K, -1, order='F')
        firstterm = self.wendog[:,1:] - self.cdf(np.dot(self.exog,
                                                  params))[:,1:]
        #NOTE: might need to switch terms if params is reshaped
        return (firstterm[:,:,None] * self.exog[:,None,:]).reshape(self.exog.shape[0], -1)

    jac = np.deprecate(score_obs, 'jac', 'score_obs', "Use score_obs method."
                       " jac will be removed in 0.7")

    def hessian(self, params):
        """
        Multinomial logit Hessian matrix of the log-likelihood

        Parameters
        -----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (J*K, J*K)
            The Hessian, second derivative of loglikelihood function with
            respect to the flattened parameters, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta_{j}\\partial\\beta_{l}}=-\\sum_{i=1}^{n}\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\left[\\boldsymbol{1}\\left(j=l\\right)-\\frac{\\exp\\left(\\beta_{l}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right]x_{i}x_{l}^{\\prime}

        where
        :math:`\\boldsymbol{1}\\left(j=l\\right)` equals 1 if `j` = `l` and 0
        otherwise.

        The actual Hessian matrix has J**2 * K x K elements. Our Hessian
        is reshaped to be square (J*K, J*K) so that the solvers can use it.

        This implementation does not take advantage of the symmetry of
        the Hessian and could probably be refactored for speed.
        """
        params = params.reshape(self.K, -1, order='F')
        X = self.exog
        pr = self.cdf(np.dot(X,params))
        partials = []
        J = self.wendog.shape[1] - 1
        K = self.exog.shape[1]
        for i in range(J):
            for j in range(J): # this loop assumes we drop the first col.
                if i == j:
                    partials.append(\
                        -np.dot(((pr[:,i+1]*(1-pr[:,j+1]))[:,None]*X).T,X))
                else:
                    partials.append(-np.dot(((pr[:,i+1]*-pr[:,j+1])[:,None]*X).T,X))
        H = np.array(partials)
        # the developer's notes on multinomial should clear this math up
        H = np.transpose(H.reshape(J,J,K,K), (0,2,1,3)).reshape(J*K,J*K)
        return H


#TODO: Weibull can replaced by a survival analsysis function
# like stat's streg (The cox model as well)
#class Weibull(DiscreteModel):
#    """
#    Binary choice Weibull model
#
#    Notes
#    ------
#    This is unfinished and untested.
#    """
##TODO: add analytic hessian for Weibull
#    def initialize(self):
#        pass
#
#    def cdf(self, X):
#        """
#        Gumbell (Log Weibull) cumulative distribution function
#        """
##        return np.exp(-np.exp(-X))
#        return stats.gumbel_r.cdf(X)
#        # these two are equivalent.
#        # Greene table and discussion is incorrect.
#
#    def pdf(self, X):
#        """
#        Gumbell (LogWeibull) probability distribution function
#        """
#        return stats.gumbel_r.pdf(X)
#
#    def loglike(self, params):
#        """
#        Loglikelihood of Weibull distribution
#        """
#        X = self.exog
#        cdf = self.cdf(np.dot(X,params))
#        y = self.endog
#        return np.sum(y*np.log(cdf) + (1-y)*np.log(1-cdf))
#
#    def score(self, params):
#        y = self.endog
#        X = self.exog
#        F = self.cdf(np.dot(X,params))
#        f = self.pdf(np.dot(X,params))
#        term = (y*f/F + (1 - y)*-f/(1-F))
#        return np.dot(term,X)
#
#    def hessian(self, params):
#        hess = nd.Jacobian(self.score)
#        return hess(params)
#
#    def fit(self, start_params=None, method='newton', maxiter=35, tol=1e-08):
## The example had problems with all zero start values, Hessian = 0
#        if start_params is None:
#            start_params = OLS(self.endog, self.exog).fit().params
#        mlefit = super(Weibull, self).fit(start_params=start_params,
#                method=method, maxiter=maxiter, tol=tol)
#        return mlefit
#

class NegativeBinomial(CountModel):
    __doc__ = """
    Negative Binomial Model for count data

%(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.

    References
    ----------

    References:

    Greene, W. 2008. "Functional forms for the negtive binomial model
        for count data". Economics Letters. Volume 99, Number 3, pp.585-590.
    Hilbe, J.M. 2011. "Negative binomial regression". Cambridge University
        Press.
    """ % {'params' : base._model_params_doc,
           'extra_params' :
           """loglike_method : string
        Log-likelihood type. 'nb2','nb1', or 'geometric'.
        Fitted value :math:`\\mu`
        Heterogeneity parameter :math:`\\alpha`

        - nb2: Variance equal to :math:`\\mu + \\alpha\\mu^2` (most common)
        - nb1: Variance equal to :math:`\\mu + \\alpha\\mu`
        - geometric: Variance equal to :math:`\\mu + \\mu^2`
    offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.

    """ + base._missing_param_doc}
    def __init__(self, endog, exog, loglike_method='nb2', offset=None,
                       exposure=None, missing='none', **kwargs):
        super(NegativeBinomial, self).__init__(endog, exog, offset=offset,
                                               exposure=exposure,
                                               missing=missing, **kwargs)
        self.loglike_method = loglike_method
        self._initialize()
        if loglike_method in ['nb2', 'nb1']:
            self.exog_names.append('alpha')
            self.k_extra = 1
        else:
            self.k_extra = 0
        # store keys for extras if we need to recreate model instance
        # we need to append keys that don't go to super
        self._init_keys.append('loglike_method')

    def _initialize(self):
        if self.loglike_method == 'nb2':
            self.hessian = self._hessian_nb2
            self.score = self._score_nbin
            self.loglikeobs = self._ll_nb2
            self._transparams = True # transform lnalpha -> alpha in fit
        elif self.loglike_method == 'nb1':
            self.hessian = self._hessian_nb1
            self.score = self._score_nb1
            self.loglikeobs = self._ll_nb1
            self._transparams = True # transform lnalpha -> alpha in fit
        elif self.loglike_method == 'geometric':
            self.hessian = self._hessian_geom
            self.score = self._score_geom
            self.loglikeobs = self._ll_geometric
        else:
            raise NotImplementedError("Likelihood type must nb1, nb2 or "
                                      "geometric")

    # Workaround to pickle instance methods
    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['hessian']
        del odict['score']
        del odict['loglikeobs']
        return odict

    def __setstate__(self, indict):
        self.__dict__.update(indict)
        self._initialize()

    def _ll_nbin(self, params, alpha, Q=0):
        endog = self.endog
        mu = self.predict(params)
        size = 1/alpha * mu**Q
        prob = size/(size+mu)
        coeff = (gammaln(size+endog) - gammaln(endog+1) -
                 gammaln(size))
        llf = coeff + size*np.log(prob) + endog*np.log(1-prob)
        return llf

    def _ll_nb2(self, params):
        if self._transparams: # got lnalpha during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        return self._ll_nbin(params[:-1], alpha, Q=0)

    def _ll_nb1(self, params):
        if self._transparams: # got lnalpha during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        return self._ll_nbin(params[:-1], alpha, Q=1)

    def _ll_geometric(self, params):
        # we give alpha of 1 because it's actually log(alpha) where alpha=0
        return self._ll_nbin(params, 1, 0)

    def loglike(self, params):
        r"""
        Loglikelihood for negative binomial model

        Parameters
        ----------
        params : array-like
            The parameters of the model. If `loglike_method` is nb1 or
            nb2, then the ancillary parameter is expected to be the
            last element.

        Returns
        -------
        llf : float
            The loglikelihood value at `params`

        Notes
        -----
        Following notation in Greene (2008), with negative binomial
        heterogeneity parameter :math:`\alpha`:

        .. math::

           \lambda_i &= exp(X\beta) \\
           \theta &= 1 / \alpha \\
           g_i &= \theta \lambda_i^Q \\
           w_i &= g_i/(g_i + \lambda_i) \\
           r_i &= \theta / (\theta+\lambda_i) \\
           ln \mathcal{L}_i &= ln \Gamma(y_i+g_i) - ln \Gamma(1+y_i) + g_iln (r_i) + y_i ln(1-r_i)

        where :math`Q=0` for NB2 and geometric and :math:`Q=1` for NB1.
        For the geometric, :math:`\alpha=0` as well.

        """
        llf = np.sum(self.loglikeobs(params))
        return llf

    def _score_geom(self, params):
        exog = self.exog
        y = self.endog[:,None]
        mu = self.predict(params)[:,None]
        dparams = exog * (y-mu)/(mu+1)
        return dparams.sum(0)

    def _score_nbin(self, params, Q=0):
        """
        Score vector for NB2 model
        """
        if self._transparams: # lnalpha came in during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        exog = self.exog
        y = self.endog[:,None]
        mu = self.predict(params)[:,None]
        a1 = 1/alpha * mu**Q
        if Q: # nb1
            dparams = exog*mu/alpha*(np.log(1/(alpha + 1)) +
                       special.digamma(y + mu/alpha) -
                       special.digamma(mu/alpha))
            dalpha = ((alpha*(y - mu*np.log(1/(alpha + 1)) -
                              mu*(special.digamma(y + mu/alpha) -
                              special.digamma(mu/alpha) + 1)) -
                       mu*(np.log(1/(alpha + 1)) +
                           special.digamma(y + mu/alpha) -
                           special.digamma(mu/alpha)))/
                       (alpha**2*(alpha + 1))).sum()

        else: # nb2
            dparams = exog*a1 * (y-mu)/(mu+a1)
            da1 = -alpha**-2
            dalpha = (special.digamma(a1+y) - special.digamma(a1) + np.log(a1)
                        - np.log(a1+mu) - (a1+y)/(a1+mu) + 1).sum()*da1

        #multiply above by constant outside sum to reduce rounding error
        if self._transparams:
            return np.r_[dparams.sum(0), dalpha*alpha]
        else:
            return np.r_[dparams.sum(0), dalpha]

    def _score_nb1(self, params):
        return self._score_nbin(params, Q=1)

    def _hessian_geom(self, params):
        exog = self.exog
        y = self.endog[:,None]
        mu = self.predict(params)[:,None]

        # for dl/dparams dparams
        dim = exog.shape[1]
        hess_arr = np.empty((dim, dim))
        const_arr = mu*(1+y)/(mu+1)**2
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i,j] = np.sum(-exog[:,i,None] * exog[:,j,None] *
                                       const_arr, axis=0)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]
        return hess_arr


    def _hessian_nb1(self, params):
        """
        Hessian of NB1 model.
        """
        if self._transparams: # lnalpha came in during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]

        params = params[:-1]
        exog = self.exog
        y = self.endog[:,None]
        mu = self.predict(params)[:,None]

        a1 = mu/alpha

        # for dl/dparams dparams
        dim = exog.shape[1]
        hess_arr = np.empty((dim+1,dim+1))
        #const_arr = a1*mu*(a1+y)/(mu+a1)**2
        # not all of dparams
        dparams = exog/alpha*(np.log(1/(alpha + 1)) +
                              special.digamma(y + mu/alpha) -
                              special.digamma(mu/alpha))

        dmudb = exog*mu
        xmu_alpha = exog*mu/alpha
        trigamma = (special.polygamma(1, mu/alpha + y) -
                    special.polygamma(1, mu/alpha))
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i,j] = np.sum(dparams[:,i,None] * dmudb[:,j,None] +
                                 xmu_alpha[:,i,None] * xmu_alpha[:,j,None] *
                                 trigamma, axis=0)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]

        # for dl/dparams dalpha
        da1 = -alpha**-2
        dldpda = np.sum(-mu/alpha * dparams + exog*mu/alpha *
                        (-trigamma*mu/alpha**2 - 1/(alpha+1)), axis=0)

        hess_arr[-1,:-1] = dldpda
        hess_arr[:-1,-1] = dldpda

        # for dl/dalpha dalpha
        digamma_part = (special.digamma(y + mu/alpha) -
                        special.digamma(mu/alpha))

        log_alpha = np.log(1/(alpha+1))
        alpha3 = alpha**3
        alpha2 = alpha**2
        mu2 = mu**2
        dada = ((alpha3*mu*(2*log_alpha + 2*digamma_part + 3) -
                2*alpha3*y + alpha2*mu2*trigamma +
                4*alpha2*mu*(log_alpha + digamma_part) +
                alpha2 * (2*mu - y) +
                2*alpha*mu2*trigamma +
                2*alpha*mu*(log_alpha + digamma_part) +
                mu2*trigamma)/(alpha**4*(alpha2 + 2*alpha + 1)))
        hess_arr[-1,-1] = dada.sum()

        return hess_arr

    def _hessian_nb2(self, params):
        """
        Hessian of NB2 model.
        """
        if self._transparams: # lnalpha came in during fit
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        a1 = 1/alpha
        params = params[:-1]

        exog = self.exog
        y = self.endog[:,None]
        mu = self.predict(params)[:,None]

        # for dl/dparams dparams
        dim = exog.shape[1]
        hess_arr = np.empty((dim+1,dim+1))
        const_arr = a1*mu*(a1+y)/(mu+a1)**2
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i,j] = np.sum(-exog[:,i,None] * exog[:,j,None] *
                                       const_arr, axis=0)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]

        # for dl/dparams dalpha
        da1 = -alpha**-2
        dldpda = np.sum(mu*exog*(y-mu)*da1/(mu+a1)**2 , axis=0)
        hess_arr[-1,:-1] = dldpda
        hess_arr[:-1,-1] = dldpda

        # for dl/dalpha dalpha
        #NOTE: polygamma(1,x) is the trigamma function
        da2 = 2*alpha**-3
        dalpha = da1 * (special.digamma(a1+y) - special.digamma(a1) +
                    np.log(a1) - np.log(a1+mu) - (a1+y)/(a1+mu) + 1)
        dada = (da2 * dalpha/da1 + da1**2 * (special.polygamma(1, a1+y) -
                    special.polygamma(1, a1) + 1/a1 - 1/(a1 + mu) +
                    (y - mu)/(mu + a1)**2)).sum()
        hess_arr[-1,-1] = dada

        return hess_arr

    #TODO: replace this with analytic where is it used?
    def score_obs(self, params):
        sc = approx_fprime_cs(params, self.loglikeobs)
        return sc

    jac = np.deprecate(score_obs, 'jac', 'score_obs', "Use score_obs method."
                       " jac will be removed in 0.7")

    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):

        # Note: don't let super handle robust covariance because it has
        # transformed params

        if self.loglike_method.startswith('nb') and method not in ['newton',
                                                                   'ncg']:
            self._transparams = True # in case same Model instance is refit
        elif self.loglike_method.startswith('nb'): # method is newton/ncg
            self._transparams = False # because we need to step in alpha space

        if start_params is None:
            # Use poisson fit as first guess.
            #TODO, Warning: this assumes exposure is logged
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            start_params = mod_poi.fit(disp=0).params
            if self.loglike_method.startswith('nb'):
                start_params = np.append(start_params, 0.1)
        mlefit = super(NegativeBinomial, self).fit(start_params=start_params,
                        maxiter=maxiter, method=method, disp=disp,
                        full_output=full_output, callback=lambda x:x,
                        **kwargs)
                        # TODO: Fix NBin _check_perfect_pred
        if self.loglike_method.startswith('nb'):
            # mlefit is a wrapped counts results
            self._transparams = False # don't need to transform anymore now
            # change from lnalpha to alpha
            if method not in ["newton", "ncg"]:
                mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])

            nbinfit = NegativeBinomialResults(self, mlefit._results)
            result = NegativeBinomialResultsWrapper(nbinfit)
        else:
            result = mlefit

        if cov_kwds is None:
            cov_kwds = {}  #TODO: make this unnecessary ?
        result._get_robustcov_results(cov_type=cov_type,
                                    use_self=True, use_t=use_t, **cov_kwds)
        return result


    def fit_regularized(self, start_params=None, method='l1',
            maxiter='defined_by_method', full_output=1, disp=1, callback=None,
            alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):

        if self.loglike_method.startswith('nb') and (np.size(alpha) == 1 and
                                                     alpha != 0):
            # don't penalize alpha if alpha is scalar
            k_params = self.exog.shape[1] + self.k_extra
            alpha = alpha * np.ones(k_params)
            alpha[-1] = 0

        # alpha for regularized poisson to get starting values
        alpha_p = alpha[:-1] if (self.k_extra and np.size(alpha) > 1) else alpha

        self._transparams = False
        if start_params is None:
            # Use poisson fit as first guess.
            #TODO, Warning: this assumes exposure is logged
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            start_params = mod_poi.fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=0, callback=callback,
                alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
            if self.loglike_method.startswith('nb'):
                start_params = np.append(start_params, 0.1)

        cntfit = super(CountModel, self).fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        if method in ['l1', 'l1_cvxopt_cp']:
            discretefit = L1NegativeBinomialResults(self, cntfit)
        else:
            raise Exception(
                    "argument method == %s, which is not handled" % method)

        return L1NegativeBinomialResultsWrapper(discretefit)


### Results Class ###

class DiscreteResults(base.LikelihoodModelResults):
    __doc__ = _discrete_results_docs % {"one_line_description" :
        "A results class for the discrete dependent variable models.",
        "extra_attr" : ""}

    def __init__(self, model, mlefit, cov_type='nonrobust', cov_kwds=None,
                 use_t=None):
        #super(DiscreteResults, self).__init__(model, params,
        #        np.linalg.inv(-hessian), scale=1.)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self._cache = resettable_cache()
        self.nobs = model.exog.shape[0]
        self.__dict__.update(mlefit.__dict__)

        if not hasattr(self, 'cov_type'):
            # do this only if super, i.e. mlefit didn't already add cov_type
            # robust covariance
            if use_t is not None:
                self.use_t = use_t
            if cov_type == 'nonrobust':
                self.cov_type = 'nonrobust'
                self.cov_kwds = {'description' : 'Standard Errors assume that the ' +
                                 'covariance matrix of the errors is correctly ' +
                                 'specified.'}
            else:
                if cov_kwds is None:
                    cov_kwds = {}
                from statsmodels.base.covtype import get_robustcov_results
                get_robustcov_results(self, cov_type=cov_type, use_self=True,
                                           **cov_kwds)



    def __getstate__(self):
        try:
            #remove unpicklable callback
            self.mle_settings['callback'] = None
        except (AttributeError, KeyError):
            pass
        return self.__dict__

    @cache_readonly
    def prsquared(self):
        return 1 - self.llf/self.llnull

    @cache_readonly
    def llr(self):
        return -2*(self.llnull - self.llf)

    @cache_readonly
    def llr_pvalue(self):
        return stats.chisqprob(self.llr, self.df_model)

    @cache_readonly
    def llnull(self):

        model = self.model
        kwds = model._get_init_kwds()
        # TODO: what parameters to pass to fit?
        mod_null = model.__class__(model.endog, np.ones(self.nobs), **kwds)
        # TODO: consider catching and warning on convergence failure?
        # in the meantime, try hard to converge. see
        # TestPoissonConstrained1a.test_smoke
        res_null = mod_null.fit(disp=0, warn_convergence=False,
                                maxiter=10000)
        return res_null.llf

    @cache_readonly
    def fittedvalues(self):
        return np.dot(self.model.exog, self.params[:self.model.exog.shape[1]])

    @cache_readonly
    def aic(self):
        return -2*(self.llf - (self.df_model+1))

    @cache_readonly
    def bic(self):
        return -2*self.llf + np.log(self.nobs)*(self.df_model+1)

    def _get_endog_name(self, yname, yname_list):
        if yname is None:
            yname = self.model.endog_names
        if yname_list is None:
            yname_list = self.model.endog_names
        return yname, yname_list

    def get_margeff(self, at='overall', method='dydx', atexog=None,
            dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available from the returned object.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semielasticity -- dy/d(lnx)
            - 'eydx' - estimate semeilasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables.
        atexog : array-like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        DiscreteMargins : marginal effects instance
            Returns an object that holds the marginal effects, standard
            errors, confidence intervals, etc. See
            `statsmodels.discrete.discrete_margins.DiscreteMargins` for more
            information.

        Notes
        -----
        When using after Poisson, returns the expected number of events
        per period, assuming that the model is loglinear.
        """
        from statsmodels.discrete.discrete_margins import DiscreteMargins
        return DiscreteMargins(self, (at, method, atexog, dummy, count))

    def summary(self, yname=None, xname=None, title=None, alpha=.05,
                yname_list=None):
        """Summarize the Regression Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `var_##` for ## in p the number of regressors
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        top_left = [('Dep. Variable:', None),
                     ('Model:', [self.model.__class__.__name__]),
                     ('Method:', ['MLE']),
                     ('Date:', None),
                     ('Time:', None),
                     #('No. iterations:', ["%d" % self.mle_retvals['iterations']]),
                     ('converged:', ["%s" % self.mle_retvals['converged']])
                      ]

        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ('Pseudo R-squ.:', ["%#6.4g" % self.prsquared]),
                     ('Log-Likelihood:', None),
                     ('LL-Null:', ["%#8.5g" % self.llnull]),
                     ('LLR p-value:', ["%#6.4g" % self.llr_pvalue])
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        #boiler plate
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        yname, yname_list = self._get_endog_name(yname, yname_list)
        # for top of table
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, #[],
                          yname=yname, xname=xname, title=title)
        # for parameters, etc
        smry.add_table_params(self, yname=yname_list, xname=xname, alpha=alpha,
                             use_t=self.use_t)

        if hasattr(self, 'constraints'):
            smry.add_extra_txt(['Model has been estimated subject to linear '
                          'equality constraints.'])

        #diagnostic table not used yet
        #smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
        #                   yname=yname, xname=xname,
        #                   title="")
        return smry

    def summary2(self, yname=None, xname=None, title=None, alpha=.05,
            float_format="%.4f"):
        """Experimental function to summarize regression results

        Parameters
        -----------
        xname : List of strings of length equal to the number of parameters
            Names of the independent variables (optional)
        yname : string
            Name of the dependent variable (optional)
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals
        float_format: string
            print format for floats in parameters summary

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """
        # Summary
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_base(results=self, alpha=alpha, float_format=float_format,
                xname=xname, yname=yname, title=title)

        if hasattr(self, 'constraints'):
            smry.add_text('Model has been estimated subject to linear '
                          'equality constraints.')

        return smry



class CountResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {
                    "one_line_description" : "A results class for count data",
                    "extra_attr" : ""}
    @cache_readonly
    def resid(self):
        """
        Residuals

        Notes
        -----
        The residuals for Count models are defined as

        .. math:: y - p

        where :math:`p = \\exp(X\\beta)`. Any exposure and offset variables
        are also handled.
        """
        return self.model.endog - self.predict()

class NegativeBinomialResults(CountResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for NegativeBinomial 1 and 2",
                    "extra_attr" : ""}

    @cache_readonly
    def lnalpha(self):
        return np.log(self.params[-1])

    @cache_readonly
    def lnalpha_std_err(self):
        return self.bse[-1] / self.params[-1]

    @cache_readonly
    def aic(self):
        # + 1 because we estimate alpha
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2*(self.llf - (self.df_model + self.k_constant + k_extra))

    @cache_readonly
    def bic(self):
        # + 1 because we estimate alpha
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2*self.llf + np.log(self.nobs)*(self.df_model +
                                                self.k_constant + k_extra)

class L1CountResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {"one_line_description" :
            "A results class for count data fit by l1 regularization",
            "extra_attr" : _l1_results_attr}
        #discretefit = CountResults(self, cntfit)

    def __init__(self, model, cntfit):
        super(L1CountResults, self).__init__(model, cntfit)
        # self.trimmed is a boolean array with T/F telling whether or not that
        # entry in params has been set zero'd out.
        self.trimmed = cntfit.mle_retvals['trimmed']
        self.nnz_params = (self.trimmed == False).sum()
        # update degrees of freedom
        self.model.df_model = self.nnz_params - 1
        self.model.df_resid = float(self.model.endog.shape[0] - self.nnz_params)
        # adjust for extra parameter in NegativeBinomial nb1 and nb2
        # extra parameter is not included in df_model
        k_extra = getattr(self.model, 'k_extra', 0)
        self.model.df_model -= k_extra
        self.model.df_resid += k_extra
        self.df_model = self.model.df_model
        self.df_resid = self.model.df_resid

class PoissonResults(CountResults):
    def predict_prob(self, n=None, exog=None, exposure=None, offset=None,
                     transform=True):
        """
        Return predicted probability of each count level for each observation

        Parameters
        ----------
        n : array-like or int
            The counts for which you want the probabilities. If n is None
            then the probabilities for each count from 0 to max(y) are
            given.

        Returns
        -------
        ndarray
            A nobs x n array where len(`n`) columns are indexed by the count
            n. If n is None, then column 0 is the probability that each
            observation is 0, column 1 is the probability that each
            observation is 1, etc.
        """
        if n is not None:
            counts = np.atleast_2d(n)
        else:
            counts = np.atleast_2d(np.arange(0, np.max(self.model.endog)+1))
        mu = self.predict(exog=exog, exposure=exposure, offset=offset,
                          transform=transform, linear=False)[:,None]
        # uses broadcasting
        return stats.poisson.pmf(counts, mu)

class L1PoissonResults(L1CountResults, PoissonResults):
    pass

class L1NegativeBinomialResults(L1CountResults, NegativeBinomialResults):
    pass

class OrderedResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {"one_line_description" : "A results class for ordered discrete data." , "extra_attr" : ""}
    pass

class BinaryResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {"one_line_description" : "A results class for binary data", "extra_attr" : ""}

    def pred_table(self, threshold=.5):
        """
        Prediction table

        Parameters
        ----------
        threshold : scalar
            Number between 0 and 1. Threshold above which a prediction is
            considered 1 and below which a prediction is considered 0.

        Notes
        ------
        pred_table[i,j] refers to the number of times "i" was observed and
        the model predicted "j". Correct predictions are along the diagonal.
        """
        model = self.model
        actual = model.endog
        pred = np.array(self.predict() > threshold, dtype=float)
        bins = np.array([0, 0.5, 1])
        return np.histogram2d(actual, pred, bins=bins)[0]


    def summary(self, yname=None, xname=None, title=None, alpha=.05,
                yname_list=None):
        smry = super(BinaryResults, self).summary(yname, xname, title, alpha,
                     yname_list)
        fittedvalues = self.model.cdf(self.fittedvalues)
        absprederror = np.abs(self.model.endog - fittedvalues)
        predclose_sum = (absprederror < 1e-4).sum()
        predclose_frac = predclose_sum / len(fittedvalues)

        #add warnings/notes
        etext = []
        if predclose_sum == len(fittedvalues): #nobs?
            wstr = "Complete Separation: The results show that there is"
            wstr += "complete separation.\n"
            wstr += "In this case the Maximum Likelihood Estimator does "
            wstr += "not exist and the parameters\n"
            wstr += "are not identified."
            etext.append(wstr)
        elif predclose_frac > 0.1:  # TODO: get better diagnosis
            wstr = "Possibly complete quasi-separation: A fraction "
            wstr += "%4.2f of observations can be\n" % predclose_frac
            wstr += "perfectly predicted. This might indicate that there "
            wstr += "is complete\nquasi-separation. In this case some "
            wstr += "parameters will not be identified."
            etext.append(wstr)
        if etext:
            smry.add_extra_txt(etext)
        return smry
    summary.__doc__ = DiscreteResults.summary.__doc__

    @cache_readonly
    def resid_dev(self):
        """
        Deviance residuals

        Notes
        -----
        Deviance residuals are defined

        .. math:: d_j = \\pm\\left(2\\left[Y_j\\ln\\left(\\frac{Y_j}{M_jp_j}\\right) + (M_j - Y_j\\ln\\left(\\frac{M_j-Y_j}{M_j(1-p_j)} \\right) \\right] \\right)^{1/2}

        where

        :math:`p_j = cdf(X\\beta)` and :math:`M_j` is the total number of
        observations sharing the covariate pattern :math:`j`.

        For now :math:`M_j` is always set to 1.
        """
        #These are the deviance residuals
        #model = self.model
        endog = self.model.endog
        #exog = model.exog
        # M = # of individuals that share a covariate pattern
        # so M[i] = 2 for i = two share a covariate pattern
        M = 1
        p = self.predict()
        #Y_0 = np.where(exog == 0)
        #Y_M = np.where(exog == M)
        #NOTE: Common covariate patterns are not yet handled
        res = -(1-endog)*np.sqrt(2*M*np.abs(np.log(1-p))) + \
                endog*np.sqrt(2*M*np.abs(np.log(p)))
        return res

    @cache_readonly
    def resid_pearson(self):
        """
        Pearson residuals

        Notes
        -----
        Pearson residuals are defined to be

        .. math:: r_j = \\frac{(y - M_jp_j)}{\\sqrt{M_jp_j(1-p_j)}}

        where :math:`p_j=cdf(X\\beta)` and :math:`M_j` is the total number of
        observations sharing the covariate pattern :math:`j`.

        For now :math:`M_j` is always set to 1.
        """
        # Pearson residuals
        #model = self.model
        endog = self.model.endog
        #exog = model.exog
        # M = # of individuals that share a covariate pattern
        # so M[i] = 2 for i = two share a covariate pattern
        # use unique row pattern?
        M = 1
        p = self.predict()
        return (endog - M*p)/np.sqrt(M*p*(1-p))

    @cache_readonly
    def resid_response(self):
        """
        The response residuals

        Notes
        -----
        Response residuals are defined to be

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`.
        """
        return self.model.endog - self.predict()

class LogitResults(BinaryResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for Logit Model",
                    "extra_attr" : ""}
    @cache_readonly
    def resid_generalized(self):
        """
        Generalized residuals

        Notes
        -----
        The generalized residuals for the Logit model are defined

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`. This is the same as the `resid_response`
        for the Logit model.
        """
        # Generalized residuals
        return self.model.endog - self.predict()

class ProbitResults(BinaryResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for Probit Model",
                    "extra_attr" : ""}
    @cache_readonly
    def resid_generalized(self):
        """
        Generalized residuals

        Notes
        -----
        The generalized residuals for the Probit model are defined

        .. math:: y\\frac{\phi(X\\beta)}{\\Phi(X\\beta)}-(1-y)\\frac{\\phi(X\\beta)}{1-\\Phi(X\\beta)}
        """
        # generalized residuals
        model = self.model
        endog = model.endog
        XB = self.predict(linear=True)
        pdf = model.pdf(XB)
        cdf = model.cdf(XB)
        return endog * pdf/cdf - (1-endog)*pdf/(1-cdf)

class L1BinaryResults(BinaryResults):
    __doc__ = _discrete_results_docs % {"one_line_description" :
    "Results instance for binary data fit by l1 regularization",
    "extra_attr" : _l1_results_attr}
    def __init__(self, model, bnryfit):
        super(L1BinaryResults, self).__init__(model, bnryfit)
        # self.trimmed is a boolean array with T/F telling whether or not that
        # entry in params has been set zero'd out.
        self.trimmed = bnryfit.mle_retvals['trimmed']
        self.nnz_params = (self.trimmed == False).sum()
        self.model.df_model = self.nnz_params - 1
        self.model.df_resid = float(self.model.endog.shape[0] - self.nnz_params)
        self.df_model = self.model.df_model
        self.df_resid = self.model.df_resid


class MultinomialResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {"one_line_description" :
            "A results class for multinomial data", "extra_attr" : ""}
    def _maybe_convert_ynames_int(self, ynames):
        # see if they're integers
        try:
            for i in ynames:
                if ynames[i] % 1 == 0:
                    ynames[i] = str(int(ynames[i]))
        except TypeError:
            pass
        return ynames

    def _get_endog_name(self, yname, yname_list, all=False):
        """
        If all is False, the first variable name is dropped
        """
        model = self.model
        if yname is None:
            yname = model.endog_names
        if yname_list is None:
            ynames = model._ynames_map
            ynames = self._maybe_convert_ynames_int(ynames)
            # use range below to ensure sortedness
            ynames = [ynames[key] for key in range(int(model.J))]
            ynames = ['='.join([yname, name]) for name in ynames]
            if not all:
                yname_list = ynames[1:] # assumes first variable is dropped
            else:
                yname_list = ynames
        return yname, yname_list

    def pred_table(self):
        """
        Returns the J x J prediction table.

        Notes
        -----
        pred_table[i,j] refers to the number of times "i" was observed and
        the model predicted "j". Correct predictions are along the diagonal.
        """
        ju = self.model.J - 1  # highest index
        # these are the actual, predicted indices
        #idx = lzip(self.model.endog, self.predict().argmax(1))
        bins = np.concatenate(([0], np.linspace(0.5, ju - 0.5, ju), [ju]))
        return np.histogram2d(self.model.endog, self.predict().argmax(1),
                              bins=bins)[0]

    @cache_readonly
    def bse(self):
        bse = np.sqrt(np.diag(self.cov_params()))
        return bse.reshape(self.params.shape, order='F')

    @cache_readonly
    def aic(self):
        return -2*(self.llf - (self.df_model+self.model.J-1))

    @cache_readonly
    def bic(self):
        return -2*self.llf + np.log(self.nobs)*(self.df_model+self.model.J-1)

    def conf_int(self, alpha=.05, cols=None):
        confint = super(DiscreteResults, self).conf_int(alpha=alpha,
                                                            cols=cols)
        return confint.transpose(2,0,1)

    def margeff(self):
        raise NotImplementedError("Use get_margeff instead")

    @cache_readonly
    def resid_misclassified(self):
        """
        Residuals indicating which observations are misclassified.

        Notes
        -----
        The residuals for the multinomial model are defined as

        .. math:: argmax(y_i) \\neq argmax(p_i)

        where :math:`argmax(y_i)` is the index of the category for the
        endogenous variable and :math:`argmax(p_i)` is the index of the
        predicted probabilities for each category. That is, the residual
        is a binary indicator that is 0 if the category with the highest
        predicted probability is the same as that of the observed variable
        and 1 otherwise.
        """
        # it's 0 or 1 - 0 for correct prediction and 1 for a missed one
        return (self.model.wendog.argmax(1) !=
                self.predict().argmax(1)).astype(float)

    def summary2(self, alpha=0.05, float_format="%.4f"):
        """Experimental function to summarize regression results

        Parameters
        -----------
        alpha : float
            significance level for the confidence intervals
        float_format: string
            print format for floats in parameters summary

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary
            results

        """

        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_dict(summary2.summary_model(self))
        # One data frame per value of endog
        eqn = self.params.shape[1]
        confint = self.conf_int(alpha)
        for i in range(eqn):
            coefs = summary2.summary_params(self, alpha, self.params[:,i],
                    self.bse[:,i], self.tvalues[:,i], self.pvalues[:,i],
                    confint[i])
            # Header must show value of endog
            level_str =  self.model.endog_names + ' = ' + str(i)
            coefs[level_str] = coefs.index
            coefs = coefs.ix[:,[-1,0,1,2,3,4,5]]
            smry.add_df(coefs, index=False, header=True, float_format=float_format)
            smry.add_title(results=self)
        return smry


class L1MultinomialResults(MultinomialResults):
    __doc__ = _discrete_results_docs % {"one_line_description" :
        "A results class for multinomial data fit by l1 regularization",
        "extra_attr" : _l1_results_attr}
    def __init__(self, model, mlefit):
        super(L1MultinomialResults, self).__init__(model, mlefit)
        # self.trimmed is a boolean array with T/F telling whether or not that
        # entry in params has been set zero'd out.
        self.trimmed = mlefit.mle_retvals['trimmed']
        self.nnz_params = (self.trimmed == False).sum()

        #Note: J-1 constants
        self.model.df_model = self.nnz_params - (self.model.J - 1)
        self.model.df_resid = float(self.model.endog.shape[0] - self.nnz_params)
        self.df_model = self.model.df_model
        self.df_resid = self.model.df_resid


#### Results Wrappers ####

class OrderedResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(OrderedResultsWrapper, OrderedResults)

class CountResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(CountResultsWrapper, CountResults)

class NegativeBinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(NegativeBinomialResultsWrapper,
                      NegativeBinomialResults)

class PoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
    #_methods = {
    #        "predict_prob" : "rows",
    #        }
    #_wrap_methods = lm.wrap.union_dicts(
    #                            lm.RegressionResultsWrapper._wrap_methods,
    #                            _methods)
wrap.populate_wrapper(PoissonResultsWrapper, PoissonResults)

class L1CountResultsWrapper(lm.RegressionResultsWrapper):
    pass

class L1PoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
    #_methods = {
    #        "predict_prob" : "rows",
    #        }
    #_wrap_methods = lm.wrap.union_dicts(
    #                            lm.RegressionResultsWrapper._wrap_methods,
    #                            _methods)
wrap.populate_wrapper(L1PoissonResultsWrapper, L1PoissonResults)

class L1NegativeBinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1NegativeBinomialResultsWrapper,
                      L1NegativeBinomialResults)

class BinaryResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {"resid_dev" : "rows",
              "resid_generalized" : "rows",
              "resid_pearson" : "rows",
              "resid_response" : "rows"
              }
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs,
                                   _attrs)
wrap.populate_wrapper(BinaryResultsWrapper, BinaryResults)

class L1BinaryResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1BinaryResultsWrapper, L1BinaryResults)

class MultinomialResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {"resid_misclassified" : "rows"}
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs,
            _attrs)
wrap.populate_wrapper(MultinomialResultsWrapper, MultinomialResults)

class L1MultinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1MultinomialResultsWrapper, L1MultinomialResults)


if __name__=="__main__":
    import numpy as np
    import statsmodels.api as sm
# Scratch work for negative binomial models
# dvisits was written using an R package, I can provide the dataset
# on request until the copyright is cleared up
#TODO: request permission to use dvisits
    data2 = np.genfromtxt('../datasets/dvisits/dvisits.csv', names=True)
# note that this has missing values for Accident
    endog = data2['doctorco']
    exog = data2[['sex','age','agesq','income','levyplus','freepoor',
            'freerepa','illness','actdays','hscore','chcond1',
            'chcond2']].view(float, np.ndarray).reshape(len(data2),-1)
    exog = sm.add_constant(exog, prepend=True)
    poisson_mod = Poisson(endog, exog)
    poisson_res = poisson_mod.fit()
#    nb2_mod = NegBinTwo(endog, exog)
#    nb2_res = nb2_mod.fit()
# solvers hang (with no error and no maxiter warn...)
# haven't derived hessian (though it will be block diagonal) to check
# newton, note that Lawless (1987) has the derivations
# appear to be something wrong with the score?
# according to Lawless, traditionally the likelihood is maximized wrt to B
# and a gridsearch on a to determin ahat?
# or the Breslow approach, which is 2 step iterative.
    nb2_params = [-2.190,.217,-.216,.609,-.142,.118,-.497,.145,.214,.144,
            .038,.099,.190,1.077] # alpha is last
    # taken from Cameron and Trivedi
# the below is from Cameron and Trivedi as well
#    endog2 = np.array(endog>=1, dtype=float)
# skipped for now, binary poisson results look off?
    data = sm.datasets.randhie.load()
    nbreg = NegativeBinomial
    mod = nbreg(data.endog, data.exog.view((float,9)))
#FROM STATA:
    params = np.asarray([-.05654133,  -.21214282, .0878311, -.02991813, .22903632,
            .06210226, .06799715, .08407035, .18532336])
    bse = [0.0062541, 0.0231818, 0.0036942, 0.0034796, 0.0305176, 0.0012397,
            0.0198008, 0.0368707, 0.0766506]
    lnalpha = .31221786
    mod.loglike(np.r_[params,np.exp(lnalpha)])
    poiss_res = Poisson(data.endog, data.exog.view((float,9))).fit()
    func = lambda x: -mod.loglike(x)
    grad = lambda x: -mod.score(x)
    from scipy import optimize
#    res1 = optimize.fmin_l_bfgs_b(func, np.r_[poiss_res.params,.1],
#                        approx_grad=True)
    res1 = optimize.fmin_bfgs(func, np.r_[poiss_res.params,.1], fprime=grad)
    from statsmodels.tools.numdiff import approx_hess_cs
#    np.sqrt(np.diag(-np.linalg.inv(approx_hess_cs(np.r_[params,lnalpha], mod.loglike))))
#NOTE: this is the hessian in terms of alpha _not_ lnalpha
    hess_arr = mod.hessian(res1)
