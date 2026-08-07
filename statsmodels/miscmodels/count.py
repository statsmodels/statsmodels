"""
Generic maximum likelihood versions of the Poisson model

Created on Mon Jul 26 08:34:59 2010

Author: josef-pktd

Notes
-----
Changes: added offset and zero-inflated version of Poisson. Kind of ok,
need better test cases; a nan appears in the ZIP bse, need to check the
Hessian calculations; found an error in the ZIP loglike; all tests pass
with the current implementation.

Known issues:

* If the true model is not zero-inflated then the numerical Hessian for
  ZIP has zeros for the inflation probability and is not invertible.
  The Hessian inverts and bse look ok if the corresponding row and
  column are dropped; pinv also works.
* GenericMLE: still get somewhere (where?)
  "CacheWriteWarning: The attribute 'bse' cannot be overwritten"
* bfgs is too fragile, does not come back
* `nm` is slow but seems to work
* need good start_params and their use in genericmle needs to be checked
  for consistency, set as attribute or method (called as attribute)
* numerical hessian needs better scaling
* check taking parts out of the loop, e.g. factorial(endog) could be
  precalculated
"""
import numpy as np
from scipy import stats
from scipy.special import factorial

from statsmodels.base.model import GenericLikelihoodModel


def maxabs(arr1, arr2):
    return np.max(np.abs(arr1 - arr2))


def maxabsrel(arr1, arr2):
    return np.max(np.abs(arr2 / arr1 - 1))


class PoissonGMLE(GenericLikelihoodModel):
    """
    Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.
    """

    # copied from discretemod.Poisson
    def nloglikeobs(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        ndarray
            The negative log likelihood of the model evaluated at `params`
            for each observation.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        XB = np.dot(self.exog, params)
        endog = self.endog
        return np.exp(XB) - endog*XB + np.log(factorial(endog))

    def predict_distribution(self, exog):
        """
        Return frozen scipy.stats distribution with mu at estimated prediction

        Parameters
        ----------
        exog : array_like
            Explanatory variables used to construct the predicted mean.

        Returns
        -------
        rv_frozen
            A frozen `scipy.stats.poisson` distribution with `mu` set to
            the predicted mean at `exog`.
        """
        if not hasattr(self, "result"):
            # TODO: why would this be ValueError instead of AttributeError?
            # TODO: Why even make this a Model attribute in the first place?
            #  It belongs on the Results class
            raise ValueError
        else:
            result = self.result
            params = result.params
            mu = np.exp(np.dot(exog, params))
            return stats.poisson(mu, loc=0)


class PoissonOffsetGMLE(GenericLikelihoodModel):
    """
    Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson but adds offset

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    Parameters
    ----------
    endog : array_like
        1-d endogenous response variable. The dependent variable.
    exog : array_like, optional
        A nobs x k array where `nobs` is the number of observations and
        `k` is the number of regressors.
    offset : array_like, optional
        Offset added to the linear predictor before computing the mean.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no
        nan checking is done. If 'drop', any observations with nans are
        dropped. If 'raise', an error is raised. Default is 'none'.
    **kwds
        Extra keyword arguments passed to the model.
    """

    def __init__(self, endog, exog=None, offset=None, missing="none", **kwds):
        # let them be none in case user wants to use inheritance
        if offset is not None:
            if offset.ndim == 1:
                offset = offset[:, None]  # need column
            self.offset = offset.ravel()
        else:
            self.offset = 0.
        super().__init__(endog, exog, missing=missing, **kwds)

# this was added temporarily for bug-hunting, but should not be needed
#    def loglike(self, params):
#        return -self.nloglikeobs(params).sum(0)

    # original copied from discretemod.Poisson
    def nloglikeobs(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        ndarray
            The negative log likelihood of the model evaluated at `params`
            for each observation.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """

        XB = self.offset + np.dot(self.exog, params)
        endog = self.endog
        nloglik = np.exp(XB) - endog*XB + np.log(factorial(endog))
        return nloglik


class PoissonZiGMLE(GenericLikelihoodModel):
    """
    Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same statistical model
    as discretemod.Poisson but adds offset and zero-inflation.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    There are numerical problems if there is no zero-inflation.

    Parameters
    ----------
    endog : array_like
        1-d endogenous response variable. The dependent variable.
    exog : array_like, optional
        A nobs x k array where `nobs` is the number of observations and
        `k` is the number of regressors. If None, a column of ones is
        used.
    offset : array_like, optional
        Offset added to the linear predictor before computing the mean.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no
        nan checking is done. If 'drop', any observations with nans are
        dropped. If 'raise', an error is raised. Default is 'none'.
    **kwds
        Extra keyword arguments passed to the model.
    """

    def __init__(self, endog, exog=None, offset=None, missing="none", **kwds):
        # let them be none in case user wants to use inheritance
        self.k_extra = 1
        super().__init__(
            endog, exog, missing=missing, extra_params_names=["zi"], **kwds
        )
        if offset is not None:
            if offset.ndim == 1:
                offset = offset[:, None]  # need column
            self.offset = offset.ravel()  # which way?
        else:
            self.offset = 0.

        # TODO: it's not standard pattern to use default exog
        if exog is None:
            self.exog = np.ones((self.nobs, 1))
        self.nparams = self.exog.shape[1]
        # what's the shape in regression for exog if only constant
        self.start_params = np.hstack((np.ones(self.nparams), 0))
        # need to add zi params to nparams
        self.nparams += 1
        self.cloneattr = ["start_params"]
        # needed for t_test and summary
        # Note: no added to super __init__ which also adjusts df_resid
        # self.exog_names.append('zi')

    # original copied from discretemod.Poisson
    def nloglikeobs(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        ndarray
            The negative log likelihood of the model evaluated at `params`
            for each observation.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        beta = params[:-1]
        gamm = 1 / (1 + np.exp(params[-1]))  # check this
        # replace with np.dot(self.exogZ, gamma)
        # print(np.shape(self.offset), self.exog.shape, beta.shape
        XB = self.offset + np.dot(self.exog, beta)
        endog = self.endog
        nloglik = -np.log(1-gamm) + np.exp(XB) - endog*XB + np.log(factorial(endog))
        nloglik[endog == 0] = - np.log(gamm + np.exp(-nloglik[endog == 0]))

        return nloglik
